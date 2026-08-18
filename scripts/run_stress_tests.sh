#!/usr/bin/env bash
# Builds and runs rampipe's stress-test harness (all scenarios in
# examples/stress_test_scenarios.yaml) against a rampiped daemon on
# whichever machine this runs on. No absolute paths -- it locates the
# repo root relative to its own location, so it works from a checkout
# anywhere as long as it stays at <repo>/scripts/run_stress_tests.sh.
# No cross-machine orchestration either: run it by hand, once per
# machine you want to test (laptop, workstation, wherever else).
#
# Auto-detects the right rampiped build: CUDA on Linux if
# /usr/local/cuda is present (setting up its PATH/LD_LIBRARY_PATH for
# this invocation), Metal/CPU everywhere else via the plain llama
# feature (Metal is auto-enabled on macOS with no separate flag needed).
#
# Usage:
#   ./scripts/run_stress_tests.sh
#
# Starts a rampiped daemon only if one isn't already running on
# /tmp/rampiped.sock -- reuses one that is, so repeated invocations
# don't pay to reload already-resident models. Doesn't stop the daemon
# it starts -- kill it yourself
# (pkill -f 'rampiped --socket /tmp/rampiped.sock') for a clean-state
# rerun instead of reusing residency from a previous run.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$SCRIPT_DIR")"
SOCKET="/tmp/rampiped.sock"

cd "$REPO_ROOT"

# Belt-and-suspenders: a plain interactive shell normally already has
# this on PATH via .bashrc/.zshrc, but a non-interactive invocation
# (e.g. `ssh host ./scripts/run_stress_tests.sh`) may not.
if ! command -v cargo > /dev/null 2>&1 && [ -f "$HOME/.cargo/env" ]; then
    # shellcheck disable=SC1091
    source "$HOME/.cargo/env"
fi

RAMPIPED_FEATURES="llama"
if [ "$(uname -s)" = "Linux" ] && [ -d /usr/local/cuda ]; then
    echo "--- CUDA toolkit found, building with GPU support ---"
    export PATH="/usr/local/cuda/bin:$PATH"
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
    RAMPIPED_FEATURES="cuda"
fi

echo "--- building rampiped (--features $RAMPIPED_FEATURES) ---"
cargo build --release --features "$RAMPIPED_FEATURES" --bin rampiped

echo "--- building stress-test harness ---"
cargo build --release --features client --example stress_test

if ! pgrep -f "target/release/rampiped --socket $SOCKET" > /dev/null; then
    echo "--- starting rampiped (not already running) ---"
    rm -f "$SOCKET"
    nohup ./target/release/rampiped --socket "$SOCKET" > /tmp/rampiped.log 2>&1 &
    disown
    sleep 3
else
    echo "--- rampiped already running on $SOCKET, reusing it ---"
fi

echo "--- stress_test (all scenarios) ---"
./target/release/examples/stress_test "$SOCKET"

echo "=== done -- output in $REPO_ROOT/stress-test-output ==="
