#!/usr/bin/env bash
# Runs `examples/prompt_harness.rs` against the same prompt, once per
# model candidate and once per sampling seed, so an overnight batch
# produces one file per (model, seed) pair to read/diff in the morning
# — comparing models against each other, and comparing each model's own
# attempt-to-attempt variance under the same seeds real taskpipe retries
# use, without needing a real taskpipe run (worktree, git checkout,
# cargo build/test) per attempt.
#
# Deliberately sequential, never parallel: two local models loaded into
# Metal at once has already caused a real, live "Insufficient Memory"
# (kIOGPUCommandBufferCallbackErrorOutOfMemory) failure in this project
# — see rampipe/README or taskpipe's own session history if that's ever
# forgotten and someone's tempted to `&`-background these calls.
#
# A failed or timed-out run (model load error, decode error, hung
# generation) is logged and skipped, not fatal to the rest of the sweep
# — the whole point of an overnight run is coming back to whatever
# actually finished, not an empty output directory because model 3 of 7
# hung at 2am.
#
# Usage:
#   scripts/model_sweep.sh --prompt <file> [--out <dir>] [--models <comma-list>]
#       [--seeds <comma-list>] [--repo-dir <path>] [--timeout <secs>]
#
#   --prompt <file>     Required. Same prompt file `prompt_harness` takes.
#   --out <dir>         Default: sweep-<UTC timestamp> in the current directory.
#   --models <list>     Default: every candidate in prompt_harness.rs's own
#                        MODEL_CANDIDATES table. Comma-separated, e.g.
#                        "giladgd_q4km,jamba_mini_1_7_q3km".
#   --seeds <list>       Default: "greedy,4202,4203,4204,4205" — matches
#                        LocalBackend::sampling_for_attempt's real 5-attempt
#                        progression (greedy first, then those 4 fixed
#                        seeds), so each model's directory mirrors what a
#                        real taskpipe run would actually try, in order.
#   --repo-dir <path>   Passed through to prompt_harness's own --repo-dir —
#                        checks AlreadyExists against a real repo's src/*.rs
#                        for multi-file (FILE:-marker) prompts. Omit for a
#                        single-file prompt, or when that check doesn't apply.
#   --timeout <secs>    Per-run wall-clock cap. Default: 900 (15 min) — a
#                        genuinely stuck decode shouldn't eat the whole night.
#
# Example (the exact draft-1 comparison this was written for):
#   scripts/model_sweep.sh \
#       --prompt ~/draft1_prompt.txt \
#       --repo-dir ~/projects/rust/chronopipe \
#       --out ~/draft1_sweep \
#       --models giladgd_q4km,jamba_mini_1_7_q3km
#
# Output layout:
#   $OUT/manifest.log            one line per run: model, seed, exit code,
#                                 duration, timestamp — tail -f this while
#                                 it's running for live progress
#   $OUT/<model>/greedy.txt      full harness stdout (raw response, stats,
#   $OUT/<model>/seed_4202.txt   extraction report) for that (model, seed)
#   ...

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAMPIPE_DIR="$(dirname "$SCRIPT_DIR")"
HARNESS_BIN="$RAMPIPE_DIR/target/release/examples/prompt_harness"

# Used when `--models` isn't passed. Comment a line out (don't delete
# it) to skip that candidate without losing track of it — real, live
# numbers as of 2026-08-11 on this machine: ~62 GiB free, and
# unsloth_q5km/unsloth_q6k are the only two of the seven real
# LOCAL_MODEL_CANDIDATES not already in the local HF cache (~20.2 GiB
# and ~23.4 GiB respectively — same Qwen3-Coder-30B weights as the
# already-cached unsloth_q4km/giladgd_q4km, just bigger quants, not a
# different model). Downloading both at once would leave only ~18 GiB
# free afterward — tight enough to risk normal system operation
# alongside whatever else runs overnight. The five active below cover
# four genuinely distinct model families with zero new downloads;
# uncomment either commented-out line once there's confirmed headroom,
# or to specifically test whether a bigger quant changes the result for
# the Qwen3-Coder-30B family already covered by q4km.
DEFAULT_MODELS=(
    "unsloth_q4km"
    "giladgd_q4km"
    "devstral_small_2507_q4km"
    "qwen3_6_35b_a3b_q4km"
    "jamba_mini_1_7_q3km"
    # "unsloth_q5km"              # ~20.2 GiB fresh download, not cached as of 2026-08-11
    # "unsloth_q6k"               # ~23.4 GiB fresh download, not cached as of 2026-08-11
)

# macOS ships no `timeout`/`gtimeout` by default (GNU coreutils isn't
# installed here) -- this is a portable, bash-only replacement: run the
# command in the background, race it against a `sleep`-based watchdog,
# and TERM (then KILL, if it's still alive 2s later) whichever one loses.
# `$TIMED_OUT` (set by the caller checking it right after) is how the
# caller tells "genuinely timed out" apart from "ran and exited nonzero
# on its own" -- a killed process's own exit code (128+signal) isn't a
# reliable enough signal on its own. A marker file, not a PID-liveness
# check after the fact: the watchdog and the real command both racing
# to finish means checking "is the watchdog process still alive" after
# `wait` returns is itself racy (the command can die from the TERM
# before the watchdog subshell's own script has finished running) --
# the marker is written *before* any signal is sent, so its existence
# deterministically means "the watchdog decided to kill it," independent
# of exactly when either process actually exits.
TIMED_OUT=0
run_with_timeout() {
    local secs="$1"
    shift
    local marker
    marker="$(mktemp -t sweep_timeout)"
    rm -f "$marker"
    TIMED_OUT=0
    "$@" &
    local cmd_pid=$!
    (
        sleep "$secs"
        if kill -0 "$cmd_pid" 2>/dev/null; then
            : > "$marker"
            kill -TERM "$cmd_pid" 2>/dev/null
            sleep 2
            kill -KILL "$cmd_pid" 2>/dev/null
        fi
    ) &
    local watchdog_pid=$!
    local exit_code=0
    wait "$cmd_pid" 2>/dev/null || exit_code=$?
    kill "$watchdog_pid" 2>/dev/null
    wait "$watchdog_pid" 2>/dev/null
    if [[ -f "$marker" ]]; then
        TIMED_OUT=1
        rm -f "$marker"
    fi
    return "$exit_code"
}

PROMPT=""
OUT=""
MODELS=""
SEEDS="greedy,4202,4203,4204,4205"
REPO_DIR=""
TIMEOUT_SECS=900

while [[ $# -gt 0 ]]; do
    case "$1" in
        --prompt) PROMPT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --seeds) SEEDS="$2"; shift 2 ;;
        --repo-dir) REPO_DIR="$2"; shift 2 ;;
        --timeout) TIMEOUT_SECS="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ -z "$PROMPT" ]]; then
    echo "usage: $0 --prompt <file> [--out <dir>] [--models <comma-list>] [--seeds <comma-list>] [--repo-dir <path>] [--timeout <secs>]" >&2
    exit 1
fi
if [[ ! -f "$PROMPT" ]]; then
    echo "prompt file not found: $PROMPT" >&2
    exit 1
fi
PROMPT="$(cd "$(dirname "$PROMPT")" && pwd)/$(basename "$PROMPT")"

if [[ -z "$OUT" ]]; then
    OUT="sweep-$(date -u +%Y%m%dT%H%M%SZ)"
fi
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
MANIFEST="$OUT/manifest.log"

echo "Building prompt_harness (release)..."
if ! (cd "$RAMPIPE_DIR" && cargo build --release --features llama --example prompt_harness); then
    echo "build failed — aborting before running anything" >&2
    exit 1
fi
if [[ ! -x "$HARNESS_BIN" ]]; then
    echo "expected binary not found after build: $HARNESS_BIN" >&2
    exit 1
fi

if [[ -z "$MODELS" ]]; then
    # `${DEFAULT_MODELS[@]+...}` guard: same bash-3.2-under-`set -u`
    # empty-array gotcha as `seed_args`/`REPO_DIR_ARGS` below — every
    # DEFAULT_MODELS line could in principle be commented out.
    MODELS="$(printf '%s,' ${DEFAULT_MODELS[@]+"${DEFAULT_MODELS[@]}"})"
    MODELS="${MODELS%,}"
fi
if [[ -z "$MODELS" ]]; then
    echo "DEFAULT_MODELS is empty (every line commented out?) and no --models was given — nothing to run" >&2
    exit 1
fi

# Validates every requested name against prompt_harness.rs's own real
# MODEL_CANDIDATES table — a plain grep, not a Rust parse, so this
# stays a shell script; good enough for a name that's always a bare
# quoted string on its own line right after the `[` or a `),`. Catches
# a typo in DEFAULT_MODELS above or a bad --models value here, with a
# clear message, instead of failing deep inside the harness on run 1
# of what might be a long unattended sweep.
KNOWN_MODELS="$(grep -oE '^\s*"[a-zA-Z0-9_]+",\s*$' "$RAMPIPE_DIR/examples/prompt_harness.rs" | grep -oE '"[a-zA-Z0-9_]+"' | tr -d '"')"
IFS=',' read -ra REQUESTED_MODELS <<< "$MODELS"
for requested in "${REQUESTED_MODELS[@]}"; do
    if ! grep -qxF "$requested" <<< "$KNOWN_MODELS"; then
        echo "no model candidate named \"$requested\" in prompt_harness.rs's own MODEL_CANDIDATES — known: $(tr '\n' ' ' <<< "$KNOWN_MODELS")" >&2
        exit 1
    fi
done

REPO_DIR_ARGS=()
if [[ -n "$REPO_DIR" ]]; then
    REPO_DIR_ARGS=(--repo-dir "$REPO_DIR")
fi

echo "Prompt:  $PROMPT"
echo "Out:     $OUT"
echo "Models:  $MODELS"
echo "Seeds:   $SEEDS"
echo "Timeout: ${TIMEOUT_SECS}s per run"
echo "Manifest: $MANIFEST"
echo

IFS=',' read -ra MODEL_LIST <<< "$MODELS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

total_runs=$(( ${#MODEL_LIST[@]} * ${#SEED_LIST[@]} ))
run_num=0

for model in "${MODEL_LIST[@]}"; do
    model_dir="$OUT/$model"
    mkdir -p "$model_dir"

    for seed in "${SEED_LIST[@]}"; do
        run_num=$((run_num + 1))
        if [[ "$seed" == "greedy" ]]; then
            out_file="$model_dir/greedy.txt"
            seed_args=()
        else
            out_file="$model_dir/seed_${seed}.txt"
            seed_args=(--seed "$seed")
        fi

        start_ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
        start_epoch=$(date +%s)
        echo "[$run_num/$total_runs] $model / $seed — starting ($start_ts)"

        # `${arr[@]+"${arr[@]}"}` rather than a plain `"${arr[@]}"` --
        # macOS's default `/bin/bash` is still Apple's ancient 3.2.57,
        # which throws "unbound variable" under `set -u` when expanding
        # a zero-element array, unlike bash 4+. This idiom is the
        # standard 3.2-safe workaround: expands to nothing at all
        # (not even an empty string) when the array is empty, on every
        # bash version.
        if run_with_timeout "$TIMEOUT_SECS" "$HARNESS_BIN" "$PROMPT" --model "$model" \
            ${seed_args[@]+"${seed_args[@]}"} ${REPO_DIR_ARGS[@]+"${REPO_DIR_ARGS[@]}"} > "$out_file" 2>&1; then
            status="ok"
        else
            exit_code=$?
            if [[ $TIMED_OUT -eq 1 ]]; then
                status="timeout"
            else
                status="failed(exit=$exit_code)"
            fi
        fi

        end_epoch=$(date +%s)
        duration=$((end_epoch - start_epoch))
        echo "[$run_num/$total_runs] $model / $seed — $status (${duration}s) -> $out_file"
        echo "$start_ts  model=$model  seed=$seed  status=$status  duration_s=$duration  out=$out_file" >> "$MANIFEST"
    done
done

echo
echo "Done. $run_num run(s) attempted — see $MANIFEST for the full log, or:"
echo "  grep -L ACCEPTED $OUT/*/*.txt   # runs where nothing was accepted (all rejected or unparseable)"
echo "  diff $OUT/<model-a>/greedy.txt $OUT/<model-b>/greedy.txt"
