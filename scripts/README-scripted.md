# Scripts for `rampiped-script`

Failures that were expensive to reproduce with a real model, written
down so they cost seconds.

    cargo run --bin rampiped-script -- --socket /tmp/fake.sock --script scripts/collapse.toml
    agent99 --socket /tmp/fake.sock --model /any/path.gguf --repo <r> --task <t> --once

The model path is not read -- the scripted daemon never loads one -- but
the flag is still required, because the agent under test is the real
binary and its argument handling is part of what is being tested.

`rampiped-script` prints what the *harness* said as it says it: the
system block, the tool list, every message and every tool result. That
is the half that catches harness bugs. What the model says is already in
the script.
