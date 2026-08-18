# rampipe

A memory-residency scheduler for multiple concurrent LLMs under a hard RAM
ceiling for on-device inference (iOS-class hardware), where holding two
copies of a model in memory at once can be the difference between running
fine and getting killed for memory pressure.

`rampipe` is about deciding what model data stays mapped, what pages in, and
what gets evicted when several models must coexist in a fixed memory budget.
Speculative decoding, per-app model routing, and adapter hot-swap are all
instances of that one problem. It is deliberately not a generic hot-swap
primitive.

**Status: measurement phase.** The mapping, prefault, and accounting
primitives are implemented, measured, and tested. There is no eviction policy
yet. That gets designed against real numbers rather than guesses, and the
numbers are below.

## Quick start

```sh
cargo test                     # 21 tests, no model download, no backend
```

The base crate is dependency-light and has no inference backend. Real
generation lives behind the `llama` feature, which links llama.cpp:

```sh
cargo run --release --features llama --example residency_vs_ttft
cargo run --release --features llama --example smoke_7b
```

Those examples download real GGUF models from the Hugging Face Hub (~491MB
and ~4.7GB respectively) on first run.

## What it does

- `SwapRegistry::load` / `evict`: `memmap2`-backed, no full-buffer read.
  `load` dedups: loading an already-resident path hands out another handle
  to the existing mapping (same `ModelId`) instead of mapping the file twice.
- `ModelId`: a stable `u64`-backed handle, not the path itself. Cheap to
  copy, and safe to hand across an FFI boundary.
- `ModelHandle`: refcounted via `Arc`. `evict(id)` fails while any handle
  for that id is still alive. This is the load-bearing safety invariant: a
  live generation holds a handle, so the scheduler cannot unmap weights out
  from under it.
- Registry aggregates `resident_count()` and `mapped_bytes()`, which report
  the ceiling rather than current RSS.
- GGUF magic validation on load (rejects non-GGUF files, empty files).
- `SwapMetrics`: map latency, prefault latency, mapped bytes, RSS delta,
  `resident_fraction`, a `warm` heuristic. See the mincore finding below
  before trusting the last two.
- `Residency::{Lazy, Prefault, Advise}`. `Lazy` just mmaps and lets the OS
  fault pages in on first touch. `Prefault` synchronously touches every
  page up front. `Advise` hints the kernel via `madvise(MADV_WILLNEED)`,
  which is a hint and not a guarantee, so unlike `Prefault` its pages
  aren't necessarily resident by the time `load()` returns.
- RSS measurement on Linux (`/proc/self/statm`) and Darwin/iOS (`mach2`'s
  `task_info(TASK_VM_INFO)`, reading `phys_footprint`), verified against
  the real `mach2` struct layout rather than guessed, and covered by two unit
  tests that check the number is plausible and actually moves when a real
  allocation is touched.

## Findings

### `mincore(2)` does not tell the truth about file-backed mappings on Darwin

The roadmap's own open item was "replace the `warm` heuristic with a real
`mincore(2)` check." Implemented that, and then a test asserting prefault
should leave pages ~fully resident failed, consistently, at ~25%. Chased it
down with a standalone repro rather than just loosening the assertion:

- `mincore` reports the same ~25% resident fraction for a file-backed mmap
  **regardless of file size** (verified at 32 pages and 500 pages).
- The number doesn't move even after `mlock(2)`, which *guarantees* every
  page is resident and pinned. If a guaranteed-resident mapping still
  reads as 75% absent, the syscall isn't answering the question truthfully.
- A plain anonymous (non-file-backed) mmap'd page reports correctly through
  the same call.

That pattern (accurate for anonymous memory, deliberately imprecise-looking
for file-backed memory) matches the shape of known page-cache side-channel
mitigations (mincore has historically been usable to fingerprint what's in
the shared page cache, leaking other processes' file-access patterns).
Root cause not confirmed against an Apple source; the empirical result is
repeatable enough to document and act on regardless. `resident_fraction`
and `warm` are implemented as asked and still exposed, but documented as
untrustworthy for file-backed mappings on Darwin. See the doc comment on
`SwapMetrics::resident_fraction`. Tests assert structural validity (a
well-formed number in range) rather than a specific residency claim.

### `Prefault` costs seconds, not milliseconds, on a cold 7B file

Early measurements were all on Qwen2.5-0.5B, and every run reused the same
small, already-OS-cached file. `examples/smoke_7b.rs` loads a real
Qwen2.5-7B-Instruct Q4_K_M GGUF (4.68GB) the same way. No code changes were
needed: `SwapRegistry` + `LlamaSession` scale cleanly, Metal offload
confirmed, no crashes, no memory pressure (25.5GB free reported by
llama.cpp). Steady-state numbers are solid: TTFT ~270ms, ~22 tok/s, coherent
output, and a second `generate()` call on the same session works cleanly,
confirming the "load once, serve many requests" shape a real backend needs.

But `map_latency` staying near-instant (97µs) hid something:
`Residency::Prefault`'s synchronous touch-loop took **8.05 seconds** on this
file's first-ever touch, which is not proportional to the ~9.5x size jump
from 491MB to 4.68GB, because every prior `Prefault` measurement was implicitly
warm-cache. At this scale, on a genuinely cold file, `Prefault` blocks for
multiple real seconds pulling the model off disk. Don't default to `Prefault`
for a 7B-class model without deciding that startup stall is acceptable.
`Lazy` (defer the cost to first inference) and `Advise` (non-blocking kernel
hint) are real alternatives now that this cost is visible instead of hidden
by warm-cache testing.

### Residency mode vs. time-to-first-token

`examples/residency_vs_ttft.rs` downloads a real Qwen2.5-0.5B-Instruct GGUF
(~491MB) and runs it through `Lazy`, `Prefault`, and `Advise` residency via
independent registries. Measured: Lazy TTFT ~78ms vs. Prefault ~34ms vs.
Advise ~37ms, all producing identical coherent text. `Advise`'s
`prefault_latency` (~16ms) is meaningfully cheaper than `Prefault`'s (~23ms),
and that comparison is unaffected by the mincore caveat since it's call
latency, not a residency claim. RSS delta came back much smaller than
`mapped_bytes` for both eager runs, consistent with `phys_footprint`
correctly excluding clean, evictable, file-backed page-cache pages, which is
the right signal for avoiding iOS jetsam, not a bug in the measurement.

### Real chat templates need a real Jinja engine

Instruct-tuned GGUFs carry their own `tokenizer.chat_template` as GGUF
metadata, the same Jinja source `transformers` uses. llama.cpp ships a
minimal Jinja subset to render it, and that subset is not always enough:
AI21's Jamba Mini uses macros and namespaces, and `apply_chat_template`
returns `ffi error -1` on it outright.

`rampipe` renders the model's own template with `minijinja`, a real Jinja
engine, and falls back to llama.cpp's renderer only if that fails. Two
details turned out to be load-bearing, and a live failure found both of them
before any documentation did.

`trim_blocks` and `lstrip_blocks` have to be on. Real HF chat templates are
authored assuming `transformers`' own
`jinja2.Environment(trim_blocks=True, lstrip_blocks=True)`. Without it,
newlines and indentation between `{% %}` control tags that don't carry their
own `-` trim markers leak into macro return values. The live failure: Jamba's
`get_last_user_index` macro fed `|int` a string padded with accumulated
block-tag whitespace instead of the `0` its `{{- ... -}}` content actually
produced.

`raise_exception` has to be registered. It isn't a builtin in any Jinja
engine, but templates call it as an ordinary function for their own input
validation, and every real chat-template caller, `transformers` included,
registers it by convention. Leaving it undefined turns a template's
deliberate validation error into an unrelated "unknown function" failure.

Below both of those sits `ChatWrap`, a hand-captured `prefix + prompt +
suffix` escape hatch for a template neither engine can render. Nothing
currently needs it; it stays because "the model ships a template we can't
execute" is a real failure mode.

The test fixture is the genuine article: Jamba Mini's template extracted
from the real `bartowski/ai21labs_AI21-Jamba-Mini-1.7-GGUF` file with
Python's `gguf` package, not hand-written to match the implementation.

## The inference backend

Behind the `llama` feature flag, so the base crate stays dependency-light and
v0 measures paging cost with zero backend coupling.

- `llama::LlamaSession` bundles a `ModelHandle` (this crate's own accounting
  mmap) with a model loaded into `llama-cpp-2`. llama.cpp always loads from
  a file path with its own internal mmap, and there's no API to hand it
  bytes we've already mapped, so the two are separate mappings of the same
  file.
  Documented, not hidden: both are read-only mmaps of the same inode, so the
  OS page cache shares the physical pages between them. Prefaulting through
  our own mapping genuinely warms what llama.cpp reads.
- `LlamaSession::generate` runs a real decode loop and reports
  `time_to_first_token` separately from total generation time, so page-in
  cost becomes attributable to an actual generation.

`rampiped` (the `llama` feature's binary) holds models resident across
process boundaries and serves generation over a Unix socket, one request per
connection. `src/client.rs` (the `client` feature) is the pure-Rust client
for it and needs no backend to build.

## Known-crude, called out on purpose

- `resident_fraction`/`warm` are backed by `mincore(2)` now, not a timing
  guess, but see the finding above: not trustworthy for file-backed
  mappings on Darwin regardless.
- Prefault is still a naive "touch one byte per page" loop, now with
  `Advise` (`madvise(WILLNEED)`) alongside it for actual comparison rather
  than as a hypothetical.
- `residency_vs_ttft`'s comparison doesn't control for OS page cache state
  going in (dropping it needs root on macOS), so the gap it shows is
  smaller than a genuine cold start.

## Not yet started

- A two-model speculative-decode residency test: small draft model plus large
  verifier, different eviction priorities, with a lock-free SPSC ring buffer
  for draft→verifier handoff so the active decode loop doesn't stall.
- Eviction policy, and a UniFFI/Swift binding for on-device dogfooding.

## On the `ModelId` design

An earlier scaffold of this design keyed everything by path; `ModelId(u64)`
replaced it specifically because an opaque integer handle is FFI-friendly for
a future Swift binding, where a path is not. The registry is `Mutex`-guarded
and thread-safe from the start rather than single-threaded with `&mut self`,
because the two-model speculative-decode case needs concurrent access.

## License

Dual-licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
