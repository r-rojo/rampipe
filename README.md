# rampipe

A memory-residency scheduler for multiple concurrent LLMs under a hard RAM
ceiling — for on-device inference (iOS-class hardware), where holding two
copies of a model in memory at once can be the difference between running
fine and getting killed for memory pressure.

Not a generic hot-swap primitive (that's [`modelswap`](../modelswap) — a
separate, simpler crate). `rampipe` is specifically about deciding what
model data stays mapped, what pages in, what gets evicted, when several
models need to coexist in a fixed memory budget. Speculative decoding,
per-app model routing, and adapter hot-swap are all instances of that one
problem.

## Phase 1 (current)

Measures the primitives — mapping cost, prefault cost, memory footprint —
with no eviction policy yet. Nothing here decides *which* model should stay
resident; that gets designed against real numbers, not guesses.

- `SwapRegistry::load` / `evict` — `memmap2`-backed, no full-buffer read.
  `load` dedups: loading an already-resident path hands out another handle
  to the existing mapping instead of mapping the file twice.
- `ModelHandle` — refcounted via `Arc`. `evict` fails while any handle for
  that path is still alive. This is the load-bearing safety invariant: a
  live generation holds a handle, so the scheduler cannot unmap weights
  out from under it.
- GGUF magic validation on load (rejects non-GGUF files, empty files).
- `SwapMetrics` — map latency, prefault latency (only for `Residency::
  Prefault`), mapped bytes, RSS delta, a `warm` heuristic.
- `Residency::{Lazy, Prefault}` — `Lazy` just mmaps and lets the OS fault
  pages in on first touch; `Prefault` touches every page up front so the
  page-fault cost is paid at load time instead of during generation. The
  gap between these two is the actual thing worth measuring.
- RSS measurement on Linux (`/proc/self/statm`) and Darwin/iOS (`mach2`'s
  `task_info(TASK_VM_INFO)`, reading `phys_footprint`) — verified against
  the real `mach2` struct layout, not guessed, and covered by two unit
  tests that check the number is plausible and actually moves when a real
  allocation is touched.

```sh
cargo test
```

10 tests (8 integration + 2 targeted at the RSS FFI call), all passing.

## Known-crude, called out on purpose

- The `warm` flag guesses residency from how fast the prefault loop ran.
  Should be a real `mincore(2)` check instead.
- Prefault is a naive "touch one byte per page" loop. Worth comparing
  against `madvise(WILLNEED)`.

Both are explicitly deferred, not accidental gaps — see the project
[`TODO.md`](../TODO.md).

## Not yet started

- Phase 1b: a real inference backend (`llama-cpp-2` behind a `llama`
  feature flag) and time-to-first-token metrics, so page-in cost becomes
  attributable to an actual generation, not just a synthetic mmap.
- Phase 2: a two-model speculative-decode residency test (small draft
  model + large verifier, different eviction priorities, `halfpipe` for
  draft→verifier handoff without stalling the active decode loop).
- Phase 3 (eviction policy) and Phase 4 (UniFFI/Swift, on-device dogfood).

## Why `halfpipe`/`logpipe` matter here too

`rampipe` and a separate NDR project share the same primitive trunk:
`halfpipe` (lock-free SPSC ring buffer) is meant to serve both a packet
pipeline elsewhere and this crate's future draft→verifier handoff;
`logpipe` is meant to serve both event logs elsewhere and this crate's
metrics stream. Keep that in mind before adding anything `rampipe`-specific
to either — they're deliberately meant to stay generic enough for both.
