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

## Phase 1

Measures the primitives — mapping cost, prefault cost, memory footprint —
with no eviction policy yet. Nothing here decides *which* model should stay
resident; that gets designed against real numbers, not guesses.

- `SwapRegistry::load` / `evict` — `memmap2`-backed, no full-buffer read.
  `load` dedups: loading an already-resident path hands out another handle
  to the existing mapping (same `ModelId`) instead of mapping the file twice.
- `ModelId` — a stable `u64`-backed handle, not the path itself. Cheap to
  copy, and safe to hand across an FFI boundary once Phase 4 (UniFFI/Swift)
  exists to hand it to.
- `ModelHandle` — refcounted via `Arc`. `evict(id)` fails while any handle
  for that id is still alive. This is the load-bearing safety invariant: a
  live generation holds a handle, so the scheduler cannot unmap weights out
  from under it.
- Registry aggregates: `resident_count()`, `mapped_bytes()` — the ceiling,
  not current RSS.
- GGUF magic validation on load (rejects non-GGUF files, empty files).
- `SwapMetrics` — map latency, prefault latency, mapped bytes, RSS delta,
  `resident_fraction`, a `warm` heuristic. See the mincore caveat below
  before trusting the last two.
- `Residency::{Lazy, Prefault, Advise}` — `Lazy` just mmaps and lets the OS
  fault pages in on first touch. `Prefault` synchronously touches every
  page up front. `Advise` hints the kernel via `madvise(MADV_WILLNEED)` —
  a non-blocking hint, not a guarantee, so unlike `Prefault` its pages
  aren't necessarily resident by the time `load()` returns.
- RSS measurement on Linux (`/proc/self/statm`) and Darwin/iOS (`mach2`'s
  `task_info(TASK_VM_INFO)`, reading `phys_footprint`) — verified against
  the real `mach2` struct layout, not guessed, and covered by two unit
  tests that check the number is plausible and actually moves when a real
  allocation is touched.

```sh
cargo test
```

12 tests, all passing.

### The `mincore(2)` finding

The roadmap's own open item was "replace the `warm` heuristic with a real
`mincore(2)` check." Implemented that — and then a test asserting prefault
should leave pages ~fully resident failed, consistently, at ~25%. Chased it
down with a standalone repro rather than just loosening the assertion:

- `mincore` reports the same ~25% resident fraction for a file-backed mmap
  **regardless of file size** (verified at 32 pages and 500 pages).
- The number doesn't move even after `mlock(2)`, which *guarantees* every
  page is resident and pinned — if a guaranteed-resident mapping still
  reads as 75% absent, the syscall isn't answering the question truthfully.
- A plain anonymous (non-file-backed) mmap'd page reports correctly through
  the same call.

That pattern — accurate for anonymous memory, deliberately-imprecise-looking
for file-backed memory — matches the shape of known page-cache side-channel
mitigations (mincore has historically been usable to fingerprint what's in
the shared page cache, leaking other processes' file-access patterns).
Root cause not confirmed against an Apple source; the empirical result is
repeatable enough to document and act on regardless. `resident_fraction`
and `warm` are implemented as asked and still exposed, but documented as
untrustworthy for file-backed mappings on Darwin — see the doc comment on
`SwapMetrics::resident_fraction`. Tests assert structural validity (a
well-formed number in range) rather than a specific residency claim.

## Phase 1b

A real inference backend, behind the `llama` feature flag so the base crate
stays dependency-light — "v0 measures paging cost with zero backend
coupling."

- `llama::LlamaSession` bundles a `ModelHandle` (this crate's own accounting
  mmap) with a model loaded into `llama-cpp-2`. llama.cpp always loads from
  a file path with its own internal mmap — there's no API to hand it bytes
  we've already mapped — so the two are separate mappings of the same file.
  Documented, not hidden: both are read-only mmaps of the same inode, so the
  OS page cache shares the physical pages between them. Prefaulting through
  our own mapping genuinely warms what llama.cpp reads.
- `LlamaSession::generate` runs a real decode loop and reports
  `time_to_first_token` separately from total generation time, so page-in
  cost becomes attributable to an actual generation.
- `examples/residency_vs_ttft.rs` — downloads a real Qwen2.5-0.5B-Instruct
  GGUF (~491MB) and runs it through `Lazy`, `Prefault`, and `Advise`
  residency via independent registries. Run for real: Lazy TTFT ~78ms vs.
  Prefault ~34ms vs. Advise ~37ms, all producing identical coherent text.
  `Advise`'s `prefault_latency` (~16ms) is meaningfully cheaper than
  `Prefault`'s (~23ms) — the actual "compare against madvise" result the
  roadmap asked for, and unaffected by the mincore caveat above since it's
  just call latency, not a residency claim. llama.cpp auto-detected Metal
  and offloaded all layers to GPU. Worth knowing: RSS delta came back much
  smaller than `mapped_bytes` for both eager runs — consistent with
  `phys_footprint` correctly excluding clean, evictable, file-backed
  page-cache pages, which is actually the right signal for avoiding iOS
  jetsam, not a bug in the measurement.

```sh
cargo run --release --features llama --example residency_vs_ttft
```

## Known-crude, called out on purpose

- `resident_fraction`/`warm` are backed by `mincore(2)` now, not a timing
  guess — but see the finding above: not trustworthy for file-backed
  mappings on Darwin regardless.
- Prefault is still a naive "touch one byte per page" loop, now with
  `Advise` (`madvise(WILLNEED)`) alongside it for actual comparison rather
  than as a hypothetical.
- `residency_vs_ttft`'s comparison doesn't control for OS page cache state
  going in (dropping it needs root on macOS), so the gap it shows is
  smaller than a genuine cold start.

Called out, not accidental — see the project [`TODO.md`](../TODO.md).

## Not yet started

- Phase 2: a two-model speculative-decode residency test (small draft
  model + large verifier, different eviction priorities, `halfpipe` for
  draft→verifier handoff without stalling the active decode loop).
- Phase 3 (eviction policy) and Phase 4 (UniFFI/Swift, on-device dogfood).

## On the `ModelId` design

The original Phase 1 scaffold from the roadmap conversation this crate is
based on was found later (as a zip, never actually compiled) and turned out
to independently make several of the same calls this crate did — same
mmap/magic/refcount design — plus one genuinely better one: a `ModelId(u64)`
handle instead of keying everything by path, specifically because it's
FFI-friendly for Phase 4. Adopted here. Kept from this crate's own version
rather than the original: a `Mutex`-guarded registry (thread-safe from the
start, since Phase 2 needs concurrent access) rather than the original's
deliberately single-threaded `&mut self` design.

## Why `halfpipe`/`logpipe` matter here too

`rampipe` and a separate NDR project share the same primitive trunk:
`halfpipe` (lock-free SPSC ring buffer) is meant to serve both a packet
pipeline elsewhere and this crate's future draft→verifier handoff;
`logpipe` is meant to serve both event logs elsewhere and this crate's
metrics stream. Keep that in mind before adding anything `rampipe`-specific
to either — they're deliberately meant to stay generic enough for both.
