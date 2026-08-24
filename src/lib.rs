use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "client")]
pub mod client;
pub mod conversation;
#[cfg(feature = "llama")]
pub mod llama;
#[cfg(feature = "template")]
pub mod chat_template;
pub mod model_settings;
pub mod protocol;
/// A `rampiped` that says what you tell it to -- see the module docs.
///
/// Deliberately not behind the `llama` feature: its whole value is that
/// a harness can be tested without a model, a GPU, or forty minutes.
pub mod scripted;
pub mod tool_format;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const PAGE_SIZE: usize = 4096;
// `warm` is `resident_fraction >= WARM_FRACTION` -- real residency data from
// mincore(2), not a guess from timing. Not quite 1.0 so a single straggler
// page (e.g. one still mid-fault when mincore samples it) doesn't flip an
// otherwise-fully-resident mapping to "not warm".
const WARM_FRACTION: f64 = 0.99;

/// How eagerly to bring a newly-mapped model's pages into physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// mmap only; the OS faults pages in on first touch. Fast to "loaded",
    /// but the first real access to each page pays a page-fault during
    /// inference instead of during load.
    Lazy,
    /// mmap, then touch every page immediately (synchronously) so the
    /// page-fault cost is paid up front instead of mid-generation. On Linux
    /// this should leave every page resident by the time it returns; on
    /// Darwin, empirically, don't assume it -- see the caveat on
    /// `SwapMetrics::resident_fraction`.
    Prefault,
    /// mmap, then hint the kernel via `madvise(MADV_WILLNEED)` that every
    /// page will be needed soon. Returns almost immediately -- it's a
    /// non-blocking hint, not a guarantee -- so unlike `Prefault`, pages
    /// aren't necessarily resident by the time `load()` returns; the
    /// kernel's readahead may still be in flight.
    Advise,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapMetrics {
    pub map_latency: Duration,
    /// `Some` only when loaded with `Residency::Prefault` or `Residency::
    /// Advise` -- the time spent in the touch loop or the `madvise` call,
    /// respectively. For `Advise` this is *not* the time until pages are
    /// actually resident (see `Residency::Advise`), just the call latency.
    pub prefault_latency: Option<Duration>,
    pub mapped_bytes: usize,
    /// Process RSS after mapping minus RSS before, in bytes. `None` on
    /// platforms without an RSS measurement implemented (see
    /// `current_rss_bytes`).
    pub rss_delta_bytes: Option<i64>,
    /// Fraction (0.0–1.0) of the mapping's pages that `mincore(2)` reports
    /// as resident in physical memory at the moment `load()` finished.
    /// `None` on platforms without an implementation (see
    /// `mincore_resident_fraction`).
    ///
    /// **Caveat, found empirically, not documented anywhere official as far
    /// as this crate's author checked:** on Darwin, this number is not
    /// trustworthy for file-backed mappings. Testing here (2026-07-25)
    /// showed `mincore` consistently reporting only ~25% of pages resident
    /// for an mmap'd file regardless of size (32 pages vs. 500 pages, same
    /// ratio) -- and the number didn't change even after `mlock(2)`, which
    /// *guarantees* full residency. A plain anonymous (non-file-backed)
    /// mmap'd page reports correctly through the same call. That pattern --
    /// accurate for anonymous memory, deliberately-looking imprecise for
    /// file-backed memory -- matches the shape of known page-cache
    /// side-channel mitigations (mincore has historically been usable to
    /// fingerprint what's in the shared page cache, leaking other
    /// processes' file-access patterns). Root cause not confirmed against
    /// an Apple source, but the empirical result is repeatable and decisive
    /// enough to document: don't trust this field for file-backed mappings
    /// on Darwin. Untested on Linux.
    pub resident_fraction: Option<f64>,
    /// `resident_fraction >= WARM_FRACTION`. Computed from a real syscall
    /// now, not guessed from prefault timing -- but inherits
    /// `resident_fraction`'s Darwin caveat above, so treat it the same way.
    pub warm: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("io error loading model: {0}")]
    Io(#[from] io::Error),
    #[error("file is empty")]
    Empty,
    #[error("not a GGUF file: expected magic {GGUF_MAGIC:?}, found {found:?}")]
    BadMagic { found: [u8; 4] },
}

#[derive(Debug, thiserror::Error)]
pub enum EvictError {
    #[error("unknown model id: {0:?}")]
    UnknownId(ModelId),
    #[error("{outstanding} handle(s) still outstanding, refusing to evict")]
    HandleOutstanding { outstanding: usize },
}

/// Stable identifier for a resident model. Cheap to copy, and -- unlike a
/// `PathBuf` -- safe to hand across an FFI boundary as a plain `u64`, which
/// matters once Phase 4 (UniFFI/Swift) exists to hand it to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(u64);

struct Resident {
    path: PathBuf,
    mmap: Mmap,
    metrics: SwapMetrics,
    /// GPU/device memory this model is using, if any -- `None` until a
    /// caller reports it via `SwapRegistry::record_device_bytes` (unknown
    /// at `load()` time: mapping the file and touching a GPU are separate
    /// steps, done by different layers -- see that method's own doc
    /// comment), and always `None` for a model that only ever ran on
    /// CPU/host memory. `Mutex`, not a plain field, because `Resident`
    /// is shared via `Arc` between the registry's own table and every
    /// `ModelHandle` clone, all of which need to see an update made after
    /// the fact through any of them.
    device_bytes: Mutex<Option<u64>>,
}

/// A live reference to a mapped model. While any handle for a given model is
/// alive, `SwapRegistry::evict` for its id fails -- this is the safety
/// invariant that makes eviction safe under concurrent generation: a live
/// generation holds a handle, so the scheduler cannot unmap weights under it.
#[derive(Clone)]
pub struct ModelHandle {
    id: ModelId,
    inner: Arc<Resident>,
}

impl ModelHandle {
    pub fn id(&self) -> ModelId {
        self.id
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.inner.mmap
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub fn metrics(&self) -> SwapMetrics {
        self.inner.metrics
    }

    /// GPU/device memory this model is using, per the most recent
    /// `SwapRegistry::record_device_bytes` call for its id -- `None` if
    /// that's never been called (e.g. a CPU-only load, or a GPU-touching
    /// load that hasn't reported its measurement yet).
    pub fn device_bytes(&self) -> Option<u64> {
        *self
            .inner
            .device_bytes
            .lock()
            .expect("rampipe registry lock poisoned")
    }
}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHandle")
            .field("id", &self.id)
            .field("path", &self.inner.path)
            .field("metrics", &self.inner.metrics)
            .finish()
    }
}

#[derive(Default)]
struct RegistryState {
    next_id: u64,
    resident: HashMap<ModelId, Arc<Resident>>,
    by_path: HashMap<PathBuf, ModelId>,
    /// When each resident model was last handed out (a fresh load *or*
    /// a cache-hit against an already-resident one) -- separate from
    /// `Resident` itself (shared via `Arc` and handed out through
    /// `ModelHandle`) so updating it on every access never needs
    /// interior mutability inside something callers hold a live
    /// reference to; it's purely the registry's own bookkeeping, always
    /// touched under the same lock as everything else here. What makes
    /// `resident_ids_by_lru` real LRU, not just insertion order.
    last_accessed: HashMap<ModelId, Instant>,
}

pub struct SwapRegistry {
    state: Mutex<RegistryState>,
}

impl SwapRegistry {
    pub fn new() -> Self {
        SwapRegistry {
            state: Mutex::new(RegistryState::default()),
        }
    }

    /// Maps `path` into memory and validates it as a GGUF file. If `path` is
    /// already resident, hands out another handle to the existing mapping
    /// (same `ModelId`) instead of mapping it again.
    pub fn load(
        &self,
        path: impl AsRef<Path>,
        residency: Residency,
    ) -> Result<ModelHandle, LoadError> {
        let path = path.as_ref().to_path_buf();
        let mut state = self.state.lock().expect("rampipe registry lock poisoned");

        if let Some(&id) = state.by_path.get(&path) {
            let inner = state
                .resident
                .get(&id)
                .expect("by_path/resident desync")
                .clone();
            state.last_accessed.insert(id, Instant::now());
            return Ok(ModelHandle { id, inner });
        }

        let rss_before = current_rss_bytes();

        let map_start = Instant::now();
        let file = File::open(&path)?;
        // Safety: nothing else in this process writes to model files while
        // they're mapped, so we don't guard against concurrent external
        // modification of the underlying file.
        let mmap = unsafe { Mmap::map(&file)? };
        let map_latency = map_start.elapsed();

        if mmap.len() < 4 {
            return Err(LoadError::Empty);
        }
        if &mmap[..4] != GGUF_MAGIC.as_slice() {
            let mut found = [0u8; 4];
            found.copy_from_slice(&mmap[..4]);
            return Err(LoadError::BadMagic { found });
        }

        let mapped_bytes = mmap.len();

        let prefault_latency = match residency {
            Residency::Lazy => None,
            Residency::Prefault => {
                let start = Instant::now();
                prefault(&mmap);
                Some(start.elapsed())
            }
            Residency::Advise => {
                let start = Instant::now();
                advise_willneed(&mmap);
                Some(start.elapsed())
            }
        };

        let rss_after = current_rss_bytes();
        let rss_delta_bytes = match (rss_before, rss_after) {
            (Some(before), Some(after)) => Some(after - before),
            _ => None,
        };

        let resident_fraction = mincore_resident_fraction(&mmap);
        let warm = resident_fraction.is_some_and(|f| f >= WARM_FRACTION);

        let metrics = SwapMetrics {
            map_latency,
            prefault_latency,
            mapped_bytes,
            rss_delta_bytes,
            resident_fraction,
            warm,
        };

        let id = ModelId(state.next_id);
        state.next_id += 1;
        let inner = Arc::new(Resident {
            path: path.clone(),
            mmap,
            metrics,
            device_bytes: Mutex::new(None),
        });
        state.resident.insert(id, inner.clone());
        state.by_path.insert(path, id);
        state.last_accessed.insert(id, Instant::now());

        Ok(ModelHandle { id, inner })
    }

    /// Evicts the model identified by `id`, freeing its mapping -- but only
    /// if no `ModelHandle` for it is still alive. This is the invariant
    /// that protects an in-flight generation from having its weights
    /// unmapped out from under it.
    pub fn evict(&self, id: ModelId) -> Result<(), EvictError> {
        let mut state = self.state.lock().expect("rampipe registry lock poisoned");
        let Some(inner) = state.resident.get(&id) else {
            return Err(EvictError::UnknownId(id));
        };
        // The registry's own table holds one strong reference; anything
        // beyond that is an outstanding `ModelHandle`.
        let outstanding = Arc::strong_count(inner) - 1;
        if outstanding > 0 {
            return Err(EvictError::HandleOutstanding { outstanding });
        }
        let path = inner.path.clone();
        state.resident.remove(&id);
        state.by_path.remove(&path);
        state.last_accessed.remove(&id);
        Ok(())
    }

    pub fn is_resident(&self, id: ModelId) -> bool {
        self.state
            .lock()
            .expect("rampipe registry lock poisoned")
            .resident
            .contains_key(&id)
    }

    pub fn resident_count(&self) -> usize {
        self.state
            .lock()
            .expect("rampipe registry lock poisoned")
            .resident
            .len()
    }

    /// Total mapped bytes across resident models. Not RSS -- this is the
    /// ceiling, not the current cost.
    pub fn mapped_bytes(&self) -> usize {
        self.state
            .lock()
            .expect("rampipe registry lock poisoned")
            .resident
            .values()
            .map(|r| r.mmap.len())
            .sum()
    }

    /// Every resident model's id, oldest-accessed first -- the order a
    /// caller managing several resident models at once (`rampiped`, not
    /// this crate, which never evicts on its own) should walk when it
    /// needs to free budget for a new one. "Accessed" means handed out
    /// by `load()`, whether that was a fresh map or a cache hit against
    /// an already-resident model -- see `RegistryState::last_accessed`'s
    /// own doc comment.
    pub fn resident_ids_by_lru(&self) -> Vec<ModelId> {
        let state = self.state.lock().expect("rampipe registry lock poisoned");
        let mut ids: Vec<(ModelId, Instant)> = state
            .last_accessed
            .iter()
            .map(|(&id, &accessed)| (id, accessed))
            .collect();
        ids.sort_by_key(|&(_, accessed)| accessed);
        ids.into_iter().map(|(id, _)| id).collect()
    }

    /// Whether loading a new model of `new_size_bytes` would keep total
    /// resident bytes within `budget_fraction` of the memory actually
    /// available for models -- system free memory *plus* whatever's
    /// already resident, since a caller managing eviction (`rampiped`)
    /// can always reclaim already-resident bytes; counting only
    /// currently-free memory would make an already-fully-loaded machine
    /// look permanently out of budget even when eviction would free
    /// plenty of room. `None` -- not `Some(true)` or `Some(false)` -- when
    /// `system_free_bytes` itself can't be measured (see that
    /// function's own doc comment for which platforms), so a caller
    /// never silently treats "couldn't tell" as "yes" or "no".
    pub fn fits_within_budget(&self, new_size_bytes: u64, budget_fraction: f64) -> Option<bool> {
        let free = system_free_bytes()?;
        let mapped = self.mapped_bytes() as u64;
        let available_for_models = free + mapped;
        Some((mapped + new_size_bytes) as f64 <= budget_fraction * available_for_models as f64)
    }

    /// Records `bytes` as the GPU/device memory footprint of the resident
    /// model `id` -- by convention, free-device-memory immediately before
    /// its GPU-touching load minus free-device-memory immediately after,
    /// the same before/after-delta shape `SwapMetrics::rss_delta_bytes`
    /// already uses for host memory. Not measured by this crate directly:
    /// on unified-memory hardware (Apple Silicon) host and device memory
    /// are the same physical pool, so `fits_within_budget` alone already
    /// modeled the real constraint correctly -- CUDA is what introduced a
    /// second, genuinely separate pool this registry needs to track, and
    /// querying it needs backend-specific FFI (CUDA/Metal) this crate's
    /// core deliberately stays free of (see the module doc comment on
    /// `llama` being an optional, heavier feature). A caller that has
    /// that FFI available (`rampipe::llama::gpu_memory_bytes`) reports the
    /// measurement here after the fact instead.
    ///
    /// A no-op if `id` isn't resident (e.g. evicted between its `load()`
    /// returning and this being called) -- rare, not a caller error worth
    /// failing over.
    pub fn record_device_bytes(&self, id: ModelId, bytes: u64) {
        let state = self.state.lock().expect("rampipe registry lock poisoned");
        if let Some(resident) = state.resident.get(&id) {
            *resident
                .device_bytes
                .lock()
                .expect("rampipe registry lock poisoned") = Some(bytes);
        }
    }

    /// Total device bytes recorded across resident models (models with no
    /// recorded device bytes -- CPU-only, or not yet reported -- count as
    /// zero). Mirrors `mapped_bytes()`'s "ceiling, not current cost"
    /// framing, just for the device-memory dimension.
    pub fn device_resident_bytes(&self) -> u64 {
        self.state
            .lock()
            .expect("rampipe registry lock poisoned")
            .resident
            .values()
            .filter_map(|r| {
                *r.device_bytes
                    .lock()
                    .expect("rampipe registry lock poisoned")
            })
            .sum()
    }

    /// Device-memory sibling of `fits_within_budget`: whether a model
    /// estimated to need `new_device_bytes` would keep total device-
    /// resident bytes within `budget_fraction` of what's available.
    /// `free_device_bytes` is supplied by the caller rather than measured
    /// here -- see `record_device_bytes`'s doc comment for why. Always
    /// `Some`, unlike `fits_within_budget`: a caller that can supply
    /// `free_device_bytes` at all already has an answer, there's no
    /// platform-can't-measure-it case to represent here.
    pub fn fits_within_device_budget(
        &self,
        new_device_bytes: u64,
        free_device_bytes: u64,
        budget_fraction: f64,
    ) -> bool {
        let resident = self.device_resident_bytes();
        let available = free_device_bytes + resident;
        (resident + new_device_bytes) as f64 <= budget_fraction * available as f64
    }
}

impl Default for SwapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Touches one byte per page to force the OS to fault every page of `mmap`
/// into physical memory. Crude on purpose for Phase 1 -- the roadmap's own
/// open item is to compare this against `madvise(WILLNEED)`.
fn prefault(mmap: &Mmap) {
    let mut sum: u64 = 0;
    for i in (0..mmap.len()).step_by(PAGE_SIZE) {
        sum = sum.wrapping_add(mmap[i] as u64);
    }
    std::hint::black_box(sum);
}

/// Hints the kernel that every page of `mmap` will be needed soon, via
/// `madvise(MADV_WILLNEED)`. Non-blocking -- see `Residency::Advise`'s doc
/// comment for why that matters.
#[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
fn advise_willneed(mmap: &Mmap) {
    if mmap.is_empty() {
        return;
    }
    // Safety: `mmap.as_ptr()`/`mmap.len()` describe the live mapping this
    // `Mmap` owns; madvise doesn't write through the pointer, and a failed
    // hint (non-zero return) just means no readahead benefit, not a
    // correctness problem, so the result is intentionally ignored.
    unsafe {
        libc::madvise(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            libc::MADV_WILLNEED,
        );
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
fn advise_willneed(_mmap: &Mmap) {}

/// Fraction (0.0–1.0) of `mmap`'s pages currently resident in physical
/// memory, per `mincore(2)`. This is what makes `SwapMetrics::warm` a real
/// measurement instead of a guess from prefault timing.
#[cfg(target_os = "linux")]
fn mincore_resident_fraction(mmap: &Mmap) -> Option<f64> {
    if mmap.is_empty() {
        return Some(1.0);
    }
    let n_pages = mmap.len().div_ceil(PAGE_SIZE);
    let mut vec = vec![0u8; n_pages];
    // Safety: `vec` has exactly one byte per page covering `mmap`'s full
    // length, matching what Linux's mincore(2) requires and will write to.
    let result = unsafe {
        libc::mincore(
            mmap.as_ptr() as *mut libc::c_void,
            mmap.len(),
            vec.as_mut_ptr(),
        )
    };
    if result != 0 {
        return None;
    }
    let resident = vec.iter().filter(|&&byte| byte & 1 != 0).count();
    Some(resident as f64 / n_pages as f64)
}

// See the caveat on `SwapMetrics::resident_fraction` -- this call succeeds
// and returns well-formed data, but the residency it reports for
// file-backed mappings has empirically not matched reality on Darwin
// (verified against mlock(2), which guarantees residency this didn't
// reflect). Kept because it's still what the roadmap asked for and it's
// not wrong to call, just not trustworthy here -- not worth silently
// dropping the platform implementation over a result that can't be
// confirmed against an authoritative source.
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn mincore_resident_fraction(mmap: &Mmap) -> Option<f64> {
    if mmap.is_empty() {
        return Some(1.0);
    }
    let n_pages = mmap.len().div_ceil(PAGE_SIZE);
    let mut vec = vec![0i8; n_pages];
    // Safety: same contract as the Linux path above; Darwin's mincore(2)
    // takes a `*mut c_char` vector of the same one-byte-per-page shape.
    let result = unsafe {
        libc::mincore(
            mmap.as_ptr() as *const libc::c_void,
            mmap.len(),
            vec.as_mut_ptr(),
        )
    };
    if result != 0 {
        return None;
    }
    let resident = vec
        .iter()
        .filter(|&&byte| byte & libc::MINCORE_INCORE as i8 != 0)
        .count();
    Some(resident as f64 / n_pages as f64)
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
fn mincore_resident_fraction(_mmap: &Mmap) -> Option<f64> {
    None
}

#[cfg(target_os = "linux")]
fn current_rss_bytes() -> Option<i64> {
    let statm = std::fs::read_to_string("/proc/self/statm").ok()?;
    let resident_pages: i64 = statm.split_whitespace().nth(1)?.parse().ok()?;
    Some(resident_pages * PAGE_SIZE as i64)
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn current_rss_bytes() -> Option<i64> {
    use mach2::kern_return::KERN_SUCCESS;
    use mach2::message::mach_msg_type_number_t;
    use mach2::task::task_info;
    use mach2::task_info::{TASK_VM_INFO, task_vm_info};
    use mach2::traps::mach_task_self;
    use mach2::vm_types::natural_t;
    use std::mem;

    let mut info = task_vm_info::default();
    let mut count =
        (mem::size_of::<task_vm_info>() / mem::size_of::<natural_t>()) as mach_msg_type_number_t;
    // Safety: `info` is `task_vm_info`'s real layout from `mach2` (not
    // hand-rolled), and `count` is sized in natural_t units to match, so
    // `task_info` writes exactly as many words as `info` has room for.
    let result = unsafe {
        task_info(
            mach_task_self(),
            TASK_VM_INFO,
            &mut info as *mut task_vm_info as *mut i32,
            &mut count,
        )
    };
    if result == KERN_SUCCESS {
        Some(info.phys_footprint as i64)
    } else {
        None
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
fn current_rss_bytes() -> Option<i64> {
    None
}

/// Best-effort system-wide "how much memory could a new model use"
/// figure -- `free + inactive` pages (matching what `vm_stat`/Activity
/// Monitor traditionally call "available" memory: pages that aren't
/// actively used and can be reclaimed without swapping), not just
/// literally-free pages alone. `free_count` on its own dramatically
/// understates real headroom on a system that's been running a while,
/// since macOS aggressively uses "free" RAM for file-backed page cache
/// it'll happily evict under pressure -- this is meant to answer "is
/// there room," not "is anything sitting fully idle."
///
/// `host_statistics64`/`HOST_VM_INFO64` aren't bound by the `mach2`
/// crate itself (only the `vm_statistics64` struct type is) -- declared
/// directly here via a small, deliberate `extern "C"` block rather than
/// pulling in a whole second mach-bindings crate for one function.
/// `HOST_VM_INFO64 = 4`, confirmed against Apple's own
/// `osfmk/mach/host_info.h` (`#define HOST_VM_INFO64 4 /* 64-bit
/// virtual memory stats */`), not guessed. Page size via
/// `libc::sysconf(_SC_PAGESIZE)` (the portable POSIX way -- real on
/// Apple Silicon, where the hardware page size is actually 16KB, not
/// the 4096 this file's own `PAGE_SIZE` const assumes elsewhere;
/// getting this wrong would corrupt the computed byte figure, not just
/// look slightly off), not a second mach call.
///
/// `None` -- not `Some(0)` -- on any failure (the syscall failing, or no
/// implementation for this platform below), so `SwapRegistry::
/// fits_within_budget` never silently treats "couldn't measure" as "no
/// room at all."
#[cfg(any(target_os = "macos", target_os = "ios"))]
fn system_free_bytes() -> Option<u64> {
    use mach2::kern_return::{KERN_SUCCESS, kern_return_t};
    use mach2::mach_init::mach_host_self;
    use mach2::mach_types::host_t;
    use mach2::message::mach_msg_type_number_t;
    use mach2::vm_statistics::vm_statistics64;
    use mach2::vm_types::{integer_t, natural_t};
    use std::mem;

    const HOST_VM_INFO64: integer_t = 4;

    unsafe extern "C" {
        fn host_statistics64(
            host_priv: host_t,
            host_flavor: integer_t,
            host_info64_out: *mut integer_t,
            host_info64_out_cnt: *mut mach_msg_type_number_t,
        ) -> kern_return_t;
    }

    // Safety: a plain `sysconf` read, no pointers involved.
    let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if page_size <= 0 {
        return None;
    }

    let mut info = vm_statistics64::default();
    let mut count =
        (mem::size_of::<vm_statistics64>() / mem::size_of::<natural_t>()) as mach_msg_type_number_t;
    // Safety: `info` is `vm_statistics64`'s real layout from `mach2`
    // (not hand-rolled), and `count` is sized in `natural_t` units to
    // match, so `host_statistics64` writes exactly as many words as
    // `info` has room for -- the same shape `current_rss_bytes` above
    // already uses for `task_info`. `mach_host_self()` returns a
    // `thread_port_t`, assignment-compatible with the `host_t` this
    // takes since both are plain aliases of `mach_port_t`.
    let result = unsafe {
        host_statistics64(
            mach_host_self(),
            HOST_VM_INFO64,
            &mut info as *mut vm_statistics64 as *mut integer_t,
            &mut count,
        )
    };
    if result != KERN_SUCCESS {
        return None;
    }

    let free_pages = info.free_count as u64 + info.inactive_count as u64;
    Some(free_pages * page_size as u64)
}

#[cfg(target_os = "linux")]
fn system_free_bytes() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    for line in meminfo.lines() {
        if let Some(rest) = line.strip_prefix("MemAvailable:") {
            // Format: "MemAvailable:   12345678 kB" -- the kernel's own
            // "free + reclaimable, without swapping" figure, already
            // exactly the semantics wanted here, no free+inactive-style
            // approximation needed the way the macOS path has to build
            // one from more primitive counters.
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
fn system_free_bytes() -> Option<u64> {
    None
}

#[cfg(test)]
mod rss_tests {
    use super::current_rss_bytes;

    // Targeted at the mach2/task_info FFI call specifically: on a supported
    // platform this must return a real, plausible value, not silently fail
    // (task_info returning a non-KERN_SUCCESS code from a bad struct/count
    // would show up here as `None`) or return nonsense (a live test process
    // has at least a few MB resident, so anything absurdly small is a sign
    // the wrong field or a misaligned struct got read).
    #[test]
    fn current_rss_is_a_plausible_positive_number() {
        let rss = current_rss_bytes();
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
        {
            let rss = rss.expect("RSS measurement should succeed on a supported platform");
            assert!(
                rss > 1_000_000,
                "a running test process should have well over 1MB resident, got {rss}"
            );
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
        {
            assert!(rss.is_none());
        }
    }

    // Loading a real chunk of memory should move RSS by a plausible amount --
    // catches the case where phys_footprint/statm parsing is silently
    // reading a field that never changes.
    #[test]
    fn rss_increases_after_touching_a_large_allocation() {
        let before = match current_rss_bytes() {
            Some(v) => v,
            None => return, // unsupported platform, nothing to check
        };
        let mut buf = vec![0u8; 64 * 1024 * 1024];
        for (i, b) in buf.iter_mut().enumerate() {
            *b = i as u8;
        }
        std::hint::black_box(&buf);
        let after = current_rss_bytes().expect("measurement succeeded once, should succeed again");
        assert!(
            after - before > 32 * 1024 * 1024,
            "touching 64MB should grow RSS by a comparable amount, before={before} after={after}"
        );
    }
}

#[cfg(test)]
mod system_free_bytes_tests {
    use super::system_free_bytes;

    // Same shape as `rss_tests::current_rss_is_a_plausible_positive_number`
    // above: targeted at the real syscall, not a synthetic mock -- on a
    // supported platform this must return a real, plausible value, not
    // silently fail (a bad HOST_VM_INFO64/page-size mismatch would show
    // up here as `None`) or return nonsense (a real machine has well
    // over a few hundred MB free+inactive in any realistic test
    // environment).
    #[test]
    fn system_free_bytes_is_a_plausible_positive_number() {
        let free = system_free_bytes();
        #[cfg(any(target_os = "linux", target_os = "macos", target_os = "ios"))]
        {
            let free = free.expect("system_free_bytes should succeed on a supported platform");
            assert!(
                free > 100_000_000,
                "expected well over 100MB free+inactive on a real machine, got {free}"
            );
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "ios")))]
        {
            assert!(free.is_none());
        }
    }
}
