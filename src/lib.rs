use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[cfg(feature = "llama")]
pub mod llama;

const GGUF_MAGIC: [u8; 4] = *b"GGUF";
const PAGE_SIZE: usize = 4096;
// Crude placeholder for the `warm` heuristic below — the roadmap's own open
// item is to replace this with a real mincore(2) check instead of guessing
// from how fast the prefault loop ran.
const WARM_THRESHOLD: Duration = Duration::from_micros(500);

/// How eagerly to bring a newly-mapped model's pages into physical memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residency {
    /// mmap only; the OS faults pages in on first touch. Fast to "loaded",
    /// but the first real access to each page pays a page-fault during
    /// inference instead of during load.
    Lazy,
    /// mmap, then touch every page immediately so the page-fault cost is
    /// paid up front instead of mid-generation.
    Prefault,
}

#[derive(Debug, Clone, Copy)]
pub struct SwapMetrics {
    pub map_latency: Duration,
    /// `Some` only when loaded with `Residency::Prefault`.
    pub prefault_latency: Option<Duration>,
    pub mapped_bytes: usize,
    /// Process RSS after mapping minus RSS before, in bytes. `None` on
    /// platforms without an RSS measurement implemented (see
    /// `current_rss_bytes`).
    pub rss_delta_bytes: Option<i64>,
    /// Crude heuristic: true if this was a `Prefault` load whose page-touch
    /// loop finished suspiciously fast, suggesting the pages were already
    /// cached by the OS. Not a real residency check — replace with
    /// `mincore(2)` before trusting this.
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

/// Stable identifier for a resident model. Cheap to copy, and — unlike a
/// `PathBuf` — safe to hand across an FFI boundary as a plain `u64`, which
/// matters once Phase 4 (UniFFI/Swift) exists to hand it to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(u64);

struct Resident {
    path: PathBuf,
    mmap: Mmap,
    metrics: SwapMetrics,
}

/// A live reference to a mapped model. While any handle for a given model is
/// alive, `SwapRegistry::evict` for its id fails — this is the safety
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
    pub fn load(&self, path: impl AsRef<Path>, residency: Residency) -> Result<ModelHandle, LoadError> {
        let path = path.as_ref().to_path_buf();
        let mut state = self.state.lock().expect("rampipe registry lock poisoned");

        if let Some(&id) = state.by_path.get(&path) {
            let inner = state.resident.get(&id).expect("by_path/resident desync").clone();
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
        };

        let rss_after = current_rss_bytes();
        let rss_delta_bytes = match (rss_before, rss_after) {
            (Some(before), Some(after)) => Some(after - before),
            _ => None,
        };

        let warm = matches!(prefault_latency, Some(d) if d < WARM_THRESHOLD);

        let metrics = SwapMetrics {
            map_latency,
            prefault_latency,
            mapped_bytes,
            rss_delta_bytes,
            warm,
        };

        let id = ModelId(state.next_id);
        state.next_id += 1;
        let inner = Arc::new(Resident { path: path.clone(), mmap, metrics });
        state.resident.insert(id, inner.clone());
        state.by_path.insert(path, id);

        Ok(ModelHandle { id, inner })
    }

    /// Evicts the model identified by `id`, freeing its mapping — but only
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
        self.state.lock().expect("rampipe registry lock poisoned").resident.len()
    }

    /// Total mapped bytes across resident models. Not RSS — this is the
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
}

impl Default for SwapRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Touches one byte per page to force the OS to fault every page of `mmap`
/// into physical memory. Crude on purpose for Phase 1 — the roadmap's own
/// open item is to compare this against `madvise(WILLNEED)`.
fn prefault(mmap: &Mmap) {
    let mut sum: u64 = 0;
    for i in (0..mmap.len()).step_by(PAGE_SIZE) {
        sum = sum.wrapping_add(mmap[i] as u64);
    }
    std::hint::black_box(sum);
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

    // Loading a real chunk of memory should move RSS by a plausible amount —
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
