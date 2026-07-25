use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

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
    #[error("path is not resident")]
    NotResident,
    #[error("{outstanding} handle(s) still outstanding, refusing to evict")]
    HandleOutstanding { outstanding: usize },
}

struct Resident {
    mmap: Arc<Mmap>,
    metrics: SwapMetrics,
}

/// A live reference to a mapped model. While any handle for a given path is
/// alive, `SwapRegistry::evict` for that path fails — this is the safety
/// invariant that makes eviction safe under concurrent generation: a live
/// generation holds a handle, so the scheduler cannot unmap weights under it.
#[derive(Debug, Clone)]
pub struct ModelHandle {
    path: PathBuf,
    mmap: Arc<Mmap>,
    metrics: SwapMetrics,
}

impl ModelHandle {
    pub fn as_bytes(&self) -> &[u8] {
        &self.mmap
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn metrics(&self) -> SwapMetrics {
        self.metrics
    }
}

pub struct SwapRegistry {
    resident: Mutex<HashMap<PathBuf, Resident>>,
}

impl SwapRegistry {
    pub fn new() -> Self {
        SwapRegistry {
            resident: Mutex::new(HashMap::new()),
        }
    }

    /// Maps `path` into memory and validates it as a GGUF file. If `path` is
    /// already resident, hands out another handle to the existing mapping
    /// instead of mapping it again.
    pub fn load(&self, path: impl AsRef<Path>, residency: Residency) -> Result<ModelHandle, LoadError> {
        let path = path.as_ref().to_path_buf();
        let mut table = self.resident.lock().expect("rampipe registry lock poisoned");

        if let Some(entry) = table.get(&path) {
            return Ok(ModelHandle {
                path,
                mmap: entry.mmap.clone(),
                metrics: entry.metrics,
            });
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
        let mmap = Arc::new(mmap);
        table.insert(
            path.clone(),
            Resident {
                mmap: mmap.clone(),
                metrics,
            },
        );

        Ok(ModelHandle { path, mmap, metrics })
    }

    /// Evicts `path`, freeing its mapping — but only if no `ModelHandle` for
    /// it is still alive. This is the invariant that protects an in-flight
    /// generation from having its weights unmapped out from under it.
    pub fn evict(&self, path: impl AsRef<Path>) -> Result<(), EvictError> {
        let path = path.as_ref();
        let mut table = self.resident.lock().expect("rampipe registry lock poisoned");
        let Some(entry) = table.get(path) else {
            return Err(EvictError::NotResident);
        };
        // The registry's own `Resident` entry holds one strong reference;
        // anything beyond that is an outstanding `ModelHandle`.
        let outstanding = Arc::strong_count(&entry.mmap) - 1;
        if outstanding > 0 {
            return Err(EvictError::HandleOutstanding { outstanding });
        }
        table.remove(path);
        Ok(())
    }

    pub fn is_resident(&self, path: impl AsRef<Path>) -> bool {
        self.resident
            .lock()
            .expect("rampipe registry lock poisoned")
            .contains_key(path.as_ref())
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
