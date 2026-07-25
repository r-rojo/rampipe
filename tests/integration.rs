use rampipe::{EvictError, LoadError, Residency, SwapRegistry};
use std::io::Write;
use tempfile::NamedTempFile;

/// A minimal file that passes rampipe's magic check: GGUF magic followed by
/// enough padding to span a few pages, so prefault has real work to do.
fn gguf_fixture(extra_bytes: usize) -> NamedTempFile {
    let mut f = NamedTempFile::new().expect("create temp file");
    f.write_all(b"GGUF").expect("write magic");
    f.write_all(&vec![0u8; extra_bytes]).expect("write padding");
    f.flush().expect("flush temp file");
    f
}

#[test]
fn rejects_non_gguf_files() {
    let mut f = NamedTempFile::new().expect("create temp file");
    f.write_all(b"NOPE").expect("write bad magic");
    f.flush().expect("flush");

    let registry = SwapRegistry::new();
    let err = registry
        .load(f.path(), Residency::Lazy)
        .expect_err("should reject non-GGUF file");
    assert!(matches!(err, LoadError::BadMagic { found } if &found == b"NOPE"));
}

#[test]
fn rejects_empty_files() {
    let f = NamedTempFile::new().expect("create temp file");
    let registry = SwapRegistry::new();
    let err = registry
        .load(f.path(), Residency::Lazy)
        .expect_err("should reject empty file");
    assert!(matches!(err, LoadError::Empty));
}

#[test]
fn loads_a_valid_gguf_file() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();
    let handle = registry.load(f.path(), Residency::Lazy).expect("load should succeed");

    assert_eq!(&handle.as_bytes()[..4], b"GGUF");
    assert_eq!(handle.metrics().mapped_bytes, 4 + 4096);
    assert!(registry.is_resident(f.path()));
}

#[test]
fn large_file_maps_quickly_regardless_of_size() {
    // A sparse ~256MB file: mmap should map it near-instantly since nothing
    // is actually read up front. A naive full-buffer read would not be this
    // fast for a file this size.
    let f = NamedTempFile::new().expect("create temp file");
    {
        use std::io::Seek;
        let mut file = f.reopen().expect("reopen");
        file.write_all(b"GGUF").expect("write magic");
        file.seek(std::io::SeekFrom::Start(256 * 1024 * 1024 - 1))
            .expect("seek");
        file.write_all(&[0u8]).expect("extend file");
    }

    let registry = SwapRegistry::new();
    let start = std::time::Instant::now();
    let handle = registry
        .load(f.path(), Residency::Lazy)
        .expect("load should succeed");
    let elapsed = start.elapsed();

    assert_eq!(handle.metrics().mapped_bytes, 256 * 1024 * 1024);
    assert!(
        elapsed < std::time::Duration::from_millis(200),
        "mmap of a sparse 256MB file took {elapsed:?}, expected near-instant \
         (a full-buffer read would be the thing to suspect if this fails)"
    );
}

#[test]
fn repeated_load_of_same_path_dedups_to_one_mapping() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();

    let first = registry.load(f.path(), Residency::Lazy).expect("first load");
    let second = registry.load(f.path(), Residency::Lazy).expect("second load");

    assert_eq!(first.as_bytes().as_ptr(), second.as_bytes().as_ptr());
}

#[test]
fn eviction_fails_while_a_handle_is_outstanding_then_succeeds_after_drop() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();

    let handle = registry.load(f.path(), Residency::Lazy).expect("load");

    let err = registry.evict(f.path()).expect_err("evict should refuse while handle is live");
    assert!(matches!(err, EvictError::HandleOutstanding { outstanding: 1 }));
    assert!(registry.is_resident(f.path()), "must still be resident after a refused evict");

    drop(handle);

    registry.evict(f.path()).expect("evict should succeed once the handle is dropped");
    assert!(!registry.is_resident(f.path()));
}

#[test]
fn evicting_a_path_that_was_never_loaded_is_an_error() {
    let registry = SwapRegistry::new();
    let err = registry.evict("/nonexistent/path.gguf").expect_err("should fail");
    assert!(matches!(err, EvictError::NotResident));
}

#[test]
fn prefault_latency_is_attributed_only_to_prefault_residency() {
    let lazy_file = gguf_fixture(4096 * 16);
    let prefault_file = gguf_fixture(4096 * 16);
    let registry = SwapRegistry::new();

    let lazy = registry.load(lazy_file.path(), Residency::Lazy).expect("lazy load");
    assert!(lazy.metrics().prefault_latency.is_none());

    let prefaulted = registry
        .load(prefault_file.path(), Residency::Prefault)
        .expect("prefault load");
    assert!(prefaulted.metrics().prefault_latency.is_some());
}
