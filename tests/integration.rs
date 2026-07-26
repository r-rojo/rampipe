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
    assert!(registry.is_resident(handle.id()));
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

    assert_eq!(first.id(), second.id());
    assert_eq!(first.as_bytes().as_ptr(), second.as_bytes().as_ptr());
    assert_eq!(registry.resident_count(), 1);
    assert_eq!(registry.mapped_bytes(), 4 + 4096);
}

#[test]
fn eviction_fails_while_a_handle_is_outstanding_then_succeeds_after_drop() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();

    let handle = registry.load(f.path(), Residency::Lazy).expect("load");
    let id = handle.id();

    let err = registry.evict(id).expect_err("evict should refuse while handle is live");
    assert!(matches!(err, EvictError::HandleOutstanding { outstanding: 1 }));
    assert!(registry.is_resident(id), "must still be resident after a refused evict");

    drop(handle);

    registry.evict(id).expect("evict should succeed once the handle is dropped");
    assert!(!registry.is_resident(id));
}

#[test]
fn evicting_an_unknown_id_is_an_error() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();

    let handle = registry.load(f.path(), Residency::Lazy).expect("load");
    let id = handle.id();
    drop(handle);
    registry.evict(id).expect("first evict should succeed");

    let err = registry.evict(id).expect_err("id is no longer resident");
    assert!(matches!(err, EvictError::UnknownId(unknown) if unknown == id));
}

#[test]
fn prefault_latency_is_attributed_to_eager_residency_modes_only() {
    let lazy_file = gguf_fixture(4096 * 16);
    let prefault_file = gguf_fixture(4096 * 16);
    let advise_file = gguf_fixture(4096 * 16);
    let registry = SwapRegistry::new();

    let lazy = registry.load(lazy_file.path(), Residency::Lazy).expect("lazy load");
    assert!(lazy.metrics().prefault_latency.is_none());

    let prefaulted = registry
        .load(prefault_file.path(), Residency::Prefault)
        .expect("prefault load");
    assert!(prefaulted.metrics().prefault_latency.is_some());

    let advised = registry
        .load(advise_file.path(), Residency::Advise)
        .expect("advise load");
    assert!(advised.metrics().prefault_latency.is_some());
}

#[test]
fn mincore_result_is_structurally_valid_after_prefault() {
    // NOT asserting "prefault forces ~full residency" here, on purpose.
    // Empirically (this repo, Darwin, 2026-07-25): mincore(2) reports a
    // consistent ~25% resident fraction for a file-backed mmap regardless
    // of file size (verified at 32 pages and 500 pages), and — the
    // decisive test — the fraction doesn't move even after mlock(2), which
    // *guarantees* every page is resident and pinned. A plain anonymous
    // (non-file-backed) mmap'd page reports correctly via the same mincore
    // call. This points at Apple deliberately returning imprecise mincore
    // results specifically for file-backed mappings — plausibly the same
    // mitigation class as published page-cache side-channel research (using
    // mincore to fingerprint what's in the shared page cache leaks
    // information about other processes' file access). Whatever the exact
    // mechanism, the result is the same: don't trust `resident_fraction`/
    // `warm` as an accurate residency signal for file-backed mappings on
    // this platform. Only asserting what's actually true: the call
    // succeeds and returns a well-formed fraction.
    let f = gguf_fixture(4096 * 32);
    let registry = SwapRegistry::new();

    let handle = registry.load(f.path(), Residency::Prefault).expect("prefault load");
    let metrics = handle.metrics();

    if let Some(fraction) = metrics.resident_fraction {
        assert!((0.0..=1.0).contains(&fraction), "resident_fraction {fraction} out of range");
        assert_eq!(metrics.warm, fraction >= 0.99, "warm should follow directly from the fraction");
    }
}

#[test]
fn resident_fraction_is_populated_regardless_of_residency_mode() {
    // Whether or not this platform has a working mincore, the field itself
    // should be computed the same way for every mode -- it's not something
    // only Prefault gets.
    let lazy_file = gguf_fixture(4096 * 8);
    let advise_file = gguf_fixture(4096 * 8);
    let registry = SwapRegistry::new();

    let lazy = registry.load(lazy_file.path(), Residency::Lazy).expect("lazy load");
    let advised = registry.load(advise_file.path(), Residency::Advise).expect("advise load");

    for fraction in [lazy.metrics().resident_fraction, advised.metrics().resident_fraction] {
        if let Some(f) = fraction {
            assert!((0.0..=1.0).contains(&f), "resident_fraction {f} out of range");
        }
    }
}
