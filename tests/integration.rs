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
    let handle = registry
        .load(f.path(), Residency::Lazy)
        .expect("load should succeed");

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

    let first = registry
        .load(f.path(), Residency::Lazy)
        .expect("first load");
    let second = registry
        .load(f.path(), Residency::Lazy)
        .expect("second load");

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

    let err = registry
        .evict(id)
        .expect_err("evict should refuse while handle is live");
    assert!(matches!(
        err,
        EvictError::HandleOutstanding { outstanding: 1 }
    ));
    assert!(
        registry.is_resident(id),
        "must still be resident after a refused evict"
    );

    drop(handle);

    registry
        .evict(id)
        .expect("evict should succeed once the handle is dropped");
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

    let lazy = registry
        .load(lazy_file.path(), Residency::Lazy)
        .expect("lazy load");
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

    let handle = registry
        .load(f.path(), Residency::Prefault)
        .expect("prefault load");
    let metrics = handle.metrics();

    if let Some(fraction) = metrics.resident_fraction {
        assert!(
            (0.0..=1.0).contains(&fraction),
            "resident_fraction {fraction} out of range"
        );
        assert_eq!(
            metrics.warm,
            fraction >= 0.99,
            "warm should follow directly from the fraction"
        );
    }
}

#[test]
fn resident_ids_by_lru_orders_by_most_recent_access() {
    let file_a = gguf_fixture(4096);
    let file_b = gguf_fixture(4096);
    let file_c = gguf_fixture(4096);
    let registry = SwapRegistry::new();

    let a = registry
        .load(file_a.path(), Residency::Lazy)
        .expect("load a");
    let b = registry
        .load(file_b.path(), Residency::Lazy)
        .expect("load b");
    let c = registry
        .load(file_c.path(), Residency::Lazy)
        .expect("load c");

    assert_eq!(
        registry.resident_ids_by_lru(),
        vec![a.id(), b.id(), c.id()],
        "load order is access order with nothing re-touched yet"
    );

    // Re-loading `a` (a cache hit, same path) counts as an access too --
    // it should move to the most-recently-used end, not stay oldest.
    registry
        .load(file_a.path(), Residency::Lazy)
        .expect("re-load a");
    assert_eq!(
        registry.resident_ids_by_lru(),
        vec![b.id(), c.id(), a.id()],
        "re-touching a should move it to most-recently-used"
    );
}

#[test]
fn resident_ids_by_lru_drops_an_evicted_model() {
    let file_a = gguf_fixture(4096);
    let file_b = gguf_fixture(4096);
    let registry = SwapRegistry::new();

    let a = registry
        .load(file_a.path(), Residency::Lazy)
        .expect("load a");
    let b = registry
        .load(file_b.path(), Residency::Lazy)
        .expect("load b");
    drop(a);
    let a_id = registry.resident_ids_by_lru()[0];
    registry.evict(a_id).expect("evict a");

    assert_eq!(registry.resident_ids_by_lru(), vec![b.id()]);
}

#[test]
fn fits_within_budget_is_true_for_a_tiny_model_at_a_generous_budget() {
    let registry = SwapRegistry::new();
    // Best-effort: only checked where `system_free_bytes` actually has an
    // implementation (see that function's own doc comment) -- `None`
    // elsewhere is itself the correct, honest answer, not a failure.
    if let Some(fits) = registry.fits_within_budget(1024, 1.0) {
        assert!(
            fits,
            "1KB should trivially fit under a 100% budget on any real machine"
        );
    }
}

#[test]
fn fits_within_budget_is_false_for_an_absurdly_large_model() {
    let registry = SwapRegistry::new();
    if let Some(fits) = registry.fits_within_budget(u64::MAX / 2, 0.8) {
        assert!(
            !fits,
            "no real machine has ~9 exabytes of free+resident memory to spare"
        );
    }
}

#[test]
fn fits_within_budget_counts_already_resident_bytes_toward_the_ceiling() {
    // A budget of 0.0 means "nothing may ever fit, including what's
    // already resident" -- proves resident bytes are actually counted in
    // the ceiling calculation, not just free system memory alone (a
    // model already loaded still occupies real space).
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();
    registry.load(f.path(), Residency::Lazy).expect("load");

    if let Some(fits) = registry.fits_within_budget(1, 0.0) {
        assert!(!fits, "a 0.0 budget fraction should never fit anything");
    }
}

#[test]
fn device_bytes_is_none_until_recorded() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();
    let handle = registry.load(f.path(), Residency::Lazy).expect("load");

    assert_eq!(
        handle.device_bytes(),
        None,
        "nothing has reported a GPU measurement yet"
    );
    registry.record_device_bytes(handle.id(), 12_345);
    assert_eq!(
        handle.device_bytes(),
        Some(12_345),
        "a handle obtained before recording still sees the update, via the shared Resident"
    );
}

#[test]
fn record_device_bytes_is_a_noop_for_an_id_that_was_evicted() {
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();
    let handle = registry.load(f.path(), Residency::Lazy).expect("load");
    let id = handle.id();
    drop(handle);
    registry.evict(id).expect("evict");

    // Must not panic -- the whole point of this being a no-op rather than
    // an error is that a caller measuring a GPU load that raced with an
    // eviction shouldn't have to handle a new failure mode for it.
    registry.record_device_bytes(id, 999);
    assert_eq!(registry.device_resident_bytes(), 0);
}

#[test]
fn device_resident_bytes_sums_only_models_with_recorded_bytes() {
    let cpu_only = gguf_fixture(4096);
    let gpu_backed = gguf_fixture(4096);
    let registry = SwapRegistry::new();
    registry
        .load(cpu_only.path(), Residency::Lazy)
        .expect("load");
    let gpu_handle = registry
        .load(gpu_backed.path(), Residency::Lazy)
        .expect("load");
    registry.record_device_bytes(gpu_handle.id(), 500);

    assert_eq!(
        registry.device_resident_bytes(),
        500,
        "the CPU-only model should contribute nothing"
    );
}

#[test]
fn fits_within_device_budget_is_true_for_a_tiny_model_at_a_generous_budget() {
    let registry = SwapRegistry::new();
    assert!(
        registry.fits_within_device_budget(1024, 16 * 1024 * 1024 * 1024, 1.0),
        "1KB should trivially fit in 16GB free at a 100% budget"
    );
}

#[test]
fn fits_within_device_budget_is_false_when_little_device_memory_is_free() {
    let registry = SwapRegistry::new();
    assert!(
        !registry.fits_within_device_budget(8 * 1024 * 1024 * 1024, 1024, 0.8),
        "an 8GB model can't fit in 1KB of free device memory"
    );
}

#[test]
fn fits_within_device_budget_counts_recorded_device_bytes_toward_the_ceiling() {
    // A budget of 0.0 means "nothing may ever fit, including what's
    // already resident" -- proves recorded device bytes are actually
    // counted in the ceiling, not just free device memory alone, mirroring
    // `fits_within_budget_counts_already_resident_bytes_toward_the_ceiling`.
    let f = gguf_fixture(4096);
    let registry = SwapRegistry::new();
    let handle = registry.load(f.path(), Residency::Lazy).expect("load");
    registry.record_device_bytes(handle.id(), 4 * 1024 * 1024 * 1024);

    assert!(
        !registry.fits_within_device_budget(1, 16 * 1024 * 1024 * 1024, 0.0),
        "a 0.0 budget fraction should never fit anything"
    );
}

#[test]
fn resident_fraction_is_populated_regardless_of_residency_mode() {
    // Whether or not this platform has a working mincore, the field itself
    // should be computed the same way for every mode -- it's not something
    // only Prefault gets.
    let lazy_file = gguf_fixture(4096 * 8);
    let advise_file = gguf_fixture(4096 * 8);
    let registry = SwapRegistry::new();

    let lazy = registry
        .load(lazy_file.path(), Residency::Lazy)
        .expect("lazy load");
    let advised = registry
        .load(advise_file.path(), Residency::Advise)
        .expect("advise load");

    for f in [
        lazy.metrics().resident_fraction,
        advised.metrics().resident_fraction,
    ]
    .into_iter()
    .flatten()
    {
        assert!(
            (0.0..=1.0).contains(&f),
            "resident_fraction {f} out of range"
        );
    }
}
