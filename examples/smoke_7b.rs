//! 7B-class smoke test — everything tested against `rampipe::llama` so far
//! (Phase 1b, the mincore/madvise work) used Qwen2.5-0.5B-Instruct, a
//! ~270MB bf16 / ~491MB Q4_K_M model. That's roughly 14x smaller than the
//! 7B-class GGUFs a real orchestrator backend (taskpipe) would actually
//! load. This exists to answer one question before taskpipe's LocalBackend
//! gets built around an untested assumption: does the same `SwapRegistry`
//! + `LlamaSession` path hold up at 7B/Q4_K_M scale — real mmap of a
//! ~4.7GB file, real Metal offload, real generation — or does something
//! about the jump in size break an assumption that happened to hold at
//! 0.5B?
//!
//! Loads once with `Residency::Prefault` (what a long-lived backend that
//! serves many requests off one resident model would actually want — pay
//! the page-in cost once at startup, not on a user's first request), then
//! runs two generate() calls against the *same* session to also sanity-check
//! the "load once, serve many requests" shape a real backend needs — the
//! second call's TTFT isn't confounded by first-touch page-in cost the way
//! the first one's would be.
//!
//!     cargo run --release --features llama --example smoke_7b

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::llama::LlamaSession;
use rampipe::{Residency, SwapRegistry};
use std::path::PathBuf;
use std::time::Instant;

const REPO_OWNER: &str = "bartowski";
const REPO_NAME: &str = "Qwen2.5-7B-Instruct-GGUF";
const FILENAME: &str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf";
const MAX_NEW_TOKENS: i32 = 60;

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file()
        .filename(FILENAME)
        .send()
        .context("downloading GGUF file (~4.7GB, first run only)")
}

fn main() -> Result<()> {
    println!(
        "Downloading/locating {REPO_OWNER}/{REPO_NAME}/{FILENAME} (~4.7GB, cached after first run)..."
    );
    let model_path = download_model()?;

    let backend = std::sync::Arc::new(
        llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?,
    );
    let registry = SwapRegistry::new();

    println!("Loading (Residency::Prefault)...");
    let load_start = Instant::now();
    let session = LlamaSession::load(&registry, backend, &model_path, Residency::Prefault)
        .context("loading 7B session")?;
    let load_wall_time = load_start.elapsed();

    let metrics = session.metrics();
    println!("\n=== SwapRegistry metrics ===");
    println!("  map_latency:       {:?}", metrics.map_latency);
    println!("  prefault_latency:  {:?}", metrics.prefault_latency);
    println!(
        "  mapped_bytes:      {} ({:.2} GB)",
        metrics.mapped_bytes,
        metrics.mapped_bytes as f64 / 1e9
    );
    println!("  rss_delta_bytes:   {:?}", metrics.rss_delta_bytes);
    println!(
        "  resident_fraction: {:?} (untrustworthy on Darwin, see doc comment)",
        metrics.resident_fraction
    );
    println!(
        "  total load() wall time (registry.load + LlamaModel::load_from_file): {load_wall_time:?}"
    );

    for (label, prompt) in [
        ("request 1", "The old lighthouse keeper"),
        (
            "request 2 (same session)",
            "Explain what a ring buffer is in one paragraph.",
        ),
    ] {
        println!("\n=== {label} ===");
        let gen_start = Instant::now();
        let result = session
            .generate(
                prompt,
                MAX_NEW_TOKENS,
                rampipe::llama::Sampling::Greedy,
                None,
                None,
                None,
            )
            .with_context(|| format!("generating for {label}"))?;
        let total_wall_time = gen_start.elapsed();
        let tok_per_sec =
            result.tokens_generated as f64 / total_wall_time.as_secs_f64().max(f64::EPSILON);
        println!("  time_to_first_token: {:?}", result.time_to_first_token);
        println!("  total generate() wall time: {total_wall_time:?}");
        println!("  tokens_generated:    {}", result.tokens_generated);
        println!("  throughput: {tok_per_sec:.1} tok/s");
        println!("  prompt: {prompt}");
        println!("  output: {}", result.text);
    }

    Ok(())
}
