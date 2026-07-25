//! Loads the same GGUF model twice — once with `Residency::Lazy`, once
//! with `Residency::Prefault` — through two independent `SwapRegistry`s
//! (independent so neither load can dedup-hit the other), then runs a real
//! generation against each and reports time-to-first-token alongside the
//! residency metrics. The point: page-in cost should show up as slower
//! TTFT for the `Lazy` load relative to `Prefault`, since prefaulting
//! warms the same OS page cache llama.cpp's own (separate) mmap reads
//! from.
//!
//! Caveat, stated plainly: this doesn't control for OS filesystem cache
//! state going into the run (dropping the page cache needs root on
//! macOS), so on a machine where the file is already cached from a
//! previous run, the Lazy/Prefault gap will be smaller than a genuine
//! cold start. Treat this as illustrative, not a rigorous cold-start
//! benchmark — that's still open work.
//!
//!     cargo run --release --features llama --example residency_vs_ttft

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::llama::LlamaSession;
use rampipe::{Residency, SwapRegistry};
use std::path::PathBuf;

const REPO_OWNER: &str = "Qwen";
const REPO_NAME: &str = "Qwen2.5-0.5B-Instruct-GGUF";
const FILENAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const PROMPT: &str = "The old lighthouse keeper";
const MAX_NEW_TOKENS: i32 = 40;

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file()
        .filename(FILENAME)
        .send()
        .context("downloading GGUF file")
}

fn run(label: &str, model_path: &PathBuf, backend: &llama_cpp_2::llama_backend::LlamaBackend, residency: Residency) -> Result<()> {
    let registry = SwapRegistry::new();
    let session = LlamaSession::load(&registry, backend, model_path, residency)
        .with_context(|| format!("loading session for {label}"))?;

    let metrics = session.metrics();
    let result = session
        .generate(backend, PROMPT, MAX_NEW_TOKENS)
        .with_context(|| format!("generating for {label}"))?;

    println!("=== {label} ===");
    println!("  map_latency:       {:?}", metrics.map_latency);
    println!("  prefault_latency:  {:?}", metrics.prefault_latency);
    println!("  mapped_bytes:      {}", metrics.mapped_bytes);
    println!("  rss_delta_bytes:   {:?}", metrics.rss_delta_bytes);
    println!("  warm:              {}", metrics.warm);
    println!("  time_to_first_token: {:?}", result.time_to_first_token);
    println!("  tokens_generated:  {}", result.tokens_generated);
    println!("  text: {}{}", PROMPT, result.text);
    println!();

    Ok(())
}

fn main() -> Result<()> {
    println!("Downloading/locating {REPO_OWNER}/{REPO_NAME}/{FILENAME} (cached after first run)...");
    let model_path = download_model()?;

    let backend = llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?;

    run("Lazy", &model_path, &backend, Residency::Lazy)?;
    run("Prefault", &model_path, &backend, Residency::Prefault)?;

    Ok(())
}
