//! Two behaviors `model_stress_test` never exercises, because it runs
//! everything strictly sequentially, one client, one request at a time:
//!
//! 1. **Concurrent requests to different models.** `rampiped` serializes
//!    every decode behind one lock by design (see `bin/rampiped.rs`'s own
//!    module doc comment) -- this fires several requests at once, each
//!    against a different model, and shows how they actually interleave
//!    (queued near-simultaneously, finished one at a time), rather than
//!    just asserting each one individually succeeds.
//! 2. **Round-robin churn across models that don't all fit.** Repeatedly
//!    cycling through models whose combined VRAM exceeds the GPU forces
//!    eviction on (close to) every request -- this watches `Status`
//!    between requests to show the resident set and free VRAM actually
//!    changing, rather than inferring it indirectly from timing alone.
//!
//! Downloads the same three models `model_stress_test` uses (cached after
//! first run).
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features llama --bin rampiped -- \
//!         --socket /tmp/rampiped.sock
//!
//!     cargo run --release --features client --example model_concurrency_stress_test -- \
//!         /tmp/rampiped.sock [--rounds N]

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::client::RampipedClient;
use rampipe::protocol::WireSampling;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

const PROMPT: &str = "write a program in rust that serves as a multi-client pub/sub messaging \
framework, call the project mpipe. acceptance criteria: multiple clients can publish and \
subscribe to differnet topics and send messages to each topic and every subscriber receives \
them. Unit tests pass.";

// Deliberately smaller than model_stress_test's -- this is about
// observing serialization/eviction timing, not full program generation.
const MAX_NEW_TOKENS: i32 = 200;
const DEFAULT_ROUNDS: u32 = 2;

struct ModelSpec {
    label: &'static str,
    repo_owner: &'static str,
    repo_name: &'static str,
    filename: &'static str,
}

const MODELS: &[ModelSpec] = &[
    ModelSpec { label: "qwen2.5-coder-7b", repo_owner: "Qwen", repo_name: "Qwen2.5-Coder-7B-Instruct-GGUF", filename: "qwen2.5-coder-7b-instruct-q4_k_m.gguf" },
    ModelSpec { label: "qwen2.5-coder-14b", repo_owner: "Qwen", repo_name: "Qwen2.5-Coder-14B-Instruct-GGUF", filename: "qwen2.5-coder-14b-instruct-q4_k_m.gguf" },
    ModelSpec { label: "llama-3.1-8b", repo_owner: "bartowski", repo_name: "Meta-Llama-3.1-8B-Instruct-GGUF", filename: "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf" },
    // See model_stress_test.rs's own MODELS entry for this one -- same
    // model, same caveat (doesn't fit genie's 16GB VRAM, fits a Mac's
    // unified memory).
    ModelSpec { label: "qwen2.5-32b-q6kl", repo_owner: "bartowski", repo_name: "Qwen2.5-32B-Instruct-GGUF", filename: "Qwen2.5-32B-Instruct-Q6_K_L.gguf" },
];

fn download_model(spec: &ModelSpec) -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(spec.repo_owner, spec.repo_name);
    repo.download_file().filename(spec.filename).send().context("downloading GGUF file")
}

fn take_flag_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == flag)?;
    args.remove(pos);
    if pos < args.len() { Some(args.remove(pos)) } else { None }
}

/// Phase 1: fire one request per model at (as close to) the same instant,
/// via a `Barrier` so every thread's actual `generate()` call starts only
/// once every thread has finished connecting and downloading -- otherwise
/// an early thread's request could finish before a slow-to-download one
/// even started, defeating the point of testing concurrent arrival.
fn run_concurrent_phase(socket_path: &Path, models: &[(&'static str, PathBuf)]) -> Result<()> {
    println!("\n=== phase 1: concurrent requests to {} different models ===", models.len());
    let barrier = Arc::new(Barrier::new(models.len()));
    let start = Instant::now();

    let handles: Vec<_> = models
        .iter()
        .cloned()
        .map(|(label, model_path)| {
            let socket_path = socket_path.to_path_buf();
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || -> Result<(&'static str, u128, u128, usize)> {
                let client = RampipedClient::connect(&socket_path).context("connecting")?;
                barrier.wait();
                let queued_at_ms = start.elapsed().as_millis();
                let outcome = client
                    .generate(&model_path, PROMPT, MAX_NEW_TOKENS, WireSampling::Greedy, None, None, None)
                    .context("generate")?;
                let done_at_ms = start.elapsed().as_millis();
                Ok((label, queued_at_ms, done_at_ms, outcome.tokens_generated))
            })
        })
        .collect();

    let mut results: Vec<(&'static str, u128, u128, usize)> = Vec::new();
    for handle in handles {
        results.push(handle.join().expect("thread panicked")?);
    }
    results.sort_by_key(|&(_, _, done_at_ms, _)| done_at_ms);

    println!("{:<20} {:>12} {:>12} {:>8}", "model", "queued (ms)", "done (ms)", "tokens");
    for (label, queued_at_ms, done_at_ms, tokens) in &results {
        println!("{label:<20} {queued_at_ms:>12} {done_at_ms:>12} {tokens:>8}");
    }
    println!(
        "  (all queued near t=0; staggered \"done\" times are rampiped's single-decode-at-a-time \
         lock serializing them, not a bug -- see bin/rampiped.rs's own module doc comment)"
    );
    Ok(())
}

/// Phase 2: sequentially cycle through every model, `rounds` times,
/// querying `Status` after each request so the resident set and free VRAM
/// are visible changing in response to real eviction, not inferred from
/// timing alone.
fn run_churn_phase(client: &RampipedClient, models: &[(&'static str, PathBuf)], rounds: u32) -> Result<()> {
    println!("\n=== phase 2: round-robin churn, {rounds} round(s) across {} models ===", models.len());
    println!("{:<20} {:>6} {:>10} {:>8}  {:<15} resident models", "model", "round", "wall (ms)", "tokens", "gpu free");
    for round in 1..=rounds {
        for (label, model_path) in models {
            let start = Instant::now();
            let outcome = client
                .generate(model_path, PROMPT, MAX_NEW_TOKENS, WireSampling::Greedy, None, None, None)
                .context("generate")?;
            let wall_ms = start.elapsed().as_millis();
            let status = client.status().context("status")?;
            let resident: Vec<String> =
                status.models.iter().filter_map(|m| m.path.file_name().map(|n| n.to_string_lossy().into_owned())).collect();
            let gpu_free = status.gpu_free_bytes.map(|b| format!("{:.1}GB", b as f64 / 1e9)).unwrap_or_else(|| "n/a".to_string());
            println!("{label:<20} {round:>6} {wall_ms:>10} {:>8}  {gpu_free:<15} {resident:?}", outcome.tokens_generated);
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let rounds: u32 =
        take_flag_value(&mut args, "--rounds").map(|s| s.parse().context("--rounds must be an integer")).transpose()?.unwrap_or(DEFAULT_ROUNDS);
    let socket_path = PathBuf::from(args.first().cloned().context("usage: model_concurrency_stress_test <socket-path> [--rounds N]")?);

    println!("=== downloading/locating models (cached after first run) ===");
    let mut models: Vec<(&'static str, PathBuf)> = Vec::new();
    for spec in MODELS {
        println!("  {}/{}/{}...", spec.repo_owner, spec.repo_name, spec.filename);
        models.push((spec.label, download_model(spec)?));
    }

    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    run_concurrent_phase(&socket_path, &models)?;
    run_churn_phase(&client, &models, rounds)?;

    println!("\nall phases completed");
    Ok(())
}
