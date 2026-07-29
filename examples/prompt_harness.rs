//! Feed an arbitrary prompt straight to the same model/config
//! `taskpipe`'s `LocalBackend` actually uses, and see the raw response —
//! for iterating on prompt wording (e.g. task 2/3 of the piper roadmap:
//! a hallucinated `tempfile` import, hallucinated `crossterm::terminal`
//! function names) without going through a full `taskpipe` run each
//! time (worktree setup, `cargo build`/`test`, retries...) just to see
//! whether a reworded prompt changes what the model writes.
//!
//! Deliberately reuses the exact same model repo/filename and
//! `LOCAL_MAX_NEW_TOKENS` value `taskpipe`'s `backend.rs` uses (kept in
//! sync by comment, not by sharing code across crates — if either
//! changes, update the other) — a harness testing against a different
//! model or token budget than what actually runs in taskpipe would be
//! misleading, not useful.
//!
//! Usage:
//!     cargo run --release --features llama --example prompt_harness -- <prompt-file> [--seed N]
//!
//! `<prompt-file>` is read whole and sent as-is — write/edit it in a
//! real editor between runs. Without `--seed`, uses `Sampling::Greedy`
//! (deterministic, matching a task's first attempt in the real retry
//! loop). With `--seed N`, uses `Sampling::Temperature` at the same
//! temperature/top_k `LocalBackend` uses for retries — useful for
//! checking whether a prompt holds up across the kind of variation a
//! real retry would actually see, not just the one greedy completion.

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;
use rampipe::llama::{LlamaSession, Sampling};
use rampipe::Residency;
use std::path::PathBuf;
use std::time::Instant;

const REPO_OWNER: &str = "bartowski";
const REPO_NAME: &str = "Qwen2.5-7B-Instruct-GGUF";
const FILENAME: &str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf";

// Kept in sync by comment with `taskpipe::backend::LOCAL_MAX_NEW_TOKENS`
// — if that changes, this should too, or a prompt that fits here could
// still get truncated for real in taskpipe (or vice versa).
const LOCAL_MAX_NEW_TOKENS: i32 = 1400;

// Matches `LocalBackend::sampling_for_attempt`'s retry values exactly —
// see that function's own doc comment for why these specific numbers.
const RETRY_TEMPERATURE: f32 = 0.7;
const RETRY_TOP_K: i32 = 40;

fn parse_args() -> Result<(PathBuf, Sampling)> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let sampling = match args.iter().position(|a| a == "--seed") {
        Some(idx) => {
            if idx + 1 >= args.len() {
                bail!("--seed requires a value");
            }
            let seed: u32 = args.remove(idx + 1).parse().context("--seed must be a u32")?;
            args.remove(idx);
            Sampling::Temperature { temperature: RETRY_TEMPERATURE, top_k: RETRY_TOP_K, seed }
        }
        None => Sampling::Greedy,
    };

    let prompt_file = args
        .into_iter()
        .next()
        .context("usage: prompt_harness <prompt-file> [--seed N]  (e.g. prompt_harness ./prompt.txt)")?;
    Ok((PathBuf::from(prompt_file), sampling))
}

fn resolve_model_path() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file()
        .filename(FILENAME)
        .send()
        .context("resolving model file (should be a cache hit if taskpipe has run before)")
}

fn main() -> Result<()> {
    let (prompt_path, sampling) = parse_args()?;
    let prompt = std::fs::read_to_string(&prompt_path)
        .with_context(|| format!("reading prompt file {}", prompt_path.display()))?;

    println!("Prompt: {} ({} chars) — sampling: {:?}", prompt_path.display(), prompt.chars().count(), sampling);

    let model_path = resolve_model_path()?;
    let backend = llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?;
    let registry = rampipe::SwapRegistry::new();

    println!("Loading model (first call only; cached after)...");
    let load_start = Instant::now();
    // Lazy, matching `LocalBackend`: a one-shot harness call gains
    // nothing from paying `Prefault`'s upfront page-in cost.
    let session =
        LlamaSession::load(&registry, &backend, &model_path, Residency::Lazy).context("loading session")?;
    println!("  load() call returned in {:?} (actual page-in is lazy, folds into first token below)", load_start.elapsed());

    let gen_start = Instant::now();
    let result = session
        .generate(&backend, &prompt, LOCAL_MAX_NEW_TOKENS, sampling)
        .context("generate() failed")?;
    let total_wall_time = gen_start.elapsed();
    let tok_per_sec = result.tokens_generated as f64 / total_wall_time.as_secs_f64().max(f64::EPSILON);

    println!("\n=== stats ===");
    println!("  time_to_first_token: {:?}", result.time_to_first_token);
    println!("  total generate() wall time: {total_wall_time:?}");
    println!("  tokens_generated: {} ({tok_per_sec:.1} tok/s)", result.tokens_generated);

    println!("\n=== raw output ===");
    println!("{}", result.text);

    Ok(())
}
