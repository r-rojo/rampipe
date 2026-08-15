//! Runs one fixed prompt through several models against a running
//! `rampiped`, one after another, and saves each model's raw response to
//! its own file for manual review -- a quick way to compare how
//! different models handle the same real code-generation task on this
//! machine, and to exercise the daemon's real load/generate/evict path
//! under back-to-back requests for different models.
//!
//! Downloads models via `hf-hub` on first run (cached after that, same
//! as `chat_smoke`).
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features llama --bin rampiped -- \
//!         --socket /tmp/rampiped.sock
//!
//!     cargo run --release --features client --example model_stress_test -- \
//!         /tmp/rampiped.sock [--max-new-tokens N] [--out-dir DIR]

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::client::RampipedClient;
use rampipe::protocol::WireSampling;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const PROMPT: &str = "write a program in rust that serves as a multi-client pub/sub messaging \
framework, call the project mpipe. acceptance criteria: multiple clients can publish and \
subscribe to differnet topics and send messages to each topic and every subscriber receives \
them. Unit tests pass.";

const DEFAULT_MAX_NEW_TOKENS: i32 = 4096;
const DEFAULT_OUT_DIR: &str = "stress-test-output";

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

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let max_new_tokens: i32 = take_flag_value(&mut args, "--max-new-tokens")
        .map(|s| s.parse().context("--max-new-tokens must be an integer"))
        .transpose()?
        .unwrap_or(DEFAULT_MAX_NEW_TOKENS);
    let out_dir = take_flag_value(&mut args, "--out-dir").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_OUT_DIR));
    let socket_path = PathBuf::from(args.first().cloned().context("usage: model_stress_test <socket-path> [--max-new-tokens N] [--out-dir DIR]")?);

    fs::create_dir_all(&out_dir).with_context(|| format!("creating output directory {}", out_dir.display()))?;

    println!("=== connecting to {} ===", socket_path.display());
    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    println!("=== prompt ===\n{PROMPT}\n");

    struct Result_ {
        label: &'static str,
        tokens_generated: usize,
        time_to_first_token_ms: u64,
        wall_ms: u128,
        out_path: PathBuf,
    }
    let mut results: Vec<Result_> = Vec::new();
    let mut failures: Vec<(&'static str, anyhow::Error)> = Vec::new();

    for spec in MODELS {
        println!("--- {} ---", spec.label);
        println!("  downloading/locating {}/{}/{} (cached after first run)...", spec.repo_owner, spec.repo_name, spec.filename);
        let model_path = match download_model(spec) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("  download failed: {err:#}");
                failures.push((spec.label, err));
                continue;
            }
        };

        println!("  generating (max_new_tokens={max_new_tokens})...");
        let start = Instant::now();
        let outcome = match client.generate(&model_path, PROMPT, max_new_tokens, WireSampling::Greedy, None, None, None) {
            Ok(o) => o,
            Err(err) => {
                eprintln!("  generate failed: {err:#}");
                failures.push((spec.label, err.into()));
                continue;
            }
        };
        let wall_ms = start.elapsed().as_millis();

        let out_path = out_dir.join(format!("{}.txt", spec.label));
        fs::write(&out_path, &outcome.text).with_context(|| format!("writing {}", out_path.display()))?;

        println!(
            "  done: {} tokens, ttft {}ms, wall {}ms -> {}",
            outcome.tokens_generated,
            outcome.time_to_first_token_ms,
            wall_ms,
            out_path.display()
        );
        results.push(Result_ { label: spec.label, tokens_generated: outcome.tokens_generated, time_to_first_token_ms: outcome.time_to_first_token_ms, wall_ms, out_path });
    }

    println!("\n=== summary ===");
    println!("{:<20} {:>10} {:>12} {:>12} {}", "model", "tokens", "ttft (ms)", "wall (ms)", "output file");
    for r in &results {
        println!("{:<20} {:>10} {:>12} {:>12} {}", r.label, r.tokens_generated, r.time_to_first_token_ms, r.wall_ms, r.out_path.display());
    }
    if !failures.is_empty() {
        println!("\n{} of {} models failed:", failures.len(), MODELS.len());
        for (label, err) in &failures {
            println!("  {label}: {err:#}");
        }
    }

    Ok(())
}
