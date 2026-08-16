//! A single stress-test harness, driven by scenarios defined in a YAML
//! file instead of hardcoded per-test Rust binaries -- replaces the
//! former `model_stress_test`/`model_concurrency_stress_test` pair,
//! which duplicated the same model list and prompt across two files and
//! could only ever run the one mode each was hardcoded for.
//!
//! Each scenario picks one of three modes against a running `rampiped`:
//!
//! - **sequential**: one fixed prompt through several models, one after
//!   another, saving each model's raw response to its own file -- a
//!   quick way to compare how different models handle the same task.
//! - **concurrent**: fires one request per model at (as close to) the
//!   same instant via a `Barrier`, showing how `rampiped`'s documented
//!   single-decode-at-a-time lock actually serializes them.
//! - **churn**: round-robin cycles through every model, `rounds` times,
//!   querying `Status` after each request so the resident set and free
//!   VRAM are visible changing in response to real eviction.
//!
//! See `examples/stress_test_scenarios.yaml` for the default scenarios
//! (YAML anchors keep the shared model list and prompt defined once).
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features llama --bin rampiped -- \
//!         --socket /tmp/rampiped.sock
//!
//!     cargo run --release --features client --example stress_test -- \
//!         /tmp/rampiped.sock [--config PATH] [--scenario NAME]

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::client::RampipedClient;
use rampipe::protocol::WireSampling;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Instant;

const DEFAULT_CONFIG_PATH: &str = "examples/stress_test_scenarios.yaml";
const DEFAULT_OUT_DIR: &str = "stress-test-output";
const DEFAULT_SEQUENTIAL_MAX_NEW_TOKENS: i32 = 4096;
// Deliberately smaller than sequential's -- concurrent/churn are about
// observing serialization/eviction timing, not full program generation.
const DEFAULT_TIMING_MAX_NEW_TOKENS: i32 = 200;
const DEFAULT_ROUNDS: u32 = 2;

#[derive(Debug, Deserialize)]
struct TestFile {
    scenarios: Vec<Scenario>,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    name: String,
    mode: Mode,
    prompt: String,
    #[serde(default)]
    max_new_tokens: Option<i32>,
    #[serde(default)]
    rounds: Option<u32>,
    #[serde(default)]
    out_dir: Option<String>,
    models: Vec<ModelSpec>,
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Mode {
    Sequential,
    Concurrent,
    Churn,
}

#[derive(Debug, Deserialize, Clone)]
struct ModelSpec {
    label: String,
    repo_owner: String,
    repo_name: String,
    filename: String,
}

fn download_model(spec: &ModelSpec) -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(&spec.repo_owner, &spec.repo_name);
    repo.download_file().filename(&spec.filename).send().context("downloading GGUF file")
}

/// Downloads every model in `specs`, up front, failing the whole
/// scenario if any one download fails -- concurrent/churn modes need
/// every model actually present before either can meaningfully start
/// (unlike sequential, which tolerates one model's download failing
/// without aborting the rest -- see `run_sequential`).
fn download_all(specs: &[ModelSpec]) -> Result<Vec<(String, PathBuf)>> {
    specs
        .iter()
        .map(|spec| {
            println!("  {}/{}/{}...", spec.repo_owner, spec.repo_name, spec.filename);
            Ok((spec.label.clone(), download_model(spec)?))
        })
        .collect()
}

fn take_flag_value(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let pos = args.iter().position(|a| a == flag)?;
    args.remove(pos);
    if pos < args.len() { Some(args.remove(pos)) } else { None }
}

/// This machine's hostname, for tagging sequential-mode output paths so
/// results from different machines (e.g. a Mac and genie, both writing
/// to the same relative `stress-test-output/`) never collide if copied
/// into one place for comparison -- std has no cross-platform hostname
/// getter, and this is example/dev tooling, so shelling out to the
/// `hostname` command (present on both macOS and Linux) is simpler than
/// adding a dependency for it. Falls back to a fixed placeholder rather
/// than failing the whole run if that's ever unavailable.
fn hostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .filter(|out| out.status.success())
        .and_then(|out| String::from_utf8(out.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

/// Sequential mode: one prompt through every model, one after another,
/// each model's own download/generate failure caught and reported
/// inline rather than aborting the rest -- unlike concurrent/churn,
/// there's no shared "all models must already be present" setup step
/// here, so there's nothing lost by handling failures per-model instead
/// of failing the whole scenario up front.
fn run_sequential(client: &RampipedClient, scenario: &Scenario) -> Result<()> {
    let max_new_tokens = scenario.max_new_tokens.unwrap_or(DEFAULT_SEQUENTIAL_MAX_NEW_TOKENS);
    let out_dir = PathBuf::from(scenario.out_dir.as_deref().unwrap_or(DEFAULT_OUT_DIR)).join(hostname());
    fs::create_dir_all(&out_dir).with_context(|| format!("creating output directory {}", out_dir.display()))?;

    println!("=== prompt ===\n{}\n", scenario.prompt);

    struct Outcome {
        label: String,
        tokens_generated: usize,
        time_to_first_token_ms: u64,
        wall_ms: u128,
        out_path: PathBuf,
    }
    let mut results: Vec<Outcome> = Vec::new();
    let mut failures: Vec<(String, anyhow::Error)> = Vec::new();

    for spec in &scenario.models {
        println!("--- {} ---", spec.label);
        println!("  downloading/locating {}/{}/{} (cached after first run)...", spec.repo_owner, spec.repo_name, spec.filename);
        let model_path = match download_model(spec) {
            Ok(p) => p,
            Err(err) => {
                eprintln!("  download failed: {err:#}");
                failures.push((spec.label.clone(), err));
                continue;
            }
        };

        println!("  generating (max_new_tokens={max_new_tokens})...");
        let start = Instant::now();
        let outcome = match client.generate(&model_path, &scenario.prompt, max_new_tokens, WireSampling::Greedy, None, None, None) {
            Ok(o) => o,
            Err(err) => {
                eprintln!("  generate failed: {err:#}");
                failures.push((spec.label.clone(), err.into()));
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
        results.push(Outcome {
            label: spec.label.clone(),
            tokens_generated: outcome.tokens_generated,
            time_to_first_token_ms: outcome.time_to_first_token_ms,
            wall_ms,
            out_path,
        });
    }

    println!("\n=== summary ===");
    println!("{:<20} {:>10} {:>12} {:>12} {}", "model", "tokens", "ttft (ms)", "wall (ms)", "output file");
    for r in &results {
        println!("{:<20} {:>10} {:>12} {:>12} {}", r.label, r.tokens_generated, r.time_to_first_token_ms, r.wall_ms, r.out_path.display());
    }
    if !failures.is_empty() {
        println!("\n{} of {} models failed:", failures.len(), scenario.models.len());
        for (label, err) in &failures {
            println!("  {label}: {err:#}");
        }
    }
    Ok(())
}

/// Concurrent mode: fire one request per model at (as close to) the
/// same instant, via a `Barrier` so every thread's actual `generate()`
/// call starts only once every thread has connected -- otherwise an
/// early thread's request could finish before a slow-to-connect one
/// even started, defeating the point of testing concurrent arrival.
fn run_concurrent_phase(socket_path: &Path, scenario: &Scenario, models: &[(String, PathBuf)]) -> Result<()> {
    let max_new_tokens = scenario.max_new_tokens.unwrap_or(DEFAULT_TIMING_MAX_NEW_TOKENS);
    let barrier = Arc::new(Barrier::new(models.len()));
    let start = Instant::now();
    let prompt = scenario.prompt.clone();

    // Each thread's own connect/generate failure is caught and returned
    // as data, not propagated -- one oversized or flaky model shouldn't
    // abort every other thread's in-flight request.
    let handles: Vec<_> = models
        .iter()
        .cloned()
        .map(|(label, model_path)| {
            let socket_path = socket_path.to_path_buf();
            let barrier = Arc::clone(&barrier);
            let prompt = prompt.clone();
            thread::spawn(move || -> (String, Result<(u128, u128, usize)>) {
                let result = (|| -> Result<(u128, u128, usize)> {
                    let client = RampipedClient::connect(&socket_path).context("connecting")?;
                    barrier.wait();
                    let queued_at_ms = start.elapsed().as_millis();
                    let outcome =
                        client.generate(&model_path, &prompt, max_new_tokens, WireSampling::Greedy, None, None, None).context("generate")?;
                    let done_at_ms = start.elapsed().as_millis();
                    Ok((queued_at_ms, done_at_ms, outcome.tokens_generated))
                })();
                (label, result)
            })
        })
        .collect();

    let mut results: Vec<(String, u128, u128, usize)> = Vec::new();
    let mut failures: Vec<(String, anyhow::Error)> = Vec::new();
    for handle in handles {
        let (label, result) = handle.join().expect("thread panicked");
        match result {
            Ok((queued_at_ms, done_at_ms, tokens)) => results.push((label, queued_at_ms, done_at_ms, tokens)),
            Err(err) => failures.push((label, err)),
        }
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
    if !failures.is_empty() {
        println!("  {} of {} models failed:", failures.len(), models.len());
        for (label, err) in &failures {
            println!("    {label}: {err:#}");
        }
    }
    Ok(())
}

/// Churn mode: sequentially cycle through every model, `rounds` times,
/// querying `Status` after each request so the resident set and free
/// VRAM are visible changing in response to real eviction, not inferred
/// from timing alone.
fn run_churn_phase(client: &RampipedClient, scenario: &Scenario, models: &[(String, PathBuf)]) -> Result<()> {
    let max_new_tokens = scenario.max_new_tokens.unwrap_or(DEFAULT_TIMING_MAX_NEW_TOKENS);
    let rounds = scenario.rounds.unwrap_or(DEFAULT_ROUNDS);

    println!("{:<20} {:>6} {:>10} {:>8}  {:<15} resident models", "model", "round", "wall (ms)", "tokens", "gpu free");
    let mut failure_count = 0;
    for round in 1..=rounds {
        for (label, model_path) in models {
            let start = Instant::now();
            // A failing model here (e.g. one too big to decode on this
            // machine) shouldn't stop the churn cycle from continuing.
            let outcome = match client.generate(model_path, &scenario.prompt, max_new_tokens, WireSampling::Greedy, None, None, None) {
                Ok(outcome) => outcome,
                Err(err) => {
                    failure_count += 1;
                    println!("{label:<20} {round:>6}  generate failed: {err:#}");
                    continue;
                }
            };
            let wall_ms = start.elapsed().as_millis();
            let status = client.status().context("status")?;
            let resident: Vec<String> =
                status.models.iter().filter_map(|m| m.path.file_name().map(|n| n.to_string_lossy().into_owned())).collect();
            let gpu_free = status.gpu_free_bytes.map(|b| format!("{:.1}GB", b as f64 / 1e9)).unwrap_or_else(|| "n/a".to_string());
            println!("{label:<20} {round:>6} {wall_ms:>10} {:>8}  {gpu_free:<15} {resident:?}", outcome.tokens_generated);
        }
    }
    if failure_count > 0 {
        println!("  {failure_count} request(s) failed across all rounds");
    }
    Ok(())
}

fn main() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let config_path = take_flag_value(&mut args, "--config").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(DEFAULT_CONFIG_PATH));
    let only_scenario = take_flag_value(&mut args, "--scenario");
    let socket_path =
        PathBuf::from(args.first().cloned().context("usage: stress_test <socket-path> [--config PATH] [--scenario NAME]")?);

    let config_text = fs::read_to_string(&config_path).with_context(|| format!("reading {}", config_path.display()))?;
    let test_file: TestFile = serde_yaml::from_str(&config_text).with_context(|| format!("parsing {}", config_path.display()))?;

    let scenarios: Vec<&Scenario> = match &only_scenario {
        Some(name) => {
            let found = test_file.scenarios.iter().find(|s| &s.name == name);
            vec![found.with_context(|| format!("no scenario named {name:?} in {}", config_path.display()))?]
        }
        None => test_file.scenarios.iter().collect(),
    };

    println!("=== connecting to {} ===", socket_path.display());
    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    for scenario in scenarios {
        println!("\n########## scenario: {} ({:?}) ##########", scenario.name, scenario.mode);
        match scenario.mode {
            Mode::Sequential => run_sequential(&client, scenario)?,
            Mode::Concurrent => {
                println!("=== downloading/locating models (cached after first run) ===");
                let models = download_all(&scenario.models)?;
                run_concurrent_phase(&socket_path, scenario, &models)?;
            }
            Mode::Churn => {
                println!("=== downloading/locating models (cached after first run) ===");
                let models = download_all(&scenario.models)?;
                run_churn_phase(&client, scenario, &models)?;
            }
        }
    }

    println!("\nall scenarios completed");
    Ok(())
}
