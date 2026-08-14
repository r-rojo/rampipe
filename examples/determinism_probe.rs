//! Diagnoses reported run-to-run non-determinism in `taskpipe`'s
//! `LocalBackend`: the same task, same prompt, has both succeeded and
//! failed across separate `taskpipe` invocations. Reproduces exactly
//! `LocalBackend`'s prompt-construction shape (see `backend.rs`) against
//! a real task that showed the variance (gamepipe issue #1, "Core ECS
//! structs"), and does ONE `generate()` call per process invocation —
//! matching how `taskpipe` actually runs (a fresh process, fresh model
//! load, every time), not a warm session reused across repeated calls
//! within one process.
//!
//! Bypasses `rampipe::llama::LlamaSession` deliberately: its `load()`
//! doesn't expose `n_gpu_layers`, and this needs to A/B Metal-offload vs.
//! CPU-only against the *exact* same generate-loop logic (copied
//! straight from `rampipe/src/llama.rs`) — not a rampipe API change for
//! a one-off diagnostic.
//!
//! Metal (default):    cargo run --release --features llama --example determinism_probe
//! CPU-only:            DETERMINISM_PROBE_CPU_ONLY=1 cargo run --release --features llama --example determinism_probe

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::{AddBos, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use std::num::NonZeroU32;
use std::path::PathBuf;

const REPO_OWNER: &str = "bartowski";
const REPO_NAME: &str = "Qwen2.5-7B-Instruct-GGUF";
const FILENAME: &str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf";
const MAX_NEW_TOKENS: i32 = 1400;

const FILE_PATH: &str = "src/ecs.rs";
const TITLE: &str = "Core ECS structs";
const BODY: &str = "## Goal\nDefine core ECS structs (Position, Velocity, Transform) as plain Rust\nstructs with basic component storage (Vec<Option<T>> per component type\nis fine — no archetype system needed).\n\n## Acceptance criteria\n- [ ] Compiles\n- [ ] Test constructing a few entities and reading components back";

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file().filename(FILENAME).send().context("resolving model file (should be a cache hit)")
}

fn main() -> Result<()> {
    let cpu_only = std::env::var("DETERMINISM_PROBE_CPU_ONLY").is_ok();
    let model_path = download_model()?;

    let backend = LlamaBackend::init().context("llama.cpp backend init")?;
    let model_params = if cpu_only { LlamaModelParams::default().with_n_gpu_layers(0) } else { LlamaModelParams::default() };
    let model = LlamaModel::load_from_file(&backend, &model_path, &model_params).context("model load")?;

    // Byte-for-byte the same prompt `LocalBackend::execute` builds for a
    // task with no dependencies and no plan (dependency_api_section and
    // plan_section both empty).
    let prompt = format!(
        "You are writing a single Rust source file: `{FILE_PATH}`.\n\n\
         Task: {TITLE}\n\n{BODY}\n\n\
         Respond with ONLY a single Rust code block containing the complete contents \
         of `{FILE_PATH}`. Do not include any explanation before or after the code block."
    );

    // Everything below is copied verbatim from
    // `rampipe::llama::LlamaSession::generate()` (n_ctx, batch chunking,
    // sampler chain, decode loop) — the whole point is an apples-to-apples
    // comparison against real `LocalBackend` behavior, not a simplified
    // stand-in.
    let ctx_params = LlamaContextParams::default().with_n_ctx(NonZeroU32::new(4096));
    let mut ctx = model.new_context(&backend, ctx_params).context("context")?;

    let tokens_list = model.str_to_token(&prompt, AddBos::Always).context("tokenize")?;
    let n_ctx = ctx.n_ctx() as i32;
    let requested = tokens_list.len() as i32 + MAX_NEW_TOKENS;
    anyhow::ensure!(requested <= n_ctx, "prompt plus max_new_tokens ({requested}) exceeds context size ({n_ctx})");

    let mut batch = LlamaBatch::new(512, 1);
    let last_index = (tokens_list.len() - 1) as i32;
    for chunk_start in (0..tokens_list.len()).step_by(512) {
        let chunk_end = (chunk_start + 512).min(tokens_list.len());
        batch.clear();
        for (offset, &token) in tokens_list[chunk_start..chunk_end].iter().enumerate() {
            let i = (chunk_start + offset) as i32;
            batch.add(token, i, &[0], i == last_index)?;
        }
        ctx.decode(&mut batch)?;
    }

    let mut decoder = encoding_rs::UTF_8.new_decoder();
    let mut sampler = LlamaSampler::chain_simple([LlamaSampler::dist(1234), LlamaSampler::greedy()]);

    let mut n_cur = tokens_list.len() as i32;
    let end = tokens_list.len() as i32 + MAX_NEW_TOKENS;
    let mut text = String::new();
    let mut tokens_generated = 0usize;

    while n_cur <= end {
        let token = sampler.sample(&ctx, batch.n_tokens() - 1);
        sampler.accept(token);

        if model.is_eog_token(token) {
            break;
        }

        text.push_str(&model.token_to_piece(token, &mut decoder, true, None)?);
        tokens_generated += 1;

        batch.clear();
        batch.add(token, n_cur, &[0], true)?;
        n_cur += 1;
        ctx.decode(&mut batch)?;
    }

    println!("===PROBE_MODE=== {}", if cpu_only { "cpu-only" } else { "metal" });
    println!("===PROBE_OUTPUT_START===");
    println!("{text}");
    println!("===PROBE_OUTPUT_END===");
    println!("===PROBE_TOKENS_GENERATED=== {tokens_generated}");

    Ok(())
}
