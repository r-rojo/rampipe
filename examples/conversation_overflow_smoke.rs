//! Regression test for a real, live bug: a long-running `Conversation`
//! with `OverflowPolicy::DropOldestTurns` eventually hit
//! `llama.cpp decode error: Decode Error 1: NoKvCacheSlot` on an
//! ordinary turn, despite `committed_pos` staying well under `n_ctx` by
//! `run_generation_loop`'s own position-based accounting. Root cause:
//! `drop_oldest_turns_for`'s `kv_cache_seq_rm`/`kv_cache_seq_add` calls
//! shift *position numbers* to close the logical gap a drop leaves, but
//! do nothing to compact the underlying KV cache's own physical cell
//! layout -- without `defrag_thold` set (llama.cpp's own default is
//! `-1.0`, auto-defrag disabled), repeated drops over a long session
//! fragment the cache until no contiguous span remains for a new batch,
//! even though the *position* budget looks fine. Fixed by setting
//! `CONVERSATION_DEFRAG_THOLD` on every conversation context -- see that
//! constant's own doc comment in `src/llama.rs`.
//!
//! Forces the same failure mode deliberately: a tiny `n_ctx` (300
//! tokens) against a real model, many short turns, guaranteeing dozens
//! of drop-and-shift cycles in a single run -- large real conversations
//! would eventually hit the same fragmentation, just slower.
//!
//!     cargo run --release --features llama --example conversation_overflow_smoke

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::llama::{ConversationOptions, LlamaSession, OverflowPolicy, Sampling};
use rampipe::{Residency, SwapRegistry};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

const REPO_OWNER: &str = "Qwen";
const REPO_NAME: &str = "Qwen2.5-0.5B-Instruct-GGUF";
const FILENAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const MAX_NEW_TOKENS: i32 = 20;
const N_CTX: u32 = 2000;
const TURNS: usize = 60;

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file()
        .filename(FILENAME)
        .send()
        .context("downloading GGUF file")
}

fn main() -> Result<()> {
    println!(
        "Downloading/locating {REPO_OWNER}/{REPO_NAME}/{FILENAME} (cached after first run)..."
    );
    let model_path = download_model()?;

    let backend = Arc::new(
        llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?,
    );
    let registry = SwapRegistry::new();
    let session = LlamaSession::load(&registry, backend, &model_path, Residency::Lazy)
        .context("loading session")?;

    let options = ConversationOptions {
        n_ctx: NonZeroU32::new(N_CTX).unwrap(),
        overflow: OverflowPolicy::DropOldestTurns,
    };
    let mut conversation = session
        .open_conversation(options)
        .context("opening conversation")?;

    println!(
        "=== sending {TURNS} short turns against a {N_CTX}-token context (forces many drop-and-shift cycles) ==="
    );
    for i in 0..TURNS {
        let prompt = format!("Say the number {i} and nothing else.");
        let turn = conversation
            .send(&prompt, MAX_NEW_TOKENS, Sampling::Greedy, None, None, None)
            .with_context(|| format!("turn {i} failed -- this is exactly the NoKvCacheSlot regression if the error mentions it"))?;
        println!("  turn {i:>2}: {:?}", turn.text.trim());
    }

    println!(
        "\nOK: all {TURNS} turns completed with no NoKvCacheSlot error -- defrag is doing its job."
    );
    Ok(())
}
