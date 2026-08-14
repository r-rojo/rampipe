//! Real smoke test for `Conversation::save_state` / `LlamaSession::
//! open_conversation_from_state`: a live conversation is saved to disk,
//! the in-memory `Conversation` is dropped (and the session reopened
//! fresh, same as a real process restart would look), then reloaded from
//! disk and asked to recall a fact from before the save with no restated
//! history in the prompt -- the one thing that can't be faked by two
//! independent `generate()` calls, and specifically proves the *disk*
//! round-trip works, not just the in-memory KV cache
//! (`chat_smoke`'s own job).
//!
//!     cargo run --release --features llama --example session_persistence_smoke

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;
use rampipe::llama::{ConversationOptions, LlamaSession, OverflowPolicy, Sampling};
use rampipe::{Residency, SwapRegistry};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

const REPO_OWNER: &str = "Qwen";
const REPO_NAME: &str = "Qwen2.5-0.5B-Instruct-GGUF";
const FILENAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const MAX_NEW_TOKENS: i32 = 60;

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file().filename(FILENAME).send().context("downloading GGUF file")
}

fn main() -> Result<()> {
    println!("Downloading/locating {REPO_OWNER}/{REPO_NAME}/{FILENAME} (cached after first run)...");
    let model_path = download_model()?;

    let tmp = tempfile::tempdir().context("tempdir for saved session")?;
    let state_path = tmp.path().join("session.state");
    let meta_path = tmp.path().join("session.meta.json");

    let backend = Arc::new(llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?);
    let registry = SwapRegistry::new();
    let options = ConversationOptions { n_ctx: NonZeroU32::new(4096).unwrap(), overflow: OverflowPolicy::Fail };

    {
        let session = LlamaSession::load(&registry, Arc::clone(&backend), &model_path, Residency::Lazy).context("loading session")?;
        let mut conversation = session.open_conversation(options).context("opening conversation")?;

        println!("\n=== turn 1 (before save) ===");
        let turn1 = conversation
            .send("My favorite number is 7492. Just say OK.", MAX_NEW_TOKENS, Sampling::Greedy, None, None, None)
            .context("turn 1")?;
        println!("  reply: {}", turn1.text.trim());

        println!("\nSaving state to {} / {}...", state_path.display(), meta_path.display());
        conversation.save_state(&state_path, &meta_path).context("save_state")?;
        let state_bytes = std::fs::metadata(&state_path).map(|m| m.len()).unwrap_or(0);
        println!("  state file: {} bytes", state_bytes);

        // `conversation` (and this whole inner-scope `session`) is
        // dropped here -- the reload below starts from a genuinely fresh
        // `LlamaSession`/`LlamaContext`, not just a reused one, the same
        // way a real process restart (or a different resident agent
        // picking up a saved-off session) would.
    }

    println!("\n=== reloading from disk into a fresh session ===");
    let session = LlamaSession::load(&registry, backend, &model_path, Residency::Lazy).context("reloading session")?;
    let mut conversation = session.open_conversation_from_state(&state_path, &meta_path).context("open_conversation_from_state")?;
    println!("  turn_count after reload: {}", conversation.turn_count());

    println!("\n=== turn 2 (after reload, no restated context) ===");
    let turn2 = conversation
        .send("What's my favorite number? Reply with just the digits, nothing else.", MAX_NEW_TOKENS, Sampling::Greedy, None, None, None)
        .context("turn 2")?;
    println!("  decoded this turn (new text only): {:?}", turn2.formatted_prompt);
    println!("  reply: {}", turn2.text.trim());

    if !turn2.text.contains("7492") {
        bail!("turn 2 (after a disk save/reload) didn't recall the number from turn 1 -- the KV cache round-trip isn't actually working: {:?}", turn2.text);
    }

    println!("\nOK: turn 2 correctly recalled context saved to disk before this process's own session was dropped and reopened.");
    Ok(())
}
