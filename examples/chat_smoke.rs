//! Real smoke test for `Conversation`: a live, multi-turn exchange
//! against a resident model, checking that the *second* turn's answer
//! actually depends on the *first* turn's content — the one thing that
//! can't be faked by two independent `generate()` calls (each of which
//! would start from a blank context and have no way to know what the
//! first prompt said).
//!
//!     cargo run --release --features llama --example chat_smoke

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

    let backend = Arc::new(llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?);
    let registry = SwapRegistry::new();
    let session = LlamaSession::load(&registry, backend, &model_path, Residency::Lazy).context("loading session")?;

    let options = ConversationOptions { n_ctx: NonZeroU32::new(4096).unwrap(), overflow: OverflowPolicy::Fail };
    let mut conversation = session.open_conversation(options).context("opening conversation")?;

    println!("\n=== turn 1 ===");
    let turn1 = conversation
        .send("My favorite number is 7492. Just say OK.", MAX_NEW_TOKENS, Sampling::Greedy)
        .context("turn 1")?;
    println!("  decoded this turn (new text only): {:?}", turn1.formatted_prompt);
    println!("  reply: {}", turn1.text.trim());

    println!("\n=== turn 2 (no restated context) ===");
    let turn2 = conversation
        .send("What's my favorite number? Reply with just the digits, nothing else.", MAX_NEW_TOKENS, Sampling::Greedy)
        .context("turn 2")?;
    println!("  decoded this turn (new text only): {:?}", turn2.formatted_prompt);
    println!("  reply: {}", turn2.text.trim());

    println!("\nturn_count: {}", conversation.turn_count());

    if !turn2.text.contains("7492") {
        bail!("turn 2 didn't recall the number from turn 1 -- conversation context isn't actually persisting: {:?}", turn2.text);
    }

    println!("\nOK: turn 2 correctly recalled context from turn 1 with no restated history in the prompt.");
    Ok(())
}
