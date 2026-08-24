//! Regression test for a real, live failure: an agent whose context
//! filled mid-run lost its system block and its tool definitions, and
//! spent the rest of its budget emitting the same malformed tool call
//! over and over.
//!
//! `Conversation::send` concatenates the rendered opening -- system
//! block plus tool list -- into the *first user turn's* text, so its
//! tokens sit at the front of `turns[0]`'s span.
//! `drop_oldest_turns_for` frees room by removing `turns[0]` and
//! `turns[1]`, from `user.start_pos`, which for the first pair is
//! position zero. So the very first eviction took the opening with it.
//! The daemon logged the removal as `positions 0..3090`, and that `0`
//! is the whole bug.
//!
//! What made it hard to see from outside: nothing errors. The model
//! keeps answering, it just no longer has any instructions, and what it
//! produces reads like a model losing coherence rather than a model
//! whose prompt was deleted.
//!
//! This forces the same eviction -- a small `n_ctx` and enough turns to
//! guarantee dozens of drop cycles -- with one distinctive fact in the
//! system block, and then asks for that fact back after the drops have
//! happened. Before `protected_prefix`, the answer is whatever a model
//! with no system prompt invents.
//!
//!     cargo run --release --features llama --example conversation_keeps_its_opening

use anyhow::{Context, Result};
use hf_hub::HFClientSync;
use rampipe::llama::{ConversationOptions, LlamaSession, OverflowPolicy, Penalties, Sampling};
use rampipe::{Residency, SwapRegistry};
use std::num::NonZeroU32;
use std::path::PathBuf;
use std::sync::Arc;

const REPO_OWNER: &str = "Qwen";
const REPO_NAME: &str = "Qwen2.5-0.5B-Instruct-GGUF";
const FILENAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const MAX_NEW_TOKENS: i32 = 24;
const N_CTX: u32 = 1200;
const TURNS: usize = 40;

/// Deliberately arbitrary. A model that has lost the system block cannot
/// arrive at this by reasoning, by echoing a later turn, or by luck --
/// which is what makes its presence in the answer evidence rather than
/// encouragement.
const CODENAME: &str = "MARIGOLD";

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file().filename(FILENAME).send().context("downloading GGUF file")
}

fn main() -> Result<()> {
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
        system: Some(format!(
            "You are a test fixture. The project codename is {CODENAME}. \
             Whenever you are asked for the project codename, answer with that one word and nothing else."
        )),
        tools: Vec::new(),
        tool_format: None,
    };
    let mut conversation = session.open_conversation(options).context("opening conversation")?;

    // Filler. Each turn is small; together they are several times the
    // window, so the front of the cache is reclaimed many times over.
    println!("=== {TURNS} filler turns against a {N_CTX}-token context ===");
    for i in 0..TURNS {
        conversation
            .send(
                &format!("Count from {i} to {}. Numbers only.", i + 6),
                MAX_NEW_TOKENS,
                Sampling::Greedy { penalties: Penalties::default() },
                None,
                None,
                None,
            )
            .with_context(|| format!("filler turn {i}"))?;
    }

    let answer = conversation
        .send("What is the project codename?", MAX_NEW_TOKENS, Sampling::Greedy { penalties: Penalties::default() }, None, None, None)
        .context("asking for the codename after the drops")?;
    let text = answer.text.trim().to_string();
    println!("\nafter {TURNS} turns of eviction, asked for the codename: {text:?}");

    anyhow::ensure!(
        text.to_uppercase().contains(CODENAME),
        "the system block did not survive context eviction -- answer was {text:?}, which contains no \
         {CODENAME}. The opening is being reclaimed along with the first turn; see `protected_prefix`."
    );
    println!("OK: the opening survived {TURNS} turns of eviction.");
    Ok(())
}
