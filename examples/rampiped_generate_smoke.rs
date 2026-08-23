//! Live verification of `rampiped`'s one-shot `generate()` path over the
//! wire: downloads a small model (same one `chat_smoke` uses, cached
//! after the first run via `hf-hub` -- see that example's doc comment),
//! sends one prompt with a checkable answer, and confirms the daemon
//! actually ran inference and returned real text, not just that the
//! request round-tripped.
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features llama --bin rampiped -- \
//!         --socket /tmp/rampiped-generate-smoke.sock
//!
//!     cargo run --features client --example rampiped_generate_smoke -- \
//!         /tmp/rampiped-generate-smoke.sock

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;
use rampipe::client::RampipedClient;
use std::path::PathBuf;

const REPO_OWNER: &str = "Qwen";
const REPO_NAME: &str = "Qwen2.5-0.5B-Instruct-GGUF";
const FILENAME: &str = "qwen2.5-0.5b-instruct-q4_k_m.gguf";
const MAX_NEW_TOKENS: i32 = 20;

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file()
        .filename(FILENAME)
        .send()
        .context("downloading GGUF file")
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let socket_path = PathBuf::from(
        args.next()
            .context("usage: rampiped_generate_smoke <socket-path>")?,
    );

    println!(
        "Downloading/locating {REPO_OWNER}/{REPO_NAME}/{FILENAME} (cached after first run)..."
    );
    let model_path = download_model()?;

    println!("=== connecting to {} ===", socket_path.display());
    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    println!("=== generate() over the wire ===");
    let outcome = client
        .generate(
            &model_path,
            "What is the capital of France? Reply with just the city name.",
            Some(MAX_NEW_TOKENS),
            None,
            None,
            None,
            None,
        )
        .context("generate")?;
    println!("  reply: {:?}", outcome.text.trim());
    println!("  tokens generated: {}", outcome.tokens_generated);
    println!(
        "  time to first token: {}ms",
        outcome.time_to_first_token_ms
    );

    if outcome.tokens_generated == 0 {
        bail!("rampiped generated zero tokens -- something's wrong with the request path");
    }
    if !outcome.text.to_lowercase().contains("paris") {
        bail!(
            "expected the reply to mention Paris, got {:?}",
            outcome.text
        );
    }

    println!("\nOK: rampiped generated real, correct text over the wire.");
    Ok(())
}
