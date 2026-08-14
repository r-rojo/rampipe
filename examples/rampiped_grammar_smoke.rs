//! Proves the `rampiped` *socket* path is grammar-constrained end to
//! end, not just the in-process `LlamaSession::generate` path --
//! `examples/grammar_smoke.rs` only exercises the latter. Requires a
//! `rampiped` already running against the given `--socket` (started
//! separately, e.g. `cargo run --release --features llama --bin
//! rampiped -- --socket /tmp/rampiped-verify.sock`).
//!
//!     cargo run --release --features client --example rampiped_grammar_smoke -- \
//!         /tmp/rampiped-verify.sock /path/to/model.gguf

use anyhow::{Context, Result, bail};
use rampipe::client::RampipedClient;
use rampipe::protocol::{GrammarCompletion, WireSampling};
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let socket_path = PathBuf::from(args.next().context("usage: rampiped_grammar_smoke <socket-path> <model-path>")?);
    let model_path = PathBuf::from(args.next().context("usage: rampiped_grammar_smoke <socket-path> <model-path>")?);

    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    println!("=== classifier grammar over the socket ===");
    let prompt = "Decide whether answering a message honestly requires writing a file, creating a directory, or \
                  running a shell command. Answer with exactly one word: YES or NO.\n\nMessage: what is the \
                  capital of France?\nAnswer:";
    let outcome = client
        .generate(
            &model_path,
            prompt,
            5,
            WireSampling::Greedy,
            Some("root ::= \"YES\" | \"NO\"\n"),
            None,
            Some(GrammarCompletion::ExactMatch(vec!["YES".to_string(), "NO".to_string()])),
        )
        .context("classifier generation over socket")?;
    println!("  raw output: {:?}", outcome.text);
    if outcome.text.trim() != "YES" && outcome.text.trim() != "NO" {
        bail!("rampiped's own sampler was not grammar-constrained: {:?}", outcome.text);
    }
    println!("  OK: exact match -- the daemon's own session.generate() applied the grammar, not just the client");

    println!("\n=== malformed grammar surfaces a real error over the socket ===");
    match client.generate(&model_path, "hello", 5, WireSampling::Greedy, Some("root ::= \"unterminated"), None, None) {
        Ok(_) => bail!("expected malformed grammar to error, but generation succeeded"),
        Err(e) => println!("  OK: got expected error: {e}"),
    }

    println!("\nall checks passed");
    Ok(())
}
