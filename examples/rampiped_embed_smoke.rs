//! Live verification of the `Embed` wire message: that `rampiped`
//! serves it at all, that the vectors come back normalized at the
//! model's own width, and -- the reason it exists -- that cosine
//! similarity actually separates a morphological variant from an
//! unrelated phrase.
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features cuda --bin rampiped -- \
//!         --socket ~/.rampipe/rampiped.sock
//!
//!     cargo run --features client --example rampiped_embed_smoke -- \
//!         ~/.rampipe/rampiped.sock /path/to/bge-small-en-v1.5-q8_0.gguf

use anyhow::{Context, Result, bail};
use rampipe::client::RampipedClient;
use rampipe::protocol::WirePooling;
use std::path::PathBuf;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: rampiped_embed_smoke <socket-path> <embedding-model.gguf>";
    let socket_path = PathBuf::from(args.next().context(usage)?);
    let model_path = PathBuf::from(args.next().context(usage)?);

    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    let texts: Vec<String> = [
        "access code",
        "access codes",
        "door code",
        "packing tape",
        "north elevator",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();

    println!("=== embedding {} texts ===", texts.len());
    let outcome = client
        .embed(&model_path, &texts, WirePooling::Model, true)
        .context("embedding over the socket")?;

    if outcome.vectors.len() != texts.len() {
        bail!(
            "expected {} vectors, got {}",
            texts.len(),
            outcome.vectors.len()
        );
    }
    println!("  n_embd = {}", outcome.n_embd);

    for (text, vector) in texts.iter().zip(&outcome.vectors) {
        let norm = cosine(vector, vector).sqrt();
        if (norm - 1.0).abs() > 1e-3 {
            bail!("{text:?} came back un-normalized: norm {norm}");
        }
    }
    println!("  OK: every vector is unit length, so a dot product is cosine");

    // The measured failure this whole path exists for: reflectpipe's
    // lexical cue matching missed `access code` against `access codes`
    // on the plural alone.
    let plural = cosine(&outcome.vectors[0], &outcome.vectors[1]);
    let synonym = cosine(&outcome.vectors[0], &outcome.vectors[2]);
    let unrelated = cosine(&outcome.vectors[0], &outcome.vectors[3]);
    let other = cosine(&outcome.vectors[0], &outcome.vectors[4]);

    println!("\n  cos(access code, access codes)  = {plural:.3}");
    println!("  cos(access code, door code)     = {synonym:.3}");
    println!("  cos(access code, north elevator)= {other:.3}");
    println!("  cos(access code, packing tape)  = {unrelated:.3}");

    if plural <= unrelated {
        bail!(
            "a morphological variant must score above an unrelated phrase: \
             plural {plural:.3} vs unrelated {unrelated:.3}"
        );
    }
    if plural < 0.9 {
        bail!("expected near-identity for a bare plural, got {plural:.3}");
    }
    println!("\n  OK: the plural case separates cleanly from unrelated text");

    println!("\n=== a nonexistent model surfaces a real error ===");
    match client.embed(
        &PathBuf::from("/nonexistent/model.gguf"),
        &["x".to_string()],
        WirePooling::Model,
        true,
    ) {
        Ok(_) => bail!("expected a missing model to error"),
        Err(e) => println!("  OK: {e}"),
    }

    println!("\nall checks passed");
    Ok(())
}
