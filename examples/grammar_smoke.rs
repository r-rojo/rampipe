//! Real-model verification for grammar-constrained decoding (the
//! constrained-decoding work: `LlamaSession::generate`'s new `grammar`/
//! `assistant_prefill` parameters). Three things this crate's own unit
//! tests can't check without a real model and a real llama.cpp grammar
//! parser:
//!
//! 1. A small classifier-shaped grammar (`root ::= "YES" | "NO"`)
//!    actually constrains real generation to an exact match.
//! 2. The JSON-envelope grammar `iopipe::envelope::ENVELOPE_GRAMMAR`
//!    mirrors (kept here as a literal copy, not a dependency on iopipe --
//!    rampipe doesn't and shouldn't know about iopipe's concerns) both
//!    compiles against llama.cpp's real grammar parser *and*, combined
//!    with `"{"` prefill, produces genuinely valid JSON once
//!    reassembled -- including a multi-line `content` field, the whole
//!    reason for this move (see that module's doc comment).
//! 3. A deliberately malformed grammar surfaces a real `Err`, never a
//!    silent fall-back to unconstrained generation.
//!
//!     cargo run --release --features llama --example grammar_smoke

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;
use rampipe::llama::{LlamaSession, Sampling};
use rampipe::{Residency, SwapRegistry};
use std::path::PathBuf;

// The tiny 0.5B built-in model is too weak at instruction-following to
// reliably emit a real `write` step instead of just narrating "done" in
// `reply` -- confirmed live (valid JSON every time, but an empty `steps`
// array). A 7B model exercises the actual multi-line-content round-trip
// this test cares about.
const REPO_OWNER: &str = "bartowski";
const REPO_NAME: &str = "Qwen2.5-7B-Instruct-GGUF";
const FILENAME: &str = "Qwen2.5-7B-Instruct-Q4_K_M.gguf";

/// Mirrors `iopipe::envelope::ENVELOPE_GRAMMAR` exactly -- see that
/// constant's own doc comment for why `root` doesn't itself match a
/// leading `{`: `assistant_prefill` below already emits it as part of
/// the prompt, outside anything the grammar sampler ever sees (a
/// sampler only constrains *sampled* tokens, with no visibility into
/// prompt tokens that were merely decoded).
const ENVELOPE_GRAMMAR: &str = r#"root   ::= "\"steps\":" steps ("," "\"reply\":" string)? "}"
steps  ::= "[" (step ("," step)*)? "]"
step   ::= "{" "\"op\":" (write | mkdir | run) "}"
write  ::= "\"write\"," "\"path\":" string "," "\"content\":" string
mkdir  ::= "\"mkdir\"," "\"path\":" string
run    ::= "\"run\"," "\"cmd\":" string ("," "\"readonly\":" bool)?
string ::= "\"" ( [^"\\] | "\\" (["\\/bfnrt] | "u" hex hex hex hex) )* "\""
hex    ::= [0-9a-fA-F]
bool   ::= "true" | "false"
"#;

fn download_model() -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(REPO_OWNER, REPO_NAME);
    repo.download_file().filename(FILENAME).send().context("downloading GGUF file")
}

fn main() -> Result<()> {
    // Safety: called before LlamaBackend::init(), on the only thread
    // touching the environment so far -- matches every other example's
    // and brush-ai's own use of this.
    unsafe {
        rampipe::llama::suppress_logs();
    }

    let model_path = download_model()?;
    let backend = rampipe::llama::LlamaBackend::init().context("initializing llama.cpp backend")?;
    let registry = SwapRegistry::new();
    let session = LlamaSession::load(&registry, std::sync::Arc::new(backend), &model_path, Residency::Lazy)
        .context("loading session")?;

    println!("=== 1. classifier grammar (YES|NO) ===");
    let classify_grammar = "root ::= \"YES\" | \"NO\"\n";
    let prompt = "Decide whether answering a message honestly requires writing a file, creating a directory, or \
                  running a shell command. Answer with exactly one word: YES or NO.\n\nMessage: what is the \
                  capital of France?\nAnswer:";
    let is_complete = |text: &str| text == "YES" || text == "NO";
    let result =
        session.generate(prompt, 5, Sampling::Greedy, Some(classify_grammar), None, Some(&is_complete)).context("classifier generation")?;
    println!("  raw output: {:?}", result.text);
    if result.text.trim() != "YES" && result.text.trim() != "NO" {
        bail!("classifier grammar failed to constrain output to an exact YES/NO match: {:?}", result.text);
    }
    println!("  OK: exact match");

    println!("\n=== 2. envelope grammar + \"{{\" prefill (multi-line content) ===");
    let envelope_prompt = "Respond with a single JSON object of the form {\"steps\": [...], \"reply\": \"...\"}. \
         Each element of \"steps\" is one action: {\"op\": \"write\", \"path\": \"<path>\", \"content\": \"<full \
         file text, newlines as \\n>\"} to write a file. Emit nothing before or after the JSON object.\n\n\
         Example -- request: write a haiku about autumn to autumn.txt\n\
         Example -- response: {\"steps\":[{\"op\":\"write\",\"path\":\"autumn.txt\",\"content\":\"leaves \
         fall\\nquiet gold drifting down\\nautumn takes its bow\\n\"}],\"reply\":\"Wrote autumn.txt.\"}\n\n\
         Write a 3-line poem about the ocean to a file named poem.txt.";
    let is_complete = |generated: &str| serde_json::from_str::<serde_json::Value>(&format!("{{{generated}")).is_ok();
    let result = session
        .generate(envelope_prompt, 300, Sampling::Greedy, Some(ENVELOPE_GRAMMAR), Some("{"), Some(&is_complete))
        .context("envelope generation")?;
    println!("  raw output: {}", result.text);
    let value: serde_json::Value = serde_json::from_str(result.text.trim()).context("envelope output failed to parse as JSON")?;
    let content = value["steps"][0]["content"].as_str().context("no steps[0].content in envelope output")?;
    println!("  parsed content field: {content:?}");
    if !content.contains('\n') {
        bail!("expected a multi-line content field (real embedded newlines after JSON unescaping), got: {content:?}");
    }
    println!("  OK: valid JSON, content round-trips with real embedded newlines");

    println!("\n=== 3. malformed grammar surfaces a real error ===");
    let broken_grammar = "root ::= \"unterminated";
    match session.generate("hello", 5, Sampling::Greedy, Some(broken_grammar), None, None) {
        Ok(_) => bail!("expected malformed grammar to error, but generation succeeded"),
        Err(e) => println!("  OK: got expected error: {e}"),
    }

    println!("\nall checks passed");
    Ok(())
}
