//! Live verification of the daemon-backed conversation path added for
//! the brush-ai/plush-aichat KV-cache refactor:
//!
//! 1. Opens a `RampipedConversation` against a small model, sends two
//!    turns, and confirms the second turn's reply demonstrates real
//!    context carryover (a fact stated in turn 1, recalled in turn 2
//!    with no restated history) -- same style of check as
//!    `examples/chat_smoke.rs`, but over the wire, proving the wire
//!    protocol is actually persisting the KV cache, not just plumbing
//!    text through.
//! 2. While that conversation is still open (pinning its model resident
//!    via a live `Arc<LlamaSession>` clone held on its own connection
//!    thread), fires an unrelated one-shot `generate()` against a
//!    *different* model with an aggressively low `--budget-fraction`,
//!    confirming the request completes rather than wedging/deadlocking
//!    or hard-failing over the pinned model it can't evict.
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features llama --bin rampiped -- \
//!         --socket /tmp/rampiped-conv-smoke.sock --budget-fraction 0.0001
//!
//!     cargo run --release --features client --example rampiped_conversation_smoke -- \
//!         /tmp/rampiped-conv-smoke.sock <model-a.gguf> <model-b.gguf>

use anyhow::{Context, Result, bail};
use rampipe::client::{RampipedClient, RampipedConversation};
use rampipe::protocol::{WireOverflowPolicy};
use std::path::PathBuf;

const MAX_NEW_TOKENS: i32 = 60;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let usage = "usage: rampiped_conversation_smoke <socket-path> <model-a.gguf> <model-b.gguf>";
    let socket_path = PathBuf::from(args.next().context(usage)?);
    let model_a = PathBuf::from(args.next().context(usage)?);
    let model_b = PathBuf::from(args.next().context(usage)?);

    println!("=== turn 1: context carryover over the wire ===");
    let mut conversation =
        RampipedConversation::open(&socket_path, &model_a, 4096, WireOverflowPolicy::Fail, None, None, Vec::new(), None)
            .context("opening conversation")?;

    let turn1 = conversation
        .send(
            "My favorite number is 7492. Just say OK.",
            Some(MAX_NEW_TOKENS),
            None,
            None,
            None,
            None,
        )
        .context("turn 1")?;
    println!(
        "  decoded this turn (new text only): {:?}",
        turn1.formatted_prompt
    );
    println!("  reply: {}", turn1.text.trim());

    println!("\n=== turn 2 (no restated context) ===");
    let turn2 = conversation
        .send(
            "What's my favorite number? Reply with just the digits, nothing else.",
            Some(MAX_NEW_TOKENS),
            None,
            None,
            None,
            None,
        )
        .context("turn 2")?;
    println!(
        "  decoded this turn (new text only): {:?}",
        turn2.formatted_prompt
    );
    println!("  reply: {}", turn2.text.trim());

    if !turn2.text.contains("7492") {
        bail!(
            "turn 2 didn't recall the number from turn 1 -- conversation context isn't actually persisting: {:?}",
            turn2.text
        );
    }
    println!(
        "\nOK: turn 2 correctly recalled context from turn 1 with no restated history in the prompt."
    );

    println!("\n=== grammar + assistant_prefill on a conversation turn ===");
    let grammar_turn = conversation
        .send(
            "Is 7492 an even number? Answer with exactly one word: YES or NO.",
            Some(5),
            None,
            Some("root ::= \"YES\" | \"NO\"\n"),
            None,
            Some(rampipe::protocol::GrammarCompletion::ExactMatch(vec![
                "YES".to_string(),
                "NO".to_string(),
            ])),
        )
        .context("grammar-constrained turn")?;
    println!("  reply: {:?}", grammar_turn.text);
    if grammar_turn.text.trim() != "YES" {
        bail!(
            "expected the grammar-constrained turn to answer exactly \"YES\" (7492 is even), got {:?}",
            grammar_turn.text
        );
    }
    println!(
        "  OK: grammar constraint applied on a conversation turn, not just one-shot generate()."
    );

    println!(
        "\n=== concurrency: unrelated one-shot generate() while this conversation is still open ==="
    );
    let client = RampipedClient::connect(&socket_path).context("connecting one-shot client")?;
    let outcome = client
        .generate(
            &model_b,
            "Say hello in exactly one word.",
            Some(10),
            None,
            None,
            None,
            None,
        )
        .context(
            "one-shot generate against a different model while a conversation pins the first",
        )?;
    println!("  reply: {}", outcome.text.trim());
    println!(
        "  OK: one-shot request completed without wedging/deadlocking despite an unrelated pinned conversation."
    );

    println!("\n=== conversation is still usable after the concurrent request ===");
    let turn3 = conversation
        .send(
            "One more time, what's my favorite number? Just the digits.",
            Some(MAX_NEW_TOKENS),
            None,
            None,
            None,
            None,
        )
        .context("turn 3")?;
    println!("  reply: {}", turn3.text.trim());
    if !turn3.text.contains("7492") {
        bail!(
            "turn 3 lost context after the concurrent request: {:?}",
            turn3.text
        );
    }
    println!("  OK: conversation state survived the concurrent unrelated request.");

    println!("\nall checks passed");
    Ok(())
}
