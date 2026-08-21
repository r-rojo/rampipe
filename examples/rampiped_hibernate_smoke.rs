//! Manual smoke test for `Snapshot`/`restore_from`: opens a conversation,
//! sends a turn, snapshots it to disk, reopens a *fresh* conversation
//! restored from that snapshot, and confirms the model still recalls the
//! first turn -- proving the KV cache round-tripped through disk rather
//! than just proving the connection stayed open.
//!
//! Requires a real running `rampiped` and a real model:
//!
//!     cargo run --release --features client --example rampiped_hibernate_smoke -- \
//!         ~/.rampipe/rampiped.sock /path/to/model.gguf

use anyhow::Context;
use rampipe::client::RampipedConversation;
use rampipe::protocol::{GrammarCompletion, SnapshotRef, WireOverflowPolicy, WireSampling};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let usage = "usage: rampiped_hibernate_smoke <socket_path> <model_path>";
    let mut args = std::env::args().skip(1);
    let socket_path = PathBuf::from(args.next().context(usage)?);
    let model_path = PathBuf::from(args.next().context(usage)?);

    let state_path = std::env::temp_dir().join("rampiped_hibernate_smoke.state");
    let meta_path = std::env::temp_dir().join("rampiped_hibernate_smoke.meta.json");

    println!("=== opening a fresh conversation ===");
    let mut conversation = RampipedConversation::open(
        &socket_path,
        &model_path,
        4096,
        WireOverflowPolicy::Fail,
        None,
    )
    .context("opening conversation")?;

    let turn1 = conversation
        .send(
            "Say the single word: pong",
            8,
            WireSampling::Greedy,
            None,
            None,
            None::<GrammarCompletion>,
        )
        .context("first turn")?;
    println!("turn 1: {:?}", turn1.text);

    println!("=== snapshotting to disk ===");
    conversation
        .snapshot(&state_path, &meta_path)
        .context("snapshotting conversation")?;
    println!(
        "saved to {} / {}",
        state_path.display(),
        meta_path.display()
    );

    println!("=== reopening a FRESH conversation restored from that snapshot ===");
    let mut restored = RampipedConversation::open(
        &socket_path,
        &model_path,
        4096,
        WireOverflowPolicy::Fail,
        Some(SnapshotRef {
            state_path: state_path.clone(),
            meta_path: meta_path.clone(),
        }),
    )
    .context("reopening from snapshot")?;

    let turn2 = restored
        .send(
            "What single word did I just ask you to say?",
            16,
            WireSampling::Greedy,
            None,
            None,
            None::<GrammarCompletion>,
        )
        .context("second turn, on the restored conversation")?;
    println!("turn 2 (restored): {:?}", turn2.text);

    if turn2.text.to_lowercase().contains("pong") {
        println!("=== PASS: restored conversation recalled turn 1's context ===");
    } else {
        anyhow::bail!(
            "FAIL: restored conversation did not recall \"pong\" -- got {:?}",
            turn2.text
        );
    }

    Ok(())
}
