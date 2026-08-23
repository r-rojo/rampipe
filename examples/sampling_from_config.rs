//! Proves the daemon applies a model's configured sampling when the
//! caller sends none.
//!
//!     cargo run --features client --example sampling_from_config -- <model.gguf>
use rampipe::client::RampipedConversation;
use rampipe::protocol::{WireOverflowPolicy, WirePenalties, WireSampling};

fn main() {
    let model = std::env::args().nth(1).expect("usage: sampling_from_config <model.gguf>");
    let model = std::path::PathBuf::from(model);
    let socket = rampipe::protocol::default_socket_path().expect("socket");

    let settings = rampipe::model_settings::ModelSettings::load(
        &rampipe::model_settings::ModelSettings::default_path().expect("home"),
    )
    .expect("settings");
    println!("configured for this model: {:?}", settings.sampling_for(&model));
    println!("configured max_new_tokens:  {:?}", settings.max_new_tokens_for(&model));

    let mut c = RampipedConversation::open(
        &socket, &model, 8192, WireOverflowPolicy::DropOldestTurns, None, None, Vec::new(), None,
    )
    .expect("open");

    // The whole point: nothing said about sampling.
    let none = c.send("Reply with exactly: OK", Some(8), None, None, None, None).expect("send with None");
    println!("sampling: None      -> {:?}", none.text.trim());

    // And the cap: nothing said about either, so the daemon applies the
    // model's own 8192 rather than a number this program invented.
    let long = c
        .send("Count from 1 to 30, one number per line, nothing else.", None, None, None, None, None)
        .expect("send with no cap");
    let lines = long.text.trim().lines().count();
    println!("max_new_tokens: None -> {} tokens, {lines} lines", long.tokens_generated);
    assert!(long.tokens_generated > 8, "a turn with no cap must not be limited to the 8 used above");

    // And an explicit value still wins.
    let explicit = c
        .send(
            "Reply with exactly: OK",
            Some(8),
            Some(WireSampling::Greedy { penalties: WirePenalties::default() }),
            None,
            None,
            None,
        )
        .expect("send with an explicit value");
    println!("sampling: explicit  -> {:?}", explicit.text.trim());
    println!("\nboth returned text, so the None path reached a real generation");
}
