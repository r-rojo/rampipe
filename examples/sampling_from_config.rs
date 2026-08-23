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

    let mut c = RampipedConversation::open(
        &socket, &model, 2048, WireOverflowPolicy::Fail, None, None, Vec::new(), None,
    )
    .expect("open");

    // The whole point: nothing said about sampling.
    let none = c.send("Reply with exactly: OK", 8, None, None, None, None).expect("send with None");
    println!("sampling: None      -> {:?}", none.text.trim());

    // And an explicit value still wins.
    let explicit = c
        .send(
            "Reply with exactly: OK",
            8,
            Some(WireSampling::Greedy { penalties: WirePenalties::default() }),
            None,
            None,
            None,
        )
        .expect("send with an explicit value");
    println!("sampling: explicit  -> {:?}", explicit.text.trim());
    println!("\nboth returned text, so the None path reached a real generation");
}
