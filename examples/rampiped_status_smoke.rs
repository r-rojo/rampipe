//! The fastest possible answer to "is rampiped actually working?": no
//! GGUF model required, just a live daemon on the socket. Connects,
//! sends `ClientMessage::Status`, and confirms the reply looks sane
//! (non-zero pid, and if the daemon reported its own exe path, that the
//! path still exists on disk right now).
//!
//! Requires a `rampiped` already running, e.g.:
//!     cargo run --release --features llama --bin rampiped -- \
//!         --socket /tmp/rampiped-status-smoke.sock
//!
//!     cargo run --features client --example rampiped_status_smoke -- \
//!         /tmp/rampiped-status-smoke.sock

use anyhow::{Context, Result, bail};
use rampipe::client::RampipedClient;
use std::path::PathBuf;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let socket_path = PathBuf::from(args.next().context("usage: rampiped_status_smoke <socket-path>")?);

    println!("=== connecting to {} ===", socket_path.display());
    let client = RampipedClient::connect(&socket_path).context("connecting to rampiped")?;

    let status = client.status().context("sending Status request")?;
    println!("  pid: {}", status.pid);
    println!("  exe path: {:?}", status.exe_path);
    println!("  exe modified (unix secs): {:?}", status.exe_modified_unix_secs);
    println!("  uptime: {}s", status.uptime_secs);
    println!("  requests served: {} (failed: {})", status.requests_served, status.requests_failed);
    println!("  total tokens generated: {}", status.total_tokens_generated);
    println!("  resident bytes: {}", status.resident_bytes);
    println!("  gpu free/total bytes: {:?}/{:?}", status.gpu_free_bytes, status.gpu_total_bytes);
    println!("  resident models: {:?}", status.models.iter().map(|m| &m.path).collect::<Vec<_>>());

    if status.pid == 0 {
        bail!("daemon reported pid 0 -- that's not a real process");
    }

    if let Some(exe_path) = &status.exe_path {
        if !exe_path.exists() {
            bail!("daemon's reported exe path {} doesn't exist on this machine -- talking to the wrong host?", exe_path.display());
        }
        println!("  OK: reported exe path exists on disk.");
    }

    println!("\nOK: rampiped is up and answering on the wire protocol (pid {}).", status.pid);
    Ok(())
}
