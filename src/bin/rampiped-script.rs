//! A `rampiped` that replays a script, for driving a real agent binary.
//!
//! The out-of-process half of [`rampipe::scripted`]. In-process tests use
//! `ScriptedDaemon` directly; this exists so `agent99` -- or any other
//! agent, in any language -- can be pointed at a socket with `--socket`
//! and put through a scripted failure without a model or a GPU.
//!
//!     rampiped-script --socket /tmp/fake.sock --script collapse.toml
//!
//! It prints what the harness said, as it says it, because half the
//! point is seeing where a briefing or a correction actually goes. On
//! exit it prints the transcript again as a whole, so a shell test can
//! read it after the agent has finished.

use rampipe::scripted::{Said, Script, ScriptedDaemon};
use std::path::PathBuf;

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|arg| arg == "-h" || arg == "--help") {
        eprintln!(
            "rampiped-script --socket <path> --script <file.toml>\n\n\
             Serves the rampiped conversation protocol from a written script -- no model,\n\
             no GPU. Records what the harness sends (system block, tools, messages, tool\n\
             results) and prints it, which is the half that catches harness bugs.\n\n\
             --transcript <path> also writes every exchange in full, unclipped -- the printed\n\
             form is for reading, the file is for checking.\n\n\
             Stops on Ctrl-C, or when the script runs out and the client disconnects."
        );
        return;
    }

    let socket = match take(&mut args, "--socket") {
        Some(path) => PathBuf::from(path),
        None => {
            eprintln!("rampiped-script: --socket is required");
            std::process::exit(2);
        }
    };
    // Where to write the transcript in full.
    //
    // The printed form is clipped to keep a live run readable, and twice
    // that clipping has hidden the very thing being checked -- once the
    // raw `old` of a failing edit, once whether an index reached the
    // model at all. A readable stream and a complete record are
    // different jobs.
    let transcript = take(&mut args, "--transcript").map(PathBuf::from);
    let Some(script_path) = take(&mut args, "--script") else {
        eprintln!("rampiped-script: --script is required");
        std::process::exit(2);
    };
    if !args.is_empty() {
        eprintln!("rampiped-script: unrecognized argument(s): {}", args.join(" "));
        std::process::exit(2);
    }

    let script = match Script::from_file(&PathBuf::from(&script_path)) {
        Ok(script) => script,
        Err(error) => {
            eprintln!("rampiped-script: {error}");
            std::process::exit(1);
        }
    };
    eprintln!(
        "rampiped-script: {} turn(s) from {script_path}, context {}, tool calls {}",
        script.turns.len(),
        script.context_size,
        if script.supports_tool_calls { "supported" } else { "UNSUPPORTED (deliberately)" }
    );

    let daemon = match ScriptedDaemon::bind(&socket, script) {
        Ok(daemon) => daemon,
        Err(error) => {
            eprintln!("rampiped-script: binding {}: {error}", socket.display());
            std::process::exit(1);
        }
    };
    eprintln!("rampiped-script: listening on {}", socket.display());

    // Polled rather than blocked on a signal handler: the interesting
    // output is the transcript growing, and printing it as it arrives is
    // what makes a live run readable. `Ctrl-C` ends it; so does the
    // script running out, since the daemon answers that with an error
    // and the client gives up.
    let mut printed = 0usize;
    loop {
        std::thread::sleep(std::time::Duration::from_millis(50));
        let said_so_far = daemon.transcript();
        for said in said_so_far.iter().skip(printed) {
            eprintln!("{}", render(said));
        }
        if said_so_far.len() != printed {
            if let Some(path) = &transcript {
                let full: String = said_so_far.iter().map(full_record).collect();
                let _ = std::fs::write(path, full);
            }
        }
        printed = said_so_far.len();
    }
}

/// One exchange, complete. No clipping and no indentation, so a test
/// can grep it and a person can diff it.
fn full_record(said: &Said) -> String {
    match said {
        Said::Opened { system, tools, n_ctx } => format!(
            "===== OPENED n_ctx={n_ctx} tools=[{}]\n{}\n",
            tools.join(", "),
            system.clone().unwrap_or_else(|| "(no system block)".to_string())
        ),
        Said::Message(text) => format!("===== MESSAGE\n{text}\n"),
        Said::ToolResults(results) => format!(
            "===== TOOL RESULTS ({})\n{}\n",
            results.len(),
            results.join("\n----- next result -----\n")
        ),
    }
}

fn render(said: &Said) -> String {
    match said {
        Said::Opened { system, tools, n_ctx } => format!(
            "\n── opened ── n_ctx {n_ctx}, tools [{}]\n   system block: {}",
            tools.join(", "),
            match system {
                // Whether there is one at all is the first thing worth
                // knowing: an empty opening is what left a real agent
                // with no instructions once its window filled.
                None => "(none)".to_string(),
                Some(text) => format!("{} chars\n{}", text.len(), indent(text)),
            }
        ),
        Said::Message(text) => format!("\n── message ──\n{}", indent(text)),
        Said::ToolResults(results) => {
            format!("\n── tool results ({}) ──\n{}", results.len(), indent(&results.join("\n---\n")))
        }
    }
}

/// Indented and clipped. A file's contents fed back as a tool result is
/// hundreds of lines, and the thing being read here is the shape of the
/// conversation, not the file.
fn indent(text: &str) -> String {
    const MAX_LINES: usize = 12;
    let lines: Vec<&str> = text.lines().collect();
    let mut out: Vec<String> = lines.iter().take(MAX_LINES).map(|line| format!("   | {line}")).collect();
    if lines.len() > MAX_LINES {
        out.push(format!("   | ... ({} more lines)", lines.len() - MAX_LINES));
    }
    out.join("\n")
}

fn take(args: &mut Vec<String>, flag: &str) -> Option<String> {
    let at = args.iter().position(|arg| arg == flag)?;
    if at + 1 >= args.len() {
        eprintln!("rampiped-script: {flag} needs a value");
        std::process::exit(2);
    }
    args.remove(at);
    Some(args.remove(at))
}
