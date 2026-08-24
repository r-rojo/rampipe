//! A `rampiped` that says what you tell it to.
//!
//! # Why this exists
//!
//! Everything interesting about an agent harness is in how it reacts to
//! what a model does: a reply cut off mid tool call, a reply that
//! collapses into one repeated token, a context window filling up, a
//! model that answers in prose when a tool call was needed. Every one of
//! those was found in this project by starting a real run against a 30B
//! model and reading forty minutes of log afterwards -- and each was
//! diagnosed at least once as the wrong thing, because the only evidence
//! was a transcript.
//!
//! This serves the same Unix-socket protocol as the real daemon and
//! replays a written script. No model, no GPU, no `llama` feature, and
//! the same failure is reproducible in milliseconds as many times as you
//! like.
//!
//! # It records as well as replays
//!
//! The half that matters more. A script fixes what the *model* says; the
//! transcript captures what the *harness* said -- the system block, the
//! tool definitions, every message and every tool result, in order.
//!
//! That is where this project's real bugs were. The task briefing was
//! being sent as the first user turn and evicted when the window filled.
//! Corrections were being sent as `<tool_response>` blocks with no call
//! to answer. Both were invisible from the outside and both are a
//! one-line assertion against a transcript.
//!
//! # Usage
//!
//! In a test, drive it in-process:
//!
//! ```no_run
//! use rampipe::scripted::{Script, ScriptedDaemon, Turn};
//! let script = Script::new(vec![Turn::says("all done")]);
//! let daemon = ScriptedDaemon::start(script).expect("start");
//! // point a client at `daemon.socket()`, then read `daemon.transcript()`
//! ```
//!
//! Or serve a real socket for a real agent binary, with
//! `rampiped-script --socket <path> --script <file.toml>`.

use crate::protocol::{
    ClientMessage, ConversationRequest, ConversationResponse, OpenConversationRequest, ToolCall,
};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

/// One reply the fake model will give.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Turn {
    /// What the model "says". Verbatim -- including a tool call in
    /// whatever shape the harness under test expects to parse, and
    /// including one deliberately cut off halfway.
    pub text: String,
    /// Tool calls the daemon would have parsed out of `text`.
    ///
    /// Given explicitly rather than parsed here, because parsing is the
    /// real daemon's job and duplicating it would make this agree with
    /// the harness by construction instead of by test. A script that
    /// wants to check "the harness ignores text when calls are present"
    /// can set the two to disagree, which is a thing that has happened.
    #[serde(default)]
    pub tool_calls: Vec<ToolCall>,
    /// Whether the reply stops inside an unterminated tool call.
    #[serde(default)]
    pub truncated_tool_call: bool,
    /// How full the context is *after* this turn. `None` fills in a
    /// running estimate, so a script only states it when the number is
    /// the point -- simulating the pressure that made a real agent
    /// collapse at 91% takes one field.
    #[serde(default)]
    pub committed_tokens: Option<usize>,
    /// Overrides the daemon-wide [`Script::context_size`] for this turn.
    #[serde(default)]
    pub context_size: Option<usize>,
    /// `None` estimates from `text`.
    #[serde(default)]
    pub tokens_generated: Option<usize>,
}

impl Turn {
    /// A plain reply with no tool calls.
    #[must_use]
    pub fn says(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tool_calls: Vec::new(),
            truncated_tool_call: false,
            committed_tokens: None,
            context_size: None,
            tokens_generated: None,
        }
    }

    /// A reply that calls one tool.
    #[must_use]
    pub fn calls(name: impl Into<String>, arguments: serde_json::Value) -> Self {
        let name = name.into();
        Self {
            text: format!("<tool_call>{name} {arguments}</tool_call>"),
            tool_calls: vec![ToolCall { name, arguments }],
            truncated_tool_call: false,
            committed_tokens: None,
            context_size: None,
            tokens_generated: None,
        }
    }

    /// A reply cut off inside a tool call -- what a token limit does.
    ///
    /// The tool calls list stays empty on purpose: a truncated call is
    /// one the daemon could not finish parsing, which is exactly why the
    /// flag exists separately from the list.
    #[must_use]
    pub fn truncated(text: impl Into<String>) -> Self {
        Self { truncated_tool_call: true, ..Self::says(text) }
    }

    /// A reply that has collapsed into repetition.
    ///
    /// The shape a real model produced once its instructions had been
    /// evicted: one short phrase, hundreds of times.
    #[must_use]
    pub fn degenerate(phrase: &str, times: usize) -> Self {
        Self::says(phrase.repeat(times))
    }

    /// This turn, but leaving the context this full afterwards.
    #[must_use]
    pub fn at_context(mut self, committed: usize, size: usize) -> Self {
        self.committed_tokens = Some(committed);
        self.context_size = Some(size);
        self
    }
}

/// What the fake model will say, in order.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct Script {
    #[serde(default, rename = "turn")]
    pub turns: Vec<Turn>,
    /// Reported as the window size when a turn does not say otherwise.
    #[serde(default = "default_context")]
    pub context_size: usize,
    /// Whether an open request carrying tools is answered with
    /// `supports_tool_calls: true`.
    ///
    /// `false` reproduces a real refusal path: `agent99` checks this
    /// before spending a single token and aborts, because a model whose
    /// template yields no parseable call format would burn its whole
    /// budget discovering that one empty turn at a time.
    #[serde(default = "default_true")]
    pub supports_tool_calls: bool,
}

fn default_context() -> usize {
    12288
}
fn default_true() -> bool {
    true
}

impl Script {
    #[must_use]
    pub fn new(turns: Vec<Turn>) -> Self {
        Self { turns, context_size: default_context(), supports_tool_calls: true }
    }

    /// Reads a script from a TOML file.
    ///
    /// ```toml
    /// context_size = 12288
    ///
    /// [[turn]]
    /// text = "I'll read the file first."
    ///
    /// [[turn]]
    /// text = "..."
    /// committed_tokens = 11448   # 93% of the window -- where a real run collapsed
    /// ```
    pub fn from_file(path: &Path) -> Result<Self, ScriptError> {
        let text = std::fs::read_to_string(path)
            .map_err(|source| ScriptError::Io { path: path.to_path_buf(), source })?;
        toml::from_str(&text).map_err(|source| ScriptError::Parse { path: path.to_path_buf(), source })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ScriptError {
    #[error("reading script {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("parsing script {path}: {source}")]
    Parse {
        path: PathBuf,
        #[source]
        source: toml::de::Error,
    },
}

/// What the harness said, in the order it said it.
///
/// The assertable half. A test reads this to check *where* something was
/// put, not merely that it was sent -- which is the distinction the
/// context-eviction bug turned on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Said {
    /// The conversation was opened with this system block and these
    /// tools. Both are part of the protected opening in the real daemon,
    /// so what a harness puts here is what survives a full context.
    Opened { system: Option<String>, tools: Vec<String>, n_ctx: u32 },
    /// A user turn.
    Message(String),
    /// Tool results fed back, in call order.
    ToolResults(Vec<String>),
}

impl Said {
    /// The text of this entry, whatever kind it is -- for the common
    /// assertion "did the harness ever tell the model X, anywhere".
    #[must_use]
    pub fn text(&self) -> String {
        match self {
            Said::Opened { system, .. } => system.clone().unwrap_or_default(),
            Said::Message(text) => text.clone(),
            Said::ToolResults(results) => results.join("\n"),
        }
    }
}

/// A fake daemon on a real socket.
///
/// Runs its listener on a thread and stops when dropped, so a test does
/// not have to remember to shut it down -- an abandoned listener would
/// leave a socket file behind and the next test would connect to it.
pub struct ScriptedDaemon {
    socket: PathBuf,
    /// Kept so the directory outlives the socket path inside it.
    _dir: Option<tempfile::TempDir>,
    transcript: Arc<Mutex<Vec<Said>>>,
    /// How many turns the script was asked for. A test that thinks it
    /// exercised five turns and actually exercised two is a test that
    /// proves less than it claims.
    served: Arc<Mutex<usize>>,
    shutdown: Arc<Mutex<bool>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl ScriptedDaemon {
    /// Binds a socket in a fresh temporary directory and serves `script`.
    pub fn start(script: Script) -> std::io::Result<Self> {
        let dir = tempfile::TempDir::new()?;
        let socket = dir.path().join("rampiped.sock");
        let mut daemon = Self::bind(&socket, script)?;
        daemon._dir = Some(dir);
        Ok(daemon)
    }

    /// Binds an explicit path -- what the binary uses, so a real agent
    /// can be pointed at it with `--socket`.
    pub fn bind(socket: &Path, script: Script) -> std::io::Result<Self> {
        // A leftover socket from a previous run makes `bind` fail with
        // "address in use" for a path nothing is listening on, which
        // reads as a port conflict and is not one.
        let _ = std::fs::remove_file(socket);
        let listener = UnixListener::bind(socket)?;
        // So the accept loop can notice a shutdown rather than blocking
        // on accept forever.
        listener.set_nonblocking(true)?;

        let transcript = Arc::new(Mutex::new(Vec::new()));
        let served = Arc::new(Mutex::new(0usize));
        let shutdown = Arc::new(Mutex::new(false));

        let handle = std::thread::spawn({
            let (transcript, served, shutdown) = (transcript.clone(), served.clone(), shutdown.clone());
            move || {
                loop {
                    if *shutdown.lock().expect("shutdown lock") {
                        return;
                    }
                    match listener.accept() {
                        Ok((stream, _)) => {
                            let _ = stream.set_nonblocking(false);
                            serve(stream, &script, &transcript, &served);
                        }
                        Err(error) if error.kind() == std::io::ErrorKind::WouldBlock => {
                            std::thread::sleep(std::time::Duration::from_millis(2));
                        }
                        Err(_) => return,
                    }
                }
            }
        });

        Ok(Self {
            socket: socket.to_path_buf(),
            _dir: None,
            transcript,
            served,
            shutdown,
            handle: Some(handle),
        })
    }

    /// Where to point a client.
    #[must_use]
    pub fn socket(&self) -> &Path {
        &self.socket
    }

    /// Everything the harness has said so far, in order.
    #[must_use]
    pub fn transcript(&self) -> Vec<Said> {
        self.transcript.lock().expect("transcript lock").clone()
    }

    /// How many script turns were actually consumed.
    #[must_use]
    pub fn turns_served(&self) -> usize {
        *self.served.lock().expect("served lock")
    }

    /// The system block the conversation was opened with, if any.
    ///
    /// A shortcut for the assertion this was written for: the task
    /// briefing belongs in the opening, which the real daemon never
    /// evicts, and not in the first user turn, which it drops first.
    #[must_use]
    pub fn system_block(&self) -> Option<String> {
        self.transcript().into_iter().find_map(|said| match said {
            Said::Opened { system, .. } => system,
            _ => None,
        })
    }
}

impl Drop for ScriptedDaemon {
    fn drop(&mut self) {
        *self.shutdown.lock().expect("shutdown lock") = true;
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
        let _ = std::fs::remove_file(&self.socket);
    }
}

/// One client connection: an open, then turns until it disconnects.
fn serve(stream: UnixStream, script: &Script, transcript: &Mutex<Vec<Said>>, served: &Mutex<usize>) {
    let Ok(writer) = stream.try_clone() else { return };
    let mut writer = writer;
    let mut reader = BufReader::new(stream);

    let mut line = String::new();
    if reader.read_line(&mut line).is_err() || line.trim().is_empty() {
        return;
    }
    let Ok(ClientMessage::OpenConversation(open)) = serde_json::from_str::<ClientMessage>(line.trim())
    else {
        // Anything else is a client this does not pretend to be. Saying
        // so beats a silent hang while it waits for a reply.
        let _ = reply(&mut writer, &ConversationResponse::Err {
            message: "this is a scripted rampiped -- it serves conversations only".to_string(),
        });
        return;
    };
    record(transcript, said_opened(&open));

    let response = if open.tools.is_empty() {
        ConversationResponse::Opened
    } else {
        ConversationResponse::OpenedWithTools { supports_tool_calls: script.supports_tool_calls }
    };
    if reply(&mut writer, &response).is_err() {
        return;
    }

    // A running estimate, so a script only states context numbers when
    // they are the thing under test.
    let mut committed = estimate_tokens(open.system.as_deref().unwrap_or_default());

    loop {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return,
            Ok(_) => {}
            Err(_) => return,
        }
        if line.trim().is_empty() {
            continue;
        }
        let Ok(request) = serde_json::from_str::<ConversationRequest>(line.trim()) else {
            let _ = reply(&mut writer, &ConversationResponse::Err {
                message: format!("scripted rampiped could not decode: {}", line.trim()),
            });
            return;
        };
        let turn = match request {
            ConversationRequest::Snapshot(_) => {
                let _ = reply(&mut writer, &ConversationResponse::Snapshotted);
                return;
            }
            ConversationRequest::Turn(turn) => turn,
        };

        record(transcript, match &turn.tool_results {
            Some(results) => Said::ToolResults(results.clone()),
            None => Said::Message(turn.message.clone()),
        });
        committed += estimate_tokens(&match &turn.tool_results {
            Some(results) => results.join("\n"),
            None => turn.message.clone(),
        });

        let at = {
            let mut served = served.lock().expect("served lock");
            let at = *served;
            *served += 1;
            at
        };
        let Some(scripted) = script.turns.get(at) else {
            // Loudly, and not by repeating the last reply. A harness
            // that ran past its script is doing something the script's
            // author did not predict, and that is the finding -- masking
            // it with a plausible answer would turn a discovery into a
            // passing test.
            let _ = reply(&mut writer, &ConversationResponse::Err {
                message: format!(
                    "scripted rampiped ran out: the script has {} turn(s) and the harness asked for turn {}",
                    script.turns.len(),
                    at + 1
                ),
            });
            return;
        };

        let generated = scripted.tokens_generated.unwrap_or_else(|| estimate_tokens(&scripted.text));
        committed += generated;
        let response = ConversationResponse::Turn {
            text: scripted.text.clone(),
            tokens_generated: generated,
            committed_tokens: scripted.committed_tokens.unwrap_or(committed),
            context_size: scripted.context_size.unwrap_or(script.context_size),
            time_to_first_token_ms: 0,
            tool_calls: scripted.tool_calls.clone(),
            truncated_tool_call: scripted.truncated_tool_call,
            formatted_prompt: String::new(),
        };
        if reply(&mut writer, &response).is_err() {
            return;
        }
    }
}

fn said_opened(open: &OpenConversationRequest) -> Said {
    Said::Opened {
        system: open.system.clone(),
        tools: open.tools.iter().map(|tool| tool.function.name.clone()).collect(),
        n_ctx: open.n_ctx,
    }
}

fn record(transcript: &Mutex<Vec<Said>>, said: Said) {
    transcript.lock().expect("transcript lock").push(said);
}

fn reply(writer: &mut UnixStream, response: &ConversationResponse) -> std::io::Result<()> {
    let mut payload = serde_json::to_vec(response).map_err(std::io::Error::other)?;
    payload.push(b'\n');
    writer.write_all(&payload)?;
    writer.flush()
}

/// Roughly four characters per token.
///
/// Deliberately crude: nothing here needs a real tokenizer, and pulling
/// one in would mean this could not build without the `llama` feature,
/// which is the entire point of the file. A script that cares about an
/// exact number states it.
fn estimate_tokens(text: &str) -> usize {
    text.len().div_ceil(4)
}

// Driven through the real client, so these need the feature that
// provides one. The module itself does not -- a harness in another
// crate brings its own client.
#[cfg(all(test, feature = "client"))]
mod tests {
    use super::*;
    use crate::client::RampipedConversation;
    use crate::protocol::{ToolSpec, WireOverflowPolicy};

    fn tools() -> Vec<ToolSpec> {
        vec![ToolSpec::new("read", "read a file", serde_json::json!({"type": "object", "properties": {}}))]
    }

    /// The whole point: what the harness said is recoverable, and *where*
    /// it said it is distinguishable.
    #[test]
    fn the_transcript_distinguishes_the_opening_from_the_first_turn() {
        let daemon = ScriptedDaemon::start(Script::new(vec![Turn::says("ok")])).expect("start");
        let mut conversation = RampipedConversation::open(
            daemon.socket(),
            Path::new("/nonexistent/model.gguf"),
            4096,
            WireOverflowPolicy::DropOldestTurns,
            None,
            Some("SYSTEM: the briefing".to_string()),
            tools(),
            None,
        )
        .expect("open");
        conversation.send("begin", None, None, None, None, None).expect("turn");
        drop(conversation);

        let transcript = daemon.transcript();
        assert_eq!(transcript.len(), 2, "{transcript:?}");
        assert_eq!(
            transcript[0],
            Said::Opened {
                system: Some("SYSTEM: the briefing".to_string()),
                tools: vec!["read".to_string()],
                n_ctx: 4096,
            }
        );
        assert_eq!(transcript[1], Said::Message("begin".to_string()));
        assert_eq!(daemon.system_block().as_deref(), Some("SYSTEM: the briefing"));
    }

    /// A tool result must be recorded as a result, not as user text.
    /// Sending one as the other was a real bug -- a correction arrived as
    /// a `<tool_response>` answering no call at all.
    #[test]
    fn a_tool_result_is_not_recorded_as_a_message() {
        let daemon =
            ScriptedDaemon::start(Script::new(vec![Turn::calls("read", serde_json::json!({})), Turn::says("done")]))
                .expect("start");
        let mut conversation = RampipedConversation::open(
            daemon.socket(),
            Path::new("/nonexistent/model.gguf"),
            4096,
            WireOverflowPolicy::DropOldestTurns,
            None,
            None,
            tools(),
            None,
        )
        .expect("open");
        let first = conversation.send("go", None, None, None, None, None).expect("turn");
        assert_eq!(first.tool_calls.len(), 1, "the script's call has to arrive as a call");
        conversation
            .send_tool_results(&["the file contents".to_string()], None, None, None, None)
            .expect("results");
        drop(conversation);

        let transcript = daemon.transcript();
        assert_eq!(transcript[1], Said::Message("go".to_string()));
        assert_eq!(transcript[2], Said::ToolResults(vec!["the file contents".to_string()]));
    }

    /// Context pressure on demand -- the condition under which a real
    /// agent collapsed, expressible in one line instead of forty minutes.
    #[test]
    fn a_script_can_put_the_window_wherever_it_needs_it() {
        let daemon = ScriptedDaemon::start(Script::new(vec![
            Turn::says("comfortable").at_context(7031, 12288),
            Turn::degenerate("I'm in the world and if I'm in the ", 40).at_context(11448, 12288),
        ]))
        .expect("start");
        let mut conversation = RampipedConversation::open(
            daemon.socket(),
            Path::new("/nonexistent/model.gguf"),
            12288,
            WireOverflowPolicy::DropOldestTurns,
            None,
            None,
            Vec::new(),
            None,
        )
        .expect("open");

        let healthy = conversation.send("one", None, None, None, None, None).expect("turn");
        assert_eq!((healthy.committed_tokens, healthy.context_size), (7031, 12288));
        let collapsed = conversation.send("two", None, None, None, None, None).expect("turn");
        assert_eq!(collapsed.committed_tokens, 11448);
        assert!(collapsed.text.matches("I'm in the world").count() > 30);
    }

    /// Running past the script is a finding, not something to paper over
    /// by repeating the last reply.
    #[test]
    fn asking_for_more_turns_than_the_script_has_is_an_error() {
        let daemon = ScriptedDaemon::start(Script::new(vec![Turn::says("only one")])).expect("start");
        let mut conversation = RampipedConversation::open(
            daemon.socket(),
            Path::new("/nonexistent/model.gguf"),
            4096,
            WireOverflowPolicy::DropOldestTurns,
            None,
            None,
            Vec::new(),
            None,
        )
        .expect("open");
        conversation.send("one", None, None, None, None, None).expect("the first is scripted");
        let second = conversation.send("two", None, None, None, None, None);
        let error = second.expect_err("the second is not").to_string();
        assert!(error.contains("ran out") && error.contains("turn 2"), "{error}");
    }

    /// A daemon that cannot parse tool calls is a real, reachable state,
    /// and a harness is supposed to refuse to start against one.
    #[test]
    fn a_script_can_refuse_to_support_tool_calls() {
        let script = Script { supports_tool_calls: false, ..Script::new(vec![Turn::says("hi")]) };
        let daemon = ScriptedDaemon::start(script).expect("start");
        let conversation = RampipedConversation::open(
            daemon.socket(),
            Path::new("/nonexistent/model.gguf"),
            4096,
            WireOverflowPolicy::DropOldestTurns,
            None,
            None,
            tools(),
            None,
        )
        .expect("open");
        assert!(!conversation.supports_tool_calls());
    }

    /// The file form, since that is how an out-of-process agent uses it.
    #[test]
    fn a_script_round_trips_through_toml() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("script.toml");
        std::fs::write(
            &path,
            r#"
context_size = 8192

[[turn]]
text = "I'll read it first."

[[turn]]
text = "<tool_call>read {\"path\":\"a.rs\"}"
truncated_tool_call = true
committed_tokens = 7900
"#,
        )
        .expect("write");

        let script = Script::from_file(&path).expect("parse");
        assert_eq!(script.context_size, 8192);
        assert_eq!(script.turns.len(), 2);
        assert!(script.turns[1].truncated_tool_call);
        assert_eq!(script.turns[1].committed_tokens, Some(7900));
        assert!(script.supports_tool_calls, "defaults to usable");
    }

    /// Dropping it has to free the path, or the next test binds a socket
    /// that something else already answers.
    #[test]
    fn the_socket_is_gone_once_the_daemon_is_dropped() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let path = dir.path().join("rampiped.sock");
        {
            let daemon = ScriptedDaemon::bind(&path, Script::new(vec![])).expect("bind");
            assert!(daemon.socket().exists());
        }
        assert!(!path.exists(), "a leftover socket makes the next bind fail as 'address in use'");
    }
}
