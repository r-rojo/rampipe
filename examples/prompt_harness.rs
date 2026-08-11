//! Feed an arbitrary prompt straight to the same model/config
//! `taskpipe`'s `LocalBackend` actually uses, and see the raw response —
//! for iterating on prompt wording (e.g. task 2/3 of the piper roadmap:
//! a hallucinated `tempfile` import, hallucinated `crossterm::terminal`
//! function names) without going through a full `taskpipe` run each
//! time (worktree setup, `cargo build`/`test`, retries...) just to see
//! whether a reworded prompt changes what the model writes.
//!
//! Deliberately reuses the exact same model repo/filename and
//! `LOCAL_MAX_NEW_TOKENS` value `taskpipe`'s `backend.rs` uses (kept in
//! sync by comment, not by sharing code across crates — if either
//! changes, update the other) — a harness testing against a different
//! model or token budget than what actually runs in taskpipe would be
//! misleading, not useful.
//!
//! Usage:
//!     cargo run --release --features llama --example prompt_harness -- <prompt-file> [--seed N] [--save <path>]
//!
//! `<prompt-file>` is read whole and sent as-is — write/edit it in a
//! real editor between runs. Without `--seed`, uses `Sampling::Greedy`
//! (deterministic, matching a task's first attempt in the real retry
//! loop). With `--seed N`, uses `Sampling::Temperature` at the same
//! temperature/top_k `LocalBackend` uses for retries — useful for
//! checking whether a prompt holds up across the kind of variation a
//! real retry would actually see, not just the one greedy completion.
//!
//! Prints both the raw response (useful for diagnosing things like a
//! degenerate repetition loop, which only shows up in the raw text) and
//! the extracted fenced code block, via the exact same `extract_code_block`
//! logic `taskpipe::backend` uses — a hand-rolled reimplementation here
//! could disagree with what taskpipe would actually accept. `--save
//! <path>` writes just the extracted code straight to a file, so
//! there's no manual copy-paste-and-trim step between "the model wrote
//! something" and "I have a file to `cargo test`."

use anyhow::{Context, Result, bail};
use hf_hub::HFClientSync;
use rampipe::llama::{LlamaSession, Sampling};
use rampipe::Residency;
use std::path::PathBuf;
use std::time::Instant;

struct ModelCandidate {
    repo_owner: &'static str,
    repo_name: &'static str,
    filename: &'static str,
}

/// Kept in sync by comment with `taskpipe`'s `policy/policy.lua`
/// `LOCAL_MODEL_CANDIDATES` table — every real entry there, mirrored
/// here (same repo_owner/repo_name/filename), the same hand-maintained-
/// in-parallel convention `LOCAL_MAX_NEW_TOKENS`/`RETRY_TEMPERATURE`/
/// `RETRY_TOP_K` below already use. Selectable via `--model <name>`;
/// add an entry here whenever `policy.lua`'s own table gets a new one.
const MODEL_CANDIDATES: &[(&str, ModelCandidate)] = &[
    (
        "unsloth_q4km",
        ModelCandidate {
            repo_owner: "unsloth",
            repo_name: "Qwen3-Coder-30B-A3B-Instruct-GGUF",
            filename: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf",
        },
    ),
    (
        "giladgd_q4km",
        ModelCandidate {
            repo_owner: "giladgd",
            repo_name: "Qwen3-Coder-30B-A3B-Instruct-Q4_K_M-GGUF",
            filename: "qwen3-coder-30b-a3b-instruct-q4_k_m.gguf",
        },
    ),
    (
        "unsloth_q5km",
        ModelCandidate {
            repo_owner: "unsloth",
            repo_name: "Qwen3-Coder-30B-A3B-Instruct-GGUF",
            filename: "Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf",
        },
    ),
    (
        "unsloth_q6k",
        ModelCandidate {
            repo_owner: "unsloth",
            repo_name: "Qwen3-Coder-30B-A3B-Instruct-GGUF",
            filename: "Qwen3-Coder-30B-A3B-Instruct-Q6_K.gguf",
        },
    ),
    (
        "qwen3_6_35b_a3b_q4km",
        ModelCandidate { repo_owner: "unsloth", repo_name: "Qwen3.6-35B-A3B-GGUF", filename: "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" },
    ),
    (
        "devstral_small_2507_q4km",
        ModelCandidate { repo_owner: "unsloth", repo_name: "Devstral-Small-2507-GGUF", filename: "Devstral-Small-2507-Q4_K_M.gguf" },
    ),
    (
        "jamba_mini_1_7_q3km",
        ModelCandidate {
            repo_owner: "bartowski",
            repo_name: "ai21labs_AI21-Jamba-Mini-1.7-GGUF",
            filename: "ai21labs_AI21-Jamba-Mini-1.7-Q3_K_M.gguf",
        },
    ),
];

/// Matches `policy.lua`'s own `ACTIVE_LOCAL_MODEL`.
const DEFAULT_MODEL: &str = "giladgd_q4km";

// Kept in sync by comment with `taskpipe::backend::LOCAL_MAX_NEW_TOKENS`
// — if that changes, this should too, or a prompt that fits here could
// still get truncated for real in taskpipe (or vice versa).
const LOCAL_MAX_NEW_TOKENS: i32 = 1400;

// Matches `LocalBackend::sampling_for_attempt`'s retry values exactly —
// see that function's own doc comment for why these specific numbers.
const RETRY_TEMPERATURE: f32 = 0.7;
const RETRY_TOP_K: i32 = 40;

/// `(prompt file, sampling, --save path, model candidate, --repo-dir)`.
type ParsedArgs = (PathBuf, Sampling, Option<PathBuf>, &'static ModelCandidate, Option<PathBuf>);

fn parse_args() -> Result<ParsedArgs> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();

    let sampling = match args.iter().position(|a| a == "--seed") {
        Some(idx) => {
            if idx + 1 >= args.len() {
                bail!("--seed requires a value");
            }
            let seed: u32 = args.remove(idx + 1).parse().context("--seed must be a u32")?;
            args.remove(idx);
            Sampling::Temperature { temperature: RETRY_TEMPERATURE, top_k: RETRY_TOP_K, seed }
        }
        None => Sampling::Greedy,
    };

    let save_path = match args.iter().position(|a| a == "--save") {
        Some(idx) => {
            if idx + 1 >= args.len() {
                bail!("--save requires a value");
            }
            let path = args.remove(idx + 1);
            args.remove(idx);
            Some(PathBuf::from(path))
        }
        None => None,
    };

    let model = match args.iter().position(|a| a == "--model") {
        Some(idx) => {
            if idx + 1 >= args.len() {
                bail!("--model requires a value");
            }
            let name = args.remove(idx + 1);
            args.remove(idx);
            &MODEL_CANDIDATES
                .iter()
                .find(|(candidate_name, _)| *candidate_name == name)
                .with_context(|| {
                    let known: Vec<&str> = MODEL_CANDIDATES.iter().map(|(name, _)| *name).collect();
                    format!("no model candidate named {name:?} in this harness's own MODEL_CANDIDATES — known: {}", known.join(", "))
                })?
                .1
        }
        None => &MODEL_CANDIDATES.iter().find(|(name, _)| *name == DEFAULT_MODEL).expect("DEFAULT_MODEL is a real entry above").1,
    };

    // Real preexisting `src/*.rs` files, checked the same way
    // `taskpipe::backend::multi_file_path_rejection`'s `AlreadyExists`
    // case does — without this, the harness can't reproduce the single
    // most common real rejection reason (chronopipe's draft-1: a
    // genuinely well-formed response rejected only because its file
    // name already existed), since a bare prompt file has no worktree
    // to check against.
    let repo_dir = match args.iter().position(|a| a == "--repo-dir") {
        Some(idx) => {
            if idx + 1 >= args.len() {
                bail!("--repo-dir requires a value");
            }
            let path = args.remove(idx + 1);
            args.remove(idx);
            Some(PathBuf::from(path))
        }
        None => None,
    };

    let prompt_file = args.into_iter().next().context(
        "usage: prompt_harness <prompt-file> [--seed N] [--save <path>] [--model <name>] [--repo-dir <path>]  \
         (e.g. prompt_harness ./prompt.txt --model jamba_mini_1_7_q3km --repo-dir ~/projects/rust/chronopipe)",
    )?;
    Ok((PathBuf::from(prompt_file), sampling, save_path, model, repo_dir))
}

/// Mirrors `taskpipe::backend::extract_code_block` exactly — kept in
/// sync by comment, not by sharing code across crates (same reasoning
/// as `LOCAL_MAX_NEW_TOKENS` above): this harness extracting a code
/// block by different rules than taskpipe actually uses would defeat
/// the point of testing against it.
fn extract_code_block(text: &str) -> Option<String> {
    let start = text.find("```")?;
    let after_start = &text[start + 3..];
    let content_start = after_start.find('\n').map(|i| i + 1).unwrap_or(0);
    let after_lang = &after_start[content_start..];
    let end = after_lang.find("```")?;
    Some(after_lang[..end].trim_end().to_string())
}

/// Counts fully-closed fenced code blocks in `text` — not "how many
/// files does this describe" (a model could easily emit several blocks
/// that are all just re-attempts at the same file, which is exactly
/// what a degenerate repetition loop under greedy decoding looks like,
/// not genuine multi-file output). Exists purely as a warning signal:
/// `extract_code_block`/real `taskpipe` only ever use the *first* one,
/// so more than one means something after it is being silently
/// discarded, whatever the reason.
fn count_code_blocks(text: &str) -> usize {
    let mut count = 0;
    let mut remaining = text;
    while let Some(start) = remaining.find("```") {
        let after_start = &remaining[start + 3..];
        let content_start = after_start.find('\n').map(|i| i + 1).unwrap_or(0);
        let after_lang = &after_start[content_start..];
        match after_lang.find("```") {
            Some(end) => {
                count += 1;
                remaining = &after_lang[end + 3..];
            }
            None => break,
        }
    }
    count
}

/// Mirrors `taskpipe::backend::extract_multi_file_blocks` — marker-first,
/// not fence-first (a real, live case desynced a naive left-to-right
/// fence-pairing scan: a stray outer fence around one `FILE:` marker
/// lost every file after it — see that function's own doc comment).
/// Every `FILE: <path>` line is located first, then `extract_code_block`
/// above runs against just the text between this marker and the next
/// one (or the end of the response), so a stray fence anywhere is
/// contained to whichever marker's own window it falls in.
fn extract_multi_file_blocks(text: &str) -> Vec<(String, String)> {
    let mut markers: Vec<(String, usize, usize)> = Vec::new();
    let mut line_start = 0usize;
    for line in text.split_inclusive('\n') {
        let line_end = line_start + line.len();
        if let Some(path) = line.trim().strip_prefix("FILE:") {
            markers.push((path.trim().to_string(), line_start, line_end));
        }
        line_start = line_end;
    }

    let mut result = Vec::new();
    for (i, (path, _, window_start)) in markers.iter().enumerate() {
        let window_end = markers.get(i + 1).map(|(_, next_line_start, _)| *next_line_start).unwrap_or(text.len());
        if let Some(code) = extract_code_block(&text[*window_start..window_end]) {
            result.push((path.clone(), code));
        }
    }
    result
}

#[derive(Debug, Clone, Copy)]
enum PathRejection {
    CrateRoot,
    AlreadyExists,
    Malformed,
}

impl PathRejection {
    fn describe(self) -> &'static str {
        match self {
            PathRejection::CrateRoot => "crate root (`src/main.rs`/`src/lib.rs`) — always rejected",
            PathRejection::AlreadyExists => "a file at this exact path already exists under --repo-dir's src/",
            PathRejection::Malformed => "not a valid new file — must be a flat `.rs` file directly under `src/`, no subdirectories, no `..`",
        }
    }
}

/// Mirrors `taskpipe::backend::multi_file_path_rejection` exactly.
/// `preexisting` is empty (so only `CrateRoot`/`Malformed` can ever
/// fire) unless `--repo-dir` was given.
fn multi_file_path_rejection(path: &str, preexisting: &std::collections::HashSet<String>) -> Option<PathRejection> {
    if matches!(path, "src/main.rs" | "src/lib.rs") {
        return Some(PathRejection::CrateRoot);
    }
    if path.contains("..") || path.contains('\\') {
        return Some(PathRejection::Malformed);
    }
    let candidate = std::path::Path::new(path);
    if candidate.is_absolute() {
        return Some(PathRejection::Malformed);
    }
    if candidate.extension().and_then(|e| e.to_str()) != Some("rs") {
        return Some(PathRejection::Malformed);
    }
    if candidate.parent() != Some(std::path::Path::new("src")) {
        return Some(PathRejection::Malformed);
    }
    if preexisting.contains(path) {
        return Some(PathRejection::AlreadyExists);
    }
    None
}

/// Mirrors `taskpipe::backend::existing_flat_src_files` — real,
/// already-present `.rs` files directly under `repo_dir`'s `src/`.
fn existing_flat_src_files(repo_dir: &std::path::Path) -> std::collections::HashSet<String> {
    let mut set = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(repo_dir.join("src")) {
        for entry in entries.flatten() {
            if let Some(name) = entry.file_name().to_str()
                && name.ends_with(".rs")
            {
                set.insert(format!("src/{name}"));
            }
        }
    }
    set
}

fn resolve_model_path(model: &ModelCandidate) -> Result<PathBuf> {
    let client = HFClientSync::new().context("creating Hugging Face Hub client")?;
    let repo = client.model(model.repo_owner, model.repo_name);
    repo.download_file()
        .filename(model.filename)
        .send()
        .context("resolving model file (should be a cache hit if taskpipe has run before)")
}

fn main() -> Result<()> {
    let (prompt_path, sampling, save_path, model, repo_dir) = parse_args()?;
    let prompt = std::fs::read_to_string(&prompt_path)
        .with_context(|| format!("reading prompt file {}", prompt_path.display()))?;

    println!("Prompt: {} ({} chars) — sampling: {:?} — model: {}/{}", prompt_path.display(), prompt.chars().count(), sampling, model.repo_owner, model.repo_name);

    let model_path = resolve_model_path(model)?;
    let backend = llama_cpp_2::llama_backend::LlamaBackend::init().context("llama.cpp backend init")?;
    let registry = rampipe::SwapRegistry::new();

    println!("Loading model (first call only; cached after)...");
    let load_start = Instant::now();
    // Lazy, matching `LocalBackend`: a one-shot harness call gains
    // nothing from paying `Prefault`'s upfront page-in cost.
    let session =
        LlamaSession::load(&registry, &backend, &model_path, Residency::Lazy).context("loading session")?;
    println!("  load() call returned in {:?} (actual page-in is lazy, folds into first token below)", load_start.elapsed());

    let gen_start = Instant::now();
    let result = session
        .generate(&backend, &prompt, LOCAL_MAX_NEW_TOKENS, sampling)
        .context("generate() failed")?;
    let total_wall_time = gen_start.elapsed();
    let tok_per_sec = result.tokens_generated as f64 / total_wall_time.as_secs_f64().max(f64::EPSILON);

    println!("\n=== stats ===");
    println!("  time_to_first_token: {:?}", result.time_to_first_token);
    println!("  total generate() wall time: {total_wall_time:?}");
    println!("  tokens_generated: {} ({tok_per_sec:.1} tok/s)", result.tokens_generated);

    println!("\n=== raw output ===");
    println!("{}", result.text);

    let block_count = count_code_blocks(&result.text);
    if block_count > 1 {
        println!(
            "\n⚠ {block_count} closed code blocks found, but taskpipe (and this harness) only ever use \
             the first — everything from the second block onward is being silently discarded. If the \
             blocks after the first look like re-attempts at the same content rather than genuinely \
             different files, this is likely degenerate repetition (more common under Sampling::Greedy \
             on a long/complex prompt) rather than intentional multi-file output."
        );
    }

    match extract_code_block(&result.text) {
        Some(code) => {
            println!("\n=== extracted code block ===");
            println!("{code}");

            if let Some(path) = &save_path {
                std::fs::write(path, &code).with_context(|| format!("writing extracted code to {}", path.display()))?;
                println!("\nSaved extracted code to {}", path.display());
            }
        }
        None => {
            println!(
                "\n=== extracted code block ===\n\
                 (none found — no closed ``` fence in the response; this is exactly \
                 what taskpipe's own extract_code_block would also fail to find, so \
                 a real run would hit NoCodeBlockFound here too)"
            );
            if save_path.is_some() {
                bail!("--save requested but no code block was found to save");
            }
        }
    }

    // Multi-file (`FILE: <path>` marker) mode — for a prompt using
    // taskpipe's `MULTI_FILE_FORMAT_SPEC` convention (`execute_multi_file`,
    // not the single-file `execute()` path above). Only printed when at
    // least one `FILE:` block actually parsed, so a plain single-file
    // prompt's output stays exactly as before.
    let preexisting = repo_dir.as_deref().map(existing_flat_src_files).unwrap_or_default();
    let multi_blocks = extract_multi_file_blocks(&result.text);
    if !multi_blocks.is_empty() {
        println!("\n=== multi-file extraction ({} FILE: block(s) found) ===", multi_blocks.len());
        for (path, code) in &multi_blocks {
            match multi_file_path_rejection(path, &preexisting) {
                None => println!("  ACCEPTED `{path}` ({} chars)", code.len()),
                Some(reason) => println!("  REJECTED `{path}`: {}", reason.describe()),
            }
        }
        match &repo_dir {
            Some(dir) => println!("  (checked AlreadyExists against {}'s real src/*.rs files)", dir.display()),
            None => println!(
                "  (no --repo-dir given — AlreadyExists can never fire here; pass --repo-dir <path> \
                 to check against a real repo's src/, e.g. --repo-dir ~/projects/rust/chronopipe)"
            ),
        }
        if save_path.is_some() {
            println!("  (--save only writes the single-file extraction above, not these — copy/paste from here if needed)");
        }
    }

    Ok(())
}
