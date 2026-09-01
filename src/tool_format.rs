//! Recovering a model's *tool-calling* API surface from its own chat
//! template, the same way `llama::derive_conversation_template` already
//! recovers its turn structure: render probe transcripts built from
//! sentinel strings, then read the literal spans the template put
//! around them.
//!
//! # Why derive rather than configure
//!
//! There is no single tool-calling format. Two real ones, both of which
//! this module handles, differ structurally rather than cosmetically:
//!
//! - Qwen3-Coder emits nested XML -- `<tool_call><function=NAME>
//!   <parameter=ARG>value</parameter></function></tool_call>` -- with
//!   one delimited block per argument and values never quoted or
//!   escaped.
//! - Hermes/Qwen2.5-style templates emit a single JSON object,
//!   `<tool_call>{"name": ..., "arguments": {...}}</tool_call>`.
//!
//! A hand-written parser for either is wrong for the other, and a
//! config key naming a format means every new model is a code change
//! plus a config change. But the template already *contains* both
//! answers: it renders an assistant tool call for the transcript, so
//! rendering one with known sentinel values and reading back where they
//! landed recovers the exact format that model was trained to emit.
//!
//! This is deliberately the same trick, and the same trust model, as
//! `derive_conversation_template`: probe with strings that cannot occur
//! naturally, require the result to be self-consistent, and return
//! `None` rather than guess when it isn't. `None` is not a failure --
//! it is this module correctly declining, and the caller falls back to
//! whatever the host configured (see `rampipe::protocol::ToolFormat`).
//!
//! # What this module does not do
//!
//! It never decides *whether* to offer tools, never builds a tool list,
//! and never talks to a model. It converts a template into a format
//! description, and converts a model's raw output text into
//! [`crate::protocol::ToolCall`]s using one. Everything about which
//! tools exist belongs to the caller.

use crate::protocol::{ToolCall, ToolFormat, ToolSpec};

/// Private Use Area-wrapped sentinels, deliberately containing *no*
/// JSON- or HTML-special characters -- which is where these differ from
/// `llama::derive_conversation_template`'s own `<&"'>`-laden ones, and
/// the difference is load-bearing.
///
/// That module wants special characters precisely so any escaping breaks
/// its exact-match `find()` and the derivation declines. Here, escaping
/// is not a warning sign but the *expected* behavior of an entire
/// supported family: a JSON-family template serializes the call through
/// `tojson`, which turns a `"` in a sentinel into `\"` and makes every
/// subsequent `find()` miss. Probing that family with quote-bearing
/// sentinels would report every JSON template as underivable.
///
/// The safety that buys is replaced, not dropped: [`derive_delimited_format`]
/// re-derives each span from more than one position and requires the
/// factorizations to agree, so a template that mangles its inputs still
/// fails to produce a self-consistent format.
const NAME: &str = "\u{E000}TFNAME\u{E001}";
const ARG1: &str = "\u{E000}TFARGA\u{E001}";
const VAL1: &str = "\u{E000}TFVALA\u{E001}";
const ARG2: &str = "\u{E000}TFARGB\u{E001}";
const VAL2: &str = "\u{E000}TFVALB\u{E001}";
const PLAIN: &str = "\u{E000}TFPLAIN\u{E001}";
const USER: &str = "\u{E000}TFUSER\u{E001}";
const RESULT: &str = "\u{E000}TFRESULT\u{E001}";
const REPLY: &str = "\u{E000}TFREPLY\u{E001}";
const RESULT2: &str = "\u{E000}TFRESULTB\u{E001}";

/// What a model's template says it can be told, discovered rather than
/// declared. Each field is independently probed: a template can render
/// a tool list without accepting a `system` message, or accept tool
/// results without any of the rest, and a caller needs to know which
/// specifically rather than one blanket "supports tools" bit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChatCapabilities {
    /// A `system`-role message renders as its own distinct block. When
    /// false, a caller must fold its system text into the first user
    /// message instead -- which is what every caller was doing
    /// unconditionally before this existed.
    pub system: bool,
    /// A `tools` list renders into the prompt at all. Checked by
    /// rendering the same transcript with and without tools and
    /// requiring the output to actually differ *and* to contain the
    /// tool's own name: a template that silently ignores an unknown
    /// `tools` variable produces identical output, which is exactly the
    /// case that must not read as support.
    pub tools: bool,
    /// A `tool`-role message (a call's result) renders as its own turn,
    /// so results can be fed back in the shape the model was trained to
    /// read them.
    pub tool_results: bool,
}

/// Renders `template_text` with `messages` (each a `(role, content)`
/// pair) and optional `tools`, via the caller-supplied `render`. Taking
/// the renderer as a function keeps this module free of any direct
/// minijinja dependency, so the derivation logic below is ordinary
/// string code that unit tests can drive with a trivial stub.
pub type RenderFn<'a> =
    &'a dyn Fn(&str, &serde_json::Value, Option<&serde_json::Value>) -> Option<String>;

/// Probes `template_text` for each capability in [`ChatCapabilities`].
#[must_use]
pub fn derive_capabilities(template_text: &str, render: RenderFn<'_>) -> ChatCapabilities {
    ChatCapabilities {
        system: probe_system(template_text, render),
        tools: probe_tools(template_text, render),
        tool_results: probe_tool_results(template_text, render),
    }
}

fn probe_system(template_text: &str, render: RenderFn<'_>) -> bool {
    let messages = serde_json::json!([
        { "role": "system", "content": PLAIN },
        { "role": "user", "content": USER },
    ]);
    let Some(rendered) = render(template_text, &messages, None) else {
        return false;
    };
    // Both must appear, system first: a template that drops the system
    // message entirely, or folds it in after the user's, is not one a
    // caller can put a system prompt through.
    match (rendered.find(PLAIN), rendered.find(USER)) {
        (Some(system_at), Some(user_at)) => system_at < user_at,
        _ => false,
    }
}

fn probe_tools(template_text: &str, render: RenderFn<'_>) -> bool {
    let messages = serde_json::json!([{ "role": "user", "content": USER }]);
    let tools = probe_tool_list();

    let Some(with_tools) = render(template_text, &messages, Some(&tools)) else {
        return false;
    };
    let Some(without_tools) = render(template_text, &messages, None) else {
        return false;
    };
    // Not just "the name appears" -- a template ignoring `tools`
    // renders identically, and a template that somehow echoed the name
    // without the caller's list still wouldn't be honoring it.
    with_tools != without_tools && with_tools.contains(NAME)
}

fn probe_tool_results(template_text: &str, render: RenderFn<'_>) -> bool {
    let messages = serde_json::json!([
        { "role": "user", "content": USER },
        { "role": "assistant", "content": PLAIN },
        { "role": "tool", "content": RESULT },
    ]);
    let Some(rendered) = render(template_text, &messages, None) else {
        return false;
    };
    match (rendered.find(PLAIN), rendered.find(RESULT)) {
        (Some(assistant_at), Some(result_at)) => assistant_at < result_at,
        _ => false,
    }
}

/// A one-tool list shaped like the OpenAI-style schema every template
/// this targets expects (`{type, function: {name, description,
/// parameters}}`), with a sentinel name so [`probe_tools`] can find it.
fn probe_tool_list() -> serde_json::Value {
    serde_json::json!([{
        "type": "function",
        "function": {
            "name": NAME,
            "description": "probe",
            "parameters": {
                "type": "object",
                "properties": { "probe_arg": { "type": "string", "description": "probe" } },
                "required": ["probe_arg"]
            }
        }
    }])
}

/// Recovers how this model emits a tool call, or `None` if the template
/// renders one in a shape this module can't reduce to either supported
/// family. See this module's own doc comment for why `None` is a normal
/// outcome and not an error.
///
/// The call's own boundaries are found by rendering the *same*
/// transcript twice -- once with the assistant turn carrying plain
/// content, once carrying tool calls -- and taking the common prefix
/// and common suffix. Wherever those two renders diverge is exactly the
/// span the template devotes to tool calls, with no assumption about
/// what a turn's opening or closing markers look like.
#[must_use]
pub fn derive_tool_call_format(template_text: &str, render: RenderFn<'_>) -> Option<ToolFormat> {
    let plain = render(
        template_text,
        &serde_json::json!([
            { "role": "user", "content": USER },
            { "role": "assistant", "content": PLAIN },
        ]),
        None,
    )?;
    let with_calls = render(
        template_text,
        &serde_json::json!([
            { "role": "user", "content": USER },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": {
                        "name": NAME,
                        "arguments": { ARG1: VAL1, ARG2: VAL2 }
                    }
                }]
            },
        ]),
        None,
    )?;

    // A third render, with a call taking *no* arguments. Two probes
    // alone leave the format genuinely ambiguous: with only
    // `name_close + arg_open` and `arg_close + arg_open` to go on, the
    // boundary between them can be placed anywhere their shared text
    // allows. Measured on the real Qwen3-Coder template, the greedy
    // split put *all* of `name_close` into `arg_open`, leaving an empty
    // `name_close` -- which then matches at offset 0 of every call body
    // and yields an empty function name. The zero-argument render pins
    // that boundary independently, because it contains `name_close`
    // with no argument markup after it at all.
    let no_args = render(
        template_text,
        &serde_json::json!([
            { "role": "user", "content": USER },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "type": "function",
                    "function": { "name": NAME, "arguments": {} }
                }]
            },
        ]),
        None,
    )?;

    let region = diverging_region(&plain, &with_calls)?;
    let no_args_region = diverging_region(&plain, &no_args)?;

    // Every sentinel must survive. A template that drops one, or
    // escapes a value beyond recognition, fails here rather than
    // producing a format description that quietly mismatches what the
    // model really emits.
    let name_at = region.find(NAME)?;
    let arg1_at = region.find(ARG1)?;
    let val1_at = region.find(VAL1)?;
    let arg2_at = region.find(ARG2)?;
    let val2_at = region.find(VAL2)?;

    // JSON family first, and *before* any positional check: the whole
    // emitted call is one object, in which key order carries no meaning
    // and is not even stable. Real case this cost an hour of confusion:
    // `serde_json`'s `Map` is a `BTreeMap` unless the `preserve_order`
    // feature is on, so the probe's own `{name, arguments}` is rendered
    // *alphabetically* as `{arguments, name}` -- putting the function
    // name after the argument values and failing an emission-order
    // check that is perfectly correct for the delimited family below.
    //
    // Detected by actually parsing the payload with the sentinels still
    // in place, never by looking for a brace: a delimited-family value
    // that happened to contain `{` would pass that and fail this.
    if let Some(format) = derive_json_format(&region) {
        return Some(format);
    }

    // Emission order *does* matter for the delimited family -- its
    // spans are derived from the text between consecutive sentinels, so
    // out-of-order sentinels mean the derivation below is meaningless.
    if !(name_at < arg1_at && arg1_at < val1_at && val1_at < arg2_at && arg2_at < val2_at) {
        return None;
    }
    // Delimited first, then separated. Order matters and is not
    // arbitrary: `Delimited` is the stricter shape -- it requires a
    // non-empty opener before every argument -- so a format it accepts
    // is unambiguously that one. `Separated` is what remains when there
    // is no per-argument opener at all.
    let format = derive_delimited_format(
        &region,
        &no_args_region,
        name_at,
        arg1_at,
        val1_at,
        arg2_at,
        val2_at,
    )
    .or_else(|| {
        derive_separated_format(
            &region,
            &no_args_region,
            name_at,
            arg1_at,
            val1_at,
            arg2_at,
            val2_at,
        )
    })?;
    Some(format)
}

/// The span in which `with_calls` differs from `plain` -- i.e. the
/// tool-call rendering itself, stripped of whatever turn framing both
/// renders share.
fn diverging_region<'a>(plain: &str, with_calls: &'a str) -> Option<&'a str> {
    let prefix = common_prefix_len(plain.as_bytes(), with_calls.as_bytes());
    let suffix = common_suffix_len(
        &plain.as_bytes()[prefix..],
        &with_calls.as_bytes()[prefix..],
    );
    let end = with_calls.len().checked_sub(suffix)?;
    // Byte offsets from a bytewise comparison can land inside a
    // multi-byte character; `get` yields None rather than panicking,
    // which correctly reads as "this template isn't derivable."
    with_calls.get(prefix..end)
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

fn common_suffix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .rev()
        .zip(b.iter().rev())
        .take_while(|(x, y)| x == y)
        .count()
}

/// Recovers a [`ToolFormat::Json`] if `region`'s payload really is a
/// JSON object carrying the sentinel name and arguments.
///
/// Takes the outermost braces -- first `{` to last `}` -- rather than
/// searching outward from a sentinel's position, because key order is
/// not fixed (see [`derive_tool_call_format`]) so no sentinel reliably
/// sits inside or outside the nested `arguments` object. Whether those
/// braces really delimit the call is settled by parsing, not by
/// position.
fn derive_json_format(region: &str) -> Option<ToolFormat> {
    let open = region.find('{')?;
    let close = region.rfind('}')?;
    if close < open {
        return None;
    }

    let payload = region.get(open..=close)?;
    let parsed: serde_json::Value = serde_json::from_str(payload).ok()?;
    let object = parsed.as_object()?;
    // Locate the keys by their *values*, never by assuming a key name:
    // "name"/"arguments" is the common spelling but not a guarantee.
    let name_key = object
        .iter()
        .find(|(_, value)| value.as_str() == Some(NAME))
        .map(|(key, _)| key.clone())?;
    let arguments_key = object
        .iter()
        .find(|(_, value)| {
            value
                .as_object()
                .is_some_and(|args| args.values().any(|v| v.as_str() == Some(VAL1)))
        })
        .map(|(key, _)| key.clone())?;

    Some(ToolFormat::Json {
        call_open: region[..open].to_string(),
        call_close: region.get(close + 1..)?.to_string(),
        name_key,
        arguments_key,
    })
}

/// Recovers a [`ToolFormat::Delimited`] -- the one-block-per-argument
/// family, of which Qwen3-Coder is the worked example.
///
/// Four spans are observable across the two probe renders:
///
/// ```text
/// (1) name -> arg1   =  name_close  + arg_open      (two-argument call)
/// (2) val1 -> arg2   =  arg_close   + arg_open      (two-argument call)
/// (3) after val2     =  arg_close   + call_close    (two-argument call)
/// (4) after name     =  name_close  + call_close    (zero-argument call)
/// ```
///
/// Four unknowns, four equations. `name_close` is pinned as the shared
/// head of (1) and (4) -- the only two spans that begin with it -- and
/// `arg_close` as the shared head of (2) and (3). The remaining two
/// follow by subtraction, and each is then checked against the equation
/// it did *not* come from, so a template whose rendering isn't actually
/// this uniform repeating shape fails rather than yielding spans that
/// happen to factor.
///
/// The factorization is not unique -- a boundary can sit anywhere the
/// shared text allows, so `name_close` may absorb a leading character
/// of `arg_open`. That is harmless: parsing splits on the same spans it
/// derived, so any self-consistent factorization round-trips. What is
/// *not* harmless is an empty span, which matches everywhere; those are
/// rejected outright.
fn derive_delimited_format(
    region: &str,
    no_args_region: &str,
    name_at: usize,
    arg1_at: usize,
    val1_at: usize,
    arg2_at: usize,
    val2_at: usize,
) -> Option<ToolFormat> {
    let name_to_arg1 = region.get(name_at + NAME.len()..arg1_at)?;
    let val1_to_arg2 = region.get(val1_at + VAL1.len()..arg2_at)?;
    let after_val2 = region.get(val2_at + VAL2.len()..)?;

    let no_args_name_at = no_args_region.find(NAME)?;
    let after_name_no_args = no_args_region.get(no_args_name_at + NAME.len()..)?;

    let name_close = shared_prefix(name_to_arg1, after_name_no_args);
    let arg_close = shared_prefix(val1_to_arg2, after_val2);
    let arg_open = name_to_arg1.strip_prefix(name_close)?;
    let call_close = after_val2.strip_prefix(arg_close)?;

    // An empty span would match at every offset, silently producing
    // empty names or values -- see this function's own doc comment.
    if name_close.is_empty() || arg_open.is_empty() || arg_close.is_empty() {
        return None;
    }

    // Cross-checks against the equations each span did not come from.
    if val1_to_arg2 != format!("{arg_close}{arg_open}") {
        return None;
    }
    if after_name_no_args != format!("{name_close}{call_close}") {
        return None;
    }

    let arg_name_close = region.get(arg1_at + ARG1.len()..val1_at)?;
    if arg_name_close.is_empty() {
        return None;
    }
    // Whatever separates the *second* argument's name from its value
    // must match the first's, or this isn't a uniform repeating unit.
    if region.get(arg2_at + ARG2.len()..val2_at)? != arg_name_close {
        return None;
    }

    Some(ToolFormat::Delimited {
        call_open: region[..name_at].to_string(),
        name_close: name_close.to_string(),
        arg_open: arg_open.to_string(),
        arg_name_close: arg_name_close.to_string(),
        arg_close: arg_close.to_string(),
        call_close: call_close.to_string(),
    })
}

/// Recovers a [`ToolFormat::Separated`] -- Gemma-style, where a
/// separator sits *between* arguments rather than a closer after each.
///
/// The real shape, from Gemma 4's own template:
///
/// ```text
/// with args:  <|tool_call>call:NAME{ARG1:<|"|>VAL1<|"|>,ARG2:<|"|>VAL2<|"|>}<tool_call|>
/// no args:    <|tool_call>call:NAME{}<tool_call|>
/// ```
///
/// `Delimited` cannot express it. Reading `,` as an `arg_close` is wrong
/// for the last argument, which is followed by the call's close instead;
/// reading `,` as an `arg_open` is wrong for the first, which has
/// nothing before it. The asymmetry is the definition of the family.
///
/// `call_close` comes from the **no-argument** probe rather than from
/// the end of the with-arguments region. That region trails whatever the
/// template emits after an assistant tool-call message -- Gemma's adds a
/// `<|tool_response>` opener in anticipation of one -- and a `call_close`
/// carrying that would never match text a model actually generates.
fn derive_separated_format(
    region: &str,
    no_args_region: &str,
    name_at: usize,
    arg1_at: usize,
    val1_at: usize,
    arg2_at: usize,
    val2_at: usize,
) -> Option<ToolFormat> {
    let name_close = region.get(name_at + NAME.len()..arg1_at)?;
    let arg_name_close = region.get(arg1_at + ARG1.len()..val1_at)?;
    let val1_to_arg2 = region.get(val1_at + VAL1.len()..arg2_at)?;
    let after_val2 = region.get(val2_at + VAL2.len()..)?;

    // The call's own close, from the probe that has no arguments and so
    // no value wrappers to confuse it.
    let no_args_name_at = no_args_region.find(NAME)?;
    let after_name_no_args = no_args_region.get(no_args_name_at + NAME.len()..)?;
    let call_close = after_name_no_args.strip_prefix(name_close)?;

    // What closes a value is whatever separates the last one from the
    // call's close.
    let closing_at = after_val2.find(call_close)?;
    let arg_close = &after_val2[..closing_at];

    // ...and the separator is what remains between one value's close and
    // the next argument's name.
    let arg_sep = val1_to_arg2.strip_prefix(arg_close)?;

    // The second argument must be introduced exactly as the first was,
    // or this is not a uniform repeating unit and reading it as one
    // would invent arguments. Checked here, against the *full* span,
    // before the wrapper is split back out of it below.
    if region.get(arg2_at + ARG2.len()..val2_at)? != arg_name_close {
        return None;
    }

    // A value's *opening* wrapper, split back out of `arg_name_close`.
    //
    // The probes only ever pass strings, and this family wraps strings
    // -- so what looked like "the separator between a name and its
    // value" is really that separator plus the wrapper. Numbers are
    // emitted bare: Gemma writes `line:38`, not `line:<|"|>38<|"|>`.
    //
    // Measured. With the wrapper welded on, `line:38` matched no
    // argument at all, so every `comment` call arrived without the line
    // it needed and was refused -- four turns in a row, on a review that
    // had already found the bug it was trying to report.
    let (arg_name_close, value_open) = match arg_name_close.strip_suffix(arg_close) {
        Some(bare) if !arg_close.is_empty() && !bare.is_empty() => (bare, arg_close),
        _ => (arg_name_close, ""),
    };

    // An empty span matches at every offset and would silently produce
    // empty names, values, or an unbounded call. `arg_close` may
    // legitimately be empty -- a family that does not wrap its values --
    // which is why it is not on this list.
    if name_close.is_empty()
        || arg_name_close.is_empty()
        || arg_sep.is_empty()
        || call_close.is_empty()
    {
        return None;
    }

    // A separator that could be mistaken for a value's close, or for the
    // call's, makes every boundary ambiguous.
    if arg_sep.contains(arg_name_close) || call_close.starts_with(arg_sep) {
        return None;
    }

    Some(ToolFormat::Separated {
        call_open: region[..name_at].to_string(),
        name_close: name_close.to_string(),
        arg_name_close: arg_name_close.to_string(),
        value_open: value_open.to_string(),
        arg_close: arg_close.to_string(),
        arg_sep: arg_sep.to_string(),
        call_close: call_close.to_string(),
    })
}

fn shared_prefix<'a>(a: &'a str, b: &str) -> &'a str {
    let shared = common_prefix_len(a.as_bytes(), b.as_bytes());
    a.get(..shared).unwrap_or("")
}

/// Extracts every tool call in `text` according to `format`.
///
/// Text outside a call is ignored rather than rejected: Qwen3-Coder's
/// own rendered instructions explicitly permit reasoning prose *before*
/// a call, so a strict parser would throw away valid turns. A malformed
/// or truncated call is skipped, not fatal -- a generation that ran out
/// of tokens mid-call should cost that call, not the whole turn.
#[must_use]
pub fn parse_tool_calls(text: &str, format: &ToolFormat) -> Vec<ToolCall> {
    match format {
        ToolFormat::Json {
            call_open,
            call_close,
            name_key,
            arguments_key,
        } => parse_json_calls(text, call_open, call_close, name_key, arguments_key),
        ToolFormat::Delimited {
            call_open,
            name_close,
            arg_open,
            arg_name_close,
            arg_close,
            call_close,
        } => parse_delimited_calls(
            text,
            call_open,
            name_close,
            arg_open,
            arg_name_close,
            arg_close,
            call_close,
        ),
        ToolFormat::Separated {
            call_open,
            name_close,
            arg_name_close,
            value_open,
            arg_close,
            arg_sep,
            call_close,
        } => parse_separated_calls(
            text,
            call_open,
            name_close,
            arg_name_close,
            value_open,
            arg_close,
            arg_sep,
            call_close,
        ),
    }
}

/// Walks every call in `text` and hands each one's body to `read_body`.
///
/// # Why the three families share this
///
/// They differ only in how a call's *body* becomes a name and
/// arguments. Finding the bodies is identical -- scan for the opener,
/// take everything up to the closer, continue after it -- and it was
/// written out three times, once per family, along with the same two
/// off-by-one slices and the same "no closer means severed" comment.
/// A fourth family should have to describe only what is different
/// about it.
///
/// The missing-closer case is the one worth keeping in one place: a
/// call generation ran out of tokens part-way through still has its
/// arguments read, because a severed call should cost that call rather
/// than the whole turn. Getting that wrong in one family and right in
/// the other two is precisely the kind of drift copying invites.
///
/// Every family now finds its closer through [`find_close`] rather than
/// a plain `find`. That was already the general rule -- see
/// `close_candidates`, whose own documentation notes that a closer with
/// no internal tag boundary "yields only itself, which is why this costs
/// the other families nothing". Two of the three simply were not using
/// it, so a template whose closer carried trailing framing would have
/// been handled in one family and not the others.
fn scan_calls(
    text: &str,
    call_open: &str,
    call_close: &str,
    mut read_body: impl FnMut(&str) -> Option<ToolCall>,
) -> Vec<ToolCall> {
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some(start) = rest.find(call_open) {
        let after_open = &rest[start + call_open.len()..];
        let (body, remainder) = match find_close(after_open, call_close) {
            Some((end, len)) => (&after_open[..end], &after_open[end + len..]),
            None => (after_open, ""),
        };
        if let Some(call) = read_body(body) {
            calls.push(call);
        }
        if remainder.is_empty() {
            break;
        }
        rest = remainder;
    }
    calls
}

/// A name and its arguments, once a family has read them out of a body.
///
/// `None` for a body with no name in it: an empty name is not a call,
/// and every family agreed on that separately before this existed.
fn named(name: String, arguments: serde_json::Map<String, serde_json::Value>) -> Option<ToolCall> {
    if name.is_empty() {
        return None;
    }
    Some(ToolCall {
        name,
        arguments: serde_json::Value::Object(arguments),
    })
}

fn parse_json_calls(
    text: &str,
    call_open: &str,
    call_close: &str,
    name_key: &str,
    arguments_key: &str,
) -> Vec<ToolCall> {
    scan_calls(text, call_open, call_close, |body| {
        let value = serde_json::from_str::<serde_json::Value>(body.trim()).ok()?;
        let name = value.get(name_key)?.as_str()?.to_string();
        let arguments = value
            .get(arguments_key)
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        (!name.is_empty()).then_some(ToolCall { name, arguments })
    })
}

/// Where a call ends, and how much of `call_close` closed it.
///
/// # Why the whole closer is not always there
///
/// `call_close` is derived by rendering a call through the template and
/// reading what comes after the arguments. For most families that is
/// exactly the terminator the model emits. Gemma's template appends one
/// more token the model never writes: it renders
///
/// ```text
/// <|tool_call>call:NAME{}<tool_call|><|tool_response>
/// ```
///
/// unconditionally -- measured with zero results following as well as
/// two, so it is framing for whatever comes next, not part of the call.
/// The model emits `}<tool_call|>` and stops at end-of-turn, so a
/// literal search for the derived closer finds nothing, every call reads
/// as severed, and the final argument swallows the terminator. That is
/// how `line:38` came back as `38}<tool_call|>`.
///
/// The template cannot tell the two apart -- both probes render the same
/// bytes. So the closer is matched longest-first and allowed to fall
/// back to a shorter prefix that still ends a tag. Families whose closer
/// the model does emit in full match on the first try and are unchanged.
fn find_close(text: &str, call_close: &str) -> Option<(usize, usize)> {
    for candidate in close_candidates(call_close) {
        if let Some(at) = text.find(candidate) {
            return Some((at, candidate.len()));
        }
    }
    None
}

/// `call_close`, then its shorter tag-ending prefixes, longest first.
///
/// A cut is only allowed after a `>`, so a prefix is always a whole
/// number of tags: `}<tool_call|><|tool_response>` yields itself and
/// `}<tool_call|>`, never `}<tool_c`. A closer with no internal tag
/// boundary -- `</tool_call>`, `\n` -- yields only itself, which is why
/// this costs the other families nothing.
fn close_candidates(call_close: &str) -> Vec<&str> {
    let mut out = Vec::new();
    if !call_close.is_empty() {
        out.push(call_close);
    }
    for (at, _) in call_close.char_indices().rev() {
        if call_close[at..].starts_with('>') {
            let prefix = &call_close[..at + 1];
            if prefix != call_close && !prefix.is_empty() {
                out.push(prefix);
            }
        }
    }
    out
}

/// The `name: value` pairs in a separated call's argument block.
///
/// # Why this cannot be a `split`
///
/// Splitting the block on `arg_sep` first is the obvious reading, and it
/// is wrong for any value that contains the separator. Gemma's separator
/// is `,`, and the argument this harness cares most about is a review
/// comment -- English prose, full of commas. Measured: a comment reading
/// `... field of the timestamp. While the test X passes, ...` came back
/// truncated at the first comma, with the rest of the sentence recorded
/// as a second argument whose name was the prose before its first `:`.
/// Nothing errored. The finding was simply mutilated on its way in.
///
/// So the block is walked instead. A value wrapped in `value_open` runs
/// to its closer no matter what it contains; only a bare value -- a
/// number, in practice -- ends at the next separator.
fn split_args<'a>(
    args: &'a str,
    arg_name_close: &str,
    value_open: &str,
    arg_close: &str,
    arg_sep: &str,
) -> Vec<(String, &'a str)> {
    let mut out: Vec<(String, &'a str)> = Vec::new();
    let mut rest = args;
    while !rest.is_empty() {
        // A piece with no name separator is malformed, and ends the
        // walk rather than being fatal -- the same treatment the
        // delimited parser gives a severed argument.
        let Some(at) = rest.find(arg_name_close) else {
            break;
        };
        let key = rest[..at].trim().to_string();
        let after_name = &rest[at + arg_name_close.len()..];

        let (value, remainder) = match after_name.strip_prefix(value_open) {
            // Wrapped: runs to its closer verbatim. This family escapes
            // nothing, so trimming would corrupt file content passed as
            // an argument. A missing closer means generation was severed
            // mid-value; keep what there is.
            Some(inner) if !value_open.is_empty() => match inner.find(arg_close) {
                Some(end) => (&inner[..end], &inner[end + arg_close.len()..]),
                None => (inner, ""),
            },
            // Bare: ends at the next separator, or at the end.
            _ => match find_sep(after_name, arg_sep) {
                Some(end) => (after_name[..end].trim(), &after_name[end..]),
                None => (after_name.trim(), ""),
            },
        };
        if !key.is_empty() {
            out.push((key, value));
        }

        rest = match find_sep(remainder, arg_sep) {
            Some(at) => &remainder[at + arg_sep.len()..],
            None => "",
        };
    }
    out
}

/// The next `arg_sep` in `text`, or `None` if there is none.
fn find_sep(text: &str, arg_sep: &str) -> Option<usize> {
    if arg_sep.is_empty() {
        return None;
    }
    text.find(arg_sep)
}

/// Extracts calls in the separated family.
///
/// The value of the *last* argument runs to the call's close rather than
/// to a separator, which is the whole difference from
/// [`parse_delimited_calls`] and the reason this is its own function
/// rather than a flag on that one.
#[expect(
    clippy::too_many_arguments,
    reason = "each is one independently-derived span of the format being parsed; grouping them would just be ToolFormat::Separated destructured again"
)]
fn parse_separated_calls(
    text: &str,
    call_open: &str,
    name_close: &str,
    arg_name_close: &str,
    value_open: &str,
    arg_close: &str,
    arg_sep: &str,
    call_close: &str,
) -> Vec<ToolCall> {
    scan_calls(text, call_open, call_close, |body| {
        let name_end = body.find(name_close)?;
        let name = body[..name_end].trim().to_string();
        let mut arguments = serde_json::Map::new();
        let args = &body[name_end + name_close.len()..];
        for (key, value) in split_args(args, arg_name_close, value_open, arg_close, arg_sep) {
            arguments.insert(key, serde_json::Value::String(value.to_string()));
        }
        named(name, arguments)
    })
}

fn parse_delimited_calls(
    text: &str,
    call_open: &str,
    name_close: &str,
    arg_open: &str,
    arg_name_close: &str,
    arg_close: &str,
    call_close: &str,
) -> Vec<ToolCall> {
    scan_calls(text, call_open, call_close, |body| {
        let name_end = body.find(name_close)?;
        let name = body[..name_end].trim().to_string();
        let mut arguments = serde_json::Map::new();
        let mut args_rest = &body[name_end + name_close.len()..];
        while let Some(arg_start) = args_rest.find(arg_open) {
            let after_arg_open = &args_rest[arg_start + arg_open.len()..];
            let Some(arg_name_end) = after_arg_open.find(arg_name_close) else {
                break;
            };
            let arg_name = after_arg_open[..arg_name_end].trim().to_string();
            let after_name = &after_arg_open[arg_name_end + arg_name_close.len()..];
            // A value is whatever sits before the closer, verbatim
            // -- this family quotes and escapes nothing, so trimming
            // beyond the delimiters' own newlines would corrupt
            // file content passed as an argument.
            let (value, next) = match after_name.find(arg_close) {
                Some(value_end) => (
                    &after_name[..value_end],
                    &after_name[value_end + arg_close.len()..],
                ),
                None => (after_name, ""),
            };
            arguments.insert(arg_name, serde_json::Value::String(value.to_string()));
            if next.is_empty() {
                break;
            }
            args_rest = next;
        }
        named(name, arguments)
    })
}

/// When a turn is over because the model has stopped making tool calls
/// and started writing the harness's half of the conversation.
///
/// # The failure this exists for
///
/// Nothing here ever ended generation except the model's own
/// end-of-generation token or the token cap. That is fine for a model
/// that reliably stops after a tool call, which Qwen3-Coder does, and it
/// is why this gap went unnoticed for the entire life of the crate.
///
/// Gemma 4 12B QAT does not. Measured on `pcapgen` task 6: it emitted
/// one correct `read`, then kept generating -- and wrote its own tool
/// result, verbatim, `Read /home/rrojo/.agent99/workspace/pcapgen/
/// src/timeline.rs (lines 111-120):` followed by invented file content.
/// It role-played the whole agent loop inside a single 4096-token
/// generation: 166 tool calls in one turn, alternating two `cargo test`
/// invocations about eighty times each, until the conversation
/// overflowed its context window and the run died.
///
/// The harness then *ran* those calls. They parsed perfectly -- they
/// were syntactically ideal -- so a hundred and sixty commands the model
/// invented while imagining a debugging session were executed for real.
/// They happened to be `cargo test`. A hallucinated `write` would have
/// landed on disk, which is the same family as the truncated call that
/// wrote a source fragment and cost a run.
///
/// # Why not simply stop at the first closer
///
/// Because a turn holding two genuine calls is legitimate and used --
/// `parse_tool_calls` returns a `Vec` and both loops iterate it. Cutting
/// at the first closer would silently discard the second call of every
/// multi-call turn, trading this bug for a quieter one.
///
/// So the rule is: after a call closes, keep going while what follows
/// could still *be* another call -- whitespace, or the opener, or a
/// prefix of the opener that has not finished arriving. The moment the
/// model writes something that is none of those, it has moved on from
/// calling tools, and everything from that closer onward is discarded.
#[derive(Debug, Clone)]
pub struct TurnEnd {
    call_open: String,
    closers: Vec<String>,
}

impl TurnEnd {
    /// Reads the openers and closers out of a derived format.
    #[must_use]
    pub fn of(format: &ToolFormat) -> Self {
        let (call_open, call_close) = match format {
            ToolFormat::Json {
                call_open,
                call_close,
                ..
            }
            | ToolFormat::Delimited {
                call_open,
                call_close,
                ..
            }
            | ToolFormat::Separated {
                call_open,
                call_close,
                ..
            } => (call_open, call_close),
        };
        Self {
            call_open: call_open.clone(),
            // Same candidate list the parser and `ends_mid_call` use, so
            // a family whose model emits a shorter closer than the
            // template renders stops where it actually stops.
            closers: close_candidates(call_close)
                .into_iter()
                .map(str::to_string)
                .collect(),
        }
    }

    /// Where to cut `text` and stop, or `None` to keep generating.
    ///
    /// `None` while the answer is still open: no call has closed yet, or
    /// one has and what follows might still become another.
    #[must_use]
    pub fn reached(&self, text: &str) -> Option<usize> {
        if self.call_open.is_empty() {
            return None;
        }
        // The *last* closer, not the first: earlier calls in a
        // multi-call turn are already settled and only the newest one
        // decides whether the turn continues.
        let end = self
            .closers
            .iter()
            .filter_map(|closer| text.rfind(closer.as_str()).map(|at| at + closer.len()))
            .max()?;
        let tail = &text[end..];
        if tail.trim_start().is_empty() {
            // Nothing but whitespace yet -- undecided.
            return None;
        }
        let after = tail.trim_start();
        // Another call, or the beginning of one still arriving.
        if after.starts_with(self.call_open.as_str()) || self.call_open.starts_with(after) {
            return None;
        }
        Some(end)
    }
}

/// Whether `text` stops part-way into what would have been a tool call,
/// rather than ending because the model had nothing more to say.
///
/// These are opposite states that look identical to a caller counting
/// calls: both yield none. The measured failure, on the last turn of a
/// real run:
///
/// ```text
/// turn 21 (10 tokens)
///   Let me create the UI module properly:
///   <tool_call>
/// ```
///
/// Generation was severed after `<tool_call>` and before `<function=`,
/// so [`parse_tool_calls`] found no complete `call_open` and returned
/// nothing -- and the loop reading that concluded the model was
/// finished and reported success on a run cut off mid-sentence.
///
/// [`parse_tool_calls`] already recovers a call truncated *after* its
/// name, deliberately. This covers the earlier cut, where there is not
/// yet enough to recover anything, and the only honest reading is "ask
/// for that call again."
#[must_use]
pub fn ends_mid_call(text: &str, format: &ToolFormat) -> bool {
    let (call_open, call_close) = match format {
        ToolFormat::Json {
            call_open,
            call_close,
            ..
        }
        | ToolFormat::Delimited {
            call_open,
            call_close,
            ..
        }
        | ToolFormat::Separated {
            call_open,
            call_close,
            ..
        } => (call_open, call_close),
    };
    if call_open.is_empty() {
        return false;
    }
    let trimmed = text.trim_end();

    // A call that opened and never closed.
    //
    // This is the general case and it is the one that matters. The
    // checks below only ever caught a severed *opener* -- text ending
    // with the marker or a prefix of it -- which misses the shape that
    // actually costs something: a call that announced itself, named its
    // function, began an argument, and was cut off inside it.
    //
    // Measured, on a real run: a model was asked to write a 136-line
    // test file with a 1500-token budget. Generation stopped at exactly
    // the cap, in the middle of `assert_eq!(&buffer`. The parser
    // recovered that as a complete `write` call with a truncated
    // `content`, the harness wrote the fragment to disk, and four
    // verification attempts then failed on "this file contains an
    // unclosed delimiter" before the run was abandoned. Nothing anywhere
    // reported that the reply had been cut off.
    //
    // A caller cannot defend against that without being told, and this
    // is the only place that knows what a complete call looks like.
    // Matched through `find_close`, exactly as the parser matches it,
    // and that is the whole point rather than a tidiness note.
    //
    // A derived `call_close` can carry framing the model never emits --
    // Gemma's derives as `}<tool_call|><|tool_response>` because the
    // template appends the response opener unconditionally, while the
    // model writes `}<tool_call|>` and stops. `close_candidates` exists
    // for precisely that and the parser has used it since Gemma landed.
    // This function did not: a plain `contains` of the full closer never
    // matched, so **every** Gemma reply was reported as cut off inside a
    // call, including complete twenty-token ones.
    //
    // Measured on the task 6 rework. The model emitted one correct
    // `read src/timeline.rs`, the parser recovered it, this said the
    // turn was severed, `agent99` discarded every call in it and asked
    // again with nothing new to say -- four turns, zero tool calls
    // executed, and the model finally degenerated into two hundred
    // copies of that same call until it hit the token cap. The collapse
    // read as the failure; it was the symptom. The reviewer loop looked
    // healthy on the identical model only because it ignored this flag
    // altogether.
    //
    // One derived span, two readers, and they disagreed. Sharing the
    // matcher is what stops the next family from finding the same seam.
    if let Some(opened) = trimmed.rfind(call_open.as_str()) {
        let after = &trimmed[opened + call_open.len()..];
        if call_close.is_empty() || find_close(after, call_close).is_none() {
            return true;
        }
    }

    // ...or a non-empty prefix of the opener at the very end, which the
    // rule above cannot see because the marker never completed. Walked
    // by character boundary, never by byte offset: a marker containing a
    // multi-byte character would otherwise panic on the slice.
    call_open
        .char_indices()
        .skip(1)
        .any(|(at, _)| trimmed.ends_with(&call_open[..at]))
}

/// Renders the text that opens a conversation carrying `system` and
/// `tools` -- everything the template emits before the first user
/// message's own content.
///
/// Unlike `llama::derive_conversation_template`'s spans, this is not a
/// reusable fixed string: it depends on the real system text and the
/// real tool list, both of which are known at conversation-open time.
/// So it renders the actual template with the actual values and slices
/// at a sentinel, rather than deriving a shape and substituting into it
/// -- exact by construction, and it costs one render per conversation
/// rather than one per turn.
#[must_use]
pub fn render_opening(
    template_text: &str,
    system: Option<&str>,
    tools: Option<&[ToolSpec]>,
    render: RenderFn<'_>,
) -> Option<String> {
    let mut messages = Vec::new();
    if let Some(system) = system {
        messages.push(serde_json::json!({ "role": "system", "content": system }));
    }
    messages.push(serde_json::json!({ "role": "user", "content": USER }));

    let tools = tools.map(|tools| serde_json::json!(tools));
    let rendered = render(
        template_text,
        &serde_json::Value::Array(messages),
        tools.as_ref(),
    )?;
    let user_at = rendered.find(USER)?;
    Some(rendered[..user_at].to_string())
}

/// The literal spans a template puts around a *sequence* of tool
/// results, so several can be appended in the shape that template
/// actually renders them.
///
/// Three spans, not two, and the third is the point. `send_tool_results`
/// used to join every result with `\n` into one block, which for a turn
/// making three calls produced a single `<tool_response>` containing
/// three newline-separated answers. Qwen3-Coder's own template renders
/// each `tool`-role message as its *own* `<tool_response>` element --
/// so the model was being handed a shape slightly off from the one it
/// was trained to read, with no boundary marking where one result ended
/// and the next began.
///
/// For a single result this collapses to exactly the previous
/// behaviour (`open + result + close`), so the common case is
/// unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolResultSpans {
    /// Before the first result -- includes whatever opens the turn.
    pub open: String,
    /// Between consecutive results. Empty for a template that renders
    /// them with nothing in between.
    pub separator: String,
    /// After the last result, running to where the next assistant
    /// turn's own content begins -- so generation resumes straight
    /// into it. See below for why that boundary and not the end of the
    /// tool turn.
    pub close: String,
}

impl ToolResultSpans {
    /// Wraps `results` in this template's own shape.
    #[must_use]
    pub fn render(&self, results: &[String]) -> String {
        format!(
            "{}{}{}",
            self.open,
            results.join(&self.separator),
            self.close
        )
    }
}

/// Recovers [`ToolResultSpans`] by probing with *two* results -- the
/// only way to see what a template puts between them.
///
/// The probe carries a trailing assistant turn as well, purely to mark
/// where generation resumes. Without it `close` would stop at the end
/// of the tool turn, and a caller would have to reach for
/// `ConversationTemplate::generation_open` to reopen the assistant --
/// which is wrong here: that span begins by closing a *user* turn, and
/// this one has already closed the tool turn itself. Real templates
/// differ on that boundary (Qwen3-Coder wraps tool results in a user
/// turn and emits its own `<|im_end|>`), so it is derived rather than
/// assembled.
#[must_use]
pub fn derive_tool_result_spans(
    template_text: &str,
    render: RenderFn<'_>,
) -> Option<ToolResultSpans> {
    derive_tool_result_spans_by_role(template_text, render)
        .or_else(|| derive_tool_result_spans_native(template_text, render))
}

/// The `{"role": "tool"}` convention -- what the OpenAI-shaped families
/// use, and what this crate assumed was the only one.
fn derive_tool_result_spans_by_role(
    template_text: &str,
    render: RenderFn<'_>,
) -> Option<ToolResultSpans> {
    let rendered = render(
        template_text,
        &serde_json::json!([
            { "role": "user", "content": USER },
            { "role": "assistant", "content": PLAIN },
            { "role": "tool", "content": RESULT },
            { "role": "tool", "content": RESULT2 },
            { "role": "assistant", "content": REPLY },
        ]),
        None,
    )?;
    let assistant_at = rendered.find(PLAIN)?;
    let first_at = rendered.find(RESULT)?;
    let second_at = rendered.find(RESULT2)?;
    let reply_at = rendered.find(REPLY)?;
    if !(assistant_at < first_at && first_at < second_at && second_at < reply_at) {
        return None;
    }
    Some(ToolResultSpans {
        open: rendered
            .get(assistant_at + PLAIN.len()..first_at)?
            .to_string(),
        separator: rendered
            .get(first_at + RESULT.len()..second_at)?
            .to_string(),
        close: rendered
            .get(second_at + RESULT2.len()..reply_at)?
            .to_string(),
    })
}

/// The same spans, for a template that carries results on the assistant
/// message instead of as `tool` messages.
///
/// # Why a second convention exists at all
///
/// `derive_tool_result_spans` probes with `{"role": "tool"}`, which is
/// what the OpenAI-shaped families use. Gemma's template branches on
/// `assistant` and `user` and **nothing else**: a `tool` message is not
/// rejected, it is silently dropped, so the probe rendered a
/// conversation with no results in it and the spans came back as the
/// distance between two adjacent assistant turns.
///
/// Its own comment calls this "Google/Gemma native": results ride on the
/// message as a `tool_responses` array. Same three spans out the other
/// end, so everything downstream is unchanged -- only the question being
/// asked of the template differs.
fn derive_tool_result_spans_native(
    template_text: &str,
    render: RenderFn<'_>,
) -> Option<ToolResultSpans> {
    let rendered = render(
        template_text,
        &serde_json::json!([
            { "role": "user", "content": USER },
            {
                // No content of its own: this family emits the responses
                // *before* the message's text, so a `PLAIN` here would
                // land after them and could not anchor the opening span.
                "role": "assistant",
                "content": "",
                // An empty name, deliberately. This family embeds the
                // answering tool's name in each result block, and
                // `ToolResultSpans` has nowhere to put a per-result one
                // -- so a probe naming a tool would bake that name into
                // the separator and every real result would claim to
                // come from it. Worse, the name here is a private-use
                // sentinel, which would then appear as invisible garbage
                // in every prompt.
                //
                // The template's own `| default('unknown', true)` fills
                // the gap. The cost is real and worth stating: a Gemma
                // conversation cannot say which call a result answers,
                // so parallel calls in one turn are ambiguous to it.
                // Single-call turns, which is what this harness issues,
                // are unaffected.
                "tool_responses": [
                    { "name": "", "response": RESULT },
                    { "name": "", "response": RESULT2 },
                ],
            },
            { "role": "assistant", "content": REPLY },
        ]),
        None,
    )?;
    // Anchored on the *user* turn, for the same reason.
    let anchor_at = rendered.find(USER)?;
    let first_at = rendered.find(RESULT)?;
    let second_at = rendered.find(RESULT2)?;
    let reply_at = rendered.find(REPLY)?;
    if !(anchor_at < first_at && first_at < second_at && second_at < reply_at) {
        return None;
    }
    Some(ToolResultSpans {
        open: rendered.get(anchor_at + USER.len()..first_at)?.to_string(),
        separator: rendered
            .get(first_at + RESULT.len()..second_at)?
            .to_string(),
        close: rendered
            .get(second_at + RESULT2.len()..reply_at)?
            .to_string(),
    })
}

#[cfg(test)]
mod tests {
    /// Two calls where only the second was severed.
    #[test]
    fn a_severed_second_call_is_caught_even_though_the_first_closed() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let text = "<tool_call>\n<function=read>\n<parameter=path>\na.rs\n</parameter>\n</function>\n</tool_call>\n\
                    <tool_call>\n<function=write>\n<parameter=path>\nb.rs\n</parameter>\n<parameter=content>\nfn b() {";
        assert!(ends_mid_call(text, &format), "the last call never closed");
    }

    /// A write cut off mid-argument -- which is what a token cap does to
    /// a model writing a file longer than its budget.
    ///
    /// Measured on a real run: `agent99` asked for a 136-line test file
    /// with `MAX_NEW_TOKENS = 1500`, generation stopped at exactly the
    /// cap in the middle of an `assert_eq!(&buffer`, and the harness
    /// wrote that to disk. Four verify attempts then failed on `this
    /// file contains an unclosed delimiter` and the run was abandoned.
    ///
    /// `ends_mid_call` did not fire because it only looks for a severed
    /// call *opener*. Whether the severed call also *parses* decides
    /// whether a caller can defend itself at all.
    #[test]
    fn a_call_severed_mid_argument_is_reported_as_truncated() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let severed = "<tool_call>\n<function=write>\n<parameter=path>\ntests/a.rs\n</parameter>\n\
                       <parameter=content>\nfn a() {\n    assert_eq!(&buffer";

        let calls = parse_tool_calls(severed, &format);
        let flagged = ends_mid_call(severed, &format);
        assert!(
            calls.is_empty() || flagged,
            "a call cut off mid-argument must either fail to parse or be flagged as truncated -- \
             otherwise a half-written file reaches disk. parsed {} call(s), flagged {flagged}",
            calls.len()
        );
    }

    use super::*;

    /// A stub renderer standing in for minijinja: enough of the two real
    /// template families to drive every derivation path, with none of
    /// the Jinja machinery. Keeping the derivation logic renderer-
    /// agnostic (see [`RenderFn`]) is what makes this possible at all.
    fn qwen_xml_render(
        _template: &str,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
    ) -> Option<String> {
        let mut out = String::new();
        if let Some(tools) = tools {
            out.push_str("<|im_start|>system\n<tools>");
            for tool in tools.as_array()? {
                let function = tool.get("function").unwrap_or(tool);
                out.push_str(function.get("name")?.as_str()?);
            }
            out.push_str("</tools><|im_end|>\n");
        }
        let all = messages.as_array()?;
        for (index, message) in all.iter().enumerate() {
            let role = message.get("role")?.as_str()?;
            let content = message
                .get("content")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let role_at = |i: usize| {
                all.get(i)
                    .and_then(|m| m.get("role"))
                    .and_then(serde_json::Value::as_str)
            };
            match role {
                // Consecutive tool results share one user turn, exactly
                // as the real template does (`if loop.previtem.role !=
                // "tool"`). Modelling that is the whole reason this stub
                // exists -- a stub that opened a fresh turn per result
                // would let a wrong separator pass its own test.
                "tool" => {
                    if index == 0 || role_at(index - 1) != Some("tool") {
                        out.push_str("<|im_start|>user\n");
                    }
                    out.push_str("<tool_response>\n");
                    out.push_str(content);
                    out.push_str("\n</tool_response>\n");
                    if role_at(index + 1) != Some("tool") {
                        out.push_str("<|im_end|>\n");
                    }
                }
                "assistant" if message.get("tool_calls").is_some() => {
                    out.push_str("<|im_start|>assistant\n");
                    for call in message.get("tool_calls")?.as_array()? {
                        let function = call.get("function").unwrap_or(call);
                        out.push_str("<tool_call>\n<function=");
                        out.push_str(function.get("name")?.as_str()?);
                        out.push_str(">\n");
                        for (key, value) in function.get("arguments")?.as_object()? {
                            out.push_str("<parameter=");
                            out.push_str(key);
                            out.push_str(">\n");
                            out.push_str(value.as_str()?);
                            out.push_str("\n</parameter>\n");
                        }
                        out.push_str("</function>\n</tool_call>");
                    }
                    out.push_str("<|im_end|>\n");
                }
                _ => {
                    out.push_str("<|im_start|>");
                    out.push_str(role);
                    out.push('\n');
                    out.push_str(content);
                    out.push_str("<|im_end|>\n");
                }
            }
        }
        Some(out)
    }

    fn hermes_json_render(
        _template: &str,
        messages: &serde_json::Value,
        tools: Option<&serde_json::Value>,
    ) -> Option<String> {
        let mut out = String::new();
        if let Some(tools) = tools {
            out.push_str("<|im_start|>system\n<tools>");
            for tool in tools.as_array()? {
                let function = tool.get("function").unwrap_or(tool);
                out.push_str(function.get("name")?.as_str()?);
            }
            out.push_str("</tools><|im_end|>\n");
        }
        for message in messages.as_array()? {
            let role = message.get("role")?.as_str()?;
            let content = message
                .get("content")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            match role {
                "tool" => {
                    out.push_str("<|im_start|>tool\n");
                    out.push_str(content);
                    out.push_str("<|im_end|>\n");
                }
                "assistant" if message.get("tool_calls").is_some() => {
                    out.push_str("<|im_start|>assistant\n");
                    for call in message.get("tool_calls")?.as_array()? {
                        let function = call.get("function").unwrap_or(call);
                        out.push_str("<tool_call>\n");
                        out.push_str(
                            &serde_json::json!({
                                "name": function.get("name")?,
                                "arguments": function.get("arguments")?
                            })
                            .to_string(),
                        );
                        out.push_str("\n</tool_call>");
                    }
                    out.push_str("<|im_end|>\n");
                }
                _ => {
                    out.push_str("<|im_start|>");
                    out.push_str(role);
                    out.push('\n');
                    out.push_str(content);
                    out.push_str("<|im_end|>\n");
                }
            }
        }
        Some(out)
    }

    /// A template that accepts no tools and no system block -- the
    /// "correctly declines" path this module's doc comment describes.
    fn plain_render(
        _template: &str,
        messages: &serde_json::Value,
        _tools: Option<&serde_json::Value>,
    ) -> Option<String> {
        let mut out = String::new();
        for message in messages.as_array()? {
            let role = message.get("role")?.as_str()?;
            if role == "system" || role == "tool" {
                continue;
            }
            out.push_str(
                message
                    .get("content")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or(""),
            );
        }
        Some(out)
    }

    /// Asserts the family and the spans that are genuinely pinned, not
    /// one exact factorization: see `derive_delimited_format`'s own doc
    /// comment for why the split between `name_close` and `arg_open`
    /// (and between `arg_close` and `call_close`) is not unique. The
    /// round-trip tests below are what actually establish correctness;
    /// this one guards the invariants that make round-tripping possible
    /// at all.
    #[test]
    fn derives_the_nested_xml_family() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        match format {
            ToolFormat::Delimited {
                call_open,
                name_close,
                arg_open,
                arg_name_close,
                arg_close,
                call_close,
            } => {
                assert_eq!(call_open, "<tool_call>\n<function=");
                assert_eq!(arg_name_close, ">\n");
                // No span may be empty -- an empty one matches at every
                // offset and silently yields empty names or values.
                for (label, span) in [
                    ("name_close", &name_close),
                    ("arg_open", &arg_open),
                    ("arg_close", &arg_close),
                    ("call_close", &call_close),
                ] {
                    assert!(!span.is_empty(), "{label} must not be empty");
                }
                // The factorizations must recompose into the text the
                // template really emits between each pair of sentinels.
                assert_eq!(format!("{name_close}{arg_open}"), ">\n<parameter=");
                assert_eq!(
                    format!("{arg_close}{call_close}"),
                    "\n</parameter>\n</function>\n</tool_call>"
                );
            }
            other => panic!("expected Delimited, got {other:?}"),
        }
    }

    #[test]
    fn derives_the_json_family() {
        let format = derive_tool_call_format("", &hermes_json_render).expect("derivable");
        match format {
            ToolFormat::Json {
                call_open,
                call_close,
                name_key,
                arguments_key,
            } => {
                assert_eq!(call_open, "<tool_call>\n");
                assert_eq!(call_close, "\n</tool_call>");
                assert_eq!(name_key, "name");
                assert_eq!(arguments_key, "arguments");
            }
            other => panic!("expected Json, got {other:?}"),
        }
    }

    /// The round trip that matters: what the template renders for a call
    /// is what the parser must recover from a model emitting one.
    #[test]
    fn xml_format_round_trips_a_real_looking_call() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let emitted = "<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(emitted, &format);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(
            calls[0]
                .arguments
                .get("path")
                .and_then(serde_json::Value::as_str),
            Some("src/main.rs")
        );
    }

    #[test]
    fn json_format_round_trips_a_real_looking_call() {
        let format = derive_tool_call_format("", &hermes_json_render).expect("derivable");
        let emitted = "<tool_call>\n{\"name\":\"read\",\"arguments\":{\"path\":\"src/main.rs\"}}\n</tool_call>";
        let calls = parse_tool_calls(emitted, &format);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
        assert_eq!(
            calls[0]
                .arguments
                .get("path")
                .and_then(serde_json::Value::as_str),
            Some("src/main.rs")
        );
    }

    #[test]
    fn reasoning_prose_before_a_call_is_ignored_not_rejected() {
        // Qwen3-Coder's own rendered instructions explicitly allow this.
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let emitted = "I should look at the file first.\n<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(emitted, &format);
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].name, "read");
    }

    #[test]
    fn several_calls_in_one_turn_are_all_recovered() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let emitted = concat!(
            "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>",
            "<tool_call>\n<function=read>\n<parameter=path>\nb.py\n</parameter>\n</function>\n</tool_call>",
        );
        let calls = parse_tool_calls(emitted, &format);
        assert_eq!(calls.len(), 2);
        assert_eq!(
            calls[1]
                .arguments
                .get("path")
                .and_then(serde_json::Value::as_str),
            Some("b.py")
        );
    }

    /// A value containing the delimiters' own characters (file content
    /// with newlines and angle brackets) must survive verbatim -- this
    /// family escapes nothing, so the parser must not "clean up".
    #[test]
    fn a_multi_line_value_survives_verbatim() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let emitted = "<tool_call>\n<function=write>\n<parameter=content>\nfn main() {\n    println!(\"<hi>\");\n}\n</parameter>\n</function>\n</tool_call>";
        let calls = parse_tool_calls(emitted, &format);
        assert_eq!(
            calls[0]
                .arguments
                .get("content")
                .and_then(serde_json::Value::as_str),
            Some("fn main() {\n    println!(\"<hi>\");\n}")
        );
    }

    #[test]
    fn a_truncated_call_yields_what_was_generated_rather_than_nothing() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let emitted = "<tool_call>\n<function=read>\n<parameter=path>\nsrc/main.rs";
        let calls = parse_tool_calls(emitted, &format);
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0]
                .arguments
                .get("path")
                .and_then(serde_json::Value::as_str),
            Some("src/main.rs")
        );
    }

    /// The measured failure -- see `ends_mid_call`'s own doc comment.
    #[test]
    fn a_reply_severed_just_inside_a_call_marker_is_not_a_final_answer() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let severed = "Let me create the UI module properly:\n<tool_call>";
        assert!(
            parse_tool_calls(severed, &format).is_empty(),
            "nothing recoverable yet"
        );
        assert!(
            ends_mid_call(severed, &format),
            "but it is plainly not finished"
        );
    }

    #[test]
    fn a_reply_severed_part_way_through_the_marker_is_also_caught() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        for severed in [
            "Now I will write it.\n<tool",
            "Now I will write it.\n<tool_call>\n<func",
        ] {
            assert!(ends_mid_call(severed, &format), "{severed:?}");
        }
    }

    /// An ordinary final answer must not be mistaken for a severed one,
    /// or every completed turn would loop forever asking for a call
    /// that was never coming.
    #[test]
    fn an_ordinary_final_answer_does_not_read_as_truncated() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        for finished in [
            "The task is complete. All tests pass.",
            "I have created hello.txt with the content \"hello\".",
            "Done. <not a call>",
        ] {
            assert!(!ends_mid_call(finished, &format), "{finished:?}");
        }
    }

    /// A turn that made a real call and then stopped cleanly is
    /// finished, not severed.
    #[test]
    fn a_complete_call_does_not_read_as_truncated() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        let complete = "<tool_call>\n<function=read>\n<parameter=path>\na.py\n</parameter>\n</function>\n</tool_call>";
        assert!(!ends_mid_call(complete, &format), "{complete}");
        // Still parses, and still is not flagged: the unterminated-call
        // rule must not sweep up calls that really did close.
        assert_eq!(parse_tool_calls(complete, &format).len(), 1);
        assert!(
            !ends_mid_call("I have finished the task.", &format),
            "prose is not a severed call"
        );
    }

    #[test]
    fn truncation_detection_works_for_the_json_family_too() {
        let format = derive_tool_call_format("", &hermes_json_render).expect("derivable");
        assert!(ends_mid_call("I will read it.\n<tool_call>", &format));
        assert!(!ends_mid_call("I am finished.", &format));
    }

    #[test]
    fn text_with_no_call_at_all_yields_no_calls() {
        let format = derive_tool_call_format("", &qwen_xml_render).expect("derivable");
        assert!(parse_tool_calls("Just an ordinary answer.", &format).is_empty());
    }

    #[test]
    fn capabilities_are_detected_per_field() {
        let full = derive_capabilities("", &qwen_xml_render);
        assert_eq!(
            full,
            ChatCapabilities {
                system: true,
                tools: true,
                tool_results: true
            }
        );

        let none = derive_capabilities("", &plain_render);
        assert_eq!(
            none,
            ChatCapabilities {
                system: false,
                tools: false,
                tool_results: false
            }
        );
    }

    /// A template that ignores the `tools` variable renders identically
    /// with and without it -- which must not read as support.
    #[test]
    fn a_template_that_ignores_tools_does_not_claim_to_support_them() {
        fn ignores_tools(
            _t: &str,
            messages: &serde_json::Value,
            _tools: Option<&serde_json::Value>,
        ) -> Option<String> {
            let mut out = String::new();
            for message in messages.as_array()? {
                out.push_str(
                    message
                        .get("content")
                        .and_then(serde_json::Value::as_str)
                        .unwrap_or(""),
                );
            }
            Some(out)
        }
        assert!(!derive_capabilities("", &ignores_tools).tools);
    }

    #[test]
    fn an_underivable_template_returns_none_rather_than_guessing() {
        assert!(derive_tool_call_format("", &plain_render).is_none());
    }

    #[test]
    fn render_opening_includes_the_system_text_and_tool_names() {
        let tools = vec![ToolSpec::new(
            "read",
            "Read a file",
            serde_json::json!({
                "type": "object",
                "properties": { "path": { "type": "string", "description": "the path" } },
                "required": ["path"]
            }),
        )];
        let opening = render_opening(
            "",
            Some("You are a coding agent."),
            Some(&tools),
            &qwen_xml_render,
        )
        .expect("renders");
        assert!(opening.contains("You are a coding agent."), "{opening}");
        assert!(opening.contains("read"), "{opening}");
        // Everything up to, but not including, the first user content.
        assert!(opening.ends_with("<|im_start|>user\n"), "{opening}");
    }

    /// Everything above drives hand-written stubs, which prove the
    /// derivation logic but not that it survives a real template's
    /// macros, filters and whitespace control.
    ///
    /// This is the format-agnostic version, and the one that actually
    /// establishes the module's claim: for each real template, ask it
    /// to render a call it invented the format for, then parse that
    /// text back with the format derived from the *same* template and
    /// require the name and arguments to survive. Nothing here names a
    /// delimiter, so it holds equally for a model this crate has never
    /// seen -- which is the whole point of deriving rather than
    /// configuring.
    ///
    /// Skipped, not failed, when a fixture is absent: these are real
    /// templates extracted from GGUFs, and requiring every checkout to
    /// carry one would tie this crate's tests to a multi-gigabyte model
    /// file being present.
    /// Prose, with the separator inside it.
    ///
    /// The turn that ran a hundred and sixty invented commands.
    ///
    /// Text taken from the task 6 log: one real `read`, then the model
    /// writing its own tool result. Everything from the closer onward is
    /// fabrication and must not reach the harness.
    #[cfg(feature = "template")]
    #[test]
    fn a_model_that_writes_its_own_tool_result_ends_the_turn() {
        let Ok(template) = std::fs::read_to_string("tests/fixtures/gemma4-12b.jinja") else {
            eprintln!("skipping absent fixture");
            return;
        };
        let render = crate::chat_template::probe_renderer();
        let format = derive_tool_call_format(&template, &render).expect("derivable");
        let end = TurnEnd::of(&format);

        let call = "<|tool_call>call:read{path:<|\"|>src/timeline.rs<|\"|>}<tool_call|>";

        // Mid-call: nothing has closed, so nothing is decided.
        assert_eq!(
            end.reached("<|tool_call>call:read{path:<|\"|>src/tim"),
            None
        );
        // Closed, nothing after it yet: still undecided, the model may
        // be about to open another call.
        assert_eq!(end.reached(call), None);
        assert_eq!(end.reached(&format!("{call}\n")), None);

        // A second genuine call must survive -- this is the case that
        // makes "stop at the first closer" wrong.
        let two = format!("{call}\n<|tool_call>call:verify{{}}<tool_call|>");
        assert_eq!(
            end.reached(&two),
            None,
            "a turn may hold more than one call"
        );
        // ...including while the second opener is still arriving.
        assert_eq!(
            end.reached(&format!("{call}\n<|tool")),
            None,
            "an opener half-emitted is not prose"
        );

        // The real failure: the model narrating a result it invented.
        let invented = format!(
            "{call}\n<|channel>thought\n<channel|>Read /home/rrojo/.agent99/workspace/pcapgen/src/timeline.rs \
             (lines 111-120):\n        }})\n            .collect();"
        );
        let at = end
            .reached(&invented)
            .expect("the model moved on from calling tools");
        assert_eq!(
            &invented[..at],
            call,
            "the turn keeps the real call and drops the fabrication"
        );
    }

    /// A complete Gemma call is not a severed one.
    ///
    /// The bug this pins cost a whole run. Gemma's derived `call_close`
    /// is `}<tool_call|><|tool_response>` -- the template appends the
    /// response opener unconditionally -- and the model emits
    /// `}<tool_call|>` and stops. `ends_mid_call` compared against the
    /// full derived closer, never matched, and reported every reply as
    /// cut off mid-call. `agent99` then discarded each turn's calls and
    /// asked again, so the model saw no result, repeated itself, and
    /// eventually collapsed. Zero tool calls ran in four turns.
    ///
    /// The parser had been right about this since Gemma landed; only
    /// this function was reading the closer literally.
    #[cfg(feature = "template")]
    #[test]
    fn a_complete_gemma_call_is_not_reported_as_cut_off() {
        let Ok(template) = std::fs::read_to_string("tests/fixtures/gemma4-12b.jinja") else {
            eprintln!("skipping absent fixture");
            return;
        };
        let render = crate::chat_template::probe_renderer();
        let format = derive_tool_call_format(&template, &render).expect("derivable");

        // Verbatim from the task 6 rework log, thought channel included.
        let complete = "<|channel>thought\n<channel|><|tool_call>call:read{path:<|\"|>src/timeline.rs<|\"|>}<tool_call|>";
        assert!(
            !parse_tool_calls(complete, &format).is_empty(),
            "the parser reads this fine, which is what made the disagreement invisible"
        );
        assert!(
            !ends_mid_call(complete, &format),
            "a call the parser recovers in full is not a call generation stopped inside of"
        );

        // ...and a genuinely severed one still reports as severed.
        let severed = "<|channel>thought\n<channel|><|tool_call>call:read{path:<|\"|>src/tim";
        assert!(
            ends_mid_call(severed, &format),
            "this one really was cut off mid-argument"
        );
    }

    /// The comment a review records is a sentence, and a sentence has
    /// commas in it. Gemma separates arguments with `,`, so a naive
    /// split truncated the finding at the first one and recorded the
    /// rest of it as a second argument. This is that call, verbatim from
    /// the run that caught it.
    #[cfg(feature = "template")]
    #[test]
    fn a_separator_inside_a_wrapped_value_does_not_end_it() {
        let Ok(template) = std::fs::read_to_string("tests/fixtures/gemma4-12b.jinja") else {
            eprintln!("skipping absent fixture");
            return;
        };
        let render = crate::chat_template::probe_renderer();
        let format = derive_tool_call_format(&template, &render).expect("derivable");

        let prose = "The sort uses only `seconds`. While the test passes, \
                     the fit criterion requires microseconds, too.";
        let text = format!(
            "<|tool_call>call:comment{{comment:<|\"|>{prose}<|\"|>,file:<|\"|>src/timeline.rs<|\"|>,line:38}}<tool_call|>"
        );
        let calls = parse_tool_calls(&text, &format);

        assert_eq!(calls.len(), 1, "one call");
        let arguments = calls[0].arguments.as_object().expect("object");
        assert_eq!(
            arguments.len(),
            3,
            "three arguments -- the prose is one value, not several: {arguments:#?}"
        );
        assert_eq!(
            arguments["comment"],
            serde_json::json!(prose),
            "the whole sentence"
        );
        assert_eq!(arguments["file"], serde_json::json!("src/timeline.rs"));
        assert_eq!(arguments["line"], serde_json::json!("38"));
    }

    /// A value the family does not wrap.
    ///
    /// Gemma quotes strings and writes numbers bare -- `path:<|"|>a.rs<|"|>`
    /// beside `line:38` -- and the probes only ever pass strings, so the
    /// derived "name to value" span carried the opening quote welded on.
    /// `line:38` then matched no argument at all.
    ///
    /// Measured on a real review: it had found the bug it was asked to
    /// find, and spent its last four turns unable to say where, because
    /// every `comment` call arrived without a line and was refused.
    #[cfg(feature = "template")]
    #[test]
    fn a_bare_value_parses_beside_a_wrapped_one() {
        let Ok(template) = std::fs::read_to_string("tests/fixtures/gemma4-12b.jinja") else {
            eprintln!("skipping absent fixture");
            return;
        };
        let render = crate::chat_template::probe_renderer();
        let format = derive_tool_call_format(&template, &render).expect("derivable");

        // Exactly what the model emitted, verbatim from the run.
        let text = "<|tool_call>call:comment{comment:<|\"|>The sort only uses seconds<|\"|>,\
                    file:<|\"|>src/timeline.rs<|\"|>,line:38}<tool_call|>";
        let calls = parse_tool_calls(text, &format);
        assert_eq!(calls.len(), 1, "{calls:?}");
        assert_eq!(calls[0].name, "comment");
        assert_eq!(calls[0].arguments["comment"], "The sort only uses seconds");
        assert_eq!(calls[0].arguments["file"], "src/timeline.rs");
        assert_eq!(
            calls[0].arguments["line"], "38",
            "a bare value has to parse too"
        );
        assert_eq!(
            calls[0].arguments.as_object().expect("object").len(),
            3,
            "and nothing else: {:?}",
            calls[0].arguments
        );
    }

    /// Two calls in one turn must not bleed into each other -- the
    /// failure that turned one unparsed argument into a key hundreds of
    /// characters long.
    #[cfg(feature = "template")]
    #[test]
    fn two_calls_in_one_turn_stay_separate() {
        let Ok(template) = std::fs::read_to_string("tests/fixtures/gemma4-12b.jinja") else {
            return;
        };
        let render = crate::chat_template::probe_renderer();
        let format = derive_tool_call_format(&template, &render).expect("derivable");

        let text = "<|tool_call>call:read{path:<|\"|>a.rs<|\"|>,line:38}<tool_call|>\n\
                    thinking out loud\n\
                    <|tool_call>call:verify{}<tool_call|>";
        let calls = parse_tool_calls(text, &format);
        assert_eq!(calls.len(), 2, "{calls:?}");
        assert_eq!(calls[0].name, "read");
        assert_eq!(calls[0].arguments["path"], "a.rs");
        assert_eq!(calls[0].arguments["line"], "38");
        assert_eq!(calls[1].name, "verify");
        assert_eq!(
            calls[1].arguments.as_object().expect("object").len(),
            0,
            "no arguments at all"
        );
    }

    #[cfg(feature = "template")]
    #[test]
    fn every_real_chat_template_round_trips_its_own_call_format() {
        let render = crate::chat_template::probe_renderer();
        let mut checked = 0;

        for fixture in [
            "tests/fixtures/qwen3-coder.jinja",
            "tests/fixtures/qwen3_8_chat_template.jinja",
            // Gemma's is the reason `ToolFormat::Separated` exists: its
            // arguments are comma-separated rather than individually
            // delimited, which `Delimited` cannot express at all.
            "tests/fixtures/gemma4-12b.jinja",
        ] {
            let Ok(template) = std::fs::read_to_string(fixture) else {
                eprintln!("skipping absent fixture: {fixture}");
                continue;
            };
            checked += 1;

            let capabilities = derive_capabilities(&template, &render);
            assert!(capabilities.system, "{fixture} renders a system block");
            assert!(capabilities.tools, "{fixture} renders a tool list");

            let format = derive_tool_call_format(&template, &render)
                .unwrap_or_else(|| panic!("{fixture} must be derivable"));

            // Let the template render a call in whatever format it
            // chooses, with values a real coding agent would send --
            // multi-line content with the delimiters' own characters in
            // it, since that is what breaks a naive parser.
            let content = "fn main() {\n    println!(\"<hi>\");\n}";
            let rendered = render(
                &template,
                &serde_json::json!([
                    { "role": "user", "content": "go" },
                    { "role": "assistant", "content": "", "tool_calls": [{
                        "type": "function",
                        "function": { "name": "write", "arguments": { "path": "src/main.rs", "content": content } }
                    }]},
                ]),
                None,
            )
            .unwrap_or_else(|| panic!("{fixture} must render a call"));

            let calls = parse_tool_calls(&rendered, &format);
            assert_eq!(
                calls.len(),
                1,
                "{fixture}: format {format:?} on {rendered:?}"
            );
            assert_eq!(calls[0].name, "write", "{fixture}");
            assert_eq!(
                calls[0]
                    .arguments
                    .get("path")
                    .and_then(serde_json::Value::as_str),
                Some("src/main.rs"),
                "{fixture}"
            );
            assert_eq!(
                calls[0]
                    .arguments
                    .get("content")
                    .and_then(serde_json::Value::as_str),
                Some(content),
                "{fixture}: a multi-line value must survive verbatim"
            );

            let tools = vec![ToolSpec::new(
                "read",
                "Read a file",
                serde_json::json!({
                    "type": "object",
                    "properties": { "path": { "type": "string", "description": "the path" } },
                    "required": ["path"]
                }),
            )];
            let opening = render_opening(
                &template,
                Some("You are a coding agent."),
                Some(&tools),
                &render,
            )
            .unwrap_or_else(|| panic!("{fixture} must render an opening"));
            assert!(
                opening.contains("You are a coding agent."),
                "{fixture}: {opening}"
            );
            assert!(opening.contains("read"), "{fixture}: {opening}");

            // Result spans, against the real template rather than the
            // stub -- two results must render as two distinct elements,
            // which is the fidelity gap `ToolResultSpans` exists to
            // close.
            let spans = derive_tool_result_spans(&template, &render)
                .unwrap_or_else(|| panic!("{fixture} must yield result spans"));
            let two = spans.render(&["FIRSTRESULT".to_string(), "SECONDRESULT".to_string()]);
            assert!(
                two.contains("FIRSTRESULT") && two.contains("SECONDRESULT"),
                "{fixture}: {two}"
            );
            // Family-agnostic, which the previous version was not: it
            // counted `<tool_response>` literally, Qwen's spelling, so
            // Gemma's `<|tool_response>` scored zero and a working
            // template looked broken. What actually has to hold is that
            // the two results are separated by whatever this template
            // separates them with -- exactly once for two, never for one.
            let one = spans.render(&["ONLYRESULT".to_string()]);
            assert_eq!(two.matches(&spans.separator).count(), 1, "{fixture}: {two}");
            assert_eq!(one.matches(&spans.separator).count(), 0, "{fixture}: {one}");

            // And no probe sentinel may survive into what a model sees.
            for rendered in [&one, &two] {
                for sentinel in [NAME, ARG1, VAL1, RESULT, PLAIN] {
                    assert!(
                        !rendered.contains(sentinel),
                        "{fixture} leaks a probe sentinel: {rendered:?}"
                    );
                }
            }
        }

        assert!(
            checked > 0,
            "no fixtures present -- this test proved nothing"
        );
    }

    /// A real template that renders no tool calls at all must decline
    /// cleanly rather than derive something unusable. Jamba Mini is the
    /// worked example, and is already a fixture here for `llama`'s own
    /// turn-structure tests.
    #[cfg(feature = "template")]
    #[test]
    fn a_real_template_without_tool_calls_declines() {
        let Ok(template) =
            std::fs::read_to_string("tests/fixtures/jamba_mini_1_7_chat_template.jinja")
        else {
            eprintln!("skipping: jamba fixture not present");
            return;
        };
        let render = crate::chat_template::probe_renderer();
        // Whatever it reports, it must not panic and must not invent a
        // format whose spans are empty -- the degenerate case that
        // silently yields empty names.
        if let Some(format) = derive_tool_call_format(&template, &render)
            && let ToolFormat::Delimited {
                name_close,
                arg_open,
                arg_close,
                ..
            } = &format
        {
            assert!(
                !name_close.is_empty() && !arg_open.is_empty() && !arg_close.is_empty(),
                "{format:?}"
            );
        }
    }

    /// `close` must reach the point generation resumes -- see
    /// `derive_tool_result_spans`'s own doc comment for why it does not
    /// simply stop at the end of the tool turn.
    /// `close` must reach the point generation resumes -- see
    /// `derive_tool_result_spans`'s own doc comment.
    #[test]
    fn tool_result_spans_wrap_the_result_and_reopen_the_assistant() {
        let spans = derive_tool_result_spans("", &qwen_xml_render).expect("derivable");
        assert_eq!(
            spans.open,
            "<|im_end|>\n<|im_start|>user\n<tool_response>\n"
        );
        assert_eq!(
            spans.close,
            "\n</tool_response>\n<|im_end|>\n<|im_start|>assistant\n"
        );
    }

    /// The fidelity gap this closes: several results must render as
    /// several `<tool_response>` elements, not one blob.
    #[test]
    fn several_results_each_get_their_own_element() {
        let spans = derive_tool_result_spans("", &qwen_xml_render).expect("derivable");
        assert_eq!(spans.separator, "\n</tool_response>\n<tool_response>\n");

        let rendered = spans.render(&["first".to_string(), "second".to_string()]);
        assert_eq!(rendered.matches("<tool_response>").count(), 2, "{rendered}");
        assert!(rendered.contains("first"), "{rendered}");
        assert!(rendered.contains("second"), "{rendered}");
    }

    /// One result must render exactly as it did before this existed.
    #[test]
    fn a_single_result_renders_unchanged() {
        let spans = derive_tool_result_spans("", &qwen_xml_render).expect("derivable");
        let rendered = spans.render(&["only".to_string()]);
        assert_eq!(rendered, format!("{}only{}", spans.open, spans.close));
        assert_eq!(rendered.matches("<tool_response>").count(), 1, "{rendered}");
    }
}
