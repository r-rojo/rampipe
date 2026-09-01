//! Rendering a model's own Jinja `tokenizer.chat_template`.
//!
//! Split out of `crate::llama` (which still calls it, and is the only
//! thing that knows how to *get* a template out of a GGUF) so that
//! everything template-shaped in this crate can be built and tested
//! without llama.cpp. `crate::tool_format` is why: it derives a model's
//! tool-calling format from a real template, and the only test that
//! proves that works is one run against a real template -- which would
//! otherwise have required a full llama.cpp build to reach.
//!
//! llama.cpp's own `apply_chat_template` is not used: it returns
//! `ffi error -1` on templates using Jinja macros or namespaces, which
//! includes real, current ones (Qwen3-Coder's defines a
//! `render_item_list` macro). minijinja renders those correctly.

use minijinja::{Environment, Value, context};

/// A minijinja environment configured to match the `jinja2.Environment(
/// trim_blocks=True, lstrip_blocks=True)` convention every HuggingFace
/// chat template is written against, plus the two compatibility pieces
/// real templates turn out to need:
///
/// - `pycompat`'s unknown-method callback, so Python `str` methods a
///   template calls (`.startswith()`, `.strip()`, ...) resolve. Without
///   it, Qwen 3.8's own template fails with "string has no method named
///   startswith" -- found by removing a `.ok()` that was swallowing the
///   real error.
/// - `raise_exception`, which templates call to reject inputs they
///   consider invalid; mapping it to a real error means that rejection
///   surfaces as a failed render rather than an undefined-function
///   panic.
fn environment() -> Environment<'static> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    env.add_function(
        "raise_exception",
        |msg: String| -> Result<Value, minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                msg,
            ))
        },
    );
    env
}

/// Renders `template_text` against a full `messages` array, optional
/// `tools`, and `add_generation_prompt`. `None` on any failure --
/// unparseable template, a `raise_exception` the template itself
/// raised, an unsupported construct -- because every caller here treats
/// "this template can't be rendered" as a fall-back-to-something-else
/// case rather than an error to propagate.
///
/// `messages` and `tools` are `serde_json::Value` rather than typed
/// structs: a chat template walks whatever shape it is handed
/// (Qwen3-Coder's renders every parameter key it doesn't recognize as
/// its own XML element), so imposing a type here would constrain
/// templates this crate does not own.
#[must_use]
pub fn render_json(
    template_text: &str,
    messages: &serde_json::Value,
    tools: Option<&serde_json::Value>,
    add_generation_prompt: bool,
) -> Option<String> {
    let mut env = environment();
    env.add_template("chat", template_text).ok()?;
    let tmpl = env.get_template("chat").ok()?;

    let messages = Value::from_serialize(messages);
    let ctx = match tools {
        Some(tools) => context! {
            messages => messages,
            tools => Value::from_serialize(tools),
            add_generation_prompt => add_generation_prompt,
        },
        // Genuinely absent, not an empty list: a template branching on
        // `tools is defined` must see the same thing it would if the
        // caller had never mentioned tools at all.
        None => context! {
            messages => messages,
            add_generation_prompt => add_generation_prompt,
        },
    };
    tmpl.render(ctx).ok()
}

/// [`render_json`] for the plain `(role, content)` case -- what
/// `crate::llama`'s own turn-structure derivation uses, where no
/// message ever carries tool calls.
#[must_use]
pub fn render_messages(
    template_text: &str,
    messages: &[(&str, &str)],
    add_generation_prompt: bool,
) -> Option<String> {
    let messages = serde_json::Value::Array(
        messages
            .iter()
            .map(|(role, content)| serde_json::json!({ "role": role, "content": content }))
            .collect(),
    );
    render_json(template_text, &messages, None, add_generation_prompt)
}

/// A [`crate::tool_format::RenderFn`] backed by this module -- what a
/// caller passes to the derivation functions there. Never adds a
/// generation prompt: every probe transcript ends with a completed
/// turn, and a trailing "now generate" marker would land inside the
/// spans being derived.
#[must_use]
pub fn probe_renderer()
-> impl Fn(&str, &serde_json::Value, Option<&serde_json::Value>) -> Option<String> {
    |template_text: &str, messages: &serde_json::Value, tools: Option<&serde_json::Value>| {
        render_json(template_text, messages, tools, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_a_trivial_template() {
        let rendered = render_messages(
            "{% for m in messages %}{{ m.role }}:{{ m.content }};{% endfor %}",
            &[("user", "hi")],
            false,
        )
        .expect("renders");
        assert_eq!(rendered, "user:hi;");
    }

    #[test]
    fn a_template_raising_an_exception_fails_rather_than_panicking() {
        assert!(
            render_messages("{{ raise_exception('nope') }}", &[("user", "hi")], false).is_none()
        );
    }

    #[test]
    fn tools_are_absent_rather_than_empty_when_not_supplied() {
        let template = "{% if tools is defined %}HAS{% else %}NONE{% endif %}";
        assert_eq!(
            render_json(template, &serde_json::json!([]), None, false).as_deref(),
            Some("NONE")
        );
        assert_eq!(
            render_json(
                template,
                &serde_json::json!([]),
                Some(&serde_json::json!([])),
                false
            )
            .as_deref(),
            Some("HAS")
        );
    }

    /// The `pycompat` case: a real template calling a Python `str`
    /// method. Without the callback registered in `environment`, this
    /// renders `None`.
    #[test]
    fn python_string_methods_resolve() {
        let rendered = render_messages(
            "{% if messages[0].content.startswith('hi') %}yes{% else %}no{% endif %}",
            &[("user", "hi there")],
            false,
        )
        .expect("renders");
        assert_eq!(rendered, "yes");
    }
}
