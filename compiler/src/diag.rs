//! LILA diagnostics: position formatting + rustc-style rendering.
//!
//! The pipeline threads `Result<T, String>` — every error message carries a
//! position prefix in one of two shapes:
//!   "lex error at line L:COL: ..."        (lexer)
//!   "parse error at line L:COL: ..."      (parser — position of the token
//!                                          that broke the parse)
//!   "line L:COL: ..."                     (checker/codegen — statement site)
//!   "internal: ..."                       (compiler invariant violated —
//!                                          a bug, not a user error)
//! `render()` turns any positioned message into an excerpt with a caret
//! pointing at the column, so `lilac build` failures read like a real
//! toolchain's, not a bare one-liner.

/// Internal invariant message — these are compiler bugs, never user errors.
pub fn internal(msg: &str) -> String {
    format!("internal: {} (this is a lilac bug — please report it)", msg)
}

/// One parsed position from an error message, if present.
///
/// Accepts both shapes:
///   "line L:C: ..."  -> (L, C)
///   "line L: ..."    -> (L, 1)   // checker/codegen emit a line with no column
///
/// The line-only form matters: ~130 error sites in the checker and codegen
/// report a statement line but no column. Previously that made `parse_pos`
/// return `None`, so `render()` silently fell back to a bare one-liner and
/// lost the source excerpt entirely. Defaulting the column to 1 keeps the
/// excerpt (the useful half) and points the caret at the start of the line.
///
/// A run of digits is only treated as a column when it is terminated by ':'
/// (a real "L:C:" pair) or ends the message — otherwise prose like
/// "line 4: 99 is out of range" would be mis-parsed as column 99.
fn parse_pos(err: &str) -> Option<(u32, u32)> {
    // accept "line L[:C]" anywhere in the message; the specific prefixes
    // ("lex error at line ...", "parse error at line ...", "line ...") all
    // funnel into the same capture
    let idx = err.find("line ")?;
    let rest = &err[idx + 5..];
    let mut it = rest.splitn(2, ':');
    let line: u32 = it.next()?.trim().parse().ok()?;
    let col: u32 = match it.next() {
        None => 1,
        Some(part) => {
            let part = part.trim_start();
            let digits: String = part.chars().take_while(|c| c.is_ascii_digit()).collect();
            let tail = &part[digits.len()..];
            if digits.is_empty() {
                1
            } else if tail.starts_with(':') || tail.is_empty() {
                digits.parse().unwrap_or(1)
            } else {
                1
            }
        }
    };
    Some((line, col))
}

/// Render an error against the source text: the message line plus the
/// offending source line with a caret at the column. Falls back to the bare
/// message when no position or the line is out of range (e.g. internal
/// errors, IO errors).
pub fn render(src: &str, err: &str) -> String {
    let Some((line, col)) = parse_pos(err) else {
        return err.to_string();
    };
    let Some((i, text)) = src.lines().enumerate().find(|(i, _)| *i + 1 == line as usize) else {
        return format!("{}\n  --> line {} (beyond end of source)", err, line);
    };
    let _ = i;
    // clamp the caret to the line's printable width
    let width = text.chars().count().max(1);
    let col_idx = (col.saturating_sub(1) as usize).min(width - 1);
    let indent: String = " ".repeat(col_idx);
    // one line of context above (when it exists), like rustc
    let prev = if line > 1 {
        src.lines().nth(line as usize - 2).map(|p| format!("   {} | {}\n", line - 1, p)).unwrap_or_default()
    } else {
        String::new()
    };
    format!(
        "{}\n{}   {} | {}\n   {} | {}^",
        err,
        prev,
        line,
        text,
        line,
        indent
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positioned_message_roundtrip() {
        let m = "line 3:7: bad thing";
        assert_eq!(parse_pos(&m), Some((3, 7)));
    }

    #[test]
    fn render_places_caret() {
        let src = "fn update() {\n    px = ;\n}";
        let out = render(src, "line 2:9: expected expression");
        assert!(out.contains("    px = ;"));
        assert!(out.contains("^"));
        assert!(out.contains("line 2:9"));
    }

    #[test]
    fn render_plain_errors_untouched() {
        assert_eq!(render("src", "cannot read 'x'"), "cannot read 'x'");
        assert!(render("src", &internal("boom")).starts_with("internal:"));
    }

    #[test]
    fn lexer_shaped_positions_parse() {
        assert_eq!(parse_pos("lex error at line 12:5: nope"), Some((12, 5)));
        assert_eq!(parse_pos("parse error at line 4:2: expected RParen"), Some((4, 2)));
    }

    // --- regressions: line-only checker/codegen errors must still locate ---

    #[test]
    fn line_only_position_defaults_to_column_one() {
        // the checker emits "line L: msg" with no column — it must still
        // resolve so render() can show the excerpt.
        assert_eq!(parse_pos("line 4: array 'a' index 99 out of bounds"), Some((4, 1)));
        assert_eq!(parse_pos("line 12: unknown variable 'x'"), Some((12, 1)));
    }

    #[test]
    fn prose_after_the_line_is_not_mistaken_for_a_column() {
        // "99" is prose, not a column: only a "L:C:" pair counts.
        assert_eq!(parse_pos("line 4: 99 is out of range"), Some((4, 1)));
        assert_eq!(parse_pos("line 2: 2 entities declared"), Some((2, 1)));
    }

    #[test]
    fn line_only_errors_render_an_excerpt() {
        let src = "fn update() {\n    px = 1;\n}";
        let out = render(src, "line 2: unknown variable 'px'");
        assert!(out.contains("    px = 1;"), "expected excerpt, got:\n{}", out);
        assert!(out.contains("^"), "expected caret, got:\n{}", out);
    }
}
