//! LILA lexer. Tokens + #Atom interning.
//! NOTE: deliberately keyword-free — the parser matches identifier TEXT in
//! keyword positions, so game code may use names like `wave` or `fill`
//! without clashing with contextual keywords (`wave:` in sfx decls etc).

#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    Ident(String),
    Atom(u16), // interned atom index (name string kept in table below)
    Int(u32),
    Fix(i32), // Q24.8
    Str(String),
    LParen, RParen, LBrace, RBrace, LBracket, RBracket,
    Comma, Semi, Colon, Dot,
    Plus, Minus, Star, Slash, Percent,
    Amp, Pipe, Caret, Shl, Shr,
    Lt, Gt, Le, Ge, EqEq, NotEq,
    Eq, PlusEq, MinusEq, StarEq,
    AndAnd, OrOr, Not,
    Arrow, // v8 OPTIMIZER: fsm guard arrow (target -> STATE)
    Eof,
}

pub struct Lexer<'a> {
    src: &'a [u8],
    pos: usize,
    line: u32,
    line_start: usize, // byte offset where the current line begins (col = pos - line_start + 1)
    tok_start: (u32, u32), // position of the token currently being scanned
    pub atom_names: Vec<String>,
    atom_map: std::collections::HashMap<String, u16>,
    toks: Vec<(Tok, u32, u32)>, // (token, line, col)
}

fn is_ident_start(c: u8) -> bool { c.is_ascii_alphabetic() || c == b'_' }
fn is_ident_cont(c: u8) -> bool { c.is_ascii_alphanumeric() || c == b'_' }

impl<'a> Lexer<'a> {
    pub fn new(src: &'a str) -> Self {
        Lexer {
            src: src.as_bytes(),
            pos: 0,
            line: 1,
            line_start: 0,
            tok_start: (1, 1),
            atom_names: Vec::new(),
            atom_map: std::collections::HashMap::new(),
            toks: Vec::new(),
        }
    }

    /// (line, col) of the current byte — 1-based, columns count bytes
    /// (ASCII source assumption; multibyte UTF-8 inside strings still works,
    /// idents are ASCII by construction).
    fn pos_pair(&self) -> (u32, u32) {
        (self.line, (self.pos - self.line_start + 1) as u32)
    }

    fn err(&self, msg: &str) -> String {
        let (line, col) = self.pos_pair();
        format!("lex error at line {}:{}: {}", line, col, msg)
    }

    fn intern_atom(&mut self, name: String) -> Result<u16, String> {
        if let Some(&id) = self.atom_map.get(&name) {
            return Ok(id);
        }
        // format guard: atom ids are u16 on the wire, and names ride in the
        // file as length-prefixed str8 — both would silently corrupt on
        // overflow, so the lexer rejects early with a precise message
        if self.atom_names.len() >= 65535 {
            let (line, col) = self.pos_pair();
            return Err(format!(
                "lex error at line {}:{}: more than 65535 distinct #atoms — the .libyte format addresses atoms by u16",
                line, col
            ));
        }
        if name.len() > 255 {
            let (line, col) = self.pos_pair();
            return Err(format!(
                "lex error at line {}:{}: atom name '{}' is {} bytes (max 255 — the format stores a u8 length)",
                line, col, &name[..32], name.len()
            ));
        }
        let id = self.atom_names.len() as u16;
        self.atom_names.push(name.clone());
        self.atom_map.insert(name, id);
        Ok(id)
    }

    pub fn tokenize(mut self) -> Result<(Vec<(Tok, u32, u32)>, Vec<String>), String> {
        while self.pos < self.src.len() {
            let c = self.src[self.pos];
            // snapshot the token's START position: the match arms below
            // advance self.pos before calling push()
            let (start_line, start_col) = self.pos_pair();
            self.tok_start = (start_line, start_col);
            match c {
                b'\n' => {
                    self.line += 1;
                    self.pos += 1;
                    self.line_start = self.pos;
                }
                b' ' | b'\t' | b'\r' => { self.pos += 1; }
                b'/' if self.src.get(self.pos + 1) == Some(&b'/') => {
                    while self.pos < self.src.len() && self.src[self.pos] != b'\n' { self.pos += 1; }
                }
                b'(' => { self.pos += 1; self.push(Tok::LParen); }
                b')' => { self.pos += 1; self.push(Tok::RParen); }
                b'{' => { self.pos += 1; self.push(Tok::LBrace); }
                b'}' => { self.pos += 1; self.push(Tok::RBrace); }
                b'[' => { self.pos += 1; self.push(Tok::LBracket); }
                b']' => { self.pos += 1; self.push(Tok::RBracket); }
                b',' => { self.pos += 1; self.push(Tok::Comma); }
                b';' => { self.pos += 1; self.push(Tok::Semi); }
                b':' => { self.pos += 1; self.push(Tok::Colon); }
                b'.' => { self.pos += 1; self.push(Tok::Dot); }
                b'+' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::PlusEq); }
                    else { self.push(Tok::Plus); }
                }
                b'-' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::MinusEq); }
                    else if self.src.get(self.pos) == Some(&b'>') { self.pos += 1; self.push(Tok::Arrow); }
                    else { self.push(Tok::Minus); }
                }
                b'*' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::StarEq); }
                    else { self.push(Tok::Star); }
                }
                b'/' => { self.pos += 1; self.push(Tok::Slash); }
                b'%' => { self.pos += 1; self.push(Tok::Percent); }
                b'&' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'&') { self.pos += 1; self.push(Tok::AndAnd); }
                    else { self.push(Tok::Amp); }
                }
                b'|' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'|') { self.pos += 1; self.push(Tok::OrOr); }
                    else { self.push(Tok::Pipe); }
                }
                b'^' => { self.pos += 1; self.push(Tok::Caret); }
                b'<' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'<') { self.pos += 1; self.push(Tok::Shl); }
                    else if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::Le); }
                    else { self.push(Tok::Lt); }
                }
                b'>' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'>') { self.pos += 1; self.push(Tok::Shr); }
                    else if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::Ge); }
                    else { self.push(Tok::Gt); }
                }
                b'=' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::EqEq); }
                    else { self.push(Tok::Eq); }
                }
                b'!' => {
                    self.pos += 1;
                    if self.src.get(self.pos) == Some(&b'=') { self.pos += 1; self.push(Tok::NotEq); }
                    else { self.push(Tok::Not); }
                }
                b'"' => {
                    self.pos += 1;
                    let start = self.pos;
                    while self.pos < self.src.len() && self.src[self.pos] != b'"' {
                        if self.src[self.pos] == b'\n' {
                            return Err(self.err("unterminated string literal"));
                        }
                        self.pos += 1;
                    }
                    if self.pos >= self.src.len() {
                        return Err(self.err("unterminated string literal"));
                    }
                    if self.pos - start > 255 {
                        return Err(self.err("string literal longer than 255 bytes (the .libyte format stores a u8 length)"));
                    }
                    let s = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                    self.pos += 1; // closing quote
                    self.push(Tok::Str(s));
                }
                b'#' => {
                    self.pos += 1;
                    let start = self.pos;
                    while self.pos < self.src.len() && is_ident_cont(self.src[self.pos]) { self.pos += 1; }
                    if start == self.pos {
                        return Err(self.err("expected atom name after '#'"));
                    }
                    let name = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                    // atom string atoms also ride the str8 wire format
                    if name.len() > 255 {
                        return Err(self.err("atom name longer than 255 bytes"));
                    }
                    let id = self.intern_atom(name)?;
                    self.push(Tok::Atom(id));
                }
                b'0'..=b'9' => {
                    if c == b'0' && self.src.get(self.pos + 1) == Some(&b'x') {
                        self.pos += 2;
                        let hs = self.pos;
                        while self.pos < self.src.len() && self.src[self.pos].is_ascii_hexdigit() { self.pos += 1; }
                        let v = u32::from_str_radix(
                            std::str::from_utf8(&self.src[hs..self.pos]).map_err(|_| self.err("bad hex"))?, 16,
                        ).map_err(|_| self.err("hex literal overflow"))?;
                        self.push(Tok::Int(v));
                        continue;
                    }
                    let start = self.pos;
                    let mut is_fix = false;
                    while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() { self.pos += 1; }
                    // '.' followed by a digit => fixed-point literal (Q24.8, max 2 frac digits)
                    if self.src.get(self.pos) == Some(&b'.')
                        && self.src.get(self.pos + 1).map_or(false, |c| c.is_ascii_digit())
                    {
                        is_fix = true;
                        self.pos += 1; // consume '.'
                        let frac_start = self.pos;
                        while self.pos < self.src.len() && self.src[self.pos].is_ascii_digit() { self.pos += 1; }
                        let n_frac = self.pos - frac_start;
                        if n_frac > 2 {
                            return Err(self.err("fixed literals allow max 2 decimal digits (Q24.8)"));
                        }
                    }
                    let text = std::str::from_utf8(&self.src[start..self.pos])
                        .map_err(|_| self.err("bad number"))?;
                    if is_fix {
                        let f: f64 = text.parse().map_err(|_| self.err("bad fixed literal"))?;
                        let q = (f * 256.0).round();
                        if q < -2147483648.0 || q > 2147483647.0 {
                            return Err(self.err("fixed literal out of Q24.8 range"));
                        }
                        self.push(Tok::Fix(q as i32));
                    } else {
                        let v: u32 = text.parse().map_err(|_| self.err("bad int literal"))?;
                        self.push(Tok::Int(v));
                    }
                }
                c if is_ident_start(c) => {
                    let start = self.pos;
                    while self.pos < self.src.len() && is_ident_cont(self.src[self.pos]) { self.pos += 1; }
                    let name = String::from_utf8_lossy(&self.src[start..self.pos]).into_owned();
                    self.push(Tok::Ident(name));
                }
                other => {
                    return Err(self.err(&format!("unexpected character '{}'", other as char)));
                }
            }
        }
        self.toks.push((Tok::Eof, self.line, self.pos_pair().1));
        Ok((self.toks, self.atom_names))
    }

    fn push(&mut self, t: Tok) {
        let (line, col) = self.tok_start;
        self.toks.push((t, line, col));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex(src: &str) -> Result<(Vec<(Tok, u32, u32)>, Vec<String>), String> {
        Lexer::new(src).tokenize()
    }

    #[test]
    fn columns_count_from_one() {
        let (toks, _) = lex("fn update() {\n  px += 1.0;\n}").unwrap();
        // 'fn' at 1:1, 'px' at 2:3
        assert_eq!(toks[0], (Tok::Ident("fn".into()), 1, 1));
        let px = toks.iter().find(|t| matches!(&t.0, Tok::Ident(s) if s == "px")).unwrap();
        assert_eq!((px.1, px.2), (2, 3));
    }

    #[test]
    fn newline_resets_column() {
        let (toks, _) = lex("a\nb\n  c").unwrap();
        assert_eq!(toks[1].2, 1); // 'b' at line 2 col 1
        assert_eq!(toks[2].2, 3); // 'c' at line 3 col 3
    }

    #[test]
    fn all_simple_tokens_carry_position() {
        let (toks, _) = lex("( ) { } [ ] , ; : . + - * / % & | ^ < > = !").unwrap();
        assert!(toks.iter().all(|t| t.1 == 1 && t.2 >= 1));
    }

    #[test]
    fn atom_interning_and_names() {
        let (toks, names) = lex("#left #left #fire").unwrap();
        assert_eq!(names, vec!["left", "fire"]);
        assert!(matches!(toks[0].0, Tok::Atom(0)));
        assert!(matches!(toks[1].0, Tok::Atom(0)));
        assert!(matches!(toks[2].0, Tok::Atom(1)));
    }

    #[test]
    fn overlong_string_rejected() {
        let long = format!("\"{}\"", "x".repeat(256));
        let err = lex(&long).unwrap_err();
        assert!(err.contains("255"), "{}", err);
    }

    #[test]
    fn fixed_literal_limits() {
        assert!(lex("1.234").is_err()); // 3 frac digits
        assert!(matches!(lex("1.25").unwrap().0[0].0, Tok::Fix(320)));
        assert!(lex("999999999999.0").is_err()); // Q24.8 overflow
    }

    #[test]
    fn comments_are_skipped() {
        let (toks, _) = lex("a // rest is comment\nb").unwrap();
        assert_eq!(toks.len(), 3); // a, b, Eof
    }
}
