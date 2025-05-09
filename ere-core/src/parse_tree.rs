use std::fmt::{Display, Write};

use proc_macro2::TokenStream;
use quote::{quote, ToTokens};

/// On input string `text: &'a str` with options `{<prefix> => T, ..}`,
/// it returns `Option<(&'a str, T)>` based on which literal prefix is matched
///
/// Example:
/// ```
/// let text = "asdf";
/// let test = match_prefix!(text, {
///     "fdsa" => 0,
///     "qwerty" => 1,
///     "as" => 2,
///     "asd" => 3,
/// });
/// assert_eq!(test, Some("df", 2));
/// ```
macro_rules! match_prefix {
    ($text:ident, { }) => (::core::option::Option::None);
    ($text:ident, {
        $x:literal => $y:expr,
        $($xs:literal => $ys:expr,)*
    }) => {
        if let ::core::option::Option::Some(rest) = str::strip_prefix($text, $x) {
            ::core::option::Option::Some((rest, $y))
        } $(else if let ::core::option::Option::Some(rest) = str::strip_prefix($text, $xs) {
            ::core::option::Option::Some((rest, $ys))
        })* else {
            ::core::option::Option::None
        }
    };
}

/// A represents a [POSIX-compliant ERE](https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap09.html).
/// Primarily intended for use as a parser.
pub struct ERE(pub(crate) Vec<EREBranch>);
impl ERE {
    fn take<'a>(rest: &'a str) -> Option<(&'a str, ERE)> {
        let mut branches = Vec::new();
        let (mut rest, branch) = EREBranch::take(rest)?;
        branches.push(branch);
        while let Some(new_rest) = rest.strip_prefix('|') {
            rest = new_rest;
            let Some((new_rest, branch)) = EREBranch::take(rest) else {
                break;
            };
            rest = new_rest;
            branches.push(branch);
        }
        return Some((rest, ERE(branches)));
    }

    pub fn parse_str(input: &str) -> Option<Self> {
        let Some(("", ere)) = ERE::take(&input) else {
            return None;
        };
        return Some(ere);
    }
}
impl syn::parse::Parse for ERE {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let literal: syn::LitStr = input.parse()?;
        let string = literal.value();
        return ERE::parse_str(&string).ok_or_else(|| {
            syn::Error::new(
                literal.span(),
                "Failed to parse POSIX Extended Regex Expression",
            )
        });
    }
}
impl Display for ERE {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut it = self.0.iter();
        let Some(first) = it.next() else {
            return Ok(());
        };
        write!(f, "{first}")?;
        for part in it {
            write!(f, "|{part}")?;
        }
        return Ok(());
    }
}

pub(crate) struct EREBranch(pub(crate) Vec<EREPart>);
impl EREBranch {
    fn take<'a>(rest: &'a str) -> Option<(&'a str, EREBranch)> {
        let mut parts = Vec::new();
        let (mut rest, part) = EREPart::take(rest)?;
        parts.push(part);
        while let Some((new_rest, part)) = EREPart::take(rest) {
            rest = new_rest;
            parts.push(part);
        }
        return Some((rest, EREBranch(parts)));
    }
}
impl Display for EREBranch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for part in &self.0 {
            write!(f, "{part}")?;
        }
        return Ok(());
    }
}

pub(crate) enum EREPart {
    Single(EREExpression),
    Quantified(EREExpression, Quantifier),
    Start,
    End,
}
impl EREPart {
    fn take<'a>(rest: &'a str) -> Option<(&'a str, EREPart)> {
        if let Some(rest) = rest.strip_prefix('^') {
            return Some((rest, EREPart::Start));
        } else if let Some(rest) = rest.strip_prefix('$') {
            return Some((rest, EREPart::End));
        }

        let (rest, expr) = if let Some(rest) = rest.strip_prefix('(') {
            let (rest, ere) = ERE::take(rest)?;
            let rest = rest.strip_prefix(')')?;
            (rest, EREExpression::Subexpression(ere))
        } else {
            let (rest, atom) = Atom::take(rest)?;
            (rest, EREExpression::Atom(atom))
        };

        let part = match Quantifier::take(rest) {
            Some((rest, quantifier)) => (rest, EREPart::Quantified(expr, quantifier)),
            None => (rest, EREPart::Single(expr)),
        };
        return Some(part);
    }
}
impl Display for EREPart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            EREPart::Single(expr) => write!(f, "{expr}"),
            EREPart::Quantified(expr, quantifier) => write!(f, "{expr}{quantifier}"),
            EREPart::Start => f.write_char('^'),
            EREPart::End => f.write_char('$'),
        };
    }
}

pub(crate) enum EREExpression {
    Atom(Atom),
    Subexpression(ERE),
}
impl Display for EREExpression {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            EREExpression::Atom(atom) => write!(f, "{atom}"),
            EREExpression::Subexpression(ere) => write!(f, "({ere})"),
        };
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub(crate) enum Quantifier {
    Star,
    Plus,
    QuestionMark,
    /// The equivalent to a range specifier with fixed size
    Multiple(u32),
    /// `(u32, None)` is unbounded, `(u32, Some(u32))` is bounded
    Range(u32, Option<u32>),
}

impl Quantifier {
    /// The minimum this quantifier matches, inclusive
    #[inline]
    const fn min(&self) -> u32 {
        return match self {
            Quantifier::Star => 0,
            Quantifier::Plus => 1,
            Quantifier::QuestionMark => 0,
            Quantifier::Multiple(n) => *n,
            Quantifier::Range(n, _) => *n,
        };
    }
    /// The maximum this quantifier matches, inclusive. If `None`, it is unbounded
    #[inline]
    const fn max(&self) -> Option<u32> {
        return match self {
            Quantifier::Star => None,
            Quantifier::Plus => None,
            Quantifier::QuestionMark => Some(1),
            Quantifier::Multiple(n) => Some(*n),
            Quantifier::Range(_, m) => *m,
        };
    }

    #[inline]
    fn take<'a>(rest: &'a str) -> Option<(&'a str, Quantifier)> {
        let mut it = rest.chars();
        match it.next() {
            Some('*') => return Some((it.as_str(), Quantifier::Star)),
            Some('+') => return Some((it.as_str(), Quantifier::Plus)),
            Some('?') => return Some((it.as_str(), Quantifier::QuestionMark)),
            Some('{') => {
                let (inside, rest) = it.as_str().split_once('}')?;
                match inside.split_once(',') {
                    None => return Some((rest, Quantifier::Multiple(inside.parse().ok()?))),
                    Some((min, "")) => {
                        return Some((rest, Quantifier::Range(min.parse().ok()?, None)))
                    }
                    Some((min, max)) => {
                        return Some((
                            rest,
                            Quantifier::Range(min.parse().ok()?, Some(max.parse().ok()?)),
                        ))
                    }
                }
            }
            _ => return None,
        }
    }
}
impl Display for Quantifier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Quantifier::Star => f.write_char('*'),
            Quantifier::Plus => f.write_char('+'),
            Quantifier::QuestionMark => f.write_char('?'),
            Quantifier::Multiple(n) => write!(f, "{{{n}}}"),
            Quantifier::Range(n, None) => write!(f, "{{{n},}}"),
            Quantifier::Range(n, Some(m)) => write!(f, "{{{n},{m}}}"),
        };
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum CharClass {
    /// Matches anything but `NUL` (`'\0'`)
    Dot,
}
impl CharClass {
    pub const fn check(&self, c: char) -> bool {
        return match self {
            CharClass::Dot => c != '\0',
        };
    }
}
impl Display for CharClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            CharClass::Dot => f.write_char('.'),
        };
    }
}
impl ToTokens for CharClass {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            CharClass::Dot => tokens.extend(quote! {::ere_core::parse_tree::CharClass::Dot}),
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub enum Atom {
    /// Includes normal char and escaped chars
    NormalChar(char),
    CharClass(CharClass),
    /// A matching bracket expression
    MatchingList(Vec<BracketExpressionTerm>),
    /// A nonmatching bracket expression
    NonmatchingList(Vec<BracketExpressionTerm>),
}
impl From<char> for Atom {
    fn from(value: char) -> Self {
        return Atom::NormalChar(value);
    }
}
impl From<CharClass> for Atom {
    fn from(value: CharClass) -> Self {
        return Atom::CharClass(value);
    }
}

impl Atom {
    fn take<'a>(rest: &'a str) -> Option<(&'a str, Atom)> {
        let mut it = rest.chars();
        match it.next() {
            Some('\\') => match it.next() {
                Some(c) if is_escapable_character(c) => {
                    return Some((it.as_str(), Atom::NormalChar(c)))
                }
                _ => return None,
            },
            Some('.') => return Some((it.as_str(), CharClass::Dot.into())),
            Some('[') => {
                let mut rest = it.as_str();
                let mut items = Vec::new();
                let none_of = if let Some(new_rest) = rest.strip_prefix('^') {
                    rest = new_rest;
                    true
                } else {
                    false
                };
                if let Some(new_rest) = rest.strip_prefix(']') {
                    rest = new_rest;
                    items.push(BracketExpressionTerm::Single(']'));
                }
                loop {
                    if let Some(new_rest) = rest.strip_prefix(']') {
                        // End of the bracket expression
                        rest = new_rest;
                        break;
                    } else if let Some((new_rest, class)) = BracketCharClass::take(rest) {
                        // A bracket char class
                        rest = new_rest;
                        items.push(BracketExpressionTerm::CharClass(class));
                    } else {
                        // Normal
                        let mut it = rest.chars();
                        let first = it.next()?;
                        rest = it.as_str();
                        if let '-' = it.next()? {
                            let second = it.next()?;
                            rest = it.as_str();
                            if second == ']' {
                                // it's just two characters at the end
                                items.push(BracketExpressionTerm::Single(first));
                                items.push(BracketExpressionTerm::Single('-'));
                                break;
                            } else {
                                // it's a range
                                items.push(BracketExpressionTerm::Range(first, second));
                            }
                        } else {
                            items.push(BracketExpressionTerm::Single(first));
                        }
                    }
                }

                if none_of {
                    return Some((rest, Atom::NonmatchingList(items)));
                } else {
                    return Some((rest, Atom::MatchingList(items)));
                }
            }
            Some(c) if !is_special_character(c) => return Some((it.as_str(), Atom::NormalChar(c))),
            None | Some(_) => return None,
        }
    }
    pub fn check(&self, c: char) -> bool {
        return match self {
            Atom::NormalChar(a) => *a == c,
            Atom::CharClass(char_class) => char_class.check(c),
            Atom::MatchingList(vec) => vec.into_iter().any(|b| b.check(c)),
            Atom::NonmatchingList(vec) => !vec.into_iter().any(|b| b.check(c)),
        };
    }
}
impl Display for Atom {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            Atom::NormalChar(c) if is_escapable_character(*c) => write!(f, "\\{c}"),
            Atom::NormalChar(c) => f.write_char(*c),
            Atom::CharClass(c) => c.fmt(f),
            Atom::MatchingList(vec) => {
                f.write_char('[')?;
                for term in vec {
                    write!(f, "{term}")?;
                }
                f.write_char(']')
            }
            Atom::NonmatchingList(vec) => {
                f.write_str("[^")?;
                for term in vec {
                    write!(f, "{term}")?;
                }
                f.write_char(']')
            }
        };
    }
}

/// From https://pubs.opengroup.org/onlinepubs/9799919799/basedefs/V1_chap09.html#tag_09_03_05
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BracketCharClass {
    /// [:alnum:]
    Alphanumeric,
    /// [:cntrl:]
    Control,
    /// [:lower:]
    Lower,
    /// [:space:]
    Space,
    /// [:alpha:]
    Alphabet,
    /// [:digit:]
    Digit,
    /// [:print:]
    Print,
    /// [:upper:]
    Upper,
    /// [:blank:]
    Blank,
    /// [:graph:]
    Graphic,
    /// [:punct:]
    Punctuation,
    /// [:xdigit:]
    HexDigit,
}
impl BracketCharClass {
    /// Checks matches to the char classes.
    pub const fn check_ascii(&self, c: char) -> bool {
        return match self {
            BracketCharClass::Alphanumeric => c.is_ascii_alphanumeric(),
            BracketCharClass::Control => c.is_ascii_control(),
            BracketCharClass::Lower => c.is_ascii_lowercase(),
            BracketCharClass::Space => c.is_ascii_whitespace() || c == '\x0b', // POSIX includes vertical tab
            BracketCharClass::Alphabet => c.is_ascii_alphabetic(),
            BracketCharClass::Digit => c.is_ascii_digit(),
            BracketCharClass::Print => matches!(c, '\x20'..='\x7E'),
            BracketCharClass::Upper => c.is_ascii_uppercase(),
            BracketCharClass::Blank => c == ' ' || c == '\t',
            BracketCharClass::Graphic => c.is_ascii_graphic(),
            BracketCharClass::Punctuation => c.is_ascii_punctuation(),
            BracketCharClass::HexDigit => c.is_ascii_hexdigit(),
        };
    }
    fn take<'a>(rest: &'a str) -> Option<(&'a str, BracketCharClass)> {
        let rest = rest.strip_prefix("[:")?;
        return match_prefix!(rest, {
            "alnum:]" => BracketCharClass::Alphanumeric,
            "cntrl:]" => BracketCharClass::Control,
            "lower:]" => BracketCharClass::Lower,
            "alpha:]" => BracketCharClass::Alphabet,
            "digit:]" => BracketCharClass::Digit,
            "print:]" => BracketCharClass::Print,
            "upper:]" => BracketCharClass::Upper,
            "blank:]" => BracketCharClass::Blank,
            "graph:]" => BracketCharClass::Graphic,
            "punct:]" => BracketCharClass::Punctuation,
            "xdigit:]" => BracketCharClass::HexDigit,
        });
    }
}
impl Display for BracketCharClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            BracketCharClass::Alphanumeric => f.write_str("[:alnum:]"),
            BracketCharClass::Control => f.write_str("[:cntrl:]"),
            BracketCharClass::Lower => f.write_str("[:lower:]"),
            BracketCharClass::Space => f.write_str("[:space:]"),
            BracketCharClass::Alphabet => f.write_str("[:alpha:]"),
            BracketCharClass::Digit => f.write_str("[:digit:]"),
            BracketCharClass::Print => f.write_str("[:print:]"),
            BracketCharClass::Upper => f.write_str("[:upper:]"),
            BracketCharClass::Blank => f.write_str("[:blank:]"),
            BracketCharClass::Graphic => f.write_str("[:graph:]"),
            BracketCharClass::Punctuation => f.write_str("[:punct:]"),
            BracketCharClass::HexDigit => f.write_str("[:xdigit:]"),
        };
    }
}
impl ToTokens for BracketCharClass {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        match self {
            BracketCharClass::Alphanumeric => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Alphanumeric})
            }
            BracketCharClass::Control => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Control})
            }
            BracketCharClass::Lower => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Lower})
            }
            BracketCharClass::Space => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Space})
            }
            BracketCharClass::Alphabet => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Alphabet})
            }
            BracketCharClass::Digit => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Digit})
            }
            BracketCharClass::Print => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Print})
            }
            BracketCharClass::Upper => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Upper})
            }
            BracketCharClass::Blank => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Blank})
            }
            BracketCharClass::Graphic => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Graph})
            }
            BracketCharClass::Punctuation => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::Punctuation})
            }
            BracketCharClass::HexDigit => {
                tokens.extend(quote! {::ere_core::parse_tree::BracketCharClass::XDigit})
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum BracketExpressionTerm {
    Single(char),
    Range(char, char),
    CharClass(BracketCharClass),
}
impl BracketExpressionTerm {
    pub const fn check(&self, c: char) -> bool {
        return match self {
            BracketExpressionTerm::Single(a) => *a == c,
            BracketExpressionTerm::Range(a, b) => *a <= c && c <= *b,
            BracketExpressionTerm::CharClass(class) => class.check_ascii(c),
        };
    }
}
impl Display for BracketExpressionTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            BracketExpressionTerm::Single(c) => f.write_char(*c),
            BracketExpressionTerm::Range(first, second) => write!(f, "{first}-{second}"),
            BracketExpressionTerm::CharClass(class) => class.fmt(f),
        };
    }
}
impl From<char> for BracketExpressionTerm {
    fn from(value: char) -> Self {
        return BracketExpressionTerm::Single(value);
    }
}
impl From<BracketCharClass> for BracketExpressionTerm {
    fn from(value: BracketCharClass) -> Self {
        return BracketExpressionTerm::CharClass(value);
    }
}
impl ToTokens for BracketExpressionTerm {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.extend(match self {
            BracketExpressionTerm::Single(c) => quote! {
                ::ere_core::parse_tree::BracketExpressionTerm::Single(#c)
            },
            BracketExpressionTerm::Range(a, z) => quote! {
                ::ere_core::parse_tree::BracketExpressionTerm::Range(#a, #z)
            },
            BracketExpressionTerm::CharClass(char_class) => quote! {
                ::ere_core::parse_tree::BracketExpressionTerm::CharClass(#char_class)
            },
        });
    }
}

/// The characters that can only occur if quoted
#[inline]
const fn is_special_character(c: char) -> bool {
    return c == '^'
        || c == '.'
        || c == '['
        || c == '$'
        || c == '('
        || c == ')'
        || c == '|'
        || c == '*'
        || c == '+'
        || c == '?'
        || c == '{'
        || c == '\\';
}

/// The characters that can only occur if quoted
#[inline]
const fn is_escapable_character(c: char) -> bool {
    return is_special_character(c) || c == ']' || c == '}';
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruction() {
        fn test_reconstruction(text: &str) {
            let (rest, ere) = ERE::take(text).unwrap();
            assert!(rest.is_empty(), "{text} did not get used (left {rest})");
            let reconstructed = ere.to_string();
            assert_eq!(text, &reconstructed);
        }
        test_reconstruction("asdf");
        test_reconstruction("as+df123$");
        test_reconstruction("asd?f*123");
        test_reconstruction("^asdf123.*");

        test_reconstruction("a[|]");
        test_reconstruction("[^]a-z1-4A-X-]asdf");
        test_reconstruction("cd[X-]er");

        test_reconstruction("my word is [[:alnum:]_]");
        test_reconstruction("my word is [[:lower:][:digit:]_]");

        test_reconstruction("a|b");
        test_reconstruction("a|b|c");
        test_reconstruction("a(a(b|c)|d)|(g|f)b");
        test_reconstruction("(a|b)|(c|d){3}");

        test_reconstruction("a[y-z]{1,3}");
        test_reconstruction("a{3,}");
        test_reconstruction("a(efg){1,}");
    }

    #[test]
    fn parse_quantifiers() {
        assert_eq!(Quantifier::take(""), None);
        assert_eq!(Quantifier::take("+asdf"), Some(("asdf", Quantifier::Plus)));
        assert_eq!(Quantifier::take("*"), Some(("", Quantifier::Star)));
        assert_eq!(Quantifier::take("e?"), None);
        assert_eq!(Quantifier::take("{"), None);
        assert_eq!(Quantifier::take("{}"), None);
        assert_eq!(Quantifier::take("{1}"), Some(("", Quantifier::Multiple(1))));
        assert_eq!(
            Quantifier::take("{9}ee"),
            Some(("ee", Quantifier::Multiple(9)))
        );
        assert_eq!(
            Quantifier::take("{10,}ee"),
            Some(("ee", Quantifier::Range(10, None)))
        );
        assert_eq!(
            Quantifier::take("{0,11}ef"),
            Some(("ef", Quantifier::Range(0, Some(11))))
        );
        assert_eq!(Quantifier::take("{0,e11}ef"), None);
        assert_eq!(Quantifier::take("{0;11}ef"), None);
    }

    #[test]
    fn parse_atom_simple() {
        assert_eq!(Atom::take("a"), Some(("", Atom::NormalChar('a'))));
        assert_eq!(Atom::take(r"abcd"), Some(("bcd", Atom::NormalChar('a'))));
        assert_eq!(Atom::take(r"\\"), Some(("", Atom::NormalChar('\\'))));
        assert_eq!(
            Atom::take(r"\[asdf\]"),
            Some((r"asdf\]", Atom::NormalChar('[')))
        );
        assert_eq!(Atom::take(r"\."), Some(("", Atom::NormalChar('.'))));
        assert_eq!(Atom::take(r" "), Some(("", Atom::NormalChar(' '))));
        assert_eq!(Atom::take(r"\"), None);

        assert_eq!(Atom::take("."), Some(("", Atom::CharClass(CharClass::Dot))));
        assert_eq!(
            Atom::take(".."),
            Some((".", Atom::CharClass(CharClass::Dot)))
        );
    }

    #[test]
    fn parse_atom_brackets() {
        assert_eq!(
            Atom::take("[ab]"),
            Some((
                "",
                Atom::MatchingList(vec![
                    BracketExpressionTerm::Single('a'),
                    BracketExpressionTerm::Single('b'),
                ])
            ))
        );
        assert_eq!(
            Atom::take("[]ab]"),
            Some((
                "",
                Atom::MatchingList(vec![
                    BracketExpressionTerm::Single(']'),
                    BracketExpressionTerm::Single('a'),
                    BracketExpressionTerm::Single('b'),
                ])
            ))
        );
        assert_eq!(
            Atom::take("[]ab-]"),
            Some((
                "",
                Atom::MatchingList(vec![
                    BracketExpressionTerm::Single(']'),
                    BracketExpressionTerm::Single('a'),
                    BracketExpressionTerm::Single('b'),
                    BracketExpressionTerm::Single('-'),
                ])
            ))
        );

        assert_eq!(
            Atom::take("[]a-y]"),
            Some((
                "",
                Atom::MatchingList(vec![
                    BracketExpressionTerm::Single(']'),
                    BracketExpressionTerm::Range('a', 'y'),
                ])
            ))
        );
        assert_eq!(
            Atom::take("[]+--]"),
            Some((
                "",
                Atom::MatchingList(vec![
                    BracketExpressionTerm::Single(']'),
                    BracketExpressionTerm::Range('+', '-'),
                ])
            ))
        );

        assert_eq!(
            Atom::take("[^]a-y]"),
            Some((
                "",
                Atom::NonmatchingList(vec![
                    BracketExpressionTerm::Single(']'),
                    BracketExpressionTerm::Range('a', 'y'),
                ])
            ))
        );
        assert_eq!(
            Atom::take("[^]+--]"),
            Some((
                "",
                Atom::NonmatchingList(vec![
                    BracketExpressionTerm::Single(']'),
                    BracketExpressionTerm::Range('+', '-'),
                ])
            ))
        );
    }
}
