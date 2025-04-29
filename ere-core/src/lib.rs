//! This crate provides the core functionality to the `ere` crate.

use proc_macro::TokenStream;
use quote::quote;
extern crate proc_macro;

pub mod nfa_static;
pub mod parse_tree;
pub mod simplified_tree;
pub mod working_nfa;

enum RegexEngines {
    NFA(nfa_static::NFAStatic),
}

/// A regular expression (specifically, a [POSIX ERE](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX-Extended_Regular_Expressions)).
///
/// Internally, this may contain one of several engines depending on the expression.
pub struct Regex(RegexEngines);
impl Regex {
    /// Returns whether or not the text is matched by the regular expression.
    pub fn test(&self, text: &str) -> bool {
        return match &self.0 {
            RegexEngines::NFA(nfa) => nfa.test(text),
        };
    }
}

pub const fn __construct_nfa_regex(nfa: nfa_static::NFAStatic) -> Regex {
    return Regex(RegexEngines::NFA(nfa));
}

pub fn __compile_regex(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);

    let serialized_nfa = nfa_static::NFAStatic::serialize_as_token_stream(&nfa);

    return quote! {
        ::ere_core::__construct_nfa_regex(#serialized_nfa)
    }
    .into();
}
