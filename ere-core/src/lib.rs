//! This crate provides the core functionality to the `ere` crate.

use proc_macro::TokenStream;
use quote::quote;
extern crate proc_macro;

pub mod nfa_static;
pub mod parse_tree;
pub mod simplified_tree;
pub mod working_nfa;

enum RegexEngines<const N: usize> {
    NFA(nfa_static::NFAStatic<N>),
}

/// A regular expression (specifically, a [POSIX ERE](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX-Extended_Regular_Expressions)).
///
/// Internally, this may contain one of several engines depending on the expression.
///
/// The const generic `N` represents the number of capture groups (including capture group 0 which is the entire expression).
/// It defaults to `1` (for just capture group 0), but you will need to specify it in the type for expressions with more capture groups.
pub struct Regex<const N: usize = 1>(RegexEngines<N>);
impl<const N: usize> Regex<N> {
    /// Returns whether or not the text is matched by the regular expression.
    pub fn test(&self, text: &str) -> bool {
        return match &self.0 {
            RegexEngines::NFA(nfa) => nfa.test(text),
        };
    }
}

pub const fn __construct_nfa_regex<const N: usize>(nfa: nfa_static::NFAStatic<N>) -> Regex<N> {
    return Regex(RegexEngines::NFA(nfa));
}

pub fn __compile_regex(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);

    let serialized_nfa = nfa_static::serialize_nfa_as_token_stream(&nfa);

    return quote! {
        ::ere_core::__construct_nfa_regex(#serialized_nfa)
    }
    .into();
}
