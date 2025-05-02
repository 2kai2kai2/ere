//! This crate provides the core functionality to the `ere` crate.

use proc_macro::TokenStream;
use quote::quote;
extern crate proc_macro;

pub mod nfa_static;
pub mod parse_tree;
pub mod pike_vm;
pub mod simplified_tree;
pub mod working_nfa;

enum RegexEngines<const N: usize> {
    NFA(nfa_static::NFAStatic<N>),
    PikeVM(pike_vm::PikeVM<N>),
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
            RegexEngines::PikeVM(pike_vm) => pike_vm.test(text),
        };
    }

    pub fn exec<'a>(&self, text: &'a str) -> Option<[Option<&'a str>; N]> {
        return match &self.0 {
            RegexEngines::NFA(nfa) => unimplemented!(),
            RegexEngines::PikeVM(pike_vm) => pike_vm.exec(text),
        };
    }
}

pub const fn __construct_pikevm_regex<const N: usize>(vm: pike_vm::PikeVM<N>) -> Regex<N> {
    return Regex(RegexEngines::PikeVM(vm));
}
pub const fn __construct_nfa_regex<const N: usize>(nfa: nfa_static::NFAStatic<N>) -> Regex<N> {
    return Regex(RegexEngines::NFA(nfa));
}

pub fn __compile_regex(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    // println!("{}", nfa.to_tikz(true));

    if true {
        let engine = pike_vm::serialize_pike_vm_token_stream(&nfa);
        return quote! { ::ere_core::__construct_pikevm_regex(#engine) }.into();
    } else {
        let engine = nfa_static::serialize_nfa_as_token_stream(&nfa);
        return quote! { ::ere_core::__construct_nfa_regex(#engine) }.into();
    };
}
