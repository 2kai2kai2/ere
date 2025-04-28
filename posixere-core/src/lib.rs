//! This crate provides the core functionality to the `posixere` crate.

use proc_macro::TokenStream;
use quote::quote;
extern crate proc_macro;

pub mod parse_tree;
pub mod pure_nfa;
pub mod pure_nfa_static;
pub mod simplified_tree;

pub fn __pure_regex_matcher(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = match pure_nfa::PureNFA::new(&tree) {
        Ok(nfa) => nfa,
        Err(pure_nfa::PureNFAError::UnexpectedAnchor) => {
            return quote! {
                compile_error!("Pure NFA does not support anchors.");
            }
            .into();
        }
    };

    return pure_nfa_static::PureNFAStatic::serialize_as_token_stream(&nfa).into();
}
