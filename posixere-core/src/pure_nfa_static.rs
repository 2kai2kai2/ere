//! Implements a version of [`crate::pure_nfa::PureNFA`] that can be serialized statically into a binary.

use crate::{
    parse_tree::{Atom, BracketExpressionTerm, CharClass},
    pure_nfa::{PureNFA, PureNFAEpsilonTransition, PureNFATransition},
};
use quote::quote;

/// A statically serializable version of [`crate::parse_tree::Atom`]
#[derive(Debug)]
pub enum StaticAtom {
    /// Includes normal char and escaped chars
    NormalChar(char),
    CharClass(CharClass),
    /// A matching bracket expression
    MatchingList(&'static [BracketExpressionTerm]),
    /// A nonmatching bracket expression
    NonmatchingList(&'static [BracketExpressionTerm]),
}
impl StaticAtom {
    pub fn check(&self, c: char) -> bool {
        return match self {
            StaticAtom::NormalChar(a) => *a == c,
            StaticAtom::CharClass(char_class) => char_class.check(c),
            StaticAtom::MatchingList(arr) => arr.into_iter().any(|b| b.check(c)),
            StaticAtom::NonmatchingList(arr) => !arr.into_iter().any(|b| b.check(c)),
        };
    }
    /// Serialize as [`Atom`], deserialize as [`StaticAtom`]
    fn serialize_as_token_stream(atom: &Atom) -> proc_macro2::TokenStream {
        return match atom {
            Atom::NormalChar(c) => quote! {
                posixere_core::pure_nfa_static::StaticAtom::NormalChar(#c)
            },
            Atom::CharClass(char_class) => quote! {
                posixere_core::pure_nfa_static::StaticAtom::CharClass(#char_class),
            },
            Atom::MatchingList(bracket_expression_terms) => {
                let terms: proc_macro2::TokenStream = bracket_expression_terms
                    .into_iter()
                    .map(|term| quote! { #term, })
                    .collect();
                quote! {{
                    static terms: &'static [posixere_core::parse_tree::BracketExpressionTerm] = &[#terms];
                    posixere_core::pure_nfa_static::StaticAtom::MatchingList(terms)
                }}
            }
            Atom::NonmatchingList(bracket_expression_terms) => {
                let terms: proc_macro2::TokenStream = bracket_expression_terms
                    .into_iter()
                    .map(|term| quote! { #term, })
                    .collect();
                quote! {{
                    static terms: &'static [posixere_core::parse_tree::BracketExpressionTerm] = &[#terms];
                    posixere_core::pure_nfa_static::StaticAtom::NonmatchingList(terms)
                }}
            }
        };
    }
}

#[derive(Debug)]
pub struct PureNFATransitionStatic {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) symbol: StaticAtom,
}
impl PureNFATransitionStatic {
    /// Only intended for internal use by macros.
    pub const fn __load(from: usize, to: usize, symbol: StaticAtom) -> PureNFATransitionStatic {
        return PureNFATransitionStatic { from, to, symbol };
    }
    fn serialize_as_token_stream(transition: &PureNFATransition) -> proc_macro2::TokenStream {
        let PureNFATransition { from, to, symbol } = transition;
        let symbol = StaticAtom::serialize_as_token_stream(symbol);
        return quote! {
            posixere_core::pure_nfa_static::PureNFATransitionStatic::__load(
                #from,
                #to,
                #symbol,
            )
        }
        .into();
    }
}

#[derive(Debug)]
pub struct PureNFAStatic {
    transitions: &'static [PureNFATransitionStatic],
    epsilons: &'static [PureNFAEpsilonTransition],
    states: usize,
}
impl PureNFAStatic {
    /// Using the classical NFA algorithm.
    pub fn check(&self, text: &str) -> bool {
        let mut list = vec![false; self.states];
        let mut new_list = vec![false; self.states];
        list[0] = true;

        // Adds all states reachable by epsilon transitions
        let propogate_epsilon = |list: &mut Vec<bool>| loop {
            let mut has_new = false;
            for t in self.epsilons {
                if list[t.from] && !list[t.to] {
                    list[t.to] = true;
                    has_new = true;
                }
            }
            if !has_new {
                break;
            }
        };
        propogate_epsilon(&mut list);

        for c in text.chars() {
            for PureNFATransitionStatic { from, to, symbol } in self.transitions {
                if list[*from] && symbol.check(c) {
                    new_list[*to] = true;
                }
            }
            let tmp = list;
            list = new_list;
            new_list = tmp;
            new_list.fill(false);
            propogate_epsilon(&mut list);
        }
        return *list.last().unwrap_or(&false);
    }

    /// Only intended for internal use by macros.
    pub const fn __load(
        transitions: &'static [PureNFATransitionStatic],
        epsilons: &'static [PureNFAEpsilonTransition],
        states: usize,
    ) -> PureNFAStatic {
        return PureNFAStatic {
            transitions,
            epsilons,
            states,
        };
    }

    /// Converts a [`PureNFA`] into a format that, when returned by a proc macro, will
    /// create the corresponding [`PureNFAStatic`].
    pub(crate) fn serialize_as_token_stream(nfa: &PureNFA) -> proc_macro2::TokenStream {
        let PureNFA {
            transitions,
            epsilons,
            states,
        } = nfa;

        let transitions_defs: proc_macro2::TokenStream = transitions
            .into_iter()
            .map(|t| {
                let t = PureNFATransitionStatic::serialize_as_token_stream(t);
                return quote! { #t, };
            })
            .collect();
        let epsilon_defs: proc_macro2::TokenStream = epsilons
            .into_iter()
            .map(|PureNFAEpsilonTransition { from, to }| {
                quote! {
                    posixere_core::pure_nfa::PureNFAEpsilonTransition::new(#from, #to),
                }
            })
            .collect();

        return quote! {{
            static transitions: &'static [posixere_core::pure_nfa_static::PureNFATransitionStatic] = &[#transitions_defs];
            static epsilons: &'static [posixere_core::pure_nfa::PureNFAEpsilonTransition] = &[#epsilon_defs];

            static static_nfa: posixere_core::pure_nfa_static::PureNFAStatic = posixere_core::pure_nfa_static::PureNFAStatic::__load(transitions, epsilons, #states);
            &static_nfa
        }};
    }
}
