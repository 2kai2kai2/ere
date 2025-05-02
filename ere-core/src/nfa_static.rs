//! Implements a version of [`crate::working_nfa::WorkingNFA`] that can be serialized statically into a binary.

use crate::{
    parse_tree::{Atom, BracketExpressionTerm, CharClass},
    working_nfa::{EpsilonTransition, EpsilonType, WorkingNFA, WorkingTransition},
};
use quote::quote;

/// A statically serializable version of [`crate::parse_tree::Atom`]
#[derive(Debug)]
pub enum AtomStatic {
    /// Includes normal char and escaped chars
    NormalChar(char),
    CharClass(CharClass),
    /// A matching bracket expression
    MatchingList(&'static [BracketExpressionTerm]),
    /// A nonmatching bracket expression
    NonmatchingList(&'static [BracketExpressionTerm]),
}
impl AtomStatic {
    pub fn check(&self, c: char) -> bool {
        return match self {
            AtomStatic::NormalChar(a) => *a == c,
            AtomStatic::CharClass(char_class) => char_class.check(c),
            AtomStatic::MatchingList(arr) => arr.into_iter().any(|b| b.check(c)),
            AtomStatic::NonmatchingList(arr) => !arr.into_iter().any(|b| b.check(c)),
        };
    }
    /// Serialize as [`Atom`], deserialize as [`StaticAtom`]
    pub(crate) fn serialize_as_token_stream(atom: &Atom) -> proc_macro2::TokenStream {
        return match atom {
            Atom::NormalChar(c) => quote! {
                ere_core::nfa_static::AtomStatic::NormalChar(#c)
            },
            Atom::CharClass(char_class) => quote! {
                ere_core::nfa_static::AtomStatic::CharClass(#char_class),
            },
            Atom::MatchingList(bracket_expression_terms) => {
                let terms: proc_macro2::TokenStream = bracket_expression_terms
                    .into_iter()
                    .map(|term| quote! { #term, })
                    .collect();
                quote! {{
                    const terms: &'static [ere_core::parse_tree::BracketExpressionTerm] = &[#terms];
                    ere_core::nfa_static::AtomStatic::MatchingList(terms)
                }}
            }
            Atom::NonmatchingList(bracket_expression_terms) => {
                let terms: proc_macro2::TokenStream = bracket_expression_terms
                    .into_iter()
                    .map(|term| quote! { #term, })
                    .collect();
                quote! {{
                    const terms: &'static [ere_core::parse_tree::BracketExpressionTerm] = &[#terms];
                    ere_core::nfa_static::AtomStatic::NonmatchingList(terms)
                }}
            }
        };
    }
}

#[derive(Debug)]
pub struct NFATransitionStatic {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) symbol: AtomStatic,
}
impl NFATransitionStatic {
    /// Only intended for internal use by macros.
    pub const fn __load(from: usize, to: usize, symbol: AtomStatic) -> NFATransitionStatic {
        return NFATransitionStatic { from, to, symbol };
    }
    fn serialize_as_token_stream(transition: &WorkingTransition) -> proc_macro2::TokenStream {
        let WorkingTransition { from, to, symbol } = transition;
        let symbol = AtomStatic::serialize_as_token_stream(symbol);
        return quote! {
            ere_core::nfa_static::NFATransitionStatic::__load(
                #from,
                #to,
                #symbol,
            )
        }
        .into();
    }
}

/// The statically allocated representation of an NFA
///
/// Generic `N` represents the number of capture groups (will always be at least 1)
#[derive(Debug)]
pub struct NFAStatic<const N: usize> {
    transitions: &'static [NFATransitionStatic],
    epsilons: &'static [EpsilonTransition],
    states: usize,
}
impl<const N: usize> NFAStatic<N> {
    /// Using the classical NFA algorithm.
    pub fn test(&self, text: &str) -> bool {
        let mut list = vec![false; self.states];
        let mut new_list = vec![false; self.states];
        list[0] = true;

        // Adds all states reachable by epsilon transitions
        let propogate_epsilon = |list: &mut Vec<bool>, idx: usize| loop {
            let mut has_new = false;
            for EpsilonTransition { from, to, special } in self.epsilons {
                if list[*from]
                    && !list[*to]
                    && (match special {
                        EpsilonType::StartAnchor => idx == 0,
                        EpsilonType::EndAnchor => idx == text.len(),
                        _ => true,
                    })
                {
                    list[*to] = true;
                    has_new = true;
                }
            }
            if !has_new {
                break;
            }
        };

        for (i, c) in text.char_indices() {
            propogate_epsilon(&mut list, i);
            for NFATransitionStatic { from, to, symbol } in self.transitions {
                if list[*from] && symbol.check(c) {
                    new_list[*to] = true;
                }
            }
            let tmp = list;
            list = new_list;
            new_list = tmp;
            new_list.fill(false);
        }
        propogate_epsilon(&mut list, text.len());
        return *list.last().unwrap_or(&false);
    }

    /// Only intended for internal use by macros.
    pub const fn __load(
        transitions: &'static [NFATransitionStatic],
        epsilons: &'static [EpsilonTransition],
        states: usize,
    ) -> NFAStatic<N> {
        return NFAStatic {
            transitions,
            epsilons,
            states,
        };
    }
}

/// Converts a [`WorkingNFA`] into a format that, when returned by a proc macro, will
/// create the corresponding [`NFAStatic`].
pub(crate) fn serialize_nfa_as_token_stream(nfa: &WorkingNFA) -> proc_macro2::TokenStream {
    let WorkingNFA {
        transitions,
        epsilons,
        states,
    } = nfa;

    let capture_groups = nfa.num_capture_groups();
    let transitions_defs: proc_macro2::TokenStream = transitions
        .into_iter()
        .map(|t| {
            let t = NFATransitionStatic::serialize_as_token_stream(t);
            return quote! { #t, };
        })
        .collect();
    let epsilon_defs: proc_macro2::TokenStream =
        epsilons.into_iter().map(|e| quote! { #e, }).collect();

    return quote! {{
        const transitions: &'static [ere_core::nfa_static::NFATransitionStatic] = &[#transitions_defs];
        const epsilons: &'static [ere_core::working_nfa::EpsilonTransition] = &[#epsilon_defs];

        ere_core::nfa_static::NFAStatic::<#capture_groups>::__load(transitions, epsilons, #states)
    }};
}
