//! Implements a version of [`crate::working_nfa::WorkingNFA`] that can be serialized statically into a binary.

use crate::{
    parse_tree::{Atom, BracketExpressionTerm, CharClass},
    working_nfa::{WorkingEpsilonTransition, WorkingEpsilonType, WorkingNFA, WorkingTransition},
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
    fn serialize_as_token_stream(atom: &Atom) -> proc_macro2::TokenStream {
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

/// An epsilon transition for the [`WorkingNFA`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EpsilonTypeStatic {
    None,
    StartCapture(usize),
    EndCapture(usize),
}
impl EpsilonTypeStatic {
    fn serialize_as_token_stream(epsilon_type: &WorkingEpsilonType) -> proc_macro2::TokenStream {
        return match epsilon_type {
            WorkingEpsilonType::None => quote! { ::ere_core::nfa_static::EpsilonTypeStatic::None },
            WorkingEpsilonType::StartAnchor | WorkingEpsilonType::EndAnchor => quote! {
                compile_error!("StaticNFA does not support anchors--
                these should have been removed already while it was a WorkingNFA.
                This may be an internal ere crate bug.");
            },
            WorkingEpsilonType::StartCapture(group_num) => quote! {
                ::ere_core::nfa_static::EpsilonTypeStatic::StartCapture(#group_num)
            },
            WorkingEpsilonType::EndCapture(group_num) => quote! {
                ::ere_core::nfa_static::EpsilonTypeStatic::EndCapture(#group_num)
            },
        };
    }
}

/// An epsilon transition for the [`WorkingNFA`]
#[derive(Debug, Clone, Copy)]
pub struct EpsilonTransitionStatic {
    pub(crate) from: usize,
    pub(crate) to: usize,
    pub(crate) special: EpsilonTypeStatic,
}
impl EpsilonTransitionStatic {
    /// Only intended for internal use by macros.
    pub const fn __load(
        from: usize,
        to: usize,
        special: EpsilonTypeStatic,
    ) -> EpsilonTransitionStatic {
        return EpsilonTransitionStatic { from, to, special };
    }
    fn serialize_as_token_stream(
        transition: &WorkingEpsilonTransition,
    ) -> proc_macro2::TokenStream {
        let WorkingEpsilonTransition { from, to, special } = transition;
        let special = EpsilonTypeStatic::serialize_as_token_stream(special);
        return quote! {
            ere_core::nfa_static::EpsilonTransitionStatic::__load(
                #from,
                #to,
                #special,
            )
        }
        .into();
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

#[derive(Debug)]
pub struct NFAStatic {
    transitions: &'static [NFATransitionStatic],
    epsilons: &'static [EpsilonTransitionStatic],
    states: usize,
}
impl NFAStatic {
    /// Using the classical NFA algorithm.
    pub fn test(&self, text: &str) -> bool {
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
            for NFATransitionStatic { from, to, symbol } in self.transitions {
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
        transitions: &'static [NFATransitionStatic],
        epsilons: &'static [EpsilonTransitionStatic],
        states: usize,
    ) -> NFAStatic {
        return NFAStatic {
            transitions,
            epsilons,
            states,
        };
    }

    /// Converts a [`WorkingNFA`] into a format that, when returned by a proc macro, will
    /// create the corresponding [`NFAStatic`].
    ///
    /// Warning: `nfa` should have anchors removed already.
    pub(crate) fn serialize_as_token_stream(nfa: &WorkingNFA) -> proc_macro2::TokenStream {
        let WorkingNFA {
            transitions,
            epsilons,
            states,
        } = nfa;

        let transitions_defs: proc_macro2::TokenStream = transitions
            .into_iter()
            .map(|t| {
                let t = NFATransitionStatic::serialize_as_token_stream(t);
                return quote! { #t, };
            })
            .collect();
        let epsilon_defs: proc_macro2::TokenStream = epsilons
            .into_iter()
            .map(|e| {
                let e = EpsilonTransitionStatic::serialize_as_token_stream(e);
                return quote! { #e, };
            })
            .collect();

        return quote! {{
            const transitions: &'static [ere_core::nfa_static::NFATransitionStatic] = &[#transitions_defs];
            const epsilons: &'static [ere_core::nfa_static::EpsilonTransitionStatic] = &[#epsilon_defs];

            ere_core::nfa_static::NFAStatic::__load(transitions, epsilons, #states)
        }};
    }
}
