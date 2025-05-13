//! Implements a version of [`crate::working_nfa::WorkingNFA`] that can be serialized statically into a binary.

use std::fmt::Write as _;

use crate::{
    parse_tree::{Atom, BracketExpressionTerm, CharClass},
    working_nfa::{EpsilonTransition, EpsilonType, WorkingNFA, WorkingState, WorkingTransition},
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
impl std::fmt::Display for AtomStatic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match self {
            AtomStatic::NormalChar(c) if crate::parse_tree::is_escapable_character(*c) => {
                write!(f, "\\{c}")
            }
            AtomStatic::NormalChar(c) => f.write_char(*c),
            AtomStatic::CharClass(c) => c.fmt(f),
            AtomStatic::MatchingList(vec) => {
                f.write_char('[')?;
                for term in *vec {
                    write!(f, "{term}")?;
                }
                f.write_char(']')
            }
            AtomStatic::NonmatchingList(vec) => {
                f.write_str("[^")?;
                for term in *vec {
                    write!(f, "{term}")?;
                }
                f.write_char(']')
            }
        };
    }
}

#[derive(Debug)]
pub struct NFATransitionStatic {
    pub(crate) to: usize,
    pub(crate) symbol: AtomStatic,
}
impl NFATransitionStatic {
    /// Only intended for internal use by macros.
    pub const fn __load(to: usize, symbol: AtomStatic) -> NFATransitionStatic {
        return NFATransitionStatic { to, symbol };
    }
    fn serialize_as_token_stream(transition: &WorkingTransition) -> proc_macro2::TokenStream {
        let WorkingTransition { to, symbol } = transition;
        let symbol = AtomStatic::serialize_as_token_stream(symbol);
        return quote! {
            ere_core::nfa_static::NFATransitionStatic::__load(
                #to,
                #symbol,
            )
        }
        .into();
    }
}
impl std::fmt::Display for NFATransitionStatic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return write!(f, "-{}> {}", self.symbol, self.to);
    }
}

#[derive(Debug, Clone)]
pub struct NFAStateStatic {
    pub(crate) transitions: &'static [NFATransitionStatic],
    pub(crate) epsilons: &'static [EpsilonTransition],
}
impl NFAStateStatic {
    pub const fn __load(
        transitions: &'static [NFATransitionStatic],
        epsilons: &'static [EpsilonTransition],
    ) -> NFAStateStatic {
        return NFAStateStatic {
            transitions,
            epsilons,
        };
    }
    fn serialize_as_token_stream(state: &WorkingState) -> proc_macro2::TokenStream {
        let WorkingState {
            transitions,
            epsilons,
        } = state;
        let transitions: proc_macro2::TokenStream = transitions
            .iter()
            .map(|t| {
                let t = NFATransitionStatic::serialize_as_token_stream(t);
                return quote! { #t, };
            })
            .collect();
        let epsilons: proc_macro2::TokenStream = epsilons.iter().map(|e| quote! { #e, }).collect();
        return quote! {{
            const transitions: &'static [::ere_core::nfa_static::NFATransitionStatic] = &[#transitions];
            const epsilons: &'static [::ere_core::working_nfa::EpsilonTransition] = &[#epsilons];
            ::ere_core::nfa_static::NFAStateStatic::__load(
                transitions,
                epsilons,
            )
        }}
        .into();
    }
}

/// The statically allocated representation of an NFA
///
/// Generic `N` represents the number of capture groups (will always be at least 1)
#[derive(Debug)]
pub struct NFAStatic<const N: usize> {
    states: &'static [NFAStateStatic],
}
impl<const N: usize> NFAStatic<N> {
    /// Using the classical NFA algorithm.
    pub fn test(&self, text: &str) -> bool {
        let mut list = vec![false; self.states.len()];
        let mut new_list = vec![false; self.states.len()];
        list[0] = true;

        // Adds all states reachable by epsilon transitions
        let propogate_epsilon = |list: &mut Vec<bool>, idx: usize| {
            let mut stack: Vec<usize> = list
                .iter()
                .enumerate()
                .filter_map(|(i, set)| set.then_some(i))
                .collect();

            while let Some(from) = stack.pop() {
                for EpsilonTransition { to, special } in self.states[from].epsilons {
                    if list[from]
                        && !list[*to]
                        && (match special {
                            EpsilonType::StartAnchor => idx == 0,
                            EpsilonType::EndAnchor => idx == text.len(),
                            _ => true,
                        })
                    {
                        stack.push(*to);
                        list[*to] = true;
                    }
                }
            }
        };

        for (i, c) in text.char_indices() {
            propogate_epsilon(&mut list, i);
            for (from, state) in self.states.iter().enumerate() {
                if !list[from] {
                    continue;
                }
                for NFATransitionStatic { to, symbol } in state.transitions {
                    if list[from] && symbol.check(c) {
                        new_list[*to] = true;
                    }
                }
            }
            list.copy_from_slice(&new_list);
            new_list.fill(false);
        }
        propogate_epsilon(&mut list, text.len());
        return *list.last().unwrap_or(&false);
    }

    /// Only intended for internal use by macros.
    pub const fn __load(states: &'static [NFAStateStatic]) -> NFAStatic<N> {
        return NFAStatic { states };
    }
}

impl<const N: usize> std::fmt::Display for NFAStatic<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, state) in self.states.iter().enumerate() {
            writeln!(f, "State {i}")?;
            for epsilon in state.epsilons {
                writeln!(f, "  {epsilon}")?;
            }
            for transition in state.transitions {
                writeln!(f, "  {transition}")?;
            }
        }
        return Ok(());
    }
}

/// Converts a [`WorkingNFA`] into a format that, when returned by a proc macro, will
/// create the corresponding [`NFAStatic`].
pub(crate) fn serialize_nfa_as_token_stream(nfa: &WorkingNFA) -> proc_macro2::TokenStream {
    let WorkingNFA { states } = nfa;

    let capture_groups = nfa.num_capture_groups();
    let state_defs: proc_macro2::TokenStream = states
        .iter()
        .map(|state| {
            let state = NFAStateStatic::serialize_as_token_stream(state);
            return quote! { #state, };
        })
        .collect();
    return quote! {{
        const states: &'static [ere_core::nfa_static::NFAStateStatic] = &[#state_defs];

        ere_core::nfa_static::NFAStatic::<#capture_groups>::__load(states)
    }};
}
