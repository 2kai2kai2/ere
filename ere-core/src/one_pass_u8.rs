//! This implements an engine for [one-pass](https://swtch.com/~rsc/regexp/regexp3.html#:~:text=Use%20a%20one%2Dpass%20NFA%20if%20possible) regexes.

use crate::working_u8_nfa::U8NFA;
use proc_macro2::TokenStream;
use quote::quote;

pub struct U8OnePass<const N: usize> {
    test_fn: fn(&str) -> bool,
    exec_fn: for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
}
impl<const N: usize> U8OnePass<N> {
    pub fn test(&self, text: &str) -> bool {
        return (self.test_fn)(text);
    }
    pub fn exec<'a>(&self, text: &'a str) -> Option<[Option<&'a str>; N]> {
        return (self.exec_fn)(text);
    }
}

/// Only intended for internal use by macros.
pub const fn __load_u8onepass<const N: usize>(
    test_fn: fn(&str) -> bool,
    exec_fn: for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
) -> U8OnePass<N> {
    return U8OnePass { test_fn, exec_fn };
}

fn vmstate_label(idx: usize) -> proc_macro2::Ident {
    let label = format!("State{idx}");
    return proc_macro2::Ident::new(&label, proc_macro2::Span::call_site());
}

/// We only need to retain states with outgoing symbol transitions
/// As well as the initial and accept states
fn compute_excluded_states(nfa: &U8NFA) -> Vec<bool> {
    let mut out = vec![true; nfa.states.len()];
    out[0] = false;
    for state in &nfa.states {
        for t in &state.transitions {
            out[t.to] = false;
        }
    }

    return out;
}

#[derive(Clone, PartialEq, Eq)]
struct ThreadUpdates {
    pub state: usize,
    pub update_captures: Vec<(bool, bool)>,
    pub start_only: bool,
    pub end_only: bool,
}
impl ThreadUpdates {
    pub fn new(state: usize, num_captures: usize) -> ThreadUpdates {
        return ThreadUpdates {
            state,
            update_captures: vec![(false, false); num_captures],
            start_only: false,
            end_only: false,
        };
    }
}

/// If the NFA is [one-pass](https://swtch.com/~rsc/regexp/regexp3.html#:~:text=Use%20a%20one%2Dpass%20NFA%20if%20possible), this function will serialize it. Otherwise, returns `None`.
///
/// An NFA/regex is one-pass if there will only ever be one thread active at a time for the NFA.
/// This means each state will only ever end up in at most one next state given an input symbol
/// (has one outgoing symbol transition, including those reachable by epsilon transitions).
///
/// A [`U8NFA`] should be one-pass whenever its corresponding [`WorkingNFA`](`crate::working_nfa::WorkingNFA`) is.
/// There are some optimizations that may make a non-one-pass NFA into a one-pass one:
/// For example, `^(a|a|b)$` into `^(a|b)$`
pub(crate) fn serialize_one_pass_token_stream(nfa: &U8NFA) -> Option<TokenStream> {
    let num_captures = nfa.num_capture_groups();
    let mut symbol_transitions = vec![Vec::new(); nfa.states.len()];
    // states with epsilons into the accept state
    // because the other transitions do any special behavior before the symbol
    // but the space after the last character needs its own
    let mut accept_transitions = vec![Vec::new(); nfa.states.len()];
    for (state_idx, _) in nfa.states.iter().enumerate() {
        let mut stack = vec![ThreadUpdates::new(state_idx, num_captures)];
        let mut reached = vec![ThreadUpdates::new(state_idx, num_captures)];
        while let Some(thread) = stack.pop() {
            if thread.state + 1 == nfa.states.len() {
                accept_transitions[state_idx].push(thread.clone());
            }
            for ep in &nfa.states[thread.state].epsilons {
                let mut new_thread = thread.clone();
                new_thread.state = ep.to;
                match ep.special {
                    crate::working_nfa::EpsilonType::None => (),
                    crate::working_nfa::EpsilonType::StartAnchor => new_thread.start_only = true,
                    crate::working_nfa::EpsilonType::EndAnchor => new_thread.end_only = true,
                    crate::working_nfa::EpsilonType::StartCapture(c) => {
                        new_thread.update_captures[c].0 = true
                    }
                    crate::working_nfa::EpsilonType::EndCapture(c) => {
                        new_thread.update_captures[c].1 = true
                    }
                }
                if !reached.contains(&new_thread) {
                    reached.push(new_thread.clone());
                    stack.push(new_thread);
                }
            }
            for tr in &nfa.states[thread.state].transitions {
                let new_transition = (
                    tr.symbol.0.clone(),
                    ThreadUpdates {
                        state: tr.to,
                        update_captures: thread.update_captures.clone(),
                        start_only: thread.start_only,
                        end_only: thread.end_only,
                    },
                );
                // we can skip if it's exactly the same, otherwise it's a conflict.
                if !symbol_transitions[state_idx].contains(&new_transition) {
                    symbol_transitions[state_idx].push(new_transition);
                }
            }
        }
    }
    for state_transitions in &mut symbol_transitions {
        state_transitions.sort_by_key(|(range, _)| *range.start());
        let overlap = !state_transitions.windows(2).all(|ranges| {
            if let &[(a, _), (b, _)] = &ranges {
                return a.end() < b.start();
            } else {
                unreachable!("Vec::windows does not use const generics so we have to do this.");
            }
        });
        if overlap {
            return None;
        }
    }

    // == codegen ==
    let U8NFA { states, .. } = nfa;
    let excluded_states = compute_excluded_states(nfa);
    let enum_states: proc_macro2::TokenStream = std::iter::IntoIterator::into_iter(0..states.len())
        .filter(|i| !excluded_states[*i])
        .map(|i| {
            let label = vmstate_label(i);
            return quote! { #label, };
        })
        .collect();

    let make_test_match_statements = |state_idx: usize| -> TokenStream {
        let mut out = TokenStream::new();
        let this_state = vmstate_label(state_idx);
        for (range, thread) in &symbol_transitions[state_idx] {
            if excluded_states[thread.state] {
                continue; // no point in going here, we included the relevant propogated states already
            }
            if thread.end_only {
                continue; // only allowed in final match
            }
            let range_start = *range.start();
            let range_end = *range.end();
            let conditions = if thread.start_only {
                quote! {if i == 0}
            } else {
                TokenStream::new()
            };
            let to = vmstate_label(thread.state);
            out.extend(quote! {
                (VMStates::#this_state, #range_start..=#range_end) #conditions => {
                    state = VMStates::#to;
                }
            });
        }
        return out;
    };
    let make_test_match_statements_final = |state_idx: usize| -> TokenStream {
        let mut out = TokenStream::new();
        let this_state = vmstate_label(state_idx);
        // Off the top of my head I think it should only ever have at most one, but idk
        for thread in &accept_transitions[state_idx] {
            let conditions = if thread.start_only {
                quote! {if i == 0}
            } else {
                TokenStream::new()
            };
            out.extend(quote! {
                VMStates::#this_state #conditions => true,
            });
        }
        return out;
    };

    let make_exec_match_statements = |state_idx: usize| -> TokenStream {
        let mut out = TokenStream::new();
        let this_state = vmstate_label(state_idx);
        for (range, thread) in &symbol_transitions[state_idx] {
            if excluded_states[thread.state] {
                continue; // no point in going here, we included the relevant propogated states already
            }
            if thread.end_only {
                continue; // only allowed in final match
            }
            let range_start = *range.start();
            let range_end = *range.end();
            let conditions = if thread.start_only {
                quote! {if i == 0}
            } else {
                TokenStream::new()
            };
            let mut capture_updates = TokenStream::new();
            for (group_num, (start, end)) in thread.update_captures.iter().enumerate() {
                if *start {
                    capture_updates.extend(quote! {
                        captures[#group_num].0 = i;
                    });
                }
                if *end {
                    capture_updates.extend(quote! {
                        captures[#group_num].1 = i;
                    });
                }
            }
            let to = vmstate_label(thread.state);
            out.extend(quote! {
                (VMStates::#this_state, #range_start..=#range_end) #conditions => {
                    #capture_updates
                    state = VMStates::#to;
                }
            });
        }

        return out;
    };
    let make_exec_match_statements_final = |state_idx: usize| -> TokenStream {
        let mut out = TokenStream::new();
        let this_state = vmstate_label(state_idx);
        for thread in &accept_transitions[state_idx] {
            // end only is always the case here
            let conditions = if thread.start_only {
                quote! {if i == 0}
            } else {
                TokenStream::new()
            };
            let mut capture_updates = TokenStream::new();
            for (group_num, (start, end)) in thread.update_captures.iter().enumerate() {
                if *start {
                    capture_updates.extend(quote! {
                        captures[#group_num].0 = text.len();
                    });
                }
                if *end {
                    capture_updates.extend(quote! {
                        captures[#group_num].1 = text.len();
                    });
                }
            }
            out.extend(quote! {
                VMStates::#this_state #conditions => {
                    #capture_updates
                }
            });
        }

        return out;
    };

    let test_match_statements: TokenStream = (0..states.len())
        .filter(|state_idx| !excluded_states[*state_idx])
        .map(make_test_match_statements)
        .collect();
    let test_match_statements_final: TokenStream = (0..states.len())
        .filter(|state_idx| !excluded_states[*state_idx])
        .map(make_test_match_statements_final)
        .collect();

    let exec_match_statements: TokenStream = (0..states.len())
        .filter(|state_idx| !excluded_states[*state_idx])
        .map(make_exec_match_statements)
        .collect();
    let exec_match_statements_final: TokenStream = (0..states.len())
        .filter(|state_idx| !excluded_states[*state_idx])
        .map(make_exec_match_statements_final)
        .collect();

    return quote! {{
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum VMStates {
            #enum_states
        }

        fn test(text: &str) -> bool {
            let mut state: VMStates = VMStates::State0;

            for (i, c) in text.bytes().enumerate() {
                match (state, c) {
                    #test_match_statements
                    _ => return false,
                }
            }
            return match state {
                #test_match_statements_final
                _ => false,
            };
        }
        fn exec<'a>(text: &'a str) -> Option<[Option<&'a str>; #num_captures]> {
            let mut state: VMStates = VMStates::State0;
            let mut captures: [(usize, usize); #num_captures] = [(usize::MAX, usize::MAX); #num_captures];

            for (i, c) in text.bytes().enumerate() {
                match (state, c) {
                    #exec_match_statements
                    _ => return ::core::option::Option::None,
                }
            }
            match state {
                #exec_match_statements_final
                _ => return ::core::option::Option::None,
            }

            let mut capture_strs = [::core::option::Option::None; #num_captures];
            for (i, (start, end)) in captures.into_iter().enumerate() {
                if start != usize::MAX {
                    assert_ne!(end, usize::MAX);
                    // assert!(start <= end);
                    capture_strs[i] = text.get(start..end);
                    assert!(capture_strs[i].is_some());
                } else {
                    assert_eq!(end, usize::MAX);
                }
            }
            return Some(capture_strs);
        }

        ::ere_core::one_pass_u8::__load_u8onepass(test, exec)
    }}.into();
}
