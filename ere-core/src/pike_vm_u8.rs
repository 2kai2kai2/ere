//! Implements a Pike VM-like regex engine for `u8`s.
//!
//! Not exactly the PikeVM, but close enough that I am naming it that.
//! It works similarly, except that since we are building at compile-time, there are benefits from inlining splitting.
//!
//! Due to the optimizations done earlier for the [`U8NFA`], we also have a slightly different NFA structure.
//! Currently we flatten all epsilon transitions for the VM so that epsilon transitions are at most a single step between symbols.
//! I'll have to review to ensure we avoid this causing large binary size overhead,
//! but it should be worst-case `O(n^2)` in the number of states, and far fewer on average.

use crate::{
    working_nfa::EpsilonType,
    working_u8_nfa::{U8Transition, U8NFA},
};
use quote::{quote, ToTokens, TokenStreamExt as _};
use std::fmt::Write;

#[derive(Clone)]
pub struct U8PikeVMThread<const N: usize, S: Send + Sync + Copy + Eq> {
    pub state: S,
    pub captures: [(usize, usize); N],
}
impl<const N: usize, S: Send + Sync + Copy + Eq + std::fmt::Debug> std::fmt::Debug
    for U8PikeVMThread<N, S>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        struct CapturesDebug<'a, const N: usize>(&'a [(usize, usize); N]);
        impl<'a, const N: usize> std::fmt::Debug for CapturesDebug<'a, N> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_char('[')?;
                for (i, endpoints) in self.0.iter().enumerate() {
                    if i != 0 {
                        f.write_str(", ")?;
                    }
                    match endpoints {
                        (usize::MAX, usize::MAX) => f.write_str("(_, _)")?,
                        (start, usize::MAX) => write!(f, "({start}, _)")?,
                        (usize::MAX, end) => write!(f, "(_, {end})")?,
                        (start, end) => write!(f, "({start}, {end})")?,
                    }
                }
                return f.write_char(']');
            }
        }
        return f
            .debug_struct("PikeVMThread")
            .field("state", &self.state)
            .field("captures", &CapturesDebug(&self.captures))
            .finish();
    }
}

/// The NFA and some precomputed data to go with it.
///
/// Helps avoid recomputing the same data all over
/// and also packages data together for convenience.
struct CachedNFA<'a> {
    nfa: &'a U8NFA,
    excluded_states: Vec<bool>,
    capture_groups: usize,
}
impl<'a> CachedNFA<'a> {
    fn new(nfa: &'a U8NFA) -> CachedNFA<'a> {
        let excluded_states = compute_excluded_states(nfa);
        assert_eq!(nfa.states.len(), excluded_states.len());
        let capture_groups = nfa.num_capture_groups();
        return CachedNFA {
            nfa,
            excluded_states,
            capture_groups,
        };
    }
}

/// Since we are shortcutting the epsilon transitions, we can skip printing
/// states that have only epsilon transitions and are not the start/end states
fn compute_excluded_states(nfa: &U8NFA) -> Vec<bool> {
    let mut out = vec![true; nfa.states.len()];
    out[0] = false;
    out[nfa.states.len() - 1] = false;
    for (from, state) in nfa.states.iter().enumerate() {
        for t in &state.transitions {
            out[from] = false;
            out[t.to] = false;
        }
    }

    return out;
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ImplVMStateLabel(usize);
impl ToTokens for ImplVMStateLabel {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        let ImplVMStateLabel(idx) = self;
        let label = format!("State{idx}");
        tokens.append(proc_macro2::Ident::new(
            &label,
            proc_macro2::Span::call_site(),
        ));
    }
}

mod impl_test {
    use quote::ToTokens;

    use super::*;

    /// Implements symbol transitions for a single state
    struct ImplTransitionStateSymbol<'a> {
        transition: &'a U8Transition,
    }
    impl<'a> ToTokens for ImplTransitionStateSymbol<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let &ImplTransitionStateSymbol { transition } = self;
            let U8Transition { symbol, to } = transition;
            let start = symbol.start();
            let end = symbol.end();
            tokens.extend(quote! {{
                if #start <= c && c <= #end {
                    new_list[#to] = true;
                }
            }});
        }
    }

    /// Assumes the `VMStates` enum is already created locally in the token stream
    ///
    /// Creates the function `transition_symbols_test` for running symbol transitions on the pike VM
    ///
    /// ```ignore
    /// fn transition_symbols_test(
    ///     list: &[bool],
    ///     new_list: &mut [bool],
    ///     c: u8,
    /// ) {
    ///     // ...
    /// }
    /// ```
    pub(super) struct TransitionSymbols<'a>(pub &'a CachedNFA<'a>);
    impl<'a> ToTokens for TransitionSymbols<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let TransitionSymbols(nfa) = self;
            let CachedNFA {
                nfa,
                excluded_states,
                ..
            } = nfa;

            let transition_symbols_defs_test = nfa
                .states
                .iter()
                .enumerate()
                .filter(|(i, _)| !excluded_states[*i])
                .map(|(i, state)| {
                    let state_transitions = state
                        .transitions
                        .iter()
                        .map(|t| ImplTransitionStateSymbol { transition: t });

                    return quote! {
                        if list[#i] {
                            #(#state_transitions)*
                        }
                    };
                });

            tokens.extend(quote! {
                fn transition_symbols_test(
                    list: &[bool],
                    new_list: &mut [bool],
                    c: u8,
                ) {
                    #(#transition_symbols_defs_test)*
                }
            });
        }
    }

    /// Implements epsilon transitions for a single state
    ///
    /// Becomes:
    /// ```ignore
    /// if list[#from_state] {
    ///     // ...
    /// }
    /// ```
    pub(super) struct ImplTransitionStateEpsilon<'a> {
        pub(super) from_state: ImplVMStateLabel,
        pub(super) thread_updates: &'a [ThreadUpdates],
    }
    impl<'a> ToTokens for ImplTransitionStateEpsilon<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let &ImplTransitionStateEpsilon {
                from_state,
                thread_updates,
            } = self;
            let ImplVMStateLabel(from_state) = from_state;

            // Write epsilon-propogation of threads to the token stream for test
            let start_end_threads = thread_updates
                .iter()
                .filter(|t| t.start_only && t.end_only)
                .map(ThreadUpdates::serialize_thread_update_test);
            let start_threads = thread_updates
                .iter()
                .filter(|t| t.start_only && !t.end_only)
                .map(ThreadUpdates::serialize_thread_update_test);
            let end_threads = thread_updates
                .iter()
                .filter(|t| !t.start_only && t.end_only)
                .map(ThreadUpdates::serialize_thread_update_test);
            let normal_threads = thread_updates
                .iter()
                .filter(|t| !t.start_only && !t.end_only)
                .map(ThreadUpdates::serialize_thread_update_test);

            tokens.extend(quote! {
                if list[#from_state] {
                    if is_start && is_end {
                        #(#start_end_threads)*
                    }
                    if is_start {
                        #(#start_threads)*
                    }
                    if is_end {
                        #(#end_threads)*
                    }
                    #(#normal_threads)*
                }
            });
        }
    }

    /// Implements a function that runs all epsilon transitions for all threads.
    ///
    /// ```ignore
    /// fn transition_epsilons_test(
    ///     list: &mut [bool],
    ///     idx: usize,
    ///     len: usize,
    /// ) {
    ///     // ...
    /// }
    /// ```
    pub(super) struct TransitionEpsilons<'a>(pub &'a CachedNFA<'a>);
    impl<'a> ToTokens for TransitionEpsilons<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let TransitionEpsilons(nfa) = self;
            let CachedNFA {
                nfa,
                excluded_states,
                ..
            } = nfa;
            assert_eq!(nfa.states.len(), excluded_states.len());
            let num_states = nfa.states.len();

            let states_epsilon_transitions = std::iter::zip(nfa.states.iter(), excluded_states)
                .enumerate()
                .filter(|(_, (_, &excluded))| !excluded)
                .map(|(i, _)| {
                    // all reachable states with next transition as epsilon
                    let mut new_threads = calculate_epsilon_propogations(nfa, i);
                    new_threads.retain(|t| {
                        !nfa.states[t.state].transitions.is_empty() || t.state + 1 == num_states
                    });

                    let label: ImplVMStateLabel = ImplVMStateLabel(i);

                    let state_epsilon_transitions_test = impl_test::ImplTransitionStateEpsilon {
                        from_state: label,
                        thread_updates: &new_threads,
                    };
                    state_epsilon_transitions_test.to_token_stream()
                });

            tokens.extend(quote! {
                fn transition_epsilons_test(
                    list: &mut [bool],
                    idx: usize,
                    len: usize,
                ) {
                    let is_start = idx == 0;
                    let is_end = idx == len;
                    #(#states_epsilon_transitions)*
                }
            });
        }
    }
}

mod impl_exec {
    use quote::ToTokens;

    use super::*;

    struct ImplTransition<'a> {
        transition: &'a U8Transition,
    }
    impl<'a> ToTokens for ImplTransition<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let ImplTransition { transition } = self;
            let U8Transition { symbol, to } = transition;
            let start = symbol.start();
            let end = symbol.end();
            let to_label = ImplVMStateLabel(*to);
            tokens.extend(quote! {{
                if #start <= c && c <= #end {
                    out.push(
                        ::ere::pike_vm_u8::U8PikeVMThread {
                            state: VMStates::#to_label,
                            captures: thread.captures.clone(),
                        },
                    );
                }
            }});
        }
    }

    /// Assumes the `VMStates` enum is already created locally in the token stream
    ///
    /// Creates the function `transition_symbols_exec` for running symbol transitions on the pike VM
    ///
    /// ```ignore
    /// fn transition_symbols_exec(
    ///     threads: &[::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>],
    ///     c: u8,
    /// ) -> ::std::vec::Vec<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>> {
    ///     // ...
    /// }
    /// ```
    pub(super) struct TransitionSymbols<'a>(pub &'a CachedNFA<'a>);
    impl<'a> ToTokens for TransitionSymbols<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let TransitionSymbols(nfa) = self;
            let CachedNFA {
                nfa,
                capture_groups,
                excluded_states,
            } = nfa;

            let transition_symbols_defs_exec = nfa
                .states
                .iter()
                .enumerate()
                .filter(|(i, _)| !excluded_states[*i])
                .map(|(i, state)| {
                    let label = ImplVMStateLabel(i);
                    let state_transitions = state
                        .transitions
                        .iter()
                        .map(|t| ImplTransition { transition: t });

                    return quote! {
                        VMStates::#label => {
                            #(#state_transitions)*
                        }
                    };
                });

            tokens.extend(quote! {
                fn transition_symbols_exec(
                    threads: &[::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>],
                    c: u8,
                ) -> ::std::vec::Vec<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>> {
                    let mut out = ::std::vec::Vec::<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>>::new();
                    for thread in threads {
                        match thread.state {
                            #(#transition_symbols_defs_exec)*
                        }
                    }
                    return out;
                }
            });
        }
    }

    /// Implements epsilon transitions for a single state
    ///
    /// Becomes:
    /// ```ignore
    /// VMStates::#from_state => {
    ///     // ...
    /// }
    /// ```
    pub(super) struct ImplTransitionStateEpsilon<'a> {
        pub(super) from_state: ImplVMStateLabel,
        pub(super) thread_updates: &'a [ThreadUpdates],
    }
    impl<'a> ToTokens for ImplTransitionStateEpsilon<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let &ImplTransitionStateEpsilon {
                from_state,
                thread_updates,
            } = self;

            // Write epsilon-propogation of threads to the token stream for exec
            let start_end_threads = thread_updates
                .iter()
                .filter(|t| t.start_only && t.end_only)
                .map(ThreadUpdates::serialize_thread_update_exec);
            let start_threads = thread_updates
                .iter()
                .filter(|t| t.start_only && !t.end_only)
                .map(ThreadUpdates::serialize_thread_update_exec);
            let end_threads = thread_updates
                .iter()
                .filter(|t| !t.start_only && t.end_only)
                .map(ThreadUpdates::serialize_thread_update_exec);
            let normal_threads = thread_updates
                .iter()
                .filter(|t| !t.start_only && !t.end_only)
                .map(ThreadUpdates::serialize_thread_update_exec);

            tokens.extend(quote! {
                VMStates::#from_state => {
                    if is_start && is_end {
                        #(#start_end_threads)*
                    }
                    if is_start {
                        #(#start_threads)*
                    }
                    if is_end {
                        #(#end_threads)*
                    }
                    #(#normal_threads)*
                }
            });
        }
    }

    /// Implements a function that runs all epsilon transitions for all threads.
    ///
    /// ```ignore
    /// fn transition_epsilons_exec(
    ///     threads: &[::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>],
    ///     idx: usize,
    ///     len: usize,
    /// ) -> ::std::vec::Vec<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>> {
    ///     // ...
    /// }
    /// ```
    pub(super) struct TransitionEpsilons<'a>(pub &'a CachedNFA<'a>);
    impl<'a> ToTokens for TransitionEpsilons<'a> {
        fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
            let TransitionEpsilons(nfa) = self;
            let CachedNFA {
                nfa,
                capture_groups,
                excluded_states,
            } = nfa;
            assert_eq!(nfa.states.len(), excluded_states.len());
            let num_states = nfa.states.len();

            let states_epsilon_transitions = std::iter::zip(nfa.states.iter(), excluded_states)
                .enumerate()
                .filter(|(_, (_, &excluded))| !excluded)
                .map(|(i, _)| {
                    // all reachable states with next transition as epsilon
                    let mut new_threads = calculate_epsilon_propogations(nfa, i);
                    new_threads.retain(|t| {
                        !nfa.states[t.state].transitions.is_empty() || t.state + 1 == num_states
                    });

                    let label: ImplVMStateLabel = ImplVMStateLabel(i);

                    let state_epsilon_transitions_exec = impl_exec::ImplTransitionStateEpsilon {
                        from_state: label,
                        thread_updates: &new_threads,
                    };
                    state_epsilon_transitions_exec.to_token_stream()
                });

            tokens.extend(quote! {
                fn transition_epsilons_exec(
                    threads: &[::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>],
                    idx: usize,
                    len: usize,
                ) -> ::std::vec::Vec<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>> {
                    let is_start = idx == 0;
                    let is_end = idx == len;
                    let mut occupied_states = ::std::vec![false; #num_states];
                    let mut out = ::std::vec::Vec::<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>>::new();
                    for thread in threads {
                        match thread.state {
                            #(#states_epsilon_transitions)*
                        }
                    }
                    return out;
                }
            });
        }
    }
}

#[derive(Clone, PartialEq, Eq)]
struct ThreadUpdates {
    pub state: usize,
    pub update_captures: Vec<(bool, bool)>,
    pub start_only: bool,
    pub end_only: bool,
}
impl ThreadUpdates {
    /// Creates a block which takes `list: &mut [bool; STATE_NUM]` from its local context, updates it in-place using `self` (compile-time).
    pub fn serialize_thread_update_test(&self) -> proc_macro2::TokenStream {
        let new_state = self.state;
        return quote! {{
            list[#new_state] = true;
        }};
    }
    /// Creates a block which takes `thread` from its local context, updates it using `self` (compile-time),
    /// and appends it to `out` from its local context.
    pub fn serialize_thread_update_exec(&self) -> proc_macro2::TokenStream {
        let new_state_idx = self.state;
        let new_state = ImplVMStateLabel(self.state);
        let mut capture_updates = proc_macro2::TokenStream::new();
        for (i, (start, end)) in self.update_captures.iter().cloned().enumerate() {
            if start {
                capture_updates.extend(quote! {
                    new_thread.captures[#i].0 = idx;
                });
            }
            if end {
                capture_updates.extend(quote! {
                    new_thread.captures[#i].1 = idx;
                });
            }
        }

        return quote! {
            if !occupied_states[#new_state_idx] {
                let mut new_thread = thread.clone();
                new_thread.state = VMStates::#new_state;

                #capture_updates

                out.push(new_thread);
                occupied_states[#new_state_idx] = true;
            }
        };
    }
}

fn calculate_epsilon_propogations(nfa: &U8NFA, state: usize) -> Vec<ThreadUpdates> {
    let U8NFA { states } = nfa;
    let capture_groups = nfa.num_capture_groups();
    // reduce epsilons to occur in a single step
    let mut new_threads = vec![ThreadUpdates {
        state,
        update_captures: vec![(false, false); capture_groups],
        start_only: false,
        end_only: false,
    }];
    fn traverse(
        thread: ThreadUpdates,
        states: &Vec<crate::working_u8_nfa::U8State>,
        out: &mut Vec<ThreadUpdates>,
    ) {
        out.push(thread.clone());
        for e in &states[thread.state].epsilons {
            let mut new_thread = thread.clone();
            new_thread.state = e.to;
            match e.special {
                EpsilonType::None => {}
                EpsilonType::StartAnchor => new_thread.start_only = true,
                EpsilonType::EndAnchor => new_thread.end_only = true,
                EpsilonType::StartCapture(capture_group) => {
                    new_thread.update_captures[capture_group].0 = true
                }
                EpsilonType::EndCapture(capture_group) => {
                    new_thread.update_captures[capture_group].1 = true
                }
            }

            if !out.contains(&new_thread) {
                traverse(new_thread, states, out);
            }
        }
    }
    traverse(
        ThreadUpdates {
            state,
            update_captures: vec![(false, false); capture_groups],
            start_only: false,
            end_only: false,
        },
        states,
        &mut new_threads,
    );
    return new_threads;
}

/// Converts a [`U8NFA`] into a format that, when returned by a proc macro, will
/// create the corresponding Pike VM.
pub(crate) fn serialize_pike_vm_token_stream(nfa: &U8NFA) -> proc_macro2::TokenStream {
    let nfa = CachedNFA::new(nfa);

    let capture_groups = nfa.capture_groups;
    let enum_states = nfa
        .excluded_states
        .iter()
        .enumerate()
        .filter_map(|(i, excluded)| match excluded {
            true => None,
            false => Some(ImplVMStateLabel(i)),
        });
    let state_count = nfa.nfa.states.len(); // TODO: not all of these are used, so we may be able to slightly reduce usage.
    let accept_state = ImplVMStateLabel(state_count - 1);

    let transition_symbols_test = impl_test::TransitionSymbols(&nfa);
    let transition_symbols_exec = impl_exec::TransitionSymbols(&nfa);

    let transition_epsilons_test = impl_test::TransitionEpsilons(&nfa);
    let transition_epsilons_exec = impl_exec::TransitionEpsilons(&nfa);

    return quote! {{
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum VMStates {
            #(#enum_states,)*
        }

        #transition_symbols_test
        #transition_symbols_exec
        #transition_epsilons_test
        #transition_epsilons_exec

        fn test(text: &str) -> bool {
            let mut list = [false; #state_count];
            let mut new_list = [false; #state_count];
            list[0] = true;

            transition_epsilons_test(&mut list, 0, text.len());
            for (i, c) in text.bytes().enumerate() {
                transition_symbols_test(&list, &mut new_list, c);
                if new_list.iter().all(|b| !b) {
                    return false;
                }
                ::std::mem::swap(&mut list, &mut new_list);
                transition_epsilons_test(&mut list, i + 1, text.len());
                new_list.fill(false);
            }

            return list[#state_count - 1];
        }
        fn exec<'a>(text: &'a str) -> Option<[Option<&'a str>; #capture_groups]> {
            let mut threads = ::std::vec::Vec::<::ere::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>>::new();
            threads.push(::ere::pike_vm_u8::U8PikeVMThread {
                state: VMStates::State0,
                captures: [(usize::MAX, usize::MAX); #capture_groups],
            });

            let new_threads = transition_epsilons_exec(&threads, 0, text.len());
            threads = new_threads;

            for (i, c) in text.bytes().enumerate() {
                let new_threads = transition_symbols_exec(&threads, c);
                threads = new_threads;
                let new_threads = transition_epsilons_exec(&threads, i + 1, text.len());
                threads = new_threads;
                if threads.is_empty() {
                    return ::core::option::Option::None;
                }
            }

            let final_capture_bounds = threads
                .into_iter()
                .find(|t| t.state == VMStates::#accept_state)?
                .captures;
            let mut captures = [::core::option::Option::None; #capture_groups];
            for (i, (start, end)) in final_capture_bounds.into_iter().enumerate() {
                if start != usize::MAX {
                    assert_ne!(end, usize::MAX);
                    // assert!(start <= end);
                    captures[i] = text.get(start..end);
                    assert!(captures[i].is_some());
                } else {
                    assert_eq!(end, usize::MAX);
                }
            }
            return ::core::option::Option::Some(captures);
        }

        (test, exec)
    }};
}
