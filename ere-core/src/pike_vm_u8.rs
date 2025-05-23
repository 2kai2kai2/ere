use std::collections::VecDeque;

use quote::quote;

use crate::{
    working_nfa::EpsilonType,
    working_u8_nfa::{U8Transition, U8NFA},
};

#[derive(Clone, Debug)]
pub struct U8PikeVMThread<const N: usize, S: Send + Sync + Copy + Eq> {
    pub state: S,
    pub captures: [(usize, usize); N],
}

/// Not exactly the PikeVM, but close enough that I am naming it that.
/// It works similarly, except that since we are building at compile-time, there are benefits from inlining splitting.
///
/// Due to the optimizations done earlier for the [`U8NFA`], we also have a slightly different NFA structure.
/// Currently we flatten all epsilon transitions for the VM so that epsilon transitions are at most a single step between symbols.
/// I'll have to review to ensure we avoid this causing large binary size overhead,
/// but it should be worst-case `O(n^2)` in the number of states, and far fewer on average.
pub struct U8PikeVM<const N: usize> {
    test_fn: fn(&str) -> bool,
    exec_fn: for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
}
impl<const N: usize> U8PikeVM<N> {
    pub fn test(&self, text: &str) -> bool {
        return (self.test_fn)(text);
    }
    pub fn exec<'a>(&self, text: &'a str) -> Option<[Option<&'a str>; N]> {
        return (self.exec_fn)(text);
    }

    /// Only intended for internal use by macros.
    pub const fn __load(
        test_fn: fn(&str) -> bool,
        exec_fn: for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
    ) -> U8PikeVM<N> {
        return U8PikeVM { test_fn, exec_fn };
    }
}

fn vmstate_label(idx: usize) -> proc_macro2::Ident {
    let label = format!("State{idx}");
    return proc_macro2::Ident::new(&label, proc_macro2::Span::call_site());
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

/// Assumes the `VMStates` enum is already created locally in the token stream
///
/// Creates the function for running symbol transitions on the pike VM
fn serialize_pike_vm_symbol_propogation(
    nfa: &U8NFA,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let U8NFA { states } = nfa;
    let capture_groups = nfa.num_capture_groups();
    let excluded_states = compute_excluded_states(nfa);

    fn make_symbol_transition_test(t: &U8Transition) -> proc_macro2::TokenStream {
        let U8Transition { symbol, to } = t;
        let start = symbol.start();
        let end = symbol.end();
        return quote! {
            {
                if #start <= c && c <= #end {
                    new_list[#to] = true;
                }
            }
        };
    }

    fn make_symbol_transition_exec(t: &U8Transition) -> proc_macro2::TokenStream {
        let U8Transition { symbol, to } = t;
        let start = symbol.start();
        let end = symbol.end();
        let to_label = vmstate_label(*to);
        return quote! {
            {
                if #start <= c && c <= #end {
                    out.push(
                        ::ere_core::pike_vm_u8::U8PikeVMThread {
                            state: VMStates::#to_label,
                            captures: thread.captures.clone(),
                        },
                    );
                }
            }
        };
    }

    let transition_symbols_defs_test: proc_macro2::TokenStream = states
        .iter()
        .enumerate()
        .filter(|(i, _)| !excluded_states[*i])
        .map(|(i, state)| {
            let state_transitions: proc_macro2::TokenStream = state
                .transitions
                .iter()
                .map(make_symbol_transition_test)
                .collect();

            return quote! {
                if list[#i] {
                    #state_transitions
                }
            };
        })
        .collect();

    let transition_symbols_defs_exec: proc_macro2::TokenStream = states
        .iter()
        .enumerate()
        .filter(|(i, _)| !excluded_states[*i])
        .map(|(i, state)| {
            let label = vmstate_label(i);
            let state_transitions: proc_macro2::TokenStream = state
                .transitions
                .iter()
                .map(make_symbol_transition_exec)
                .collect();

            return quote! {
                VMStates::#label => {
                    #state_transitions
                }
            };
        })
        .collect();

    let transition_symbols_test = quote! {
        fn transition_symbols_test(
            list: &[bool],
            new_list: &mut [bool],
            c: u8,
        ) {
            #transition_symbols_defs_test
        }
    };
    let transition_symbols_exec = quote! {
        fn transition_symbols_exec(
            threads: &[::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>],
            c: u8,
        ) -> ::std::vec::Vec<::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>> {
            let mut out = ::std::vec::Vec::<::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>>::new();
            for thread in threads {
                match thread.state {
                    #transition_symbols_defs_exec
                }
            }
            return out;
        }
    };

    return (transition_symbols_test, transition_symbols_exec);
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
        let new_state = vmstate_label(self.state);
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

        return quote! {{
            let mut new_thread = thread.clone();
            new_thread.state = VMStates::#new_state;

            #capture_updates

            out.push(new_thread);
        }};
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
    let mut queue = VecDeque::new();
    queue.push_back(new_threads[0].clone());
    while let Some(thread) = queue.pop_front() {
        // enumerate next step of new threads
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

            if new_threads.contains(&new_thread) {
                continue;
            }
            queue.push_back(new_thread.clone());
            new_threads.push(new_thread);
        }
    }
    return new_threads;
}

/// Assumes the `VMStates` enum is already created locally in the token stream
fn serialize_pike_vm_epsilon_propogation(
    nfa: &U8NFA,
) -> (proc_macro2::TokenStream, proc_macro2::TokenStream) {
    let U8NFA { states } = nfa;
    let capture_groups = nfa.num_capture_groups();
    let excluded_states = compute_excluded_states(nfa);

    // Generate code to propogate/split a thread according to epsilon transitions
    let mut transition_epsilons_test = proc_macro2::TokenStream::new();
    let mut transition_epsilons_exec = proc_macro2::TokenStream::new();
    for (i, _) in states.iter().enumerate() {
        if excluded_states[i] {
            // since we propogate, some states are now useless if they are intermediate
            // with only epsilon transitions
            continue;
        }
        let mut new_threads = calculate_epsilon_propogations(nfa, i);
        new_threads.retain(|t| !excluded_states[t.state]);

        // Write epsilon-propogation of threads to the token stream for test
        let start_end_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| t.start_only && t.end_only)
            .map(ThreadUpdates::serialize_thread_update_test)
            .collect();
        let start_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| t.start_only && !t.end_only)
            .map(ThreadUpdates::serialize_thread_update_test)
            .collect();
        let end_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| !t.start_only && t.end_only)
            .map(ThreadUpdates::serialize_thread_update_test)
            .collect();
        let normal_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| !t.start_only && !t.end_only)
            .map(ThreadUpdates::serialize_thread_update_test)
            .collect();

        let label = vmstate_label(i);
        transition_epsilons_test.extend(quote! {
            if list[#i] {
                if is_start && is_end {
                    #start_end_threads
                }
                if is_start {
                    #start_threads
                }
                if is_end {
                    #end_threads
                }
                #normal_threads
            }
        });

        // Write epsilon-propogation of threads to the token stream for exec
        let start_end_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| t.start_only && t.end_only)
            .map(ThreadUpdates::serialize_thread_update_exec)
            .collect();
        let start_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| t.start_only && !t.end_only)
            .map(ThreadUpdates::serialize_thread_update_exec)
            .collect();
        let end_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| !t.start_only && t.end_only)
            .map(ThreadUpdates::serialize_thread_update_exec)
            .collect();
        let normal_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| !t.start_only && !t.end_only)
            .map(ThreadUpdates::serialize_thread_update_exec)
            .collect();

        transition_epsilons_exec.extend(quote! {
            VMStates::#label => {
                if is_start && is_end {
                    #start_end_threads
                }
                if is_start {
                    #start_threads
                }
                if is_end {
                    #end_threads
                }
                #normal_threads
            }
        });
    }

    let transition_epsilons_test = quote! {
        fn transition_epsilons_test(
            list: &mut [bool],
            idx: usize,
            len: usize,
        ) {
            let is_start = idx == 0;
            let is_end = idx == len;
            #transition_epsilons_test
        }
    };
    let transition_epsilons_exec = quote! {
        fn transition_epsilons_exec(
            threads: &[::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>],
            idx: usize,
            len: usize,
        ) -> ::std::vec::Vec<::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>> {
            let is_start = idx == 0;
            let is_end = idx == len;
            let mut out = ::std::vec::Vec::<::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>>::new();
            for thread in threads {
                match thread.state {
                    #transition_epsilons_exec
                }
            }
            return out;
        }
    };

    return (transition_epsilons_test, transition_epsilons_exec);
}

/// Converts a [`U8NFA`] into a format that, when returned by a proc macro, will
/// create the corresponding Pike VM.
pub(crate) fn serialize_pike_vm_token_stream(nfa: &U8NFA) -> proc_macro2::TokenStream {
    let U8NFA { states, .. } = nfa;
    let capture_groups = nfa.num_capture_groups();
    let excluded_states = compute_excluded_states(nfa);

    let enum_states: proc_macro2::TokenStream = std::iter::IntoIterator::into_iter(0..states.len())
        .filter(|i| !excluded_states[*i])
        .map(|i| {
            let label = vmstate_label(i);
            return quote! { #label, };
        })
        .collect();
    let state_count = states.len(); // TODO: not all of these are used, so we may be able to slightly reduce usage.
    let accept_state = vmstate_label(states.len() - 1);

    let (transition_symbols_test, transition_symbols_exec) =
        serialize_pike_vm_symbol_propogation(nfa);
    let (transition_epsilons_test, transition_epsilons_exec) =
        serialize_pike_vm_epsilon_propogation(nfa);

    return quote! {{
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum VMStates {
            #enum_states
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
            let mut threads = ::std::vec::Vec::<::ere_core::pike_vm_u8::U8PikeVMThread<#capture_groups, VMStates>>::new();
            threads.push(::ere_core::pike_vm_u8::U8PikeVMThread {
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
                    return None;
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
            return Some(captures);
        }

        ::ere_core::pike_vm_u8::U8PikeVM::<#capture_groups>::__load(test, exec)
    }};
}
