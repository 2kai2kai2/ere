use quote::quote;

use crate::{
    nfa_static,
    working_nfa::{EpsilonType, WorkingNFA, WorkingTransition},
};

#[derive(Clone, Debug)]
pub struct PikeVMThread<const N: usize, S: Send + Sync + Copy + Eq> {
    pub state: S,
    pub captures: [(usize, usize); N],
}

/// Not exactly the PikeVM, but close enough that I am naming it that.
/// It works similarly, except that since we are building at compile-time, there are benefits from inlining splitting.
/// 
/// Due to the optimizations done earlier for the [`WorkingNFA`], we also have a slightly different NFA structure.
/// Currently we flatten all epsilon transitions for the VM so that epsilon transitions are at most a single step between symbols.
/// I'll have to review to ensure we avoid this causing large binary size overhead,
/// but it should be worst-case `O(n^2)` in the number of states, and far fewer on average.
pub struct PikeVM<const N: usize> {
    test_fn: fn(&str) -> bool,
    exec_fn: for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
}
impl<const N: usize> PikeVM<N> {
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
    ) -> PikeVM<N> {
        return PikeVM { test_fn, exec_fn };
    }
}

fn vmstate_label(idx: usize) -> proc_macro2::Ident {
    let label = format!("State{idx}");
    return proc_macro2::Ident::new(&label, proc_macro2::Span::call_site());
}

/// Since we are shortcutting the epsilon transitions, we can skip printing
/// states that have only epsilon transitions and are not the start/end states
fn compute_excluded_states(nfa: &WorkingNFA) -> Vec<bool> {
    let mut out = vec![true; nfa.states];
    out[0] = false;
    out[nfa.states - 1] = false;
    for t in &nfa.transitions {
        out[t.from] = false;
        out[t.to] = false;
    }
    return out;
}

/// Assumes the `VMStates` enum is already created locally in the token stream
///
/// Creates the function for running symbol transitions on the pike VM
fn serialize_pike_vm_symbol_propogation(nfa: &WorkingNFA) -> proc_macro2::TokenStream {
    let WorkingNFA {
        transitions,
        states,
        ..
    } = nfa;
    let capture_groups = nfa.num_capture_groups();
    let excluded_states = compute_excluded_states(nfa);

    fn make_symbol_transition(t: &WorkingTransition) -> proc_macro2::TokenStream {
        let WorkingTransition { symbol, to, .. } = t;
        let symbol = nfa_static::AtomStatic::serialize_as_token_stream(symbol);
        let to_label = vmstate_label(*to);
        return quote! {
            {
                let symbol = #symbol;
                if symbol.check(c) {
                    out.push(
                        ::ere_core::pike_vm::PikeVMThread {
                            state: VMStates::#to_label,
                            captures: thread.captures.clone(),
                        },
                    );
                }
            }
        };
    }

    let transition_symbols_defs: proc_macro2::TokenStream =
        std::iter::IntoIterator::into_iter(0..*states)
            .filter(|i| !excluded_states[*i])
            .map(|i| {
                let label = vmstate_label(i);
                let state_transitions: proc_macro2::TokenStream = transitions
                    .iter()
                    .filter(|t| t.from == i)
                    .map(make_symbol_transition)
                    .collect();

                return quote! {
                    VMStates::#label => {
                        #state_transitions
                    }
                };
            })
            .collect();

    return quote! {
        fn transition_symbols(
            threads: &[::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>],
            c: char,
        ) -> ::std::vec::Vec<::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>> {
            let mut out = ::std::vec::Vec::<::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>>::new();
            for thread in threads {
                match thread.state {
                    #transition_symbols_defs
                }
            }
            return out;
        }
    };
}

/// Assumes the `VMStates` enum is already created locally in the token stream
fn serialize_pike_vm_epsilon_propogation(nfa: &WorkingNFA) -> proc_macro2::TokenStream {
    let WorkingNFA {
        epsilons, states, ..
    } = nfa;
    let capture_groups = nfa.num_capture_groups();
    let excluded_states = compute_excluded_states(nfa);

    #[derive(Clone, PartialEq, Eq)]
    struct ThreadUpdates {
        pub state: usize,
        pub update_captures: Vec<(bool, bool)>,
        pub start_only: bool,
        pub end_only: bool,
    }

    // Generate code to propogate/split a thread according to epsilon transitions
    let mut transition_epsilons_defs = proc_macro2::TokenStream::new();
    for i in 0..*states {
        if excluded_states[i] {
            // since we propogate, some states are now useless if they are intermediate
            // with only epsilon transitions
            continue;
        }
        // reduce epsilons to occur in a single step
        let mut new_threads = vec![ThreadUpdates {
            state: i,
            update_captures: vec![(false, false); capture_groups],
            start_only: false,
            end_only: false,
        }];
        let mut stack = new_threads.clone();
        while let Some(thread) = stack.pop() {
            // enumerate next step of new threads
            for e in epsilons {
                if e.from != thread.state {
                    continue;
                }

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
                stack.push(new_thread.clone());
                new_threads.push(new_thread);
            }
        }

        new_threads.retain(|t| !excluded_states[t.state]);

        // Write epsilon-propogation of threads to the token stream
        fn serialize_thread_update(thread: &ThreadUpdates) -> proc_macro2::TokenStream {
            let new_state = vmstate_label(thread.state);
            let mut capture_updates = proc_macro2::TokenStream::new();
            for (i, (start, end)) in thread.update_captures.iter().cloned().enumerate() {
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
        let start_end_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| t.start_only && t.end_only)
            .map(serialize_thread_update)
            .collect();
        let start_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| t.start_only && !t.end_only)
            .map(serialize_thread_update)
            .collect();
        let end_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| !t.start_only && t.end_only)
            .map(serialize_thread_update)
            .collect();
        let normal_threads: proc_macro2::TokenStream = new_threads
            .iter()
            .filter(|t| !t.start_only && !t.end_only)
            .map(serialize_thread_update)
            .collect();

        let label = vmstate_label(i);
        transition_epsilons_defs.extend(quote! {
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

    return quote! {
        fn transition_epsilons(
            threads: &[::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>],
            idx: usize,
            len: usize,
        ) -> ::std::vec::Vec<::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>> {
            let is_start = idx == 0;
            let is_end = idx == len;
            let mut out = ::std::vec::Vec::<::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>>::new();
            for thread in threads {
                match thread.state {
                    #transition_epsilons_defs
                }
            }
            return out;
        }
    };
}

/// Converts a [`WorkingNFA`] into a format that, when returned by a proc macro, will
/// create the corresponding Pike VM.
pub(crate) fn serialize_pike_vm_token_stream(nfa: &WorkingNFA) -> proc_macro2::TokenStream {
    let WorkingNFA { states, .. } = nfa;
    let capture_groups = nfa.num_capture_groups();
    let excluded_states = compute_excluded_states(nfa);

    let enum_states: proc_macro2::TokenStream = std::iter::IntoIterator::into_iter(0..*states)
        .filter(|i| !excluded_states[*i])
        .map(|i| {
            let label = vmstate_label(i);
            return quote! { #label, };
        })
        .collect();
    let accept_state = vmstate_label(*states - 1);

    let transition_symbols = serialize_pike_vm_symbol_propogation(nfa);
    let transition_epsilons = serialize_pike_vm_epsilon_propogation(nfa);

    return quote! {{
        #[derive(Clone, Copy, PartialEq, Eq, Debug)]
        enum VMStates {
            #enum_states
        }

        #transition_symbols
        #transition_epsilons

        fn test(text: &str) -> bool {
            todo!();
        }
        fn exec<'a>(text: &'a str) -> Option<[Option<&'a str>; #capture_groups]> {
            let mut threads = ::std::vec::Vec::<::ere_core::pike_vm::PikeVMThread<#capture_groups, VMStates>>::new();
            threads.push(::ere_core::pike_vm::PikeVMThread {
                state: VMStates::State0,
                captures: [(usize::MAX, usize::MAX); #capture_groups],
            });

            let new_threads = transition_epsilons(&threads, 0, text.len());
            threads = new_threads;

            for (i, c) in text.char_indices() {
                let new_threads = transition_symbols(&threads, c);
                threads = new_threads;
                let new_threads = transition_epsilons(&threads, i + c.len_utf8(), text.len());
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

        ::ere_core::pike_vm::PikeVM::<#capture_groups>::__load(test, exec)
    }};
}
