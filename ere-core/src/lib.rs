//! This crate provides the core functionality to the `ere` crate.

use proc_macro::TokenStream;
use quote::quote;
extern crate proc_macro;

pub mod config;
pub mod nfa_static;
pub mod one_pass_u8;
pub mod parse_tree;
pub mod pike_vm;
pub mod pike_vm_u8;
pub mod simplified_tree;
pub mod visualization;
pub mod working_nfa;
pub mod working_u8_nfa;

enum RegexEngines<const N: usize> {
    NFA(nfa_static::NFAStatic<N>),
    PikeVM(pike_vm::PikeVM<N>),
    U8PikeVM(pike_vm_u8::U8PikeVM<N>),
    U8OnePass(one_pass_u8::U8OnePass<N>),
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
            RegexEngines::U8PikeVM(pike_vm) => pike_vm.test(text),
            RegexEngines::U8OnePass(one_pass) => one_pass.test(text),
        };
    }

    pub fn exec<'a>(&self, text: &'a str) -> Option<[Option<&'a str>; N]> {
        return match &self.0 {
            RegexEngines::NFA(nfa) => unimplemented!(),
            RegexEngines::PikeVM(pike_vm) => pike_vm.exec(text),
            RegexEngines::U8PikeVM(pike_vm) => pike_vm.exec(text),
            RegexEngines::U8OnePass(one_pass) => one_pass.exec(text),
        };
    }
}
impl<const N: usize> std::fmt::Display for Regex<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        return match &self.0 {
            RegexEngines::NFA(nfastatic) => nfastatic.fmt(f),
            RegexEngines::PikeVM(_) => f.write_str("<Compiled VM>"),
            RegexEngines::U8PikeVM(_) => f.write_str("<Compiled VM>"),
            RegexEngines::U8OnePass(_) => f.write_str("<Compiled VM>"),
        };
    }
}

pub const fn __construct_pikevm_regex<const N: usize>(vm: pike_vm::PikeVM<N>) -> Regex<N> {
    return Regex(RegexEngines::PikeVM(vm));
}
pub const fn __construct_u8pikevm_regex<const N: usize>(vm: pike_vm_u8::U8PikeVM<N>) -> Regex<N> {
    return Regex(RegexEngines::U8PikeVM(vm));
}
pub const fn __construct_nfa_regex<const N: usize>(nfa: nfa_static::NFAStatic<N>) -> Regex<N> {
    return Regex(RegexEngines::NFA(nfa));
}
pub const fn __construct_u8onepass_regex<const N: usize>(
    nfa: one_pass_u8::U8OnePass<N>,
) -> Regex<N> {
    return Regex(RegexEngines::U8OnePass(nfa));
}

/// Tries to pick the best engine.
pub fn __compile_regex(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);

    // Currently use a conservative check: only use u8 engines when it will only match ascii strings
    fn is_state_ascii(state: &working_nfa::WorkingState) -> bool {
        return state
            .transitions
            .iter()
            .flat_map(|t| t.symbol.to_ranges())
            .all(|range| range.end().is_ascii());
    }
    let is_ascii = nfa.states.iter().all(is_state_ascii);

    let u8_nfa = working_u8_nfa::U8NFA::new(&nfa);

    if let Some(engine) = one_pass_u8::serialize_one_pass_token_stream(&u8_nfa) {
        return quote! { ::ere_core::__construct_u8onepass_regex(#engine) }.into();
    }

    if is_ascii {
        let engine = pike_vm_u8::serialize_pike_vm_token_stream(&u8_nfa);
        return quote! { ::ere_core::__construct_u8pikevm_regex(#engine) }.into();
    } else if true {
        let engine = pike_vm::serialize_pike_vm_token_stream(&nfa);
        return quote! { ::ere_core::__construct_pikevm_regex(#engine) }.into();
    } else {
        let engine = nfa_static::serialize_nfa_as_token_stream(&nfa);
        return quote! { ::ere_core::__construct_nfa_regex(#engine) }.into();
    };
}

/// Always uses the [`pike_vm::PikeVM`] engine and returns an tokenized instance of it
/// instead of [`Regex`]
pub fn __compile_regex_engine_pike_vm(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    return pike_vm::serialize_pike_vm_token_stream(&nfa).into();
}

/// Always uses the [`pike_vm_u8::U8PikeVM`] engine and returns an tokenized instance of it
/// instead of [`Regex`]
pub fn __compile_regex_engine_pike_vm_u8(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    let nfa = working_u8_nfa::U8NFA::new(&nfa);
    return pike_vm_u8::serialize_pike_vm_token_stream(&nfa).into();
}

/// Always uses the [`one_pass_u8::U8OnePass`] engine and returns an tokenized instance of it
/// instead of [`Regex`].
///
/// Will return a compiler error if regex was not one-pass and could not be optimized to become one-pass.
pub fn __compile_regex_engine_one_pass_u8(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    let nfa = working_u8_nfa::U8NFA::new(&nfa);
    return one_pass_u8::serialize_one_pass_token_stream(&nfa)
        .unwrap_or(
            syn::parse::Error::new(
                proc_macro2::Span::call_site(),
                "Regex was not one-pass and could not be optimized to become one pass. 
Try using a different engine.",
            )
            .to_compile_error(),
        )
        .into();
}

#[cfg(feature = "unstable-attr-regex")]
pub fn __compile_regex_attr(attr: TokenStream, input: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(attr);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);

    let capture_groups = nfa.num_capture_groups();
    let optional_captures: Vec<bool> = (0..capture_groups)
        .map(|group_num| nfa.capture_group_is_optional(group_num))
        .collect();

    let input_copy = input.clone();
    let regex_struct: syn::DeriveInput = syn::parse_macro_input!(input_copy);
    let syn::Data::Struct(data_struct) = regex_struct.data else {
        return syn::parse::Error::new_spanned(
            regex_struct,
            "Attribute regexes currently only support structs.",
        )
        .to_compile_error()
        .into();
    };
    let syn::Fields::Unnamed(fields) = data_struct.fields else {
        return syn::parse::Error::new_spanned(
            data_struct.fields,
            "Attribute regexes currently require unnamed structs (tuple syntax).",
        )
        .to_compile_error()
        .into();
    };
    if fields.unnamed.len() != optional_captures.len() {
        return syn::parse::Error::new_spanned(
            fields.unnamed,
            format!(
                "Expected struct to have {} unnamed fields, based on number of captures in regular expression.",
                optional_captures.len()
            ),
        )
        .to_compile_error()
        .into();
    }
    // for field in &fields.unnamed {
    //     if let syn::Type::Reference(ty) = &field.ty {
    //         if matches!(*ty.elem, syn::parse_quote!(str)) {
    //             continue;
    //         }
    //     }
    // }

    let mut out: proc_macro2::TokenStream = input.into();

    // Currently use a conservative check: only use u8 engines when it will only match ascii strings
    fn is_state_ascii(state: &working_nfa::WorkingState) -> bool {
        return state
            .transitions
            .iter()
            .flat_map(|t| t.symbol.to_ranges())
            .all(|range| range.end().is_ascii());
    }
    let is_ascii = nfa.states.iter().all(is_state_ascii);

    let struct_args: proc_macro2::TokenStream = optional_captures
        .iter()
        .enumerate()
        .map(|(group_num, opt)| if *opt {
            quote! { result[#group_num], }
        } else {
            quote! {
                result[#group_num]
                .expect(
                    "If you are seeing this, there is probably an internal bug in the `ere-core` crate where a capture group was mistakenly marked as non-optional. Please report the bug."
                ),
            }
        })
        .collect();

    // TODO: is it possible to avoid all this wrapping?
    let struct_name = regex_struct.ident;
    if is_ascii {
        let nfa = working_u8_nfa::U8NFA::new(&nfa);
        let engine = pike_vm_u8::serialize_pike_vm_token_stream(&nfa);
        let implementation = quote! {
            impl<'a> #struct_name<'a> {
                const ENGINE: ::ere_core::pike_vm_u8::U8PikeVM::<#capture_groups> = #engine;
                pub fn test(text: &str) -> bool {
                    return Self::ENGINE.test(text);
                }
                pub fn exec(text: &'a str) -> ::core::option::Option<#struct_name<'a>> {
                    let result: [::core::option::Option<&'a str>; #capture_groups] = Self::ENGINE.exec(text)?;
                    return ::core::option::Option::<#struct_name<'a>>::Some(#struct_name(
                        #struct_args
                    ));
                }
            }
        };
        out.extend(implementation);
    } else {
        let engine = pike_vm::serialize_pike_vm_token_stream(&nfa);
        let implementation = quote! {
            impl<'a> #struct_name<'a> {
                const ENGINE: ::ere_core::pike_vm::PikeVM::<#capture_groups> = #engine;
                pub fn test(text: &str) -> bool {
                    return Self::ENGINE.test(text);
                }
                pub fn exec(text: &'a str) -> ::core::option::Option<#struct_name<'a>> {
                    let result: [::core::option::Option<&'a str>; #capture_groups] = Self::ENGINE.exec(text)?;
                    return ::core::option::Option::<#struct_name<'a>>::Some(#struct_name(
                        #struct_args
                    ));
                }
            }
        };
        out.extend(implementation);
    }

    return out.into();
}
