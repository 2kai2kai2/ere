//! This crate provides the core functionality to the `ere` crate.

use proc_macro::TokenStream;
use quote::quote;
extern crate proc_macro;

pub mod config;
pub mod fixed_offset;
pub mod nfa_static;
pub mod one_pass_u8;
pub mod parse_tree;
pub mod pike_vm;
pub mod pike_vm_u8;
pub mod simplified_tree;
pub mod visualization;
pub mod working_nfa;
pub mod working_u8_nfa;

/// A regular expression (specifically, a [POSIX ERE](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX-Extended_Regular_Expressions)).
///
/// Internally, this may contain one of several engines depending on the expression.
///
/// The const generic `N` represents the number of capture groups (including capture group 0 which is the entire expression).
/// It defaults to `1` (for just capture group 0), but you will need to specify it in the type for expressions with more capture groups.
pub struct Regex<const N: usize = 1> {
    test_fn: fn(&str) -> bool,
    exec_fn: for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
}
impl<const N: usize> Regex<N> {
    /// Returns whether or not the text is matched by the regular expression.
    #[inline]
    pub fn test(&self, text: &str) -> bool {
        return (self.test_fn)(text);
    }
    #[inline]
    pub fn exec<'a>(&self, text: &'a str) -> Option<[Option<&'a str>; N]> {
        return (self.exec_fn)(text);
    }
}

/// Intended to be used in macros only.
#[inline]
pub const fn __construct_regex<const N: usize>(
    fn_pair: (
        fn(&str) -> bool,
        for<'a> fn(&'a str) -> Option<[Option<&'a str>; N]>,
    ),
) -> Regex<N> {
    return Regex {
        test_fn: fn_pair.0,
        exec_fn: fn_pair.1,
    };
}

/// Tries to pick the best engine that doesn't rely on sub-engines.
///
/// Returns a stream that evaluates to a pair `(test_fn, exec_fn)`
fn pick_base_engine(
    ere: parse_tree::ERE,
) -> (
    proc_macro2::TokenStream,
    simplified_tree::SimplifiedTreeNode,
    working_nfa::WorkingNFA,
    working_u8_nfa::U8NFA,
) {
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

    let base_engine = if let Some(engine) = one_pass_u8::serialize_one_pass_token_stream(&u8_nfa) {
        engine
    } else if is_ascii {
        pike_vm_u8::serialize_pike_vm_token_stream(&u8_nfa)
    } else {
        pike_vm::serialize_pike_vm_token_stream(&nfa)
    };
    return (base_engine, tree, nfa, u8_nfa);
}

/// Tries to pick the best engine that doesn't rely on sub-engines.
///
/// Returns a stream that evaluates to a pair `(test_fn, exec_fn)`
fn pick_engine(ere: parse_tree::ERE) -> proc_macro2::TokenStream {
    let (base_engine, _, _, u8_nfa) = pick_base_engine(ere);

    // Consider nested engines
    if let Some(offsets) = fixed_offset::get_fixed_offsets(&u8_nfa) {
        return fixed_offset::serialize_fixed_offset_token_stream(
            base_engine,
            offsets,
            u8_nfa.num_capture_groups(),
        );
    };
    return base_engine;
}

/// Tries to pick the best engine.
pub fn __compile_regex(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let fn_pair = pick_engine(ere);
    return quote! {
        {
            ::ere_core::__construct_regex(#fn_pair)
        }
    }
    .into();
}

/// Always uses the [`pike_vm`] engine
pub fn __compile_regex_engine_pike_vm(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    let fn_pair = pike_vm::serialize_pike_vm_token_stream(&nfa);
    return quote! {
        ::ere_core::__construct_regex(#fn_pair)
    }
    .into();
}

/// Always uses the [`pike_vm_u8`] engine
pub fn __compile_regex_engine_pike_vm_u8(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    let nfa = working_u8_nfa::U8NFA::new(&nfa);
    let fn_pair = pike_vm_u8::serialize_pike_vm_token_stream(&nfa);
    return quote! {
        ::ere_core::__construct_regex(#fn_pair)
    }
    .into();
}

/// Always uses the [`one_pass_u8`]
///
/// Will return a compiler error if regex was not one-pass and could not be optimized to become one-pass.
pub fn __compile_regex_engine_one_pass_u8(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere);
    let nfa = working_nfa::WorkingNFA::new(&tree);
    let nfa = working_u8_nfa::U8NFA::new(&nfa);
    let Some(fn_pair) = one_pass_u8::serialize_one_pass_token_stream(&nfa) else {
        return syn::parse::Error::new(
            proc_macro2::Span::call_site(),
            "Regex was not one-pass and could not be optimized to become one pass. 
Try using a different engine.",
        )
        .to_compile_error()
        .into();
    };
    return quote! {
        ::ere_core::__construct_regex(#fn_pair)
    }
    .into();
}

/// Always uses the [`fixed_offset`]
///
/// Will return a compiler error if regex was not fixed offset.
pub fn __compile_regex_engine_fixed_offset(stream: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(stream);
    let (base_engine, _, _, u8_nfa) = pick_base_engine(ere);

    let Some(offsets) = fixed_offset::get_fixed_offsets(&u8_nfa) else {
        return syn::parse::Error::new(
            proc_macro2::Span::call_site(),
            "Regex capture groups were not fixed offset. Try using a different engine.",
        )
        .to_compile_error()
        .into();
    };
    let fn_pair = fixed_offset::serialize_fixed_offset_token_stream(
        base_engine,
        offsets,
        u8_nfa.num_capture_groups(),
    );
    return quote! {
        ::ere_core::__construct_regex(#fn_pair)
    }
    .into();
}

#[cfg(feature = "unstable-attr-regex")]
pub fn __compile_regex_attr(attr: TokenStream, input: TokenStream) -> TokenStream {
    let ere: parse_tree::ERE = syn::parse_macro_input!(attr);
    let tree = simplified_tree::SimplifiedTreeNode::from(ere.clone());
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

    // TODO: is it possible to more naturally extract struct args as optional or not?
    let fn_pair = pick_engine(ere);
    let struct_name = regex_struct.ident;

    let implementation = quote! {
        impl<'a> #struct_name<'a> {
            const ENGINE: (
                fn(&str) -> bool,
                fn(&'a str) -> ::core::option::Option<[::core::option::Option<&'a str>; #capture_groups]>,
            ) = #fn_pair;
            #[inline]
            pub fn test(text: &str) -> bool {
                return (Self::ENGINE.0)(text);
            }
            pub fn exec(text: &'a str) -> ::core::option::Option<#struct_name<'a>> {
                let result: [::core::option::Option<&'a str>; #capture_groups] = (Self::ENGINE.1)(text)?;
                return ::core::option::Option::<#struct_name<'a>>::Some(#struct_name(
                    #struct_args
                ));
            }
        }
    };
    out.extend(implementation);

    return out.into();
}
