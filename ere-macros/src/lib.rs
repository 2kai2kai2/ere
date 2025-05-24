use ere_core;
use proc_macro::TokenStream;

extern crate proc_macro;

/// This is the primary entrypoint to the `ere` crate.
/// Checks and compiles a regular expression into a [`Regex<N>`](`ere_core::Regex<N>`).
///
/// This compilation happens during build using proc macros,
/// resulting in rust code equivalent to your regex.
/// This code can then by further optimized by rustc when compiled directly into the binary.
///
/// The generic `const N: usize` will be the number of capture groups present in the regular expression
/// (including capture group 0 which is the entire matched text).
/// You will need to properly specify this in the generics for the regex (default if unspecified is 1).
/// When using [`Regex<N>::exec`](`ere_core::Regex<N>::exec`), this is the length of the captures returned.
///
/// ```
/// use ere_core::Regex; // usually `ere::Regex`
/// use ere_macros::compile_regex; // usually `ere::compile_regex`
///
/// const MY_REGEX: Regex<2> = compile_regex!("a(b?)c");
/// ```
#[proc_macro]
pub fn compile_regex(stream: TokenStream) -> TokenStream {
    return ere_core::__compile_regex(stream);
}
