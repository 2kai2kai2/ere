use ere_core;
use proc_macro::TokenStream;

extern crate proc_macro;

#[proc_macro]
pub fn compile_regex(stream: TokenStream) -> TokenStream {
    return ere_core::__compile_regex(stream);
}
