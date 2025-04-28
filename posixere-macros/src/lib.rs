use posixere_core;
use proc_macro::TokenStream;

extern crate proc_macro;

#[proc_macro]
pub fn pure_regex_matcher(stream: TokenStream) -> TokenStream {
    return posixere_core::__pure_regex_matcher(stream);
}
