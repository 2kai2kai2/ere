#![doc = include_str!("../README.md")]

pub use ::ere_core::nfa_static;
pub use ::ere_core::parse_tree;
pub use ::ere_core::pike_vm;
pub use ::ere_core::pike_vm_u8;
pub use ::ere_core::simplified_tree;
pub use ::ere_core::working_nfa;
pub use ::ere_core::Regex;
pub use ::ere_macros::*;

/// Includes the basic things you'll need.
///
/// Unless you want to use a specific engine or more specific internals,
/// you will probably never need anything else.
pub mod prelude {
    pub use ::ere_core::Regex;
    pub use ::ere_macros::compile_regex;
    pub use ::ere_macros::regex;
}
