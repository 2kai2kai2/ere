This crate provides tools for compiling and using regular expressions.
It is intended as a simple but compiler-checked version of the [`regex`](https://crates.io/crates/regex) crate, as it does regular expression compilation at compile-time, but only supports [POSIX Extended Regular Expressions](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX-Extended_Regular_Expressions).

> [!IMPORTANT]
> Work-in-progress
>
> This project is not feature-complete. It currently supports regular expression `test`, but does not yet fully implement capture groups. Additionally, it currently only has one engine implemented.

## Usage

```rust
use ere::prelude::*;

const PHONE_REGEX: Regex = compile_regex!(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$");

fn test() {
    assert!(PHONE_REGEX.test("012-345-6789"));
    assert!(PHONE_REGEX.test("987-654-3210"));
    assert!(PHONE_REGEX.test("+1 555-555-5555"));
    assert!(PHONE_REGEX.test("123-555-9876"));

    assert!(!PHONE_REGEX.test("abcd"));
    assert!(!PHONE_REGEX.test("0123456789"));
    assert!(!PHONE_REGEX.test("012--345-6789"));
    assert!(!PHONE_REGEX.test("(555) 555-5555"));
    assert!(!PHONE_REGEX.test("1 555-555-5555"));
}
```

To minimize memory overhead and binary size, it is recommended to create a single instance of each regular expression (using a `const` variable) rather than creating multiple.

## Alternatives

`ere` is intended as an alternative to [`regex`](https://crates.io/crates/regex) that provides compile-time checking and regex compilation. However, `ere` is less featureful, so here are a few reasons you might prefer `regex`:

-   You require more complex regular expressions with features like backreferences and word boundary checking (which are unavailable in POSIX EREs).
-   You need run-time-compiled regular expressions (such as when provided by the user).
-   Your regular expression runs significantly more efficiently on a specific regex engine not currently available in `ere`.
