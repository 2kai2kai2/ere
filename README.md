This crate provides tools for compiling and using regular expressions.
It is intended as a simple but compiler-checked version of the [`regex`](https://crates.io/crates/regex) crate, as it does regular expression compilation at compile-time, but only supports [POSIX Extended Regular Expressions](https://en.wikibooks.org/wiki/Regular_Expressions/POSIX-Extended_Regular_Expressions).

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

const COLOR_REGEX: Regex = compile_regex!(
    r"^#?([[:xdigit:]]{2})([[:xdigit:]]{2})([[:xdigit:]]{2})([[:xdigit:]]{2})?$",
);
fn exec() {
    assert_eq!(
        COLOR_REGEX.exec("#000000"),
        Some([
            Some("#000000"),
            Some("00"),
            Some("00"),
            Some("00"),
            None,
        ]),
    );
    assert_eq!(
        COLOR_REGEX.exec("1F2e3D"),
        Some([
            Some("1F2E3D"),
            Some("1F"),
            Some("2e"),
            Some("3D"),
            None,
        ]),
    );
    assert_eq!(
        COLOR_REGEX.exec("ffffff80"),
        Some([
            Some("ffffff80"),
            Some("ff"),
            Some("ff"),
            Some("ff"),
            Some("80"),
        ]),
    );

    assert_eq!(PHONE_REGEX.exec("green"), None);
    assert_eq!(PHONE_REGEX.exec("%FFFFFF"), None);
    assert_eq!(PHONE_REGEX.exec("#2"), None);
}
```

To minimize memory overhead and binary size, it is recommended to create a single instance of each regular expression (using a `const` variable) rather than creating multiple.

## Roadmap

#### Features
- [ ] POSIX-compilant ambiguous submatching rules: we currently implement the Perl-like greedy submatching when submatches are ambiguous (this should only affect `exec`, not `test`, and only more complex regexes). While known implementations are more expensive, I plan on also supporting the POSIX rules.
- [ ] Case-insensitive mode: while regexes can be modified to support case-insensitivity (and this can also be done on ascii by just lower-casing the text first), I intend to implement case-insensitive mode.
- [ ] Non-capturing groups: `(?:non-capturing)` while not POSIX ERE standard-compilant, this is a relatively commonly used feature (and can improve performance by not tracking matches). If added, this will be behind a feature flag.

#### Performance Improvements
- [ ] `u8`-based engines: performance improvements can be made for many regexes (such as those with only ascii) by using `u8`s instead of extracting variably one to four-byte `char`s from strings.
- [ ] Additional limited-feature engines. This is relatively open-ended, but major improvements can be made for regexes with certain properties.

## Alternatives

`ere` is intended as an alternative to [`regex`](https://crates.io/crates/regex) that provides compile-time checking and regex compilation. However, `ere` is less featureful, so here are a few reasons you might prefer `regex`:

-   You require more complex regular expressions with features like backreferences and word boundary checking (which are unavailable in POSIX EREs).
-   You need run-time-compiled regular expressions (such as when provided by the user).
-   Your regular expression runs significantly more efficiently on a specific regex engine not currently available in `ere`.
