use std::hash::Hasher;

use ere::prelude::*;
use ere_macros::{
    compile_regex_fixed_offset, compile_regex_pikevm, compile_regex_u8onepass,
    compile_regex_u8pikevm,
};

#[test]
fn phone_number() {
    const REGEXES: [Regex<2>; 3] = [
        compile_regex_pikevm!(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$"),
        compile_regex_u8pikevm!(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$"),
        compile_regex_u8onepass!(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$"),
    ];

    for regex in REGEXES {
        assert!(regex.test("012-345-6789"));
        assert!(regex.test("987-654-3210"));
        assert!(regex.test("+1 555-555-5555"));
        assert!(regex.test("123-555-9876"));

        assert!(!regex.test("abcd"));
        assert!(!regex.test("0123456789"));
        assert!(!regex.test("012--345-6789"));
        assert!(!regex.test("(555) 555-5555"));
        assert!(!regex.test("1 555-555-5555"));
    }
}

#[test]
fn byte_value_exec() {
    const REGEXES: [Regex<2>; 2] = [
        compile_regex_pikevm!(r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"),
        compile_regex_u8pikevm!(r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"),
        // one pass not working yet, needs further optimizations
    ];
    for regex in REGEXES {
        for i in 0u8..=255u8 {
            let text = i.to_string();
            assert!(regex.test(&text));
            assert_eq!(
                regex.exec(&text),
                Some([Some(text.as_str()), Some(text.as_str())])
            );
        }

        assert_eq!(regex.exec("abcd"), None);
        assert_eq!(regex.exec("00"), None);
        assert_eq!(regex.exec("256"), None);
    }
}

#[test]
fn ipv4_exec() {
    const REGEXES: [Regex<5>; 2] = [
        compile_regex_pikevm!(
            r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"
        ),
        compile_regex_u8pikevm!(
            r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"
        ),
        // one pass not working yet, needs further optimizations
    ];

    for regex in REGEXES {
        for i in 0..=10000 {
            // testing deterministic pseudo-random numbers via hashing
            let mut hasher = std::hash::DefaultHasher::new();
            hasher.write_u32(i);
            let i = hasher.finish();
            let [_, _, a, b, c, d, _, _] = i.to_be_bytes();
            let a = a.to_string();
            let b = b.to_string();
            let c = c.to_string();
            let d = d.to_string();
            let text = format!("{a}.{b}.{c}.{d}");
            assert!(regex.test(&text));
            assert_eq!(
                regex.exec(&text),
                Some([Some(text.as_str()), Some(&a), Some(&b), Some(&c), Some(&d)])
            );
        }
        assert_eq!(regex.exec("abcd"), None);
        assert_eq!(regex.exec("1.1.1"), None);
        assert_eq!(regex.exec("..."), None);
        assert_eq!(regex.exec("1::"), None);
        assert_eq!(regex.exec("256.0.0.0"), None);
    }
}

#[test]
fn needle() {
    const REGEXES: [Regex; 2] = [
        compile_regex_pikevm!(r"nee+dle"),
        compile_regex_u8pikevm!(r"nee+dle"),
        // not one-pass because it is not start/end anchored
    ];
    for regex in REGEXES {
        assert!(regex.test("needle"));
        assert!(regex.test("haystackhaysneedletackhaystack"));
        assert!(regex.test("haystackneeeeeeeeedlehaystack"));
        assert!(regex.test("needneedlele"));

        assert!(!regex.test("haystackhaystack"));
        assert!(!regex.test("0123456789"));
        assert!(!regex.test("nothinghere"));
        assert!(!regex.test("npuowahpeoifjap098uq09p3ior"));
        assert!(!regex.test("nedle"));
    }
}

#[test]
fn dot() {
    const REGEXES: [Regex; 3] = [
        compile_regex_pikevm!("^.$"),
        compile_regex_u8pikevm!("^.$"),
        compile_regex_u8onepass!("^.$"),
    ];
    for regex in REGEXES {
        for c in '\u{0001}'..=char::MAX {
            let text = c.to_string();
            assert!(regex.test(&text));
            assert_eq!(regex.exec(&text), Some([Some(text.as_str())]));
        }

        assert!(!regex.test("\0"));
        assert!(!regex.test("12"));
        assert!(!regex.test("å©"));
        assert!(!regex.test(""));
    }
}

#[test]
fn duplicate_paths() {
    const REGEXES: [Regex<3>; 4] = [
        compile_regex_pikevm!("^(ab|bc|ab|bc)(xy|yz|yz|xy)$"),
        compile_regex_u8pikevm!("^(ab|bc|ab|bc)(xy|yz|yz|xy)$"),
        // one-pass because it can be simplified to one-pass
        // since its branching paths are actually the same and get merged
        compile_regex_u8onepass!("^(ab|bc|ab|bc)(xy|yz|yz|xy)$"),
        compile_regex_fixed_offset!("^(ab|bc|ab|bc)(xy|yz|yz|xy)$"),
    ];
    for regex in &REGEXES {
        assert!(regex.test("abxy"));
        assert!(regex.test("abyz"));
        assert!(regex.test("bcxy"));
        assert!(regex.test("bcyz"));

        assert!(!regex.test("acxy"));
        assert!(!regex.test("abxz"));
        assert!(!regex.test("bc"));
        assert!(!regex.test("yz"));
    }
}

#[test]
fn hex_color() {
    const REGEXES: [Regex<4>; 4] = [
        compile_regex_pikevm!("^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$"),
        compile_regex_u8pikevm!("^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$"),
        compile_regex_u8onepass!("^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$"),
        compile_regex_fixed_offset!("^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$"),
    ];
    for regex in &REGEXES {
        assert_eq!(
            regex.exec("#abcdef"),
            Some([Some("#abcdef"), Some("ab"), Some("cd"), Some("ef")]),
        );
        assert_eq!(
            regex.exec("#FfFfFf"),
            Some([Some("#FfFfFf"), Some("Ff"), Some("Ff"), Some("Ff")]),
        );
        assert_eq!(
            regex.exec("#000000"),
            Some([Some("#000000"), Some("00"), Some("00"), Some("00")]),
        );

        assert_eq!(regex.exec("#qaaaaa"), None);
        assert_eq!(regex.exec("#12345"), None);
        assert_eq!(regex.exec("#1234567"), None);
    }
}

#[test]
fn greedy() {
    const REGEX1: Regex<2> = compile_regex!(r"^(a|ab)b?cd$");
    assert_eq!(REGEX1.exec("abcd"), Some([Some("abcd"), Some("a")]));

    const REGEX2: Regex<2> = compile_regex!(r"^(ab|a)b?cd$");
    assert_eq!(REGEX2.exec("abcd"), Some([Some("abcd"), Some("ab")]));

    const REGEX3: Regex<3> = compile_regex!(r"^(a*)(a*)$");
    assert_eq!(
        REGEX3.exec("aaaaaaaa"),
        Some([Some("aaaaaaaa"), Some("aaaaaaaa"), Some("")])
    );

    const REGEX4: Regex<1> = compile_regex!(r"a*");
    assert_eq!(REGEX4.exec("aaaaaaaa"), Some([Some("aaaaaaaa")]));
    assert_eq!(REGEX4.exec("aabaaaaaa"), Some([Some("aa")]));

    // with shortest
    const REGEX5: Regex<1> = compile_regex!(r"a*?");
    assert_eq!(REGEX5.exec("aaaaaaaa"), Some([Some("")]));

    const REGEX6: Regex<1> = compile_regex!(r"a*?|a*");
    assert_eq!(REGEX6.exec("aaaaaaaa"), Some([Some("")]));
    assert_eq!(REGEX6.exec("aaaabaaaa"), Some([Some("")]));

    // Matches the second alternation because it is greedy, so it matches the `a` first
    // and does not wait to find the `b`. This is consistent with other engines.
    const REGEX7: Regex<1> = compile_regex!(r"ba*?b|a*");
    assert_eq!(REGEX7.exec("abaaabaaaa"), Some([Some("a")]));

    const REGEX8: Regex<1> = compile_regex!(r"a??");
    assert_eq!(REGEX8.exec("abaaabaaaa"), Some([Some("")]));
}
