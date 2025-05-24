use ere::prelude::*;

#[test]
fn phone_number() {
    const PHONE_REGEX: Regex<2> = compile_regex!(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$");

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

#[cfg(feature = "unstable-attr-regex")]
#[test]
fn phone_number_struct() {
    #[derive(PartialEq, Eq, Debug)]
    #[regex(r"^(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}$")]
    struct PhoneMatcher<'a>(&'a str, Option<&'a str>);

    assert!(PhoneMatcher::test("012-345-6789"));
    assert!(PhoneMatcher::test("987-654-3210"));
    assert!(PhoneMatcher::test("+1 555-555-5555"));
    assert!(PhoneMatcher::test("123-555-9876"));

    assert!(!PhoneMatcher::test("abcd"));
    assert!(!PhoneMatcher::test("0123456789"));
    assert!(!PhoneMatcher::test("012--345-6789"));
    assert!(!PhoneMatcher::test("(555) 555-5555"));
    assert!(!PhoneMatcher::test("1 555-555-5555"));

    assert_eq!(
        PhoneMatcher::exec("012-345-6789"),
        Some(PhoneMatcher("012-345-6789", None))
    );
    assert_eq!(
        PhoneMatcher::exec("987-654-3210"),
        Some(PhoneMatcher("987-654-3210", None))
    );
    assert_eq!(
        PhoneMatcher::exec("+1 555-555-5555"),
        Some(PhoneMatcher("+1 555-555-5555", Some("+1 ")))
    );
    assert_eq!(
        PhoneMatcher::exec("123-555-9876"),
        Some(PhoneMatcher("123-555-9876", None))
    );

    assert_eq!(PhoneMatcher::exec("abcd"), None);
    assert_eq!(PhoneMatcher::exec("0123456789"), None);
    assert_eq!(PhoneMatcher::exec("012--345-6789"), None);
    assert_eq!(PhoneMatcher::exec("(555) 555-5555"), None);
    assert_eq!(PhoneMatcher::exec("1 555-555-5555"), None);
}

#[test]
fn byte_value_exec() {
    const BYTE_REGEX: Regex<2> =
        compile_regex!(r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$");

    assert_eq!(BYTE_REGEX.exec("1"), Some([Some("1"), Some("1")]),);
    assert_eq!(BYTE_REGEX.exec("255"), Some([Some("255"), Some("255")]),);
    assert_eq!(BYTE_REGEX.exec("0"), Some([Some("0"), Some("0")]),);
    assert_eq!(BYTE_REGEX.exec("12"), Some([Some("12"), Some("12"),]),);

    assert_eq!(BYTE_REGEX.exec("abcd"), None);
    assert_eq!(BYTE_REGEX.exec("00"), None);
    assert_eq!(BYTE_REGEX.exec("256"), None);
}

#[test]
fn ipv4_exec() {
    const IPV4_REGEX: Regex<5> = compile_regex!(
        r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"
    );

    assert_eq!(
        IPV4_REGEX.exec("1.1.1.1"),
        Some([Some("1.1.1.1"), Some("1"), Some("1"), Some("1"), Some("1")]),
    );
    assert_eq!(
        IPV4_REGEX.exec("255.255.255.255"),
        Some([
            Some("255.255.255.255"),
            Some("255"),
            Some("255"),
            Some("255"),
            Some("255")
        ]),
    );
    assert_eq!(
        IPV4_REGEX.exec("192.168.0.1"),
        Some([
            Some("192.168.0.1"),
            Some("192"),
            Some("168"),
            Some("0"),
            Some("1")
        ]),
    );
    assert_eq!(
        IPV4_REGEX.exec("12.34.56.78"),
        Some([
            Some("12.34.56.78"),
            Some("12"),
            Some("34"),
            Some("56"),
            Some("78")
        ]),
    );

    assert_eq!(IPV4_REGEX.exec("abcd"), None);
    assert_eq!(IPV4_REGEX.exec("1.1.1"), None);
    assert_eq!(IPV4_REGEX.exec("..."), None);
    assert_eq!(IPV4_REGEX.exec("1::"), None);
    assert_eq!(IPV4_REGEX.exec("256.0.0.0"), None);
}

#[test]
fn needle() {
    const NEEDLE_REGEX: Regex = compile_regex!(r"nee+dle");

    assert!(NEEDLE_REGEX.test("needle"));
    assert!(NEEDLE_REGEX.test("haystackhaysneedletackhaystack"));
    assert!(NEEDLE_REGEX.test("haystackneeeeeeeeedlehaystack"));
    assert!(NEEDLE_REGEX.test("needneedlele"));

    assert!(!NEEDLE_REGEX.test("haystackhaystack"));
    assert!(!NEEDLE_REGEX.test("0123456789"));
    assert!(!NEEDLE_REGEX.test("nothinghere"));
    assert!(!NEEDLE_REGEX.test("npuowahpeoifjap098uq09p3ior"));
    assert!(!NEEDLE_REGEX.test("nedle"));
}

#[test]
fn dot() {
    const DOT_REGEX: Regex = compile_regex!("^.$");

    assert!(DOT_REGEX.test("a"));
    assert!(DOT_REGEX.test("b"));
    assert!(DOT_REGEX.test("©"));
    assert!(DOT_REGEX.test("\u{0001}"));

    assert!(!DOT_REGEX.test("\0"));
    assert!(!DOT_REGEX.test("12"));
    assert!(!DOT_REGEX.test("å©"));
    assert!(!DOT_REGEX.test(""));
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

    // non-greedy
    const REGEX5: Regex<1> = compile_regex!(r"a*?");
    assert_eq!(REGEX5.exec("aaaaaaaa"), Some([Some("")]));

    const REGEX6: Regex<1> = compile_regex!(r"a*?|a*");
    assert_eq!(REGEX6.exec("aaaaaaaa"), Some([Some("")]));
    assert_eq!(REGEX6.exec("aaaabaaaa"), Some([Some("")]));

    // Matches the second alternation because it is greedy, so it matches the `a` first
    // and does not wait to find the `b`. This is consistent with other engines.
    const REGEX7: Regex<1> = compile_regex!(r"ba*?b|a*");
    assert_eq!(REGEX7.exec("abaaabaaaa"), Some([Some("a")]));
}
