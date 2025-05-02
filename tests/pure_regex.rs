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
