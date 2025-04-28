use ere::*;

#[test]
fn phone_number() {
    let regex = pure_regex_matcher!(r"(\+1 )?[0-9]{3}-[0-9]{3}-[0-9]{4}");

    assert!(regex.check("012-345-6789"));
    assert!(regex.check("987-654-3210"));
    assert!(regex.check("+1 555-555-5555"));
    assert!(regex.check("123-555-9876"));

    assert!(!regex.check("abcd"));
    assert!(!regex.check("0123456789"));
    assert!(!regex.check("012--345-6789"));
    assert!(!regex.check("(555) 555-5555"));
    assert!(!regex.check("1 555-555-5555"));
}
