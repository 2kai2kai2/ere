use ere::prelude::*;

fn ipv4() {
    const REGEX: Regex<5> = compile_regex!(
        r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"
    );

    let start = std::time::SystemTime::now();
    let regex_runtime = ::regex::Regex::new(r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$").unwrap();
    let duration = start.elapsed().unwrap();
    println!("Runtime compilation duration: {}µs", duration.as_micros());

    let haystacks: Vec<String> = (0..=u32::MAX)
        .step_by(83)
        .map(|i| {
            let [a, b, c, d] = i.to_be_bytes();
            return format!("{a}.{b}.{c}.{d}");
        })
        .collect();

    let start = std::time::SystemTime::now();
    for haystack in haystacks.iter() {
        assert!(REGEX.test(haystack))
    }
    let duration = start.elapsed().unwrap();
    println!("Test (ere) duration:   {}µs", duration.as_micros());

    let start = std::time::SystemTime::now();
    for haystack in haystacks.iter() {
        assert!(regex_runtime.is_match(haystack))
    }
    let duration = start.elapsed().unwrap();
    println!("Test (regex) duration: {}µs", duration.as_micros());
}

fn big_haystack_1() {
    const REGEX: Regex<1> = compile_regex!("ab+c");

    let start = std::time::SystemTime::now();
    let regex_runtime = ::regex::Regex::new("ab+c").unwrap();
    let duration = start.elapsed().unwrap();
    println!("Runtime compilation duration: {}µs", duration.as_micros());

    let haystacks = [
        "abc".to_owned() + &"ac".repeat(10000),
        "ac".repeat(2500) + "abc" + &"ac".repeat(7500),
        "ac".repeat(5000) + "abc" + &"ac".repeat(5000),
        "ac".repeat(7500) + "abc" + &"ac".repeat(2500),
        "ac".repeat(10000) + "abc",
        "abc".to_owned() + &"az".repeat(10000),
        "az".repeat(2500) + "abc" + &"az".repeat(7500),
        "az".repeat(5000) + "abc" + &"az".repeat(5000),
        "az".repeat(7500) + "abc" + &"az".repeat(2500),
        "az".repeat(10000) + "abc",
        "abc".to_owned() + &"zz".repeat(10000),
        "zz".repeat(2500) + "abc" + &"zz".repeat(7500),
        "zz".repeat(5000) + "abc" + &"zz".repeat(5000),
        "zz".repeat(7500) + "abc" + &"zz".repeat(2500),
        "zz".repeat(10000) + "abc",
    ];

    let start = std::time::SystemTime::now();
    for _ in 0..1000 {
        for haystack in haystacks.iter() {
            assert!(REGEX.test(haystack))
        }
    }
    let duration = start.elapsed().unwrap();
    println!("Test (ere) duration:   {}µs", duration.as_micros());

    let start = std::time::SystemTime::now();
    for _ in 0..1000 {
        for haystack in haystacks.iter() {
            assert!(regex_runtime.is_match(haystack))
        }
    }
    let duration = start.elapsed().unwrap();
    println!("Test (regex) duration: {}µs", duration.as_micros());
}

fn main() {
    println!("ipv4");
    ipv4();
    println!("Big haystack 1");
    big_haystack_1();
}
