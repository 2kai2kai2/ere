use std::hash::Hasher;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use ere::{compile_regex, Regex};
use pprof::criterion::{Output, PProfProfiler};

fn ipv4(c: &mut Criterion) {
    const REGEX: Regex<5> = compile_regex!(
        r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$"
    );

    let regex_runtime = ::regex::Regex::new(r"^(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])\.(25[0-5]|2[0-4][0-9]|1[0-9]{2}|[1-9][0-9]|[0-9])$").unwrap();

    let mut group = c.benchmark_group("ipv4");
    for i in 0..5 {
        let mut hasher = ::std::hash::DefaultHasher::new();
        hasher.write_u8(i);
        let [b0, b1, b2, b3, ..] = hasher.finish().to_le_bytes();
        let haystack = format!("{b0}.{b1}.{b2}.{b3}");

        group.bench_with_input(
            BenchmarkId::new("ere (test)", &haystack),
            &haystack,
            |b, s| b.iter(|| assert!(REGEX.test(s))),
        );
        group.bench_with_input(
            BenchmarkId::new("regex (test)", &haystack),
            &haystack,
            |b, s| b.iter(|| assert!(regex_runtime.is_match(s))),
        );
    }
}

fn big_haystack_1(c: &mut Criterion) {
    const REGEX: Regex<1> = compile_regex!("ab+c");

    let regex_runtime = ::regex::Regex::new("ab+c").unwrap();

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

    let mut group = c.benchmark_group("big_haystack_1");
    for (i, haystack) in haystacks.iter().enumerate() {
        group.bench_with_input(
            BenchmarkId::new("ere (test)", i),
            haystack.as_str(),
            |b, s| b.iter(|| assert!(REGEX.test(s))),
        );
        group.bench_with_input(
            BenchmarkId::new("regex (test)", i),
            haystack.as_str(),
            |b, s| b.iter(|| assert!(regex_runtime.is_match(s))),
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default().with_profiler(PProfProfiler::new(100, Output::Flamegraph(None)));
    targets = ipv4, big_haystack_1
}
criterion_main!(benches);
