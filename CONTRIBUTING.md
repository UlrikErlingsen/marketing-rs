# Contributing

Use Rust 1.88 or later. Run formatting, strict Clippy and all tests before proposing a change:

```sh
cargo fmt --all --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
```

## Sources and answers

Each model needs a named primary peer-reviewed article or a standard textbook. Cite authors, year, title, journal or publisher, and DOI where available. Explain assumptions, required inputs, units, limits and situations where the result should not be trusted. Add a short compiling example and an answer-key test against a reference implementation or a published worked example.

Keep reference answers and their provenance separate from the implementation. Record source revisions, fixture hashes, data origin and comparison tolerances. Use synthetic data or clearly identified published aggregate worked examples, never private customer records. Use only data with terms that permit redistribution. Keep borrowed prose, scans and logos out of a contribution.

Do not rewrite an expected answer or loosen a tolerance just to make a test pass. Investigate discrepancies against the primary source. A deliberate correction must explain the changed output and its reason in the changelog. Bug fixes need a regression that fails before the fix and passes afterward. Preserve existing public behavior unless an intentional versioned change is documented.

Keep inputs simple and refuse invalid or failed calculations clearly. Numerical routines belong in the library; file imports, databases and application policy do not. Retain source attribution and licence notices when adapting code. Package files must include required notices and keep repository fixtures out of the crate archive.

## Sign-off

Sign off each commit under the [Developer Certificate of Origin](https://developercertificate.org/):

```sh
git commit -s
```

Use your own name and email. The sign-off records your certification that you may submit the contribution under the project's licence. Contributions use MIT OR Apache-2.0, at the recipient's option.
