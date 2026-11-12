# Source and dependency notices

The source code is copyright 2026 Ulrik Øhrn Erlingsen. Each file under `src/`
and the example carries the MIT copyright and permission notice. The software is
available under MIT OR Apache-2.0; both complete licence texts are included at
the package root.

The fitted models use a bounded Nelder-Mead optimizer that is part of this crate.
Finite-range admission checks refuse unsupported combinations and nonfinite
calculated values, as documented in the changelog.

`statrs` 0.18.0 supplies log-gamma and log-beta functions. It is MIT licensed;
its copyright and permission notice are retained in `statrs-MIT.txt`. The
dependency is used with default features disabled. No statrs source is vendored.

Academic citations name the published mathematical methods. Reference inputs
are synthetic. Publisher artwork and borrowed datasets are not included.
