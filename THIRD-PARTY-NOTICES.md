# Third-party notices — Crystallography.Native

This repository's own source code is licensed under the MIT License (see
[`LICENSE.md`](LICENSE.md)), Copyright (c) 2002-2026 Yusuke SETO.

It additionally vendors third-party source that is **compiled into** the resulting
`Crystallography.Native.dll` (and its `*.avx2.dll` / `*.avx512.dll` variants; on
win-arm64 a single NEON-optimized build with no AVX flavors). These libraries are
statically compiled in rather than shipped as separate binaries, but their notices
are preserved here.

## Eigen

| Field | Value |
| --- | --- |
| Component | Eigen |
| Version | 5.0.1-dev+master snapshot, upstream commit `5a19218d5a28bfc38ca6b1475ea040c8f73257ba` (2026-11-06); from `Eigen/Version`: WORLD 3, MAJOR 5, MINOR 0, PATCH 1. Updated 2026-11-06 (261106Ch); the version string alone does not identify a development snapshot |
| Source archive | <https://gitlab.com/api/v4/projects/15462818/repository/archive.zip?sha=5a19218d5a28bfc38ca6b1475ea040c8f73257ba> (sha256 `453b9ab5765c8752d65a39713af2afbd644c56a99df88d3ca7c578374794aede`) |
| Purpose | Linear algebra, compiled into `Crystallography.Native.dll` |
| Form | Vendored header-only source, compiled in |
| Trees | `Eigen/` (supported modules), `contrib/` (including `MatrixFunctions`, i.e. the matrix exponential), and upstream `unsupported/` forwarding headers |
| Upstream | <https://libeigen.gitlab.io/> |
| License | MPL-2.0 (Mozilla Public License, version 2.0), with permissive portions noted below |

Previous snapshot (260905Cl): `17d8ddbd22c277c6bd20e19e5b8d128b0135a9f4` (2026-09-03), also labeled 5.0.1-dev+master. The latest stable release checked on 2026-11-06 is 5.0.1 (2025-11-11); this update retains the existing development-snapshot track instead of downgrading to that older release. Upstream files are imported without local source edits. License texts are preserved under [`eigen-licenses/`](eigen-licenses/COPYING.README).

The Eigen headers are distributed primarily under the Mozilla Public License,
version 2.0; its full text is available at <https://www.mozilla.org/MPL/2.0/>. This
includes the `contrib/Eigen/MatrixFunctions` module this project relies on for the
matrix exponential — `contrib/Eigen/src/MatrixFunctions/MatrixExponential.h` and
its siblings carry the standard Eigen MPL-2.0 file header.

The vendored Eigen tree also contains portions under other **permissive** licenses,
whose notices are preserved in their respective source files:

- **Apache-2.0**: `Eigen/src/Core/arch/Default/BFloat16.h` is derived from TensorFlow,
  Copyright 2017 The TensorFlow Authors, licensed under the Apache License, Version 2.0
  (<https://www.apache.org/licenses/LICENSE-2.0>).
- **BSD-3-Clause**: the optional BLAS / LAPACKE / MKL / Pardiso backend headers (e.g.
  `Eigen/src/Core/products/*_BLAS.h`, `Eigen/src/*/*_LAPACKE.h`, `Eigen/src/misc/lapacke.h`),
  which are inert here (see below). Upstream now tags every file with an SPDX identifier
  (`MPL-2.0` / `BSD-3-Clause` / `Apache-2.0`); the MINPACK-derived `contrib/Eigen/src/LevenbergMarquardt`
  (`LicenseRef-MINPACK`, BSD-like) is present in the tree but not compiled in.

The repository's MIT license does not replace the licenses on vendored files.
MPL-covered files retain their MPL-2.0 obligations; the Apache, BSD and MINPACK
notices are retained as supplied by upstream. The unused Riccati module also
contains SciPy and LAPACK-derived code with notices in its source header. No GPL- or LGPL-licensed Eigen
code is compiled in: the only LGPL mention, in
`Eigen/src/IterativeLinearSolvers/IncompleteLUT.h`, is an algorithm-provenance note
(the reordering idea derives from SPARSKIT) inside an MPL-2.0 file, not a license
grant on Eigen code.

Optional LAPACKE / MKL Eigen backends present in the header tree are inert:
`EIGEN_USE_MKL`, `EIGEN_USE_BLAS`, and `EIGEN_USE_LAPACKE` are never defined in this
project's sources.
