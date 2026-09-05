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
| Version | 5.0.1-dev+master snapshot, upstream commit `17d8ddbd22c277c6bd20e19e5b8d128b0135a9f4` (2026-09-03); from `Eigen/Version`: WORLD 3, MAJOR 5, MINOR 0, PATCH 1. Updated 2026-09-05 (260905Cl) from the previous master snapshot of 2026-03-21..04-05 |
| Source archive | <https://gitlab.com/libeigen/eigen/-/archive/17d8ddbd22c277c6bd20e19e5b8d128b0135a9f4/eigen-17d8ddbd22c277c6bd20e19e5b8d128b0135a9f4.tar.gz> (sha256 `b059ec2590e12c9cc71377bd664e9093adb36eb3716edd8753e5cfd92fa07d8b`) |
| Purpose | Linear algebra, compiled into `Crystallography.Native.dll` |
| Form | Vendored header-only source, compiled in |
| Trees | `Eigen/` (supported modules) and `unsupported/Eigen/` (used for `MatrixFunctions`, i.e. the matrix exponential) |
| Upstream | <https://eigen.tuxfamily.org/> |
| License | MPL-2.0 (Mozilla Public License, version 2.0), with permissive portions noted below |

The Eigen headers are distributed primarily under the Mozilla Public License,
version 2.0; its full text is available at <https://www.mozilla.org/MPL/2.0/>. This
includes the `unsupported/Eigen/MatrixFunctions` module this project relies on for the
matrix exponential — `unsupported/Eigen/src/MatrixFunctions/MatrixExponential.h` and
its siblings carry the standard Eigen MPL-2.0 file header.

The vendored Eigen tree also contains portions under other **permissive** licenses,
whose notices are preserved in their respective source files:

- **Apache-2.0**: `Eigen/src/Core/arch/Default/BFloat16.h` is derived from TensorFlow,
  Copyright 2017 The TensorFlow Authors, licensed under the Apache License, Version 2.0
  (<https://www.apache.org/licenses/LICENSE-2.0>).
- **BSD-3-Clause**: the optional BLAS / LAPACKE / MKL / Pardiso backend headers (e.g.
  `Eigen/src/Core/products/*_BLAS.h`, `Eigen/src/*/*_LAPACKE.h`, `Eigen/src/misc/lapacke.h`),
  which are inert here (see below). Upstream now tags every file with an SPDX identifier
  (`MPL-2.0` / `BSD-3-Clause` / `Apache-2.0`); the MINPACK-derived `unsupported/Eigen/src/LevenbergMarquardt`
  (`LicenseRef-MINPACK`, BSD-like) is present in the tree but not compiled in.

All licenses involved (MPL-2.0, Apache-2.0, BSD) are permissive and compatible with
redistribution under this repository's MIT license; per MPL-2.0 §3.3 the MPL-covered
files may be combined with code under other licenses. No GPL- or LGPL-licensed Eigen
code is compiled in: the only LGPL mention, in
`Eigen/src/IterativeLinearSolvers/IncompleteLUT.h`, is an algorithm-provenance note
(the reordering idea derives from SPARSKIT) inside an MPL-2.0 file, not a license
grant on Eigen code.

Optional LAPACKE / MKL Eigen backends present in the header tree are inert:
`EIGEN_USE_MKL`, `EIGEN_USE_BLAS`, and `EIGEN_USE_LAPACKE` are never defined in this
project's sources.
