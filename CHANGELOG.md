# Changelog

All notable changes to this project are documented here. Format based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.1.0 - 2026-11-03

Initial release.

### Added

- `SnapshotMatrix`, which validates a trajectory and builds the consecutive
  pair matrices `X` and `Y`, with typed errors for too few snapshots,
  inconsistent dimensions, empty states and non-finite values.
- `dmd`, which fits `A = Y X^+` using normal equations on the smaller Gram
  matrix and Gaussian elimination with partial pivoting, and reports a singular
  system as `Error::SingularSystem`.
- `Dmd` with `operator`, `eigenvalues`, `modes`, `spectral_radius`,
  `initial_state` and `reconstruct`, including a fallback to operator powers
  when the eigenvector matrix cannot be used.
- `edmd`, which lifts a trajectory through observables and applies DMD to the
  lifted states.
- `Complex`, a dependency-free complex type used for eigenvalues and modes.
