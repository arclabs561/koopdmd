# koopdmd

Dynamic mode decomposition operators: DMD and EDMD (extended DMD, which
approximates the Koopman operator on caller-chosen observables).

This crate implements the operator core: fit the best linear operator
`A = Y X^+` from consecutive snapshots, eigen-decompose it, and reconstruct the
trajectory spectrally. It has no dependencies and targets small state spaces.

```toml
[dependencies]
koopdmd = "0.1"
```

```rust
use koopdmd::{dmd, SnapshotMatrix};

let mut state = vec![1.0, 1.0];
let mut trajectory = vec![state.clone()];
for _ in 0..20 {
    state = vec![0.9 * state[0], 0.5 * state[1]];
    trajectory.push(state.clone());
}

let model = dmd(&SnapshotMatrix::new(&trajectory).unwrap()).unwrap();
let largest = model
    .eigenvalues()
    .iter()
    .map(|value| value.re)
    .fold(f64::MIN, f64::max);
assert!((largest - 0.9).abs() < 1e-9);
assert_eq!(model.reconstruct(10).len(), 10);
```

## What it is

- `SnapshotMatrix`: validates a trajectory (`NaN`, inconsistent dimensions,
  too few states) and exposes the consecutive-pair matrices `X` and `Y`.
- `dmd`: fits `A = Y X^+` and returns eigenvalues, right modes, the operator,
  and a spectral reconstruction. It returns `Error::SingularSystem` instead of
  silently fitting a singular system.
- `edmd`: lifts states with a caller-provided observables map, then applies DMD
  to the lifted trajectory.
- `Complex`: a small owned complex type, so eigenvalues need no dependency.

## Boundary

This is the operator-focused first slice, not a broad control, neural Koopman,
or forecasting framework. Kernel DMD and Koopman eigenfunction approximations
are not in 0.1.0; they stay out until a consumer needs them.

The solver is SVD-free and intended for small systems. It forms normal
equations, which squares the condition number of the snapshot matrix. A
singular or grossly rank-deficient system is reported as
`Error::SingularSystem` only when Gaussian elimination with partial pivoting
meets a pivot below a fixed relative tolerance. Ill-conditioned but nonsingular
problems can still produce inaccurate operators; use a better-excited, longer
trajectory, or an SVD-based implementation, for those.

DMD fits a linear operator. Affine dynamics `x_{k+1} = A x_k + c` are not
modelled unless the state is augmented with a constant observable through
`edmd`.

Reconstruction is spectral, `x_k = Re(Phi Lambda^k b)` with
`b = Phi^+ x_0`. If the eigenvector matrix is singular or too ill-conditioned
to solve (a defective or repeated-eigenvalue system), reconstruction falls back
to repeated multiplication by the fitted operator; eigenvalues and modes remain
available, but the fallback does not use them.

The eigen-solver computes the characteristic polynomial and iterates its roots.
It is not a general Schur or SVD eigensolver and eigenvalue accuracy degrades
as the spectrum becomes clustered or the state dimension grows.

## Testing

`just check` runs formatting, Clippy with `-D warnings`, the test suite and
docs. The tests cover the exact eigenvalues of a diagonal linear system,
20-step reconstruction of a non-diagonal linear system, a complex-conjugate
rotation pair, EDMD on `x -> 0.5x + 0.2x^2` with observables `[x, x^2]`,
singular-system and snapshot-length errors, and reconstruction of a defective
operator through the fallback.

## License

MIT OR Apache-2.0.
