//! Koopman and dynamic mode decomposition operators for small state spaces.
//!
//! Given a trajectory of consecutive states, [`dmd`] fits the best linear
//! operator `A = Y X^+` between the snapshot matrices and exposes its
//! eigenvalues, right eigenvectors and a spectral reconstruction. [`edmd`]
//! applies the same fit after lifting states through an observables map.
//!
//! ```
//! use koopdmd::{dmd, SnapshotMatrix};
//!
//! let mut state = vec![1.0, 1.0];
//! let mut trajectory = vec![state.clone()];
//! for _ in 0..20 {
//!     state = vec![0.9 * state[0], 0.5 * state[1]];
//!     trajectory.push(state.clone());
//! }
//!
//! let snapshots = SnapshotMatrix::new(&trajectory).unwrap();
//! let model = dmd(&snapshots).unwrap();
//! let largest = model
//!     .eigenvalues()
//!     .iter()
//!     .map(|value| value.re)
//!     .fold(f64::MIN, f64::max);
//! assert!((largest - 0.9).abs() < 1e-9);
//! assert_eq!(model.reconstruct(10).len(), 10);
//! ```
//!
//! # Method and limits
//!
//! The operator solve uses normal equations on the smaller Gram matrix and
//! Gaussian elimination with partial pivoting, with no external linear-algebra
//! dependency. Normal equations square the condition number of the snapshot
//! matrix; a singular or grossly rank-deficient system is reported as
//! [`Error::SingularSystem`], but an ill-conditioned one can still yield an
//! inaccurate operator.
//!
//! The eigen-decomposition uses the Faddeev-LeVerrier recurrence for the
//! characteristic polynomial and Durand-Kerner iteration for its roots. It is
//! intended for small systems, not as a general Schur or SVD eigensolver.
//!
//! This crate is the operator core, not a forecasting or control product.
//! Kernel DMD and Koopman eigenfunction approximations are not in 0.1.0.

#![warn(missing_docs)]

mod complex;
mod dmd;
mod eigen;
mod error;
mod linalg;
mod snapshot;

pub use complex::Complex;
pub use dmd::{dmd, edmd, Dmd};
pub use error::{Error, Result};
pub use snapshot::SnapshotMatrix;
