//! The fitted DMD operator and its eigen-decomposition.

use crate::complex::{solve as solve_complex, Complex};
use crate::eigen::eigen_decomposition;
use crate::error::{Error, Result};
use crate::linalg::{least_squares_operator, mat_vec};
use crate::snapshot::SnapshotMatrix;

/// A fitted linear operator with its spectrum and right eigenvectors.
#[derive(Clone, Debug)]
pub struct Dmd {
    operator: Vec<Vec<f64>>,
    eigenvalues: Vec<Complex>,
    modes: Vec<Vec<Complex>>,
    initial_state: Vec<f64>,
}

impl Dmd {
    /// The fitted operator `A = Y X^+`, shape `n x n`.
    pub fn operator(&self) -> &[Vec<f64>] {
        &self.operator
    }

    /// The eigenvalues of the fitted operator, ordered by decreasing norm.
    pub fn eigenvalues(&self) -> &[Complex] {
        &self.eigenvalues
    }

    /// The right eigenvectors, one per eigenvalue in the same order.
    ///
    /// Each mode is unit norm with its largest component rotated to be real
    /// and positive. Modes of a complex-conjugate pair are conjugates up to
    /// numerical noise.
    pub fn modes(&self) -> &[Vec<Complex>] {
        &self.modes
    }

    /// The first state of the training trajectory.
    pub fn initial_state(&self) -> &[f64] {
        &self.initial_state
    }

    /// The largest eigenvalue norm.
    pub fn spectral_radius(&self) -> f64 {
        self.eigenvalues
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
    }

    /// Reconstructs the states at times `1..=steps` after the first snapshot.
    ///
    /// Uses the spectral expansion `x_k = Re(Phi Lambda^k b)` with
    /// `b = Phi^+ x_0`. When the eigenvector matrix is singular or too
    /// ill-conditioned to solve, reconstruction falls back to repeated
    /// multiplication by the fitted operator, which is exact for the fitted
    /// one-step map but does not use the eigen-decomposition.
    pub fn reconstruct(&self, steps: usize) -> Vec<Vec<f64>> {
        let mut reconstructed = Vec::with_capacity(steps);
        if steps == 0 {
            return reconstructed;
        }

        if let Some(coefficients) = self.spectral_coefficients() {
            let mut powers = vec![Complex::ONE; self.eigenvalues.len()];
            for _ in 0..steps {
                for (power, &eigenvalue) in powers.iter_mut().zip(&self.eigenvalues) {
                    *power = *power * eigenvalue;
                }
                let mut state = vec![0.0; self.initial_state.len()];
                for ((mode, &coefficient), &power) in
                    self.modes.iter().zip(&coefficients).zip(&powers)
                {
                    let weight = power * coefficient;
                    for (value, &mode_value) in state.iter_mut().zip(mode) {
                        *value += (mode_value * weight).re;
                    }
                }
                reconstructed.push(state);
            }
            return reconstructed;
        }

        let mut state = self.initial_state.clone();
        for _ in 0..steps {
            state = mat_vec(&self.operator, &state);
            reconstructed.push(state.clone());
        }
        reconstructed
    }

    fn spectral_coefficients(&self) -> Option<Vec<Complex>> {
        let n = self.initial_state.len();
        if n == 0 || self.modes.len() != n {
            return None;
        }
        let mut phi = vec![vec![Complex::ZERO; n]; n];
        for (column, mode) in self.modes.iter().enumerate() {
            for (row, &value) in mode.iter().enumerate() {
                phi[row][column] = value;
            }
        }
        let right_hand_side = self
            .initial_state
            .iter()
            .map(|&value| Complex::from(value))
            .collect();
        solve_complex(phi, right_hand_side)
    }
}

/// Fits `A = Y X^+` and its eigendecomposition from consecutive snapshots.
///
/// # Errors
///
/// Returns [`Error::SingularSystem`] when the least-squares solve meets a
/// singular or numerically rank-deficient Gram matrix, and an eigen error when
/// the spectrum cannot be isolated. Normal equations square the condition
/// number of `X`; well-excited trajectories are required for accurate fits.
pub fn dmd(snapshots: &SnapshotMatrix) -> Result<Dmd> {
    let operator = least_squares_operator(snapshots.x(), snapshots.y())?;
    let (eigenvalues, modes) = eigen_decomposition(&operator)?;
    Ok(Dmd {
        operator,
        eigenvalues,
        modes,
        initial_state: snapshots.initial_state().to_vec(),
    })
}

/// Lifts a trajectory into observables, then fits DMD on the lifted states.
///
/// The observables map receives each raw state and must return a vector of
/// fixed dimension. The returned model operates on lifted states, so its
/// modes and reconstruction live in observable space, not the original state
/// space. An affine system can be made linear by including a constant
/// observable.
///
/// # Errors
///
/// Returns [`Error::TooFewSnapshots`] for fewer than two states,
/// [`Error::EmptyObservation`] when the map returns an empty vector, and
/// [`Error::ObservableDimensionMismatch`] when the map returns differing
/// dimensions. The lifted trajectory is then validated like any other.
pub fn edmd<F>(trajectory: &[Vec<f64>], observables: F) -> Result<Dmd>
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    if trajectory.len() < 2 {
        return Err(Error::TooFewSnapshots {
            got: trajectory.len(),
        });
    }

    let mut lifted = Vec::with_capacity(trajectory.len());
    let mut dimension = None;
    for (index, state) in trajectory.iter().enumerate() {
        let observation = observables(state);
        if observation.is_empty() {
            return Err(Error::EmptyObservation { index });
        }
        match dimension {
            None => dimension = Some(observation.len()),
            Some(expected) if expected != observation.len() => {
                return Err(Error::ObservableDimensionMismatch {
                    index,
                    expected,
                    found: observation.len(),
                });
            }
            Some(_) => {}
        }
        lifted.push(observation);
    }

    let snapshots = SnapshotMatrix::new(&lifted)?;
    dmd(&snapshots)
}
