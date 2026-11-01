//! Snapshot validation and the consecutive-pair matrices.

use crate::error::{Error, Result};

/// The pair matrices `X` and `Y` formed from a trajectory.
///
/// For a trajectory `[x_0, x_1, ..., x_m]` of `m + 1` states with `n`
/// components each, `X = [x_0 ... x_{m-1}]` and `Y = [x_1 ... x_m]` both have
/// shape `n x m`. Rows are state components and columns are time steps.
#[derive(Clone, Debug)]
pub struct SnapshotMatrix {
    x: Vec<Vec<f64>>,
    y: Vec<Vec<f64>>,
    initial: Vec<f64>,
}

impl SnapshotMatrix {
    /// Validates a trajectory and builds the pair matrices.
    ///
    /// # Errors
    ///
    /// Returns [`Error::TooFewSnapshots`] for fewer than two states,
    /// [`Error::EmptyState`] for an empty state vector,
    /// [`Error::DimensionMismatch`] for inconsistent state dimensions, and
    /// [`Error::NonFinite`] for `NaN` or infinite components.
    pub fn new(trajectory: &[Vec<f64>]) -> Result<Self> {
        if trajectory.len() < 2 {
            return Err(Error::TooFewSnapshots {
                got: trajectory.len(),
            });
        }
        let dimension = trajectory[0].len();
        if dimension == 0 {
            return Err(Error::EmptyState { index: 0 });
        }
        for (index, state) in trajectory.iter().enumerate() {
            if state.len() != dimension {
                return Err(Error::DimensionMismatch {
                    index,
                    expected: dimension,
                    found: state.len(),
                });
            }
            if let Some(component) = state.iter().position(|value| !value.is_finite()) {
                return Err(Error::NonFinite { index, component });
            }
        }

        let pairs = trajectory.len() - 1;
        let mut x = vec![vec![0.0; pairs]; dimension];
        let mut y = vec![vec![0.0; pairs]; dimension];
        for (index, state) in trajectory.iter().enumerate() {
            for (row, &value) in state.iter().enumerate() {
                if index < pairs {
                    x[row][index] = value;
                }
                if index > 0 {
                    y[row][index - 1] = value;
                }
            }
        }

        Ok(Self {
            x,
            y,
            initial: trajectory[0].clone(),
        })
    }

    /// The `X` matrix, shape `n x pairs`, with rows as state components.
    pub fn x(&self) -> &[Vec<f64>] {
        &self.x
    }

    /// The `Y` matrix, shape `n x pairs`, with rows as state components.
    pub fn y(&self) -> &[Vec<f64>] {
        &self.y
    }

    /// The state dimension `n`.
    pub fn state_dimension(&self) -> usize {
        self.x.len()
    }

    /// The number of consecutive pairs `m`.
    pub fn pairs(&self) -> usize {
        self.x.first().map_or(0, Vec::len)
    }

    /// The first trajectory state.
    pub fn initial_state(&self) -> &[f64] {
        &self.initial
    }
}
