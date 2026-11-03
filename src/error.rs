use std::fmt;

/// Errors from snapshot validation, operator fitting, lifting and
/// eigen-decomposition.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Error {
    /// A trajectory needs at least two snapshots to form one pair.
    TooFewSnapshots {
        /// Number of snapshots that were provided.
        got: usize,
    },
    /// A snapshot has a different state dimension than the first snapshot.
    DimensionMismatch {
        /// Index of the offending snapshot.
        index: usize,
        /// Dimension of the first snapshot.
        expected: usize,
        /// Dimension of the offending snapshot.
        found: usize,
    },
    /// A snapshot has no state components.
    EmptyState {
        /// Index of the empty snapshot.
        index: usize,
    },
    /// A snapshot contains `NaN` or an infinity.
    NonFinite {
        /// Index of the offending snapshot.
        index: usize,
        /// Index of the offending component.
        component: usize,
    },
    /// The observables map produced vectors of differing dimension.
    ObservableDimensionMismatch {
        /// Index of the offending snapshot.
        index: usize,
        /// Dimension produced for the first snapshot.
        expected: usize,
        /// Dimension produced for the offending snapshot.
        found: usize,
    },
    /// The observables map produced an empty vector.
    EmptyObservation {
        /// Index of the snapshot whose observation was empty.
        index: usize,
    },
    /// The least-squares system is singular or numerically rank deficient.
    SingularSystem,
    /// Eigenvalue iteration did not converge to an acceptable residual.
    EigenvalueIteration,
    /// An eigenvector could not be isolated.
    EigenvectorFailure,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TooFewSnapshots { got } => {
                write!(f, "a trajectory needs at least two snapshots, got {got}")
            }
            Self::DimensionMismatch {
                index,
                expected,
                found,
            } => write!(
                f,
                "snapshot {index} has dimension {found}, expected {expected}"
            ),
            Self::EmptyState { index } => {
                write!(f, "snapshot {index} has no state components")
            }
            Self::NonFinite { index, component } => {
                write!(f, "snapshot {index} component {component} is not finite")
            }
            Self::ObservableDimensionMismatch {
                index,
                expected,
                found,
            } => write!(
                f,
                "observables produced dimension {found} at snapshot {index}, expected {expected}"
            ),
            Self::EmptyObservation { index } => {
                write!(
                    f,
                    "observables produced an empty vector at snapshot {index}"
                )
            }
            Self::SingularSystem => {
                write!(
                    f,
                    "the snapshot matrix is singular or numerically rank deficient"
                )
            }
            Self::EigenvalueIteration => write!(f, "eigenvalue iteration did not converge"),
            Self::EigenvectorFailure => write!(f, "could not isolate an eigenvector"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;
