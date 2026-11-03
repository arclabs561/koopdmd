use koopdmd::{dmd, edmd, Complex, Error, SnapshotMatrix};

fn simulate(matrix: [[f64; 2]; 2], initial: [f64; 2], steps: usize) -> Vec<Vec<f64>> {
    let mut state = initial;
    let mut trajectory = vec![state.to_vec()];
    for _ in 0..steps {
        state = [
            matrix[0][0] * state[0] + matrix[0][1] * state[1],
            matrix[1][0] * state[0] + matrix[1][1] * state[1],
        ];
        trajectory.push(state.to_vec());
    }
    trajectory
}

fn max_error(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    left.iter()
        .zip(right)
        .flat_map(|(expected, actual)| {
            expected
                .iter()
                .zip(actual)
                .map(|(expected, actual)| (expected - actual).abs())
        })
        .fold(0.0, f64::max)
}

#[test]
fn recovers_eigenvalues_of_a_linear_system() {
    let trajectory = simulate([[0.9, 0.0], [0.0, 0.5]], [1.0, 1.0], 20);
    let snapshots = SnapshotMatrix::new(&trajectory).unwrap();
    let model = dmd(&snapshots).unwrap();

    assert_eq!(model.eigenvalues().len(), 2);
    for expected in [0.9, 0.5] {
        let distance = model
            .eigenvalues()
            .iter()
            .map(|value| (*value - Complex::from(expected)).norm())
            .fold(f64::INFINITY, f64::min);
        assert!(
            distance < 1e-9,
            "eigenvalue {expected} is off by {distance}"
        );
    }
}

#[test]
fn reconstructs_a_linear_trajectory() {
    let trajectory = simulate([[0.9, 0.1], [0.0, 0.5]], [0.5, -0.25], 20);
    let snapshots = SnapshotMatrix::new(&trajectory).unwrap();
    let model = dmd(&snapshots).unwrap();

    let reconstructed = model.reconstruct(20);
    assert_eq!(reconstructed.len(), 20);
    let error = max_error(&trajectory[1..], &reconstructed);
    assert!(error < 1e-9, "reconstruction error is {error}");
}

#[test]
fn recovers_a_complex_conjugate_pair() {
    let angle: f64 = 0.35;
    let (sin, cos) = angle.sin_cos();
    let trajectory = simulate([[cos, -sin], [sin, cos]], [1.0, 0.0], 20);
    let snapshots = SnapshotMatrix::new(&trajectory).unwrap();
    let model = dmd(&snapshots).unwrap();

    let expected = Complex::new(cos, sin);
    let distance = model
        .eigenvalues()
        .iter()
        .map(|value| (*value - expected).norm())
        .fold(f64::INFINITY, f64::min);
    assert!(distance < 1e-9, "complex eigenvalue is off by {distance}");

    let reconstructed = model.reconstruct(20);
    let error = max_error(&trajectory[1..], &reconstructed);
    assert!(error < 1e-9, "rotation reconstruction error is {error}");
}

#[test]
fn edmd_recovers_a_stable_operator_on_a_nonlinear_map() {
    let mut state = 0.1;
    let mut trajectory = vec![vec![state]];
    for _ in 0..30 {
        state = 0.5 * state + 0.2 * state * state;
        trajectory.push(vec![state]);
    }

    let model = edmd(&trajectory, |value| vec![value[0], value[0] * value[0]]).unwrap();
    let operator = model.operator();

    // x' = 0.5 x + 0.2 x^2 is exactly linear in [x, x^2] on the first row.
    assert!((operator[0][0] - 0.5).abs() < 1e-9);
    assert!((operator[0][1] - 0.2).abs() < 1e-9);
    assert!(
        model.spectral_radius() < 1.0,
        "spectral radius is {}",
        model.spectral_radius()
    );
    assert_eq!(model.eigenvalues().len(), 2);
}

#[test]
fn singular_system_returns_a_typed_error() {
    let trajectory = vec![vec![1.0, 1.0], vec![2.0, 2.0], vec![3.0, 3.0]];
    let snapshots = SnapshotMatrix::new(&trajectory).unwrap();
    assert_eq!(dmd(&snapshots).unwrap_err(), Error::SingularSystem);
}

#[test]
fn rejects_mismatched_snapshot_lengths() {
    let error = SnapshotMatrix::new(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
    assert_eq!(
        error,
        Error::DimensionMismatch {
            index: 1,
            expected: 2,
            found: 1,
        }
    );
}

#[test]
fn rejects_trajectories_with_too_few_snapshots() {
    assert_eq!(
        SnapshotMatrix::new(&[]).unwrap_err(),
        Error::TooFewSnapshots { got: 0 }
    );
    assert_eq!(
        SnapshotMatrix::new(&[vec![1.0, 2.0]]).unwrap_err(),
        Error::TooFewSnapshots { got: 1 }
    );
}

#[test]
fn rejects_non_finite_states() {
    let error = SnapshotMatrix::new(&[vec![1.0, 2.0], vec![f64::NAN, 3.0]]).unwrap_err();
    assert_eq!(
        error,
        Error::NonFinite {
            index: 1,
            component: 0,
        }
    );
}

#[test]
fn edmd_rejects_inconsistent_observables() {
    let trajectory = vec![vec![1.0], vec![2.0], vec![3.0]];
    let error = edmd(&trajectory, |value| {
        if value[0] > 1.5 {
            vec![value[0], value[0] * value[0]]
        } else {
            vec![value[0]]
        }
    })
    .unwrap_err();
    assert_eq!(
        error,
        Error::ObservableDimensionMismatch {
            index: 1,
            expected: 1,
            found: 2,
        }
    );
}

#[test]
fn defective_operator_reconstructs_through_operator_powers() {
    let trajectory = simulate([[0.5, 1.0], [0.0, 0.5]], [0.0, 1.0], 20);
    let snapshots = SnapshotMatrix::new(&trajectory).unwrap();
    let model = dmd(&snapshots).unwrap();

    let reconstructed = model.reconstruct(20);
    let error = max_error(&trajectory[1..], &reconstructed);
    assert!(error < 1e-9, "defective reconstruction error is {error}");
}
