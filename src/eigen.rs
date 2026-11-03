//! Eigen-decomposition of a small real matrix without an external library.

use crate::complex::Complex;
use crate::error::{Error, Result};
use crate::linalg::{mat_mul, zeros};

/// Computes eigenvalues and right eigenvectors of a small real matrix.
///
/// The characteristic polynomial comes from the Faddeev-LeVerrier recurrence
/// and its roots from Durand-Kerner iteration followed by Newton polishing.
/// This is intended for the small state spaces DMD is used on here; it is not
/// a replacement for a Schur or SVD-based eigensolver.
pub(crate) fn eigen_decomposition(
    matrix: &[Vec<f64>],
) -> Result<(Vec<Complex>, Vec<Vec<Complex>>)> {
    let n = matrix.len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return Err(Error::EigenvalueIteration);
    }

    let coefficients = characteristic_polynomial(matrix);
    let mut eigenvalues = polynomial_roots(&coefficients)?;
    eigenvalues.sort_by(|a, b| {
        b.norm()
            .total_cmp(&a.norm())
            .then_with(|| b.re.total_cmp(&a.re))
            .then_with(|| b.im.total_cmp(&a.im))
    });

    let mut modes = Vec::with_capacity(n);
    for &eigenvalue in &eigenvalues {
        let mode = eigenvector(matrix, eigenvalue).ok_or(Error::EigenvectorFailure)?;
        modes.push(mode);
    }

    Ok((eigenvalues, modes))
}

fn trace_product(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    left.iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(k, value)| value * right[k][row])
                .sum::<f64>()
        })
        .sum()
}

/// Returns monic coefficients `[1, c_1, ..., c_n]`.
fn characteristic_polynomial(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    let mut coefficients = vec![0.0; n + 1];
    coefficients[0] = 1.0;

    let mut current = zeros(n);
    for k in 1..=n {
        current = mat_mul(matrix, &current);
        for (row, values) in current.iter_mut().enumerate() {
            values[row] += coefficients[k - 1];
        }
        coefficients[k] = -trace_product(matrix, &current) / k as f64;
    }

    coefficients
}

fn evaluate(coefficients: &[f64], value: Complex) -> Complex {
    coefficients
        .iter()
        .fold(Complex::ZERO, |acc, &coefficient| {
            acc * value + Complex::from(coefficient)
        })
}

/// Finds all roots of a monic real polynomial by Durand-Kerner iteration.
fn polynomial_roots(coefficients: &[f64]) -> Result<Vec<Complex>> {
    let n = coefficients.len().saturating_sub(1);
    if n == 0 {
        return Ok(Vec::new());
    }

    let radius = 1.0
        + (1..=n)
            .map(|i| coefficients[i].abs().powf(1.0 / i as f64))
            .fold(0.0, f64::max);
    let spiral = Complex::new(0.4, 0.9);
    let mut roots = Vec::with_capacity(n);
    for step in 0..n {
        let mut point = Complex::ONE;
        for _ in 0..step {
            point = point * spiral;
        }
        let norm = point.norm();
        roots.push(if norm > 0.0 {
            point * Complex::from(radius / norm)
        } else {
            Complex::from(radius)
        });
    }

    for _ in 0..1000 {
        let mut largest_step: f64 = 0.0;
        for index in 0..n {
            let current = roots[index];
            let mut denominator = Complex::ONE;
            for (other, &root) in roots.iter().enumerate() {
                if other != index {
                    denominator = denominator * (current - root);
                }
            }
            if denominator.norm() == 0.0 {
                roots[index] = current + Complex::new(1e-8, 1e-8);
                continue;
            }
            let step = evaluate(coefficients, current) / denominator;
            roots[index] = current - step;
            largest_step = largest_step.max(step.norm());
        }
        let magnitude = roots.iter().map(|root| root.norm()).fold(0.0, f64::max);
        if largest_step <= 1e-14 * (1.0 + magnitude) {
            break;
        }
    }

    let derivative: Vec<f64> = coefficients[..n]
        .iter()
        .enumerate()
        .map(|(index, coefficient)| coefficient * (n - index) as f64)
        .collect();
    for root in &mut roots {
        for _ in 0..4 {
            let value = evaluate(coefficients, *root);
            let slope = evaluate(&derivative, *root);
            if slope.norm() <= f64::MIN_POSITIVE {
                break;
            }
            let step = value / slope;
            *root = *root - step;
            if step.norm() <= 1e-16 * (1.0 + root.norm()) {
                break;
            }
        }
        if root.im.abs() <= 1e-12 * (1.0 + root.re.abs()) {
            root.im = 0.0;
        }
        if !root.is_finite() {
            return Err(Error::EigenvalueIteration);
        }
    }

    // Durand-Kerner converges only linearly at a multiple root and can stall
    // with a split of about sqrt(machine epsilon). Accept such roots when the
    // relative polynomial residual is still tiny.
    for root in &roots {
        let magnitude = root.norm();
        let reference = coefficients
            .iter()
            .enumerate()
            .map(|(index, coefficient)| coefficient.abs() * magnitude.powi((n - index) as i32))
            .sum::<f64>()
            .max(f64::MIN_POSITIVE);
        if evaluate(coefficients, *root).norm() > 1e-6 * reference {
            return Err(Error::EigenvalueIteration);
        }
    }

    Ok(roots)
}

/// Returns a unit right eigenvector for `eigenvalue`.
///
/// The matrix `A - lambda I` is reduced to reduced row echelon form and a free
/// column is used for the null vector. Tolerances are retried from tight to
/// loose so a numerically split multiple eigenvalue still yields a direction.
fn eigenvector(matrix: &[Vec<f64>], eigenvalue: Complex) -> Option<Vec<Complex>> {
    for factor in [1e-12, 1e-9, 1e-6] {
        if let Some(vector) = null_vector(matrix, eigenvalue, factor) {
            return Some(vector);
        }
    }
    None
}

fn null_vector(matrix: &[Vec<f64>], eigenvalue: Complex, factor: f64) -> Option<Vec<Complex>> {
    let n = matrix.len();
    let mut reduced: Vec<Vec<Complex>> = matrix
        .iter()
        .enumerate()
        .map(|(row, values)| {
            values
                .iter()
                .enumerate()
                .map(|(column, &value)| {
                    if row == column {
                        Complex::from(value) - eigenvalue
                    } else {
                        Complex::from(value)
                    }
                })
                .collect()
        })
        .collect();

    let scale = matrix
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let tolerance = factor * scale.max(1.0) * n as f64;

    let mut pivots = Vec::new();
    let mut pivot_row = 0;
    for column in 0..n {
        if pivot_row == n {
            break;
        }
        let (row, magnitude) = (pivot_row..n)
            .map(|candidate| (candidate, reduced[candidate][column].norm()))
            .fold((pivot_row, 0.0), |best, candidate| {
                if candidate.1 > best.1 {
                    candidate
                } else {
                    best
                }
            });
        if magnitude <= tolerance {
            continue;
        }
        reduced.swap(pivot_row, row);

        let pivot = reduced[pivot_row][column];
        for entry in reduced[pivot_row].iter_mut().skip(column) {
            *entry = *entry / pivot;
        }
        let pivot_entries = reduced[pivot_row].clone();
        for (row, values) in reduced.iter_mut().enumerate() {
            if row == pivot_row {
                continue;
            }
            let multiplier = values[column];
            if multiplier.norm() == 0.0 {
                continue;
            }
            for (entry, &pivot_entry) in pivot_entries.iter().enumerate().skip(column) {
                values[entry] = values[entry] - multiplier * pivot_entry;
            }
        }
        pivots.push(column);
        pivot_row += 1;
    }

    if pivots.len() == n {
        return None;
    }
    let free = (0..n).find(|column| !pivots.contains(column))?;
    let mut vector = vec![Complex::ZERO; n];
    vector[free] = Complex::ONE;
    for (row, &column) in pivots.iter().enumerate() {
        vector[column] = -reduced[row][free];
    }

    normalize(vector)
}

fn normalize(mut vector: Vec<Complex>) -> Option<Vec<Complex>> {
    let norm = vector
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    if !norm.is_finite() || norm == 0.0 {
        return None;
    }
    for value in &mut vector {
        *value = *value / Complex::from(norm);
    }

    let (index, _) = vector
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.norm().total_cmp(&right.norm()))?;
    let phase = vector[index] / Complex::from(vector[index].norm());
    if phase.norm() > 0.0 {
        for value in &mut vector {
            *value = *value / phase;
        }
    }

    Some(vector)
}
