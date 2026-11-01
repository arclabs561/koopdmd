//! Dense real matrix helpers for the least-squares operator fit.

use crate::error::{Error, Result};

/// A dense row-major matrix.
pub(crate) type Matrix = Vec<Vec<f64>>;

/// Returns an `n x n` zero matrix.
pub(crate) fn zeros(n: usize) -> Matrix {
    vec![vec![0.0; n]; n]
}

/// Multiplies two dense matrices.
pub(crate) fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Matrix {
    let rows = a.len();
    let inner = a.first().map_or(0, Vec::len);
    let columns = if inner == 0 {
        0
    } else {
        b.first().map_or(0, Vec::len)
    };
    let mut out = vec![vec![0.0; columns]; rows];
    for (row, values) in a.iter().enumerate() {
        for (k, &left) in values.iter().enumerate() {
            if left == 0.0 {
                continue;
            }
            for (column, &right) in b[k].iter().enumerate() {
                out[row][column] += left * right;
            }
        }
    }
    out
}

/// Multiplies a dense matrix by a vector.
pub(crate) fn mat_vec(a: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    a.iter()
        .map(|row| row.iter().zip(x).map(|(left, right)| left * right).sum())
        .collect()
}

/// Solves `a x = b` in place by Gaussian elimination with partial pivoting.
///
/// Each element of `rhs` is one right-hand side column. The pivot tolerance is
/// relative to the largest matrix entry; it detects an exactly singular system
/// and gross rank deficiency, not a large condition number.
#[allow(clippy::needless_range_loop)]
pub(crate) fn solve(mut a: Matrix, mut rhs: Vec<Vec<f64>>) -> Result<Vec<Vec<f64>>> {
    let n = a.len();
    if n == 0 || a.iter().any(|row| row.len() != n) || rhs.iter().any(|column| column.len() != n) {
        return Err(Error::SingularSystem);
    }

    let scale = a
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.abs())
        .fold(0.0, f64::max);
    let tolerance = 1e-12 * scale.max(1.0) * n as f64;

    for k in 0..n {
        let (pivot_row, pivot_norm) =
            (k..n)
                .map(|row| (row, a[row][k].abs()))
                .fold((k, 0.0), |best, candidate| {
                    if candidate.1 > best.1 {
                        candidate
                    } else {
                        best
                    }
                });
        if pivot_norm <= tolerance {
            return Err(Error::SingularSystem);
        }
        if pivot_row != k {
            a.swap(k, pivot_row);
            for column in rhs.iter_mut() {
                column.swap(k, pivot_row);
            }
        }

        let pivot = a[k][k];
        for row in k + 1..n {
            let factor = a[row][k] / pivot;
            if factor == 0.0 {
                continue;
            }
            a[row][k] = 0.0;
            for column in k + 1..n {
                a[row][column] -= factor * a[k][column];
            }
            for right in rhs.iter_mut() {
                right[row] -= factor * right[k];
            }
        }
    }

    for right in rhs.iter_mut() {
        for row in (0..n).rev() {
            let mut value = right[row];
            for column in row + 1..n {
                value -= a[row][column] * right[column];
            }
            right[row] = value / a[row][row];
        }
    }

    Ok(rhs)
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

fn gram(left: &[Vec<f64>], right: &[Vec<f64>]) -> Matrix {
    left.iter()
        .map(|left_row| {
            right
                .iter()
                .map(|right_row| dot(left_row, right_row))
                .collect()
        })
        .collect()
}

fn gram_transposed(x: &[Vec<f64>], columns: usize) -> Matrix {
    (0..columns)
        .map(|i| {
            (0..columns)
                .map(|j| x.iter().map(|row| row[i] * row[j]).sum())
                .collect()
        })
        .collect()
}

/// Fits the least-squares operator `A = Y X^+` for `X` and `Y` of shape
/// `n x columns`.
///
/// Uses the smaller Gram matrix: the `n x n` row form when there are at least
/// as many snapshot pairs as state dimensions, and the `columns x columns`
/// column form otherwise. Normal equations square the condition number of `X`,
/// which is why a singular result is reported rather than silently fitted.
pub(crate) fn least_squares_operator(x: &[Vec<f64>], y: &[Vec<f64>]) -> Result<Matrix> {
    let rows = x.len();
    let columns = x.first().map_or(0, Vec::len);
    if rows == 0 || columns == 0 || y.len() != rows || y.iter().any(|row| row.len() != columns) {
        return Err(Error::SingularSystem);
    }

    if columns >= rows {
        // A = Y X^T (X X^T)^-1. Solve G A^T = B^T with B = Y X^T, using the
        // rows of B as right-hand sides; the solutions are the rows of A.
        let gram_x = gram(x, x);
        let cross = gram(y, x);
        solve(gram_x, cross)
    } else {
        // A = Y (X^T X)^-1 X^T. Solve H W = X^T, then A = Y W.
        let gram_t = gram_transposed(x, columns);
        let w_columns = solve(gram_t, x.to_vec())?;
        let mut w = vec![vec![0.0; rows]; columns];
        for (column, solution) in w_columns.iter().enumerate() {
            for (row, &value) in solution.iter().enumerate() {
                w[row][column] = value;
            }
        }
        Ok(mat_mul(y, &w))
    }
}
