//! A minimal owned complex number type.

use std::fmt;
use std::ops::{Add, Div, Mul, Neg, Sub};

/// A complex number with real and imaginary parts.
///
/// This is deliberately small: enough arithmetic for eigenvalues, modes and
/// spectral reconstruction, with no dependency on a numerics crate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Complex {
    /// Real part.
    pub re: f64,
    /// Imaginary part.
    pub im: f64,
}

impl Complex {
    /// The additive identity.
    pub const ZERO: Self = Self { re: 0.0, im: 0.0 };

    /// The multiplicative identity.
    pub const ONE: Self = Self { re: 1.0, im: 0.0 };

    /// Creates a complex number from its real and imaginary parts.
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    /// Creates a real complex number.
    pub const fn from_real(re: f64) -> Self {
        Self { re, im: 0.0 }
    }

    /// The Euclidean norm.
    pub fn norm(self) -> f64 {
        self.re.hypot(self.im)
    }

    /// The squared Euclidean norm.
    pub fn norm_sqr(self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    /// The complex conjugate.
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    /// Whether both parts are finite.
    pub fn is_finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

impl From<f64> for Complex {
    fn from(value: f64) -> Self {
        Self::from_real(value)
    }
}

impl Neg for Complex {
    type Output = Self;

    fn neg(self) -> Self {
        Self {
            re: -self.re,
            im: -self.im,
        }
    }
}

impl Add for Complex {
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

impl Sub for Complex {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

impl Mul for Complex {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

impl Div for Complex {
    type Output = Self;

    fn div(self, rhs: Self) -> Self {
        let denominator = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / denominator,
            im: (self.im * rhs.re - self.re * rhs.im) / denominator,
        }
    }
}

impl fmt::Display for Complex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.im >= 0.0 {
            write!(f, "{}+{}i", self.re, self.im)
        } else {
            write!(f, "{}-{}i", self.re, -self.im)
        }
    }
}

/// Solves the square complex system `a x = b` by Gaussian elimination with
/// partial pivoting.
///
/// Returns `None` for a pivot below the relative tolerance. The tolerance is
/// intentionally conservative so that an ill-conditioned eigenvector matrix is
/// rejected rather than used for an unstable spectral reconstruction.
pub(crate) fn solve(mut a: Vec<Vec<Complex>>, mut b: Vec<Complex>) -> Option<Vec<Complex>> {
    let n = a.len();
    if n == 0 || b.len() != n {
        return None;
    }
    let scale = a
        .iter()
        .flat_map(|row| row.iter())
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    let tolerance = 1e-7 * scale.max(1.0) * n as f64;

    for k in 0..n {
        let (pivot_row, pivot_norm) =
            (k..n)
                .map(|row| (row, a[row][k].norm()))
                .fold((k, 0.0), |best, candidate| {
                    if candidate.1 > best.1 {
                        candidate
                    } else {
                        best
                    }
                });
        if pivot_norm <= tolerance {
            return None;
        }
        if pivot_row != k {
            a.swap(k, pivot_row);
            b.swap(k, pivot_row);
        }

        let pivot = a[k][k];
        for entry in a[k].iter_mut().skip(k) {
            *entry = *entry / pivot;
        }
        b[k] = b[k] / pivot;

        let pivot_entries = a[k].clone();
        for row in 0..n {
            if row == k {
                continue;
            }
            let factor = a[row][k];
            if factor.norm() == 0.0 {
                continue;
            }
            for column in k..n {
                a[row][column] = a[row][column] - factor * pivot_entries[column];
            }
            b[row] = b[row] - factor * b[k];
        }
    }

    Some(b)
}
