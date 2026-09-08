/**
 * @file dense_solve.hh
 * @brief Small runtime-sized dense linear algebra: LU with partial pivoting
 *        and Householder QR least-squares.
 *
 * GRACE's existing dense solvers (utils/LU_utils.hh, utils/rootfinding.hh) fix
 * the system size at compile time, which does not fit solves whose dimension
 * comes from a runtime parameter — for example a spectral surface with
 * (l_max+1)^2 unknowns.
 *
 * Deliberately three free functions over std::vector<double> and nothing else:
 * no matrix class, no expression templates, no external dependency. At the
 * sizes this targets (n of order 10^2, so an LU is ~10^5 flops) the arithmetic
 * is free, and the narrow surface means the implementation can later be swapped
 * for a BLAS/LAPACK or Kokkos-Kernels backend without any caller changing.
 *
 * Storage is row-major throughout, matching
 * spherical_harmonic_basis_t::Y_matrix().
 *
 * Host-only: these are not GRACE_HOST_DEVICE. They allocate, and the solves
 * they serve run once per Newton iteration on a surface of order 10^2 points,
 * far too little work to be worth a device launch.
 *
 * @copyright This file is part of GRACE.
 * GRACE is an evolution framework that uses Finite Difference
 * methods to simulate relativistic spacetimes and plasmas
 * Copyright (C) 2023-2026 Carlo Musolino and GRACE Contributors
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_UTILS_DENSE_SOLVE_HH
#define GRACE_UTILS_DENSE_SOLVE_HH

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

namespace grace {

/**
 * @brief LU factorisation with partial pivoting, in place.
 *
 * @param[in,out] A   n x n row-major matrix, overwritten with the combined
 *                    L (unit diagonal, implicit) and U factors.
 * @param[in]     n   System size.
 * @param[out]    piv Row permutation, length n.
 * @param[out]    min_pivot  Smallest |pivot| encountered, relative to the
 *                    largest matrix entry. Callers can use it to distinguish a
 *                    merely ill-conditioned system from a usable one; it is a
 *                    cheap proxy for the condition number, not a substitute.
 * @return false if the matrix is numerically singular (a pivot underflowed),
 *         in which case A and piv are left in an unspecified state.
 *
 * Partial pivoting rather than the plain Cholesky-on-normal-equations that a
 * symmetric formulation might invite: forming A^T A squares the condition
 * number, and a non-positive pivot there reports "not positive definite" for
 * what is really just loss of precision.
 */
inline bool lu_factor(std::vector<double>& A, int n,
                      std::vector<int>& piv, double& min_pivot)
{
    piv.resize(n);
    for (int i = 0; i < n; ++i) piv[i] = i;

    double scale = 0.0;
    for (int i = 0; i < n * n; ++i) scale = std::max(scale, std::abs(A[i]));
    if (scale == 0.0) { min_pivot = 0.0; return false; }

    double const tol = 1e-14 * scale;
    min_pivot = scale;

    for (int k = 0; k < n; ++k) {
        // Select the pivot row.
        int    ipiv = k;
        double amax = std::abs(A[k * n + k]);
        for (int i = k + 1; i < n; ++i) {
            double const a = std::abs(A[i * n + k]);
            if (a > amax) { amax = a; ipiv = i; }
        }

        if (amax <= tol) { min_pivot = amax; return false; }
        min_pivot = std::min(min_pivot, amax);

        if (ipiv != k) {
            for (int j = 0; j < n; ++j) std::swap(A[k * n + j], A[ipiv * n + j]);
            std::swap(piv[k], piv[ipiv]);
        }

        double const inv_pivot = 1.0 / A[k * n + k];
        for (int i = k + 1; i < n; ++i) {
            double const factor = A[i * n + k] * inv_pivot;
            A[i * n + k] = factor;
            if (factor != 0.0)
                for (int j = k + 1; j < n; ++j)
                    A[i * n + j] -= factor * A[k * n + j];
        }
    }

    min_pivot /= scale;
    return true;
}

/**
 * @brief Solve A x = b given the factorisation produced by lu_factor().
 *
 * @param[in]     LU  Factored matrix from lu_factor().
 * @param[in]     n   System size.
 * @param[in]     piv Row permutation from lu_factor().
 * @param[in,out] b   Right-hand side, overwritten with the solution.
 */
inline void lu_solve(std::vector<double> const& LU, int n,
                     std::vector<int> const& piv, std::vector<double>& b)
{
    std::vector<double> y(n);

    // Forward substitution on the permuted right-hand side (L has unit diagonal).
    for (int i = 0; i < n; ++i) {
        double sum = b[piv[i]];
        for (int j = 0; j < i; ++j) sum -= LU[i * n + j] * y[j];
        y[i] = sum;
    }

    // Back substitution.
    for (int i = n - 1; i >= 0; --i) {
        double sum = y[i];
        for (int j = i + 1; j < n; ++j) sum -= LU[i * n + j] * b[j];
        b[i] = sum / LU[i * n + i];
    }
}

/**
 * @brief Convenience wrapper: factor and solve in one call.
 * @return false if the system is numerically singular.
 */
inline bool lu_solve_in_place(std::vector<double>& A, int n, std::vector<double>& b)
{
    std::vector<int> piv;
    double min_pivot = 0.0;
    if (!lu_factor(A, n, piv, min_pivot)) return false;
    lu_solve(A, n, piv, b);
    return true;
}

/**
 * @brief Householder QR least-squares: minimise ||A x - b||_2 for m >= n.
 *
 * @param[in,out] A m x n row-major matrix, destroyed.
 * @param[in]     m Row count (must be >= n).
 * @param[in]     n Column count.
 * @param[in,out] b Right-hand side of length m; on success the first n entries
 *                  hold the solution and the remainder holds the residual in
 *                  the orthogonal complement.
 * @return false if A is rank deficient.
 *
 * Kept even where the natural formulation gives a square system: it is the
 * fallback when a Jacobian loses rank, which is the normal outcome for
 * high-l coefficients on an under-resolved surface.
 */
inline bool qr_least_squares(std::vector<double>& A, int m, int n,
                             std::vector<double>& b)
{
    if (m < n) return false;

    double scale = 0.0;
    for (int i = 0; i < m * n; ++i) scale = std::max(scale, std::abs(A[i]));
    if (scale == 0.0) return false;
    double const tol = 1e-14 * scale;

    for (int k = 0; k < n; ++k) {
        // Householder vector for column k below the diagonal.
        double norm = 0.0;
        for (int i = k; i < m; ++i) norm += A[i * n + k] * A[i * n + k];
        norm = std::sqrt(norm);
        if (norm <= tol) return false;

        double const alpha = (A[k * n + k] > 0.0) ? -norm : norm;
        double const vk    = A[k * n + k] - alpha;

        std::vector<double> v(m - k);
        v[0] = vk;
        for (int i = k + 1; i < m; ++i) v[i - k] = A[i * n + k];

        double vnorm2 = 0.0;
        for (double vi : v) vnorm2 += vi * vi;
        if (vnorm2 <= tol * tol) { A[k * n + k] = alpha; continue; }

        // Apply H = I - 2 v v^T / (v^T v) to the trailing columns and to b.
        for (int j = k; j < n; ++j) {
            double dot = 0.0;
            for (int i = k; i < m; ++i) dot += v[i - k] * A[i * n + j];
            double const f = 2.0 * dot / vnorm2;
            for (int i = k; i < m; ++i) A[i * n + j] -= f * v[i - k];
        }
        {
            double dot = 0.0;
            for (int i = k; i < m; ++i) dot += v[i - k] * b[i];
            double const f = 2.0 * dot / vnorm2;
            for (int i = k; i < m; ++i) b[i] -= f * v[i - k];
        }
        A[k * n + k] = alpha;
    }

    // Back substitution against the upper-triangular R.
    for (int i = n - 1; i >= 0; --i) {
        if (std::abs(A[i * n + i]) <= tol) return false;
        double sum = b[i];
        for (int j = i + 1; j < n; ++j) sum -= A[i * n + j] * b[j];
        b[i] = sum / A[i * n + i];
    }

    return true;
}

/**
 * @brief Cholesky factorisation of a symmetric positive-definite matrix.
 *
 * @param[in,out] A n x n row-major; the lower triangle is overwritten with L.
 * @return false if A is not positive definite (which is itself useful
 *         information -- it usually means the caller's matrix is singular
 *         rather than merely ill-conditioned).
 */
inline bool cholesky_factor(std::vector<double>& A, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double sum = A[i*n + j];
            for (int k = 0; k < j; ++k) sum -= A[i*n + k] * A[j*n + k];
            if (i == j) {
                if (!(sum > 0.0)) return false;
                A[i*n + i] = std::sqrt(sum);
            } else {
                A[i*n + j] = sum / A[j*n + j];
            }
        }
        for (int j = i + 1; j < n; ++j) A[i*n + j] = 0.0;
    }
    return true;
}

/**
 * @brief Eigen-decomposition of a real symmetric matrix, by cyclic Jacobi.
 *
 * @param[in,out] A    n x n row-major symmetric; destroyed.
 * @param[out]    eval Eigenvalues, ascending.
 * @param[out]    evec n x n row-major; column j is the eigenvector for eval[j].
 * @return false if the sweeps did not converge.
 *
 * Jacobi rather than tridiagonal-QL: it is a page of code, needs no external
 * dependency, and is unconditionally stable for symmetric input. At the sizes
 * this serves -- n of order (l_max+1)^2, so 10^2 -- its extra cost is
 * irrelevant, and it delivers small eigenvalues to high relative accuracy,
 * which matters here because the interesting eigenvalue is the one nearest
 * zero.
 */
inline bool jacobi_eigen_symmetric(std::vector<double>& A, int n,
                                   std::vector<double>& eval,
                                   std::vector<double>& evec,
                                   int max_sweeps = 100)
{
    evec.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int i = 0; i < n; ++i) evec[i*n + i] = 1.0;
    eval.assign(n, 0.0);

    auto off_diagonal_norm = [&]() {
        double s = 0.0;
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j) s += A[i*n + j] * A[i*n + j];
        return std::sqrt(2.0 * s);
    };

    double scale = 0.0;
    for (int i = 0; i < n * n; ++i) scale = std::max(scale, std::abs(A[i]));
    if (scale == 0.0) { return true; }          // already diagonal (zero)
    double const tol = 1e-14 * scale * n;

    bool converged = false;
    for (int sweep = 0; sweep < max_sweeps; ++sweep) {
        if (off_diagonal_norm() <= tol) { converged = true; break; }

        for (int p = 0; p < n - 1; ++p) {
            for (int q = p + 1; q < n; ++q) {
                double const apq = A[p*n + q];
                if (std::abs(apq) <= 1e-300) continue;

                double const app = A[p*n + p];
                double const aqq = A[q*n + q];

                double const theta = 0.5 * (aqq - app) / apq;
                double const t = (theta >= 0.0)
                    ?  1.0 / (theta + std::sqrt(1.0 + theta * theta))
                    : -1.0 / (-theta + std::sqrt(1.0 + theta * theta));
                double const c = 1.0 / std::sqrt(1.0 + t * t);
                double const s = t * c;

                for (int k = 0; k < n; ++k) {
                    double const akp = A[k*n + p];
                    double const akq = A[k*n + q];
                    A[k*n + p] = c * akp - s * akq;
                    A[k*n + q] = s * akp + c * akq;
                }
                for (int k = 0; k < n; ++k) {
                    double const apk = A[p*n + k];
                    double const aqk = A[q*n + k];
                    A[p*n + k] = c * apk - s * aqk;
                    A[q*n + k] = s * apk + c * aqk;
                }
                for (int k = 0; k < n; ++k) {
                    double const vkp = evec[k*n + p];
                    double const vkq = evec[k*n + q];
                    evec[k*n + p] = c * vkp - s * vkq;
                    evec[k*n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    if (!converged && off_diagonal_norm() > tol) return false;

    for (int i = 0; i < n; ++i) eval[i] = A[i*n + i];

    // Sort ascending, carrying the eigenvectors along.
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return eval[a] < eval[b]; });

    std::vector<double> ev_sorted(n), vec_sorted(static_cast<std::size_t>(n) * n);
    for (int j = 0; j < n; ++j) {
        ev_sorted[j] = eval[order[j]];
        for (int i = 0; i < n; ++i) vec_sorted[i*n + j] = evec[i*n + order[j]];
    }
    eval = std::move(ev_sorted);
    evec = std::move(vec_sorted);
    return true;
}

/**
 * @brief Generalised symmetric eigenproblem  A v = lambda B v,  B positive definite.
 *
 * Reduced to standard form with a Cholesky factor of B: with B = L L^T,
 * C = L^-1 A L^-T is symmetric and shares the eigenvalues, and the generalised
 * eigenvectors are L^-T y.
 *
 * @return false if B is not positive definite or the Jacobi sweeps fail.
 */
inline bool generalized_symmetric_eigen(std::vector<double> A,
                                        std::vector<double> B,
                                        int n,
                                        std::vector<double>& eval,
                                        std::vector<double>& evec)
{
    if (!cholesky_factor(B, n)) return false;   // B -> L (lower)

    // F = L^-1 A : forward substitution on each column.
    std::vector<double> F(static_cast<std::size_t>(n) * n, 0.0);
    for (int col = 0; col < n; ++col)
        for (int i = 0; i < n; ++i) {
            double sum = A[i*n + col];
            for (int k = 0; k < i; ++k) sum -= B[i*n + k] * F[k*n + col];
            F[i*n + col] = sum / B[i*n + i];
        }

    // C = F L^-T : forward substitution on each row.
    std::vector<double> C(static_cast<std::size_t>(n) * n, 0.0);
    for (int row = 0; row < n; ++row)
        for (int j = 0; j < n; ++j) {
            double sum = F[row*n + j];
            for (int k = 0; k < j; ++k) sum -= B[j*n + k] * C[row*n + k];
            C[row*n + j] = sum / B[j*n + j];
        }

    // Symmetrise away the round-off asymmetry before Jacobi.
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            double const m = 0.5 * (C[i*n + j] + C[j*n + i]);
            C[i*n + j] = C[j*n + i] = m;
        }

    std::vector<double> y;
    if (!jacobi_eigen_symmetric(C, n, eval, y)) return false;

    // v = L^-T y, back substitution per eigenvector column.
    evec.assign(static_cast<std::size_t>(n) * n, 0.0);
    for (int col = 0; col < n; ++col)
        for (int i = n - 1; i >= 0; --i) {
            double sum = y[i*n + col];
            for (int k = i + 1; k < n; ++k) sum -= B[k*n + i] * evec[k*n + col];
            evec[i*n + col] = sum / B[i*n + i];
        }
    return true;
}

}  // namespace grace

#endif  // GRACE_UTILS_DENSE_SOLVE_HH
