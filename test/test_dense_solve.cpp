/**
 * @file test_dense_solve.cpp
 * @brief Tests for the runtime-sized dense LU and QR least-squares solvers.
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
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <grace/utils/dense_solve.hh>

#include <cmath>
#include <random>
#include <vector>

using namespace grace;

namespace {

/// y = A x for a row-major m x n matrix.
std::vector<double> matvec(std::vector<double> const& A, int m, int n,
                           std::vector<double> const& x) {
    std::vector<double> y(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            y[i] += A[i * n + j] * x[j];
    return y;
}

/// Diagonally dominant (hence well conditioned and non-singular) test matrix.
std::vector<double> well_conditioned(int n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> A(n * n);
    for (int i = 0; i < n; ++i) {
        double off = 0;
        for (int j = 0; j < n; ++j) {
            if (i != j) { A[i * n + j] = d(rng); off += std::abs(A[i * n + j]); }
        }
        A[i * n + i] = off + 1.0 + std::abs(d(rng));
    }
    return A;
}

}  // namespace

//---------------------------------------------------------------------------
// LU
//---------------------------------------------------------------------------

TEST_CASE("LU solves well-conditioned systems", "[dense_solve]") {
    for (int n : {1, 2, 5, 17, 81}) {
        auto const A0 = well_conditioned(n, 900u + n);

        std::vector<double> x_exact(n);
        for (int i = 0; i < n; ++i) x_exact[i] = std::sin(0.7 * i) + 0.3 * i;

        auto b = matvec(A0, n, n, x_exact);
        auto A = A0;

        REQUIRE(lu_solve_in_place(A, n, b));

        for (int i = 0; i < n; ++i) {
            INFO("n=" << n << " i=" << i);
            REQUIRE_THAT(b[i], Catch::Matchers::WithinAbs(x_exact[i], 1e-10));
        }
    }
}

TEST_CASE("LU requires pivoting to handle a zero leading pivot", "[dense_solve]") {
    // Without row interchange this matrix breaks down immediately at A(0,0)=0,
    // even though it is perfectly well conditioned.
    int const n = 2;
    std::vector<double> A{0.0, 1.0,
                          1.0, 0.0};
    std::vector<double> b{2.0, 3.0};  // => x = (3, 2)

    REQUIRE(lu_solve_in_place(A, n, b));
    REQUIRE_THAT(b[0], Catch::Matchers::WithinAbs(3.0, 1e-13));
    REQUIRE_THAT(b[1], Catch::Matchers::WithinAbs(2.0, 1e-13));
}

TEST_CASE("LU reports singular matrices instead of returning garbage",
          "[dense_solve]") {
    SECTION("exactly singular: duplicated row") {
        int const n = 3;
        std::vector<double> A{1.0, 2.0, 3.0,
                              2.0, 4.0, 6.0,
                              1.0, 0.0, 1.0};
        std::vector<double> b{1.0, 2.0, 3.0};
        REQUIRE_FALSE(lu_solve_in_place(A, n, b));
    }
    SECTION("all zeros") {
        int const n = 4;
        std::vector<double> A(n * n, 0.0);
        std::vector<double> b(n, 1.0);
        REQUIRE_FALSE(lu_solve_in_place(A, n, b));
    }
    SECTION("zero column makes the system rank deficient") {
        int const n = 3;
        std::vector<double> A{1.0, 0.0, 3.0,
                              4.0, 0.0, 6.0,
                              7.0, 0.0, 9.0};
        std::vector<double> b{1.0, 2.0, 3.0};
        REQUIRE_FALSE(lu_solve_in_place(A, n, b));
    }
}

TEST_CASE("LU recovers the identity solution exactly", "[dense_solve]") {
    int const n = 6;
    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i) A[i * n + i] = 1.0;
    std::vector<double> b{1.0, -2.0, 3.0, -4.0, 5.0, -6.0};
    auto const expected = b;

    REQUIRE(lu_solve_in_place(A, n, b));
    for (int i = 0; i < n; ++i) REQUIRE(b[i] == expected[i]);
}

TEST_CASE("min_pivot degrades on an ill-conditioned matrix", "[dense_solve]") {
    // The Hilbert matrix is the classic ill-conditioned case. The solver should
    // still complete, but min_pivot must signal that the result is untrustworthy
    // rather than the caller having to guess.
    auto hilbert = [](int n) {
        std::vector<double> H(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                H[i * n + j] = 1.0 / (i + j + 1.0);
        return H;
    };

    std::vector<int> piv;
    double pivot_small = 0, pivot_large = 0;

    auto H4 = hilbert(4);
    REQUIRE(lu_factor(H4, 4, piv, pivot_small));

    auto W = well_conditioned(4, 31u);
    REQUIRE(lu_factor(W, 4, piv, pivot_large));

    INFO("hilbert min_pivot=" << pivot_small << " well-conditioned=" << pivot_large);
    REQUIRE(pivot_small < pivot_large);
    REQUIRE(pivot_small > 0.0);
}

//---------------------------------------------------------------------------
// QR least squares
//---------------------------------------------------------------------------

TEST_CASE("QR reproduces the exact solution for a consistent system",
          "[dense_solve]") {
    // Overdetermined but consistent: b lies exactly in the column space, so the
    // least-squares solution is the exact one.
    int const m = 8, n = 3;
    std::mt19937 rng(2024u);
    std::uniform_real_distribution<double> d(-1.0, 1.0);

    std::vector<double> A(m * n);
    for (auto& a : A) a = d(rng);

    std::vector<double> x_exact{1.5, -2.25, 0.75};
    auto b = matvec(A, m, n, x_exact);

    auto Awork = A;
    REQUIRE(qr_least_squares(Awork, m, n, b));

    for (int i = 0; i < n; ++i) {
        INFO("component " << i);
        REQUIRE_THAT(b[i], Catch::Matchers::WithinAbs(x_exact[i], 1e-10));
    }
}

TEST_CASE("QR satisfies the normal equations for an inconsistent system",
          "[dense_solve]") {
    // The defining property of the least-squares solution: the residual is
    // orthogonal to every column of A, i.e. A^T (A x - b) = 0. Checked directly
    // rather than against a second implementation.
    int const m = 10, n = 4;
    std::mt19937 rng(555u);
    std::uniform_real_distribution<double> d(-1.0, 1.0);

    std::vector<double> A(m * n);
    for (auto& a : A) a = d(rng);
    std::vector<double> b(m);
    for (auto& v : b) v = d(rng);

    auto const A0 = A;
    auto const b0 = b;

    REQUIRE(qr_least_squares(A, m, n, b));
    std::vector<double> x(b.begin(), b.begin() + n);

    auto const Ax = matvec(A0, m, n, x);
    std::vector<double> r(m);
    for (int i = 0; i < m; ++i) r[i] = Ax[i] - b0[i];

    for (int j = 0; j < n; ++j) {
        double dot = 0;
        for (int i = 0; i < m; ++i) dot += A0[i * n + j] * r[i];
        INFO("column " << j);
        REQUIRE_THAT(dot, Catch::Matchers::WithinAbs(0.0, 1e-10));
    }
}

TEST_CASE("QR agrees with LU on a square nonsingular system", "[dense_solve]") {
    int const n = 12;
    auto const A0 = well_conditioned(n, 606u);

    std::vector<double> x_exact(n);
    for (int i = 0; i < n; ++i) x_exact[i] = 0.5 * i - 2.0;
    auto const b0 = matvec(A0, n, n, x_exact);

    auto A_lu = A0; auto b_lu = b0;
    REQUIRE(lu_solve_in_place(A_lu, n, b_lu));

    auto A_qr = A0; auto b_qr = b0;
    REQUIRE(qr_least_squares(A_qr, n, n, b_qr));

    for (int i = 0; i < n; ++i) {
        INFO("component " << i);
        REQUIRE_THAT(b_qr[i], Catch::Matchers::WithinAbs(b_lu[i], 1e-10));
        REQUIRE_THAT(b_qr[i], Catch::Matchers::WithinAbs(x_exact[i], 1e-10));
    }
}

TEST_CASE("QR rejects rank-deficient and underdetermined input", "[dense_solve]") {
    SECTION("m < n is refused") {
        int const m = 2, n = 4;
        std::vector<double> A(m * n, 1.0);
        std::vector<double> b(m, 1.0);
        REQUIRE_FALSE(qr_least_squares(A, m, n, b));
    }
    SECTION("duplicated column is rank deficient") {
        int const m = 5, n = 2;
        std::vector<double> A{1.0, 1.0,
                              2.0, 2.0,
                              3.0, 3.0,
                              4.0, 4.0,
                              5.0, 5.0};
        std::vector<double> b{1.0, 2.0, 3.0, 4.0, 5.0};
        REQUIRE_FALSE(qr_least_squares(A, m, n, b));
    }
    SECTION("zero matrix is refused") {
        int const m = 4, n = 2;
        std::vector<double> A(m * n, 0.0);
        std::vector<double> b(m, 1.0);
        REQUIRE_FALSE(qr_least_squares(A, m, n, b));
    }
}

//---------------------------------------------------------------------------
// Symmetric eigenproblems
//---------------------------------------------------------------------------

TEST_CASE("Jacobi recovers a known symmetric spectrum", "[dense_solve]") {
    // Build A = Q D Q^T from an explicit orthogonal Q, so both eigenvalues and
    // eigenvectors are known independently of the solver.
    int const n = 6;
    std::vector<double> want{-3.0, -0.5, 0.0, 1.0, 2.5, 9.0};

    // Orthogonal Q from Gram-Schmidt on a fixed pseudo-random matrix.
    std::mt19937 rng(4242u);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    std::vector<double> Q(n * n);
    for (auto& v : Q) v = d(rng);
    for (int j = 0; j < n; ++j) {
        for (int k = 0; k < j; ++k) {
            double dot = 0;
            for (int i = 0; i < n; ++i) dot += Q[i*n + j] * Q[i*n + k];
            for (int i = 0; i < n; ++i) Q[i*n + j] -= dot * Q[i*n + k];
        }
        double nrm = 0;
        for (int i = 0; i < n; ++i) nrm += Q[i*n + j] * Q[i*n + j];
        nrm = std::sqrt(nrm);
        for (int i = 0; i < n; ++i) Q[i*n + j] /= nrm;
    }

    std::vector<double> A(n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            for (int k = 0; k < n; ++k)
                A[i*n + j] += Q[i*n + k] * want[k] * Q[j*n + k];

    auto const A0 = A;
    std::vector<double> eval, evec;
    REQUIRE(jacobi_eigen_symmetric(A, n, eval, evec));

    REQUIRE(eval.size() == static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        INFO("eigenvalue " << i);
        REQUIRE_THAT(eval[i], Catch::Matchers::WithinAbs(want[i], 1e-10));
    }

    // A v = lambda v for every recovered pair.
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) {
            double av = 0;
            for (int k = 0; k < n; ++k) av += A0[i*n + k] * evec[k*n + j];
            INFO("pair " << j << " row " << i);
            REQUIRE_THAT(av, Catch::Matchers::WithinAbs(eval[j] * evec[i*n + j], 1e-9));
        }
}

TEST_CASE("Cholesky rejects a non-positive-definite matrix", "[dense_solve]") {
    int const n = 3;
    SECTION("indefinite") {
        std::vector<double> A{1.0, 2.0, 0.0,
                              2.0, 1.0, 0.0,
                              0.0, 0.0, 1.0};
        REQUIRE_FALSE(cholesky_factor(A, n));
    }
    SECTION("positive definite succeeds") {
        std::vector<double> A{4.0, 1.0, 0.5,
                              1.0, 3.0, 0.2,
                              0.5, 0.2, 2.0};
        auto const A0 = A;
        REQUIRE(cholesky_factor(A, n));
        // L L^T must rebuild A.
        for (int i = 0; i < n; ++i)
            for (int j = 0; j <= i; ++j) {
                double acc = 0;
                for (int k = 0; k <= j; ++k) acc += A[i*n + k] * A[j*n + k];
                INFO("entry " << i << "," << j);
                REQUIRE_THAT(acc, Catch::Matchers::WithinAbs(A0[i*n + j], 1e-12));
            }
    }
}

TEST_CASE("Generalised symmetric eigenproblem A v = lambda B v", "[dense_solve]") {
    // With B = I the generalised problem must reduce to the standard one.
    SECTION("B = identity") {
        int const n = 4;
        std::vector<double> A{ 2.0, -1.0,  0.0,  0.0,
                              -1.0,  2.0, -1.0,  0.0,
                               0.0, -1.0,  2.0, -1.0,
                               0.0,  0.0, -1.0,  2.0};
        std::vector<double> B(n * n, 0.0);
        for (int i = 0; i < n; ++i) B[i*n + i] = 1.0;

        std::vector<double> eval, evec;
        REQUIRE(generalized_symmetric_eigen(A, B, n, eval, evec));

        // Eigenvalues of this tridiagonal matrix are 2 - 2cos(k pi/(n+1)).
        for (int k = 1; k <= n; ++k) {
            double const want = 2.0 - 2.0 * std::cos(k * M_PI / (n + 1));
            INFO("mode " << k);
            REQUIRE_THAT(eval[k-1], Catch::Matchers::WithinAbs(want, 1e-10));
        }
    }

    SECTION("general positive-definite B") {
        int const n = 3;
        std::vector<double> A{ 3.0, 0.4, 0.1,
                               0.4, 2.0, 0.3,
                               0.1, 0.3, 1.0};
        std::vector<double> B{ 2.0, 0.3, 0.0,
                               0.3, 1.5, 0.2,
                               0.0, 0.2, 1.0};
        auto const A0 = A, B0 = B;

        std::vector<double> eval, evec;
        REQUIRE(generalized_symmetric_eigen(A, B, n, eval, evec));

        // The defining property, checked directly rather than against a
        // reference implementation.
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i) {
                double av = 0, bv = 0;
                for (int k = 0; k < n; ++k) {
                    av += A0[i*n + k] * evec[k*n + j];
                    bv += B0[i*n + k] * evec[k*n + j];
                }
                INFO("pair " << j << " row " << i);
                REQUIRE_THAT(av, Catch::Matchers::WithinAbs(eval[j] * bv, 1e-9));
            }
    }

    SECTION("non-positive-definite B is refused") {
        int const n = 2;
        std::vector<double> A{1.0, 0.0, 0.0, 1.0};
        std::vector<double> B{0.0, 1.0, 1.0, 0.0};
        std::vector<double> eval, evec;
        REQUIRE_FALSE(generalized_symmetric_eigen(A, B, n, eval, evec));
    }
}
