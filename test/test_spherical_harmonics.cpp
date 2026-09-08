/**
 * @file test_spherical_harmonics.cpp
 * @brief Correctness tests for grace::spherical_harmonic_basis_t.
 *
 * This header shipped untested and unused: nothing in the tree referenced it
 * until the apparent-horizon infrastructure was built on top of it. Every
 * later layer (Strahlkoerper surface, Galerkin projection, horizon area)
 * depends on the transform being exact, so it is pinned down here first.
 *
 * The strongest checks below cross-validate the two independent code paths
 * in the header against each other: evaluate() consumes the precomputed
 * Y/dY matrices, while evaluate_single() re-runs the Legendre recursion
 * directly. They share no arithmetic, so agreement is meaningful.
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

#include <grace/utils/spherical_harmonics.hh>

#include <cmath>
#include <random>
#include <vector>

using grace::gauss_legendre;
using grace::spherical_harmonic_basis_t;

namespace {

/// Reproducible pseudo-random coefficient vector, decaying with l so the
/// surfaces it represents stay smooth and single-valued.
std::vector<double> random_coeffs(int l_max, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    std::vector<double> a((l_max + 1) * (l_max + 1));
    for (int l = 0; l <= l_max; ++l) {
        double const decay = 1.0 / (1.0 + l * l);
        for (int m = 0; m <= l; ++m) {
            a[spherical_harmonic_basis_t::coeff_index(l, m, false)] = dist(rng) * decay;
            if (m > 0)
                a[spherical_harmonic_basis_t::coeff_index(l, m, true)] = dist(rng) * decay;
        }
    }
    return a;
}

}  // namespace

//---------------------------------------------------------------------------
// Gauss-Legendre quadrature
//---------------------------------------------------------------------------

TEST_CASE("Gauss-Legendre nodes integrate degree 2n-1 exactly", "[spherical_harmonics]") {
    for (int n : {2, 3, 5, 9, 16}) {
        std::vector<double> x, w;
        gauss_legendre(n, x, w);

        REQUIRE(static_cast<int>(x.size()) == n);
        REQUIRE(static_cast<int>(w.size()) == n);

        // Weights sum to the measure of [-1,1].
        double wsum = 0;
        for (double wi : w) wsum += wi;
        REQUIRE_THAT(wsum, Catch::Matchers::WithinAbs(2.0, 1e-13));

        // int_{-1}^{1} x^p dx = 2/(p+1) for even p, 0 for odd p.
        for (int p = 0; p <= 2 * n - 1; ++p) {
            double q = 0;
            for (int i = 0; i < n; ++i) q += w[i] * std::pow(x[i], p);
            double const exact = (p % 2 == 0) ? 2.0 / (p + 1) : 0.0;
            INFO("n=" << n << " degree=" << p);
            REQUIRE_THAT(q, Catch::Matchers::WithinAbs(exact, 1e-12));
        }

        // All nodes strictly interior: the basis divides by sin(theta).
        for (double xi : x) REQUIRE(std::abs(xi) < 1.0);
    }
}

TEST_CASE("Surface quadrature weights sum to 4 pi", "[spherical_harmonics]") {
    for (int l_max : {0, 1, 2, 4, 8}) {
        spherical_harmonic_basis_t basis(l_max);
        double total = 0;
        for (int it = 0; it < basis.n_theta(); ++it)
            for (int ip = 0; ip < basis.n_phi(); ++ip)
                total += basis.weight(it, ip);
        INFO("l_max=" << l_max);
        REQUIRE_THAT(total, Catch::Matchers::WithinAbs(4.0 * M_PI, 1e-12));
    }
}

//---------------------------------------------------------------------------
// Index bookkeeping
//---------------------------------------------------------------------------

TEST_CASE("coeff_index is a bijection onto [0, (l_max+1)^2)", "[spherical_harmonics]") {
    int const l_max = 8;
    int const n_coeffs = (l_max + 1) * (l_max + 1);
    std::vector<int> hits(n_coeffs, 0);

    for (int l = 0; l <= l_max; ++l) {
        for (int m = 0; m <= l; ++m) {
            int const ic = spherical_harmonic_basis_t::coeff_index(l, m, false);
            REQUIRE(ic >= 0);
            REQUIRE(ic < n_coeffs);
            hits[ic]++;
            if (m > 0) {
                int const is = spherical_harmonic_basis_t::coeff_index(l, m, true);
                REQUIRE(is >= 0);
                REQUIRE(is < n_coeffs);
                REQUIRE(is != ic);
                hits[is]++;
            }
        }
    }
    for (int c = 0; c < n_coeffs; ++c) {
        INFO("coefficient slot " << c);
        REQUIRE(hits[c] == 1);
    }
}

TEST_CASE("Grid dimensions follow (l_max+1) x (2 l_max+1)", "[spherical_harmonics]") {
    for (int l_max : {0, 1, 2, 4, 8}) {
        spherical_harmonic_basis_t basis(l_max);
        REQUIRE(basis.l_max() == l_max);
        REQUIRE(basis.n_coeffs() == (l_max + 1) * (l_max + 1));
        REQUIRE(basis.n_theta() == l_max + 1);
        REQUIRE(basis.n_phi() == 2 * l_max + 1);
        REQUIRE(basis.n_points() == basis.n_theta() * basis.n_phi());
        // The collocation grid must not under-resolve the coefficient set.
        REQUIRE(basis.n_points() >= basis.n_coeffs());
    }
}

//---------------------------------------------------------------------------
// Orthonormality
//---------------------------------------------------------------------------

TEST_CASE("Basis functions are orthonormal under the collocation quadrature",
          "[spherical_harmonics]") {
    // The quadrature is exact for products Y_a Y_b: Gauss-Legendre with
    // (l_max+1) nodes handles degree 2*l_max+1 >= l1+l2 in cos(theta), and the
    // uniform (2*l_max+1)-point phi rule is exact for frequencies m1+m2 <= 2*l_max.
    for (int l_max : {2, 4, 6}) {
        spherical_harmonic_basis_t basis(l_max);
        int const nc = basis.n_coeffs();
        int const np = basis.n_phi();
        double const* Y = basis.Y_matrix();

        for (int a = 0; a < nc; ++a) {
            for (int b = a; b < nc; ++b) {
                double integral = 0;
                for (int p = 0; p < basis.n_points(); ++p)
                    integral += basis.weight(p / np, p % np)
                              * Y[p * nc + a] * Y[p * nc + b];
                double const expected = (a == b) ? 1.0 : 0.0;
                INFO("l_max=" << l_max << " a=" << a << " b=" << b);
                REQUIRE_THAT(integral, Catch::Matchers::WithinAbs(expected, 1e-12));
            }
        }
    }
}

//---------------------------------------------------------------------------
// Synthesis / analysis round-trip
//---------------------------------------------------------------------------

TEST_CASE("inverse(evaluate(a)) recovers a", "[spherical_harmonics]") {
    for (int l_max : {1, 2, 4, 8}) {
        spherical_harmonic_basis_t basis(l_max);
        int const nc = basis.n_coeffs();
        int const np = basis.n_points();

        auto const a = random_coeffs(l_max, 12345u + l_max);

        std::vector<double> h(np), dth(np), dph(np);
        basis.evaluate(a.data(), h.data(), dth.data(), dph.data());

        std::vector<double> a_rt(nc);
        basis.inverse(h.data(), a_rt.data());

        for (int c = 0; c < nc; ++c) {
            INFO("l_max=" << l_max << " coeff " << c);
            REQUIRE_THAT(a_rt[c], Catch::Matchers::WithinAbs(a[c], 1e-12));
        }
    }
}

TEST_CASE("A constant field projects onto Y_00 alone", "[spherical_harmonics]") {
    int const l_max = 6;
    spherical_harmonic_basis_t basis(l_max);
    std::vector<double> f(basis.n_points(), 3.5);
    std::vector<double> a(basis.n_coeffs());
    basis.inverse(f.data(), a.data());

    // Y_00 = 1/sqrt(4 pi), so a_00 = 3.5 * sqrt(4 pi).
    REQUIRE_THAT(a[0], Catch::Matchers::WithinAbs(3.5 * std::sqrt(4.0 * M_PI), 1e-12));
    for (int c = 1; c < basis.n_coeffs(); ++c) {
        INFO("coefficient " << c);
        REQUIRE_THAT(a[c], Catch::Matchers::WithinAbs(0.0, 1e-12));
    }
}

//---------------------------------------------------------------------------
// Closed-form low-l harmonics
//---------------------------------------------------------------------------

TEST_CASE("Low-l basis functions match closed form", "[spherical_harmonics]") {
    // Convention: Condon-Shortley phase retained, real harmonics normalised so
    // that the m>0 cos/sin pairs carry an extra sqrt(2).
    int const l_max = 2;
    spherical_harmonic_basis_t basis(l_max);
    int const nc = basis.n_coeffs();
    double const* Y = basis.Y_matrix();

    for (int it = 0; it < basis.n_theta(); ++it) {
        for (int ip = 0; ip < basis.n_phi(); ++ip) {
            int const p = basis.point_index(it, ip);
            double const th = basis.theta(it);
            double const ph = basis.phi(ip);
            double const x = std::cos(th);
            double const s = std::sin(th);

            auto at = [&](int l, int m, bool sine) {
                return Y[p * nc + spherical_harmonic_basis_t::coeff_index(l, m, sine)];
            };

            INFO("it=" << it << " ip=" << ip);
            REQUIRE_THAT(at(0, 0, false),
                         Catch::Matchers::WithinAbs(0.5 / std::sqrt(M_PI), 1e-13));
            REQUIRE_THAT(at(1, 0, false),
                         Catch::Matchers::WithinAbs(std::sqrt(3.0 / (4.0 * M_PI)) * x, 1e-13));
            // P_1^1 = -sin(theta) carries the Condon-Shortley sign.
            REQUIRE_THAT(at(1, 1, false),
                         Catch::Matchers::WithinAbs(
                             -std::sqrt(3.0 / (4.0 * M_PI)) * s * std::cos(ph), 1e-13));
            REQUIRE_THAT(at(1, 1, true),
                         Catch::Matchers::WithinAbs(
                             -std::sqrt(3.0 / (4.0 * M_PI)) * s * std::sin(ph), 1e-13));
            REQUIRE_THAT(at(2, 0, false),
                         Catch::Matchers::WithinAbs(
                             std::sqrt(5.0 / (4.0 * M_PI)) * 0.5 * (3.0 * x * x - 1.0), 1e-13));
        }
    }
}

//---------------------------------------------------------------------------
// evaluate() vs evaluate_single(): independent code paths
//---------------------------------------------------------------------------

TEST_CASE("evaluate_single agrees with evaluate at collocation points",
          "[spherical_harmonics]") {
    int const l_max = 6;
    spherical_harmonic_basis_t basis(l_max);
    auto const a = random_coeffs(l_max, 777u);

    int const np = basis.n_points();
    std::vector<double> h(np), dth(np), dph(np);
    basis.evaluate(a.data(), h.data(), dth.data(), dph.data());

    for (int it = 0; it < basis.n_theta(); ++it) {
        for (int ip = 0; ip < basis.n_phi(); ++ip) {
            int const p = basis.point_index(it, ip);
            double const single = basis.evaluate_single(a.data(), basis.theta(it), basis.phi(ip));
            INFO("it=" << it << " ip=" << ip);
            REQUIRE_THAT(single, Catch::Matchers::WithinAbs(h[p], 1e-12));
        }
    }
}

//---------------------------------------------------------------------------
// Angular derivatives
//---------------------------------------------------------------------------

TEST_CASE("Angular derivatives match finite differences of evaluate_single",
          "[spherical_harmonics]") {
    // evaluate() reads the precomputed dY/d2Y matrices; evaluate_single()
    // re-runs the recursion and knows nothing about derivatives. Differencing
    // the latter therefore independently validates the former.
    int const l_max = 6;
    spherical_harmonic_basis_t basis(l_max);
    auto const a = random_coeffs(l_max, 4242u);

    int const np = basis.n_points();
    std::vector<double> h(np), dth(np), dph(np), dthth(np), dthph(np), dphph(np);
    basis.evaluate(a.data(), h.data(), dth.data(), dph.data(),
                   dthth.data(), dthph.data(), dphph.data());

    double const e = 1e-4;
    auto f = [&](double t, double p) { return basis.evaluate_single(a.data(), t, p); };

    for (int it = 0; it < basis.n_theta(); ++it) {
        for (int ip = 0; ip < basis.n_phi(); ++ip) {
            int const p = basis.point_index(it, ip);
            double const t0 = basis.theta(it);
            double const p0 = basis.phi(ip);

            double const fd_th = (f(t0 + e, p0) - f(t0 - e, p0)) / (2 * e);
            double const fd_ph = (f(t0, p0 + e) - f(t0, p0 - e)) / (2 * e);
            double const fd_thth = (f(t0 + e, p0) - 2 * f(t0, p0) + f(t0 - e, p0)) / (e * e);
            double const fd_phph = (f(t0, p0 + e) - 2 * f(t0, p0) + f(t0, p0 - e)) / (e * e);
            double const fd_thph = (f(t0 + e, p0 + e) - f(t0 + e, p0 - e)
                                  - f(t0 - e, p0 + e) + f(t0 - e, p0 - e)) / (4 * e * e);

            INFO("it=" << it << " ip=" << ip);
            REQUIRE_THAT(dth[p],   Catch::Matchers::WithinAbs(fd_th,   1e-6));
            REQUIRE_THAT(dph[p],   Catch::Matchers::WithinAbs(fd_ph,   1e-6));
            REQUIRE_THAT(dthth[p], Catch::Matchers::WithinAbs(fd_thth, 1e-4));
            REQUIRE_THAT(dphph[p], Catch::Matchers::WithinAbs(fd_phph, 1e-4));
            REQUIRE_THAT(dthph[p], Catch::Matchers::WithinAbs(fd_thph, 1e-4));
        }
    }
}

TEST_CASE("Derivatives of a constant field vanish", "[spherical_harmonics]") {
    int const l_max = 4;
    spherical_harmonic_basis_t basis(l_max);
    std::vector<double> a(basis.n_coeffs(), 0.0);
    a[0] = 2.0 * std::sqrt(4.0 * M_PI);  // uniform radius 2

    int const np = basis.n_points();
    std::vector<double> h(np), dth(np), dph(np), dthth(np), dthph(np), dphph(np);
    basis.evaluate(a.data(), h.data(), dth.data(), dph.data(),
                   dthth.data(), dthph.data(), dphph.data());

    for (int p = 0; p < np; ++p) {
        INFO("point " << p);
        REQUIRE_THAT(h[p],     Catch::Matchers::WithinAbs(2.0, 1e-12));
        REQUIRE_THAT(dth[p],   Catch::Matchers::WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(dph[p],   Catch::Matchers::WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(dthth[p], Catch::Matchers::WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(dthph[p], Catch::Matchers::WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(dphph[p], Catch::Matchers::WithinAbs(0.0, 1e-12));
    }
}
