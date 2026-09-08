/**
 * @file test_horizon_multipoles.cpp
 * @brief Tests for the horizon mass and angular-momentum multipoles.
 *
 * The two normalisations are exact identities and are checked as such:
 * `M_0 = M` follows from Gauss-Bonnet fixing the integral of Rtilde at 8 pi,
 * and `J_1 = J` because `P'_1 = 1` reduces that integral to the spin itself.
 * Everything else is checked through symmetry, which for Kerr is exact:
 * equatorial reflection kills the odd mass moments and the even spin moments.
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

#include <grace/geometry/horizon_multipoles.hh>
#include <grace/physics/analytic_geometry.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

struct kerr_case_t {
    strahlkoerper_t                       surface;
    std::vector<adm_point_t>              geom;
    std::vector<surface_point_geometry_t> sg;
    akv_result_t                          akv;
    horizon_spin_akv_t                    spin;
    horizon_multipoles_t                  mp;
};

kerr_case_t build(double a, int l_max = 16) {
    kerr_schild_geometry_t ks(1.0, a);
    strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, ks.horizon_radius());

    std::vector<double> h(s.n_points());
    for (int it = 0; it < s.basis().n_theta(); ++it)
        for (int ip = 0; ip < s.basis().n_phi(); ++ip)
            h[s.basis().point_index(it, ip)] =
                ks.horizon_coordinate_radius(s.basis().theta(it), s.basis().phi(ip));
    s.coefficients() = s.project(h);

    kerr_case_t c{ s, {}, {}, {}, {}, {} };
    c.geom = ks.at(c.surface.collocation_points());
    c.sg   = surface_geometry(c.surface, c.geom);
    c.akv  = approximate_killing_vector(c.surface, c.geom, c.sg);
    double const m_irr = c.surface.irreducible_mass(c.geom);
    c.spin = horizon_spin_akv(c.surface, c.geom, c.sg, m_irr);
    c.mp   = horizon_multipoles(c.surface, c.geom, c.sg, c.akv,
                                c.spin.christodoulou_mass);
    return c;
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("M_0 is the mass and J_1 is the spin", "[horizon_multipoles]") {
    // Both are identities, not approximations: M_0 reduces to M through
    // Gauss-Bonnet, and J_1 reduces to J because P'_1 = 1.  They therefore tie
    // the multipole quadrature to the area and spin integrals computed
    // independently elsewhere.
    for (double a : {0.0, 0.4, 0.8}) {
        auto const c = build(a);
        REQUIRE(c.mp.valid);

        INFO("a=" << a << " M_0 = " << c.mp.mass[0]
             << " (M = " << c.spin.christodoulou_mass << "), J_1 = "
             << c.mp.spin[1] << " (J = " << c.spin.J << ")");
        REQUIRE_THAT(c.mp.mass[0],
                     Catch::Matchers::WithinRel(c.spin.christodoulou_mass, 1e-6));
        REQUIRE_THAT(c.mp.spin[1], Catch::Matchers::WithinAbs(c.spin.J, 1e-8));
    }
}

TEST_CASE("Equatorial symmetry kills the odd mass and even spin moments",
          "[horizon_multipoles]") {
    // A Kerr horizon is symmetric under zeta -> -zeta.  P_n is odd for odd n,
    // so the odd mass moments must vanish; P'_n is odd for even n, so the even
    // spin moments must vanish.  Nothing in the implementation enforces this,
    // which makes it a genuine test of the invariant coordinate.
    for (double a : {0.4, 0.8}) {
        auto const c = build(a);
        REQUIRE(c.mp.valid);

        double const scale = c.spin.christodoulou_mass;
        for (int n = 1; n <= HORIZON_MULTIPOLE_MAX; n += 2) {
            INFO("a=" << a << " odd mass moment M_" << n << " = " << c.mp.mass[n]);
            REQUIRE_THAT(c.mp.mass[n] / scale, Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
        for (int n = 0; n <= HORIZON_MULTIPOLE_MAX; n += 2) {
            INFO("a=" << a << " even spin moment J_" << n << " = " << c.mp.spin[n]);
            REQUIRE_THAT(c.mp.spin[n], Catch::Matchers::WithinAbs(0.0, 1e-6));
        }
    }
}

TEST_CASE("A Schwarzschild horizon has no higher moments",
          "[horizon_multipoles]") {
    // A round horizon is fully described by M_0: every higher moment measures
    // distortion, and there is none.
    auto const c = build(0.0);
    REQUIRE(c.mp.valid);

    REQUIRE_THAT(c.mp.mass[0], Catch::Matchers::WithinRel(1.0, 1e-6));

    // Compared as dimensionless distortions M_n / (M R^n).  The R^n in the
    // normalisation amplifies a fixed relative error by R per order, so the raw
    // moments look progressively worse with n while the underlying accuracy is
    // constant: at l_max = 16 every M_n / (M R^n) here sits near 2e-4, and the
    // whole family falls by ~2 orders of magnitude on going to l_max = 32.
    double const R = c.surface.areal_radius(c.geom);
    double const M = c.spin.christodoulou_mass;

    for (int n = 1; n <= HORIZON_MULTIPOLE_MAX; ++n) {
        double const norm = M * std::pow(R, n);
        INFO("M_" << n << " = " << c.mp.mass[n]
             << " (dimensionless " << c.mp.mass[n] / norm << "), J_" << n
             << " = " << c.mp.spin[n]);
        REQUIRE_THAT(c.mp.mass[n] / norm, Catch::Matchers::WithinAbs(0.0, 1e-3));
        REQUIRE_THAT(c.mp.spin[n] / norm, Catch::Matchers::WithinAbs(0.0, 1e-3));
    }
}

TEST_CASE("The mass quadrupole grows negative with spin",
          "[horizon_multipoles]") {
    // A spinning horizon is oblate, so its mass quadrupole is negative and its
    // magnitude increases with spin.  This is the moment that actually
    // quantifies horizon distortion, which is what the multipoles are for.
    double previous = 1.0;
    for (double a : {0.0, 0.3, 0.6, 0.8}) {
        auto const c = build(a);
        REQUIRE(c.mp.valid);

        INFO("a=" << a << " M_2 = " << c.mp.mass[2]);
        if (a > 0.0) REQUIRE(c.mp.mass[2] < 0.0);
        REQUIRE(c.mp.mass[2] < previous);
        previous = c.mp.mass[2];
    }
    REQUIRE(previous < -0.5);   // clearly distorted at a = 0.8
}

TEST_CASE("Multipoles are unavailable without a Killing vector",
          "[horizon_multipoles]") {
    // The invariant coordinate comes from the AKV, so a failed solve must
    // report no multipoles rather than silently producing moments against an
    // arbitrary coordinate -- the same guard QuasiLocalMeasures applies with
    // qlm_have_killing_vector.
    auto c = build(0.4, 8);
    akv_result_t bad;                 // found == false
    auto const mp = horizon_multipoles(c.surface, c.geom, c.sg, bad, 1.0);
    REQUIRE_FALSE(mp.valid);
}
