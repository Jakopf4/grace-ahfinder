/**
 * @file test_surface_geometry.cpp
 * @brief Tests for the intrinsic geometry of a Strahlkoerper.
 *
 * The decisive test here is Gauss-Bonnet: for any topological sphere, whatever
 * its shape and whatever the ambient metric,
 *
 *     integral_S Rtilde dA = 8 pi
 *
 * exactly. It validates the whole chain -- metric Hessian, slice Ricci tensor,
 * extrinsic curvature of the surface, and the Gauss relation -- without needing
 * a closed-form answer for any individual piece.
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

#include <grace/geometry/surface_geometry.hh>
#include <grace/physics/analytic_geometry.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

std::vector<adm_point_t> flat_geometry(int npoints) {
    adm_point_t f{};
    f.g[ADM_XX] = f.g[ADM_YY] = f.g[ADM_ZZ] = 1.0;
    f.ginv[ADM_XX] = f.ginv[ADM_YY] = f.ginv[ADM_ZZ] = 1.0;
    f.sqrt_det_g = 1.0;
    return std::vector<adm_point_t>(npoints, f);
}

/// integral_S Rtilde dA, using the surface's own quadrature.
double integrate_ricci(strahlkoerper_t const& s,
                       std::vector<adm_point_t> const& geom) {
    auto const sg = surface_geometry(s, geom);
    std::vector<double> f(s.n_points());
    for (int p = 0; p < s.n_points(); ++p) f[p] = sg[p].ricci_scalar;
    return s.integrate(f, geom);
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Flat-space sphere has Rtilde = 2/r^2", "[surface_geometry]") {
    // The intrinsic curvature of a round sphere of radius r, and the simplest
    // check that the Gauss relation is assembled correctly.
    for (double r : {0.5, 1.0, 3.0}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);
        auto const geom = flat_geometry(s.n_points());
        auto const sg   = surface_geometry(s, geom);

        for (int p = 0; p < s.n_points(); ++p) {
            INFO("r=" << r << " point " << p);
            REQUIRE_THAT(sg[p].ricci_scalar,
                         Catch::Matchers::WithinRel(2.0 / (r*r), 1e-9));
            REQUIRE_THAT(sg[p].gauss_curvature,
                         Catch::Matchers::WithinRel(1.0 / (r*r), 1e-9));
            // Expansion of the outward normal within a flat slice is 2/r.
            REQUIRE_THAT(sg[p].theta_s, Catch::Matchers::WithinRel(2.0 / r, 1e-9));
            REQUIRE_THAT(sg[p].sqrt_q, Catch::Matchers::WithinRel(
                             r * r * s.basis().sin_theta(p / s.basis().n_phi()), 1e-9));
        }
    }
}

TEST_CASE("Gauss-Bonnet holds for a flat-space sphere", "[surface_geometry]") {
    for (double r : {0.5, 2.0, 7.0}) {
        strahlkoerper_t s(8, {0.0, 0.0, 0.0}, r);
        auto const geom = flat_geometry(s.n_points());
        INFO("r=" << r);
        REQUIRE_THAT(integrate_ricci(s, geom),
                     Catch::Matchers::WithinRel(8.0 * M_PI, 1e-8));
    }
}

TEST_CASE("Gauss-Bonnet holds for a deformed surface", "[surface_geometry]") {
    // Still flat space, but the surface is no longer a coordinate sphere, so
    // Rtilde varies over it and the tangent/extrinsic-curvature algebra is
    // genuinely exercised.  The integral is a topological invariant and must
    // not care.
    strahlkoerper_t s(16, {0.0, 0.0, 0.0}, 2.0);
    auto& a = s.coefficients();
    a[spherical_harmonic_basis_t::coeff_index(2, 0, false)] =  0.25;
    a[spherical_harmonic_basis_t::coeff_index(3, 1, false)] = -0.12;
    a[spherical_harmonic_basis_t::coeff_index(4, 2, true)]  =  0.08;

    auto const geom = flat_geometry(s.n_points());
    auto const sg   = surface_geometry(s, geom);

    // Confirm it really is deformed: Rtilde must vary.
    double lo = 1e30, hi = -1e30;
    for (auto const& g : sg) { lo = std::min(lo, g.ricci_scalar);
                               hi = std::max(hi, g.ricci_scalar); }
    INFO("Rtilde range " << lo << " .. " << hi);
    REQUIRE(hi - lo > 0.05);

    REQUIRE_THAT(integrate_ricci(s, geom),
                 Catch::Matchers::WithinRel(8.0 * M_PI, 1e-6));
}

TEST_CASE("Gauss-Bonnet holds in curved space", "[surface_geometry]") {
    // Kerr-Schild: now the slice Ricci tensor and the Hessian of the metric are
    // both nonzero, so every term in the Gauss relation contributes.
    for (double spin : {0.0, 0.6}) {
        kerr_schild_geometry_t ks(1.0, spin);

        for (double scale : {1.0, 2.0}) {
            strahlkoerper_t s(12, {0.0, 0.0, 0.0}, scale * ks.horizon_radius());
            auto const geom = ks.at(s.collocation_points());

            INFO("spin=" << spin << " r=" << scale * ks.horizon_radius());
            REQUIRE_THAT(integrate_ricci(s, geom),
                         Catch::Matchers::WithinRel(8.0 * M_PI, 1e-5));
        }
    }
}

TEST_CASE("Kerr horizon intrinsic curvature matches the closed form",
          "[surface_geometry]") {
    // The Kerr horizon 2-geometry has Gaussian curvature
    //
    //     K = (r+^2 + a^2)(r+^2 - 3 a^2 cos^2 t) / (r+^2 + a^2 cos^2 t)^3
    //
    // in Boyer-Lindquist angle t.  The collocation angle here is the spherical
    // polar angle of the Cartesian point, which differs from t in general but
    // coincides with it on the equator (both mean z = 0), so that is where the
    // two can be compared directly.
    //
    // A spinning horizon is oblate, so the curvature is *larger* at the equator
    // than near the axis -- and for a > sqrt(3)/2 M the polar value even turns
    // negative, which is the sharpest available check that Rtilde carries real
    // structure rather than merely integrating to the right total.
    for (double a : {0.6, 0.8, 0.9}) {
        kerr_schild_geometry_t ks(1.0, a);
        double const rp = ks.horizon_radius();

        int const l_max = 16;                 // n_theta = 17, odd -> a node at z = 0
        strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, rp);

        std::vector<double> h(s.n_points());
        for (int it = 0; it < s.basis().n_theta(); ++it)
            for (int ip = 0; ip < s.basis().n_phi(); ++ip)
                h[s.basis().point_index(it, ip)] =
                    ks.horizon_coordinate_radius(s.basis().theta(it), s.basis().phi(ip));
        s.coefficients() = s.project(h);

        auto const geom = ks.at(s.collocation_points());
        auto const sg   = surface_geometry(s, geom);

        int const n_phi   = s.basis().n_phi();
        int const i_equat = s.basis().n_theta() / 2;
        int const i_pole  = 0;

        REQUIRE_THAT(s.basis().cos_theta(i_equat),
                     Catch::Matchers::WithinAbs(0.0, 1e-12));

        double const K_equat = sg[i_equat * n_phi].gauss_curvature;
        double const K_pole  = sg[i_pole  * n_phi].gauss_curvature;

        // Equator: cos t = 0.
        double const want_equat = (rp*rp + a*a) * (rp*rp) / std::pow(rp*rp, 3);

        INFO("a=" << a << "  K_equator = " << K_equat << " (want " << want_equat
             << "), K_pole = " << K_pole);
        REQUIRE_THAT(K_equat, Catch::Matchers::WithinRel(want_equat, 1e-5));
        REQUIRE(K_equat > K_pole);

        // Beyond a = sqrt(3)/2 the curvature near the axis goes negative.
        if (a > 0.5 * std::sqrt(3.0)) REQUIRE(K_pole < 0.0);

        // And Gauss-Bonnet still holds on this surface.
        REQUIRE_THAT(integrate_ricci(s, geom),
                     Catch::Matchers::WithinRel(8.0 * M_PI, 1e-5));
    }
}

//---------------------------------------------------------------------------
// Shear and circumferences
//---------------------------------------------------------------------------

TEST_CASE("Shear vanishes on a round sphere and on the Kerr horizon",
          "[surface_geometry]") {
    SECTION("flat-space sphere: expanding but shear-free") {
        double const r = 2.0;
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);
        auto const geom = flat_geometry(s.n_points());
        auto const sg   = surface_geometry(s, geom);

        for (auto const& g : sg) {
            // k_AB = (1/r) q_AB is pure trace, so the shear is zero even though
            // the surface expands at 2/r.
            REQUIRE_THAT(g.expansion, Catch::Matchers::WithinRel(2.0 / r, 1e-9));
            REQUIRE_THAT(g.shear2,    Catch::Matchers::WithinAbs(0.0, 1e-16));
        }
    }

    SECTION("Kerr horizon: non-expanding AND shear-free") {
        // The defining property of an isolated horizon, and the check that
        // replaces QuasiLocalMeasures' Psi_0 = Psi_1 = 0 without needing any
        // time derivatives.
        for (double a : {0.0, 0.5, 0.8}) {
            kerr_schild_geometry_t ks(1.0, a);
            strahlkoerper_t s(16, {0.0, 0.0, 0.0}, ks.horizon_radius());

            std::vector<double> h(s.n_points());
            for (int it = 0; it < s.basis().n_theta(); ++it)
                for (int ip = 0; ip < s.basis().n_phi(); ++ip)
                    h[s.basis().point_index(it, ip)] =
                        ks.horizon_coordinate_radius(s.basis().theta(it),
                                                     s.basis().phi(ip));
            s.coefficients() = s.project(h);

            auto const geom = ks.at(s.collocation_points());
            auto const sg   = surface_geometry(s, geom);

            double worst_exp = 0.0, worst_shear = 0.0;
            for (auto const& g : sg) {
                worst_exp   = std::max(worst_exp,   std::abs(g.expansion));
                worst_shear = std::max(worst_shear, std::sqrt(std::abs(g.shear2)));
            }
            INFO("a=" << a << " max|Theta| = " << worst_exp
                 << ", max|sigma| = " << worst_shear);
            REQUIRE(worst_exp   < 1e-6);
            REQUIRE(worst_shear < 1e-6);
        }
    }

    SECTION("a sphere off the horizon is not shear-free in Kerr") {
        // Guards against the shear being identically zero by construction.
        kerr_schild_geometry_t ks(1.0, 0.8);
        strahlkoerper_t s(16, {0.0, 0.0, 0.0}, 1.6 * ks.horizon_radius());
        auto const geom = ks.at(s.collocation_points());
        auto const sg   = surface_geometry(s, geom);

        double worst = 0.0;
        for (auto const& g : sg) worst = std::max(worst, std::sqrt(std::abs(g.shear2)));
        INFO("max|sigma| off the horizon = " << worst);
        REQUIRE(worst > 1e-3);
    }
}

TEST_CASE("Circumferences of a flat-space sphere are 2 pi r",
          "[surface_geometry]") {
    for (double r : {1.0, 3.5}) {
        strahlkoerper_t s(8, {0.0, 0.0, 0.0}, r);   // l_max even -> equator node
        auto const geom = flat_geometry(s.n_points());
        auto const sg   = surface_geometry(s, geom);
        auto const c    = surface_circumferences(s, sg);

        INFO("r=" << r);
        REQUIRE_THAT(c.equatorial, Catch::Matchers::WithinRel(2.0 * M_PI * r, 1e-9));
        REQUIRE_THAT(c.polar_xz,   Catch::Matchers::WithinRel(2.0 * M_PI * r, 1e-8));
        REQUIRE_THAT(c.polar_yz,   Catch::Matchers::WithinRel(2.0 * M_PI * r, 1e-8));
        REQUIRE_THAT(c.ratio_xz_eq, Catch::Matchers::WithinRel(1.0, 1e-8));
    }
}

TEST_CASE("Kerr horizon is oblate in proper circumference",
          "[surface_geometry]") {
    // The classic pre-quasi-local spin proxy: a spinning horizon is shorter
    // around the poles than around the equator, and the ratio decreases
    // monotonically with spin.
    double previous = 1e30;
    for (double a : {0.0, 0.4, 0.8}) {
        kerr_schild_geometry_t ks(1.0, a);
        strahlkoerper_t s(16, {0.0, 0.0, 0.0}, ks.horizon_radius());

        std::vector<double> h(s.n_points());
        for (int it = 0; it < s.basis().n_theta(); ++it)
            for (int ip = 0; ip < s.basis().n_phi(); ++ip)
                h[s.basis().point_index(it, ip)] =
                    ks.horizon_coordinate_radius(s.basis().theta(it), s.basis().phi(ip));
        s.coefficients() = s.project(h);

        auto const geom = ks.at(s.collocation_points());
        auto const sg   = surface_geometry(s, geom);
        auto const c    = surface_circumferences(s, sg);

        // Kerr's equatorial proper circumference is exactly 2 pi (r+^2+a^2)/r+.
        double const rp = ks.horizon_radius();
        double const want_eq = 2.0 * M_PI * (rp*rp + a*a) / rp;

        INFO("a=" << a << " C_eq = " << c.equatorial << " (want " << want_eq
             << "), ratio = " << c.ratio_xz_eq);
        REQUIRE_THAT(c.equatorial, Catch::Matchers::WithinRel(want_eq, 1e-5));

        // Axisymmetric, so the two meridians must agree.
        REQUIRE_THAT(c.polar_xz, Catch::Matchers::WithinRel(c.polar_yz, 1e-6));

        REQUIRE(c.ratio_xz_eq < previous);
        previous = c.ratio_xz_eq;
    }
    REQUIRE(previous < 1.0);   // clearly oblate at a = 0.8
}
