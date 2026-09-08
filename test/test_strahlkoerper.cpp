/**
 * @file test_strahlkoerper.cpp
 * @brief Tests for the spectral star-shaped surface and the analytic
 *        Kerr-Schild geometry provider it is checked against.
 *
 * The surface integrals are validated in two regimes: flat space, where closed
 * forms exist for the shapes used, and Kerr-Schild, where the horizon area is
 * known exactly (A = 8 pi M r+). The latter is the check that actually matters
 * for a horizon finder, because it exercises a non-spherical surface in a
 * curved metric.
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

#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/analytic_geometry.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

/// Flat space: gamma_ij = delta_ij, K_ij = 0, no gradients.
std::vector<adm_point_t> flat_geometry(int npoints) {
    adm_point_t f{};
    f.g[ADM_XX] = f.g[ADM_YY] = f.g[ADM_ZZ] = 1.0;
    f.ginv[ADM_XX] = f.ginv[ADM_YY] = f.ginv[ADM_ZZ] = 1.0;
    f.sqrt_det_g = 1.0;
    return std::vector<adm_point_t>(npoints, f);
}

}  // namespace

//---------------------------------------------------------------------------
// Construction and representation
//---------------------------------------------------------------------------

TEST_CASE("A coordinate sphere is represented exactly", "[strahlkoerper]") {
    double const r = 2.75;
    strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);

    auto const shape = s.evaluate();
    for (int p = 0; p < s.n_points(); ++p) {
        INFO("point " << p);
        REQUIRE_THAT(shape.h[p],   Catch::Matchers::WithinAbs(r,   1e-12));
        REQUIRE_THAT(shape.dth[p], Catch::Matchers::WithinAbs(0.0, 1e-12));
        REQUIRE_THAT(shape.dph[p], Catch::Matchers::WithinAbs(0.0, 1e-12));
    }

    // Every collocation point sits at distance r from the centre.
    for (auto const& p : s.collocation_points()) {
        double const d = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        REQUIRE_THAT(d, Catch::Matchers::WithinAbs(r, 1e-12));
    }
}

TEST_CASE("Collocation points respect a shifted centre", "[strahlkoerper]") {
    double const r = 1.5;
    std::array<double,3> const c{0.4, -1.2, 3.0};
    strahlkoerper_t s(4, c, r);

    for (auto const& p : s.collocation_points()) {
        double const d = std::sqrt((p[0]-c[0])*(p[0]-c[0])
                                 + (p[1]-c[1])*(p[1]-c[1])
                                 + (p[2]-c[2])*(p[2]-c[2]));
        REQUIRE_THAT(d, Catch::Matchers::WithinAbs(r, 1e-12));
    }
}

TEST_CASE("project() inverts evaluate()", "[strahlkoerper]") {
    strahlkoerper_t s(6, {0.0, 0.0, 0.0}, 1.0);

    // A non-trivial l=2, m=1 deformation on top of the sphere.
    auto& a = s.coefficients();
    a[spherical_harmonic_basis_t::coeff_index(2, 1, false)] =  0.13;
    a[spherical_harmonic_basis_t::coeff_index(3, 0, false)] = -0.07;
    a[spherical_harmonic_basis_t::coeff_index(4, 2, true)]  =  0.05;
    auto const expected = a;

    auto const rt = s.project(s.evaluate().h);
    for (int c = 0; c < s.n_coeffs(); ++c) {
        INFO("coefficient " << c);
        REQUIRE_THAT(rt[c], Catch::Matchers::WithinAbs(expected[c], 1e-12));
    }
}

TEST_CASE("radius_at agrees with the collocation values", "[strahlkoerper]") {
    strahlkoerper_t s(5, {0.0, 0.0, 0.0}, 2.0);
    s.coefficients()[spherical_harmonic_basis_t::coeff_index(2, 0, false)] = 0.2;

    auto const shape = s.evaluate();
    for (int it = 0; it < s.basis().n_theta(); ++it)
        for (int ip = 0; ip < s.basis().n_phi(); ++ip) {
            int const p = s.basis().point_index(it, ip);
            INFO("it=" << it << " ip=" << ip);
            REQUIRE_THAT(s.radius_at(s.basis().theta(it), s.basis().phi(ip)),
                         Catch::Matchers::WithinAbs(shape.h[p], 1e-12));
        }
}

//---------------------------------------------------------------------------
// Surface integrals in flat space
//---------------------------------------------------------------------------

TEST_CASE("Area of a flat-space sphere is 4 pi r^2", "[strahlkoerper]") {
    for (double r : {0.5, 1.0, 3.25}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);
        auto const geom = flat_geometry(s.n_points());

        INFO("r=" << r);
        REQUIRE_THAT(s.area(geom),
                     Catch::Matchers::WithinRel(4.0 * M_PI * r * r, 1e-11));
        REQUIRE_THAT(s.areal_radius(geom), Catch::Matchers::WithinRel(r, 1e-11));
        REQUIRE_THAT(s.irreducible_mass(geom),
                     Catch::Matchers::WithinRel(0.5 * r, 1e-11));
    }
}

TEST_CASE("Centroid of an off-centre flat sphere is its centre",
          "[strahlkoerper]") {
    std::array<double,3> const c{1.25, -0.5, 2.0};
    strahlkoerper_t s(6, c, 1.75);
    auto const geom = flat_geometry(s.n_points());

    auto const cen = s.centroid(geom);
    for (int d = 0; d < 3; ++d) {
        INFO("component " << d);
        REQUIRE_THAT(cen[d], Catch::Matchers::WithinAbs(c[d], 1e-10));
    }
}

TEST_CASE("Radius statistics on a sphere are all the radius", "[strahlkoerper]") {
    double const r = 2.5;
    strahlkoerper_t s(5, {0.0, 0.0, 0.0}, r);
    auto const geom = flat_geometry(s.n_points());

    auto const st = s.radius_stats(geom);
    REQUIRE_THAT(st.min,  Catch::Matchers::WithinAbs(r, 1e-11));
    REQUIRE_THAT(st.max,  Catch::Matchers::WithinAbs(r, 1e-11));
    REQUIRE_THAT(st.mean, Catch::Matchers::WithinRel(r, 1e-11));
}

TEST_CASE("Area of a flat-space prolate spheroid matches its closed form",
          "[strahlkoerper]") {
    // A genuinely non-spherical surface, so the tangent-vector and induced
    // 2-metric algebra is exercised rather than cancelling.
    //
    // Spheroid  (x^2+y^2)/a^2 + z^2/c^2 = 1  with c > a has
    //     A = 2 pi a^2 (1 + (c/(a e)) asin(e)),  e = sqrt(1 - a^2/c^2).
    // In polar form r(theta) = a c / sqrt(c^2 sin^2 t + a^2 cos^2 t), which is
    // smooth and even in cos(theta) so a moderate l_max resolves it well.
    double const a = 1.0, c = 1.4;
    int const l_max = 24;

    strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, 1.0);

    // Build the shape by projecting the exact radius onto the basis.
    std::vector<double> h(s.n_points());
    for (int it = 0; it < s.basis().n_theta(); ++it) {
        double const st = s.basis().sin_theta(it);
        double const ct = s.basis().cos_theta(it);
        double const r  = a * c / std::sqrt(c*c*st*st + a*a*ct*ct);
        for (int ip = 0; ip < s.basis().n_phi(); ++ip)
            h[s.basis().point_index(it, ip)] = r;
    }
    s.coefficients() = s.project(h);

    auto const geom = flat_geometry(s.n_points());

    double const e = std::sqrt(1.0 - (a*a)/(c*c));
    double const exact = 2.0 * M_PI * a * a * (1.0 + (c / (a * e)) * std::asin(e));

    INFO("spectral area = " << s.area(geom) << "  exact = " << exact);
    REQUIRE_THAT(s.area(geom), Catch::Matchers::WithinRel(exact, 1e-6));
}

//---------------------------------------------------------------------------
// Analytic Kerr-Schild geometry
//---------------------------------------------------------------------------

TEST_CASE("Kerr-Schild horizon radius and area follow the closed forms",
          "[analytic_geometry]") {
    SECTION("Schwarzschild") {
        kerr_schild_geometry_t ks(1.0, 0.0);
        REQUIRE_THAT(ks.horizon_radius(), Catch::Matchers::WithinRel(2.0, 1e-14));
        REQUIRE_THAT(ks.horizon_area(),
                     Catch::Matchers::WithinRel(16.0 * M_PI, 1e-14));
        REQUIRE_THAT(ks.irreducible_mass(), Catch::Matchers::WithinRel(1.0, 1e-14));
    }
    SECTION("spinning") {
        double const spin = 0.6;
        kerr_schild_geometry_t ks(1.0, spin);
        double const rp = 1.0 + std::sqrt(1.0 - spin * spin);
        REQUIRE_THAT(ks.horizon_radius(), Catch::Matchers::WithinRel(rp, 1e-14));
        REQUIRE_THAT(ks.horizon_area(),
                     Catch::Matchers::WithinRel(8.0 * M_PI * rp, 1e-14));
    }
    SECTION("mass scaling") {
        kerr_schild_geometry_t ks(2.5, 0.0);
        REQUIRE_THAT(ks.horizon_radius(), Catch::Matchers::WithinRel(5.0, 1e-14));
        REQUIRE_THAT(ks.horizon_area(),
                     Catch::Matchers::WithinRel(16.0 * M_PI * 2.5 * 2.5, 1e-14));
    }
}

TEST_CASE("Kerr-Schild geometry is self-consistent", "[analytic_geometry]") {
    kerr_schild_geometry_t ks(1.0, 0.5);

    std::vector<std::array<double,3>> pts{
        {3.0, 0.0, 0.0}, {0.0, 4.0, 1.0}, {2.0, -2.0, 2.5}, {5.0, 1.0, -3.0}};

    for (auto const& x : ks.at(pts)) {
        // gamma^ik gamma_kj = delta^i_j
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double sum = 0;
                for (int k = 0; k < 3; ++k)
                    sum += x.ginv[adm_sym(i,k)] * x.g[adm_sym(k,j)];
                INFO("i=" << i << " j=" << j);
                REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(i==j ? 1.0 : 0.0, 1e-11));
            }
        REQUIRE(x.sqrt_det_g > 0.0);
        REQUIRE(std::isfinite(x.K_trace));
    }
}

TEST_CASE("Kerr-Schild metric gradients match differences of the metric",
          "[analytic_geometry]") {
    kerr_schild_geometry_t ks(1.0, 0.4);
    double const e = 1e-4;

    for (auto const& x0 : { std::array<double,3>{3.0, 1.0, 0.5},
                            std::array<double,3>{-4.0, 2.0, -1.0} }) {
        auto const a0 = ks.at(x0);
        for (int k = 0; k < 3; ++k) {
            auto xp = x0; xp[k] += e;
            auto xm = x0; xm[k] -= e;
            auto const ap = ks.at(xp);
            auto const am = ks.at(xm);
            for (int c = 0; c < ADM_NSYM; ++c) {
                double const fd = (ap.g[c] - am.g[c]) / (2 * e);
                INFO("k=" << k << " c=" << c);
                REQUIRE_THAT(a0.dg[k][c], Catch::Matchers::WithinAbs(fd, 1e-6));
            }
        }
    }
}

//---------------------------------------------------------------------------
// The combined check: horizon area in a curved, non-spherical case
//---------------------------------------------------------------------------

TEST_CASE("Proper area of the Kerr-Schild horizon is 8 pi M r+",
          "[strahlkoerper][analytic_geometry]") {
    // For a != 0 the horizon is an ellipsoid in Cartesian Kerr-Schild
    // coordinates, so this exercises a non-spherical surface in a curved
    // metric -- the configuration a horizon finder actually meets. The exact
    // answer is independent of the coordinate shape, which is what makes it a
    // strong check of the induced-metric algebra.
    for (double spin : {0.0, 0.3, 0.6, 0.8}) {
        kerr_schild_geometry_t ks(1.0, spin);
        int const l_max = 16;
        strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, ks.horizon_radius());

        // Set the surface to the exact horizon by projecting its coordinate
        // radius onto the basis.
        std::vector<double> h(s.n_points());
        for (int it = 0; it < s.basis().n_theta(); ++it)
            for (int ip = 0; ip < s.basis().n_phi(); ++ip) {
                int const p = s.basis().point_index(it, ip);
                h[p] = ks.horizon_coordinate_radius(s.basis().theta(it),
                                                    s.basis().phi(ip));
            }
        s.coefficients() = s.project(h);

        auto const geom = ks.at(s.collocation_points());
        double const computed = s.area(geom);
        double const exact    = ks.horizon_area();

        INFO("spin=" << spin << " computed=" << computed << " exact=" << exact);
        REQUIRE_THAT(computed, Catch::Matchers::WithinRel(exact, 1e-6));

        // And the derived masses.
        REQUIRE_THAT(s.irreducible_mass(geom),
                     Catch::Matchers::WithinRel(ks.irreducible_mass(), 1e-6));
    }
}

TEST_CASE("Coordinate spheres in Schwarzschild-Kerr-Schild have area 4 pi r^2",
          "[strahlkoerper][analytic_geometry]") {
    // For a = 0 the Kerr-Schild 3-metric is gamma_ij = delta_ij + (2M/r) n_i n_j
    // with n the unit radial vector.  Vectors tangent to a coordinate sphere are
    // orthogonal to n, so the induced 2-metric is exactly the flat round metric
    // and the proper area is exactly 4 pi r^2 -- at every radius, not just
    // asymptotically.  An exact identity in a curved metric, so it isolates the
    // induced-metric algebra from any quadrature or truncation error.
    kerr_schild_geometry_t ks(1.0, 0.0);

    for (double r : {2.0, 5.0, 50.0, 1000.0}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);
        auto const geom = ks.at(s.collocation_points());

        INFO("r=" << r);
        REQUIRE_THAT(s.area(geom),
                     Catch::Matchers::WithinRel(4.0 * M_PI * r * r, 1e-11));
    }

    // At r = 2M that sphere is the horizon, so it must also reproduce 8 pi M r+.
    strahlkoerper_t horizon(6, {0.0, 0.0, 0.0}, ks.horizon_radius());
    auto const geom = ks.at(horizon.collocation_points());
    REQUIRE_THAT(horizon.area(geom),
                 Catch::Matchers::WithinRel(ks.horizon_area(), 1e-11));
}

//---------------------------------------------------------------------------
// Persistence and output
//---------------------------------------------------------------------------

TEST_CASE("Serialization round-trips shape and centre", "[strahlkoerper]") {
    strahlkoerper_t original(5, {0.3, -1.1, 2.2}, 1.7);
    auto& a = original.coefficients();
    a[spherical_harmonic_basis_t::coeff_index(2, 1, true)]  = 0.09;
    a[spherical_harmonic_basis_t::coeff_index(3, 2, false)] = -0.04;

    auto const buf = original.serialize();
    REQUIRE(buf.size() == strahlkoerper_t::serialized_size(5));

    strahlkoerper_t restored(5, {0.0, 0.0, 0.0}, 1.0);
    REQUIRE(restored.deserialize(buf));

    for (int d = 0; d < 3; ++d)
        REQUIRE(restored.centre()[d] == original.centre()[d]);
    for (int c = 0; c < original.n_coeffs(); ++c) {
        INFO("coefficient " << c);
        REQUIRE(restored.coefficients()[c] == original.coefficients()[c]);
    }
}

TEST_CASE("Deserialization rejects malformed or mismatched buffers",
          "[strahlkoerper]") {
    strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 1.0);
    auto const good = s.serialize();

    SECTION("truncated header") {
        REQUIRE_FALSE(s.deserialize(std::vector<double>{1.0, 2.0}));
    }
    SECTION("wrong l_max is rejected rather than misread") {
        // A checkpoint written at a different angular resolution must not be
        // silently reinterpreted as coefficients of this basis.
        strahlkoerper_t other(6, {0.0, 0.0, 0.0}, 1.0);
        REQUIRE_FALSE(s.deserialize(other.serialize()));
    }
    SECTION("truncated coefficient block") {
        auto bad = good;
        bad.pop_back();
        REQUIRE_FALSE(s.deserialize(bad));
    }
    SECTION("a rejected buffer leaves the surface untouched") {
        auto const before = s.coefficients();
        REQUIRE_FALSE(s.deserialize(std::vector<double>{1.0}));
        REQUIRE(s.coefficients() == before);
    }
}

//---------------------------------------------------------------------------
// Re-expressing the shape about a new origin
//---------------------------------------------------------------------------

TEST_CASE("recentre moves the origin without moving the surface",
          "[strahlkoerper]") {
    // set_centre alone keeps h as the radius about the *old* origin and so
    // translates the whole surface.  recentre re-solves for the radius in each
    // new direction, leaving the surface itself exactly where it was -- which
    // is what makes a tracked origin compatible with a warm start.
    double const r = 2.0;
    strahlkoerper_t s(16, {0.0, 0.0, 0.0}, r);
    auto const geom = flat_geometry(s.n_points());
    double const area_before = s.area(geom);

    std::array<double,3> const shift{0.5, -0.3, 0.2};
    REQUIRE(s.recentre(shift));

    for (int d = 0; d < 3; ++d) REQUIRE(s.centre()[d] == shift[d]);

    // Every collocation point must still lie on the original sphere.
    for (auto const& p : s.collocation_points()) {
        double const dist = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
        REQUIRE_THAT(dist, Catch::Matchers::WithinAbs(r, 1e-6));
    }

    // And the proper area is unchanged, since the surface has not moved.
    auto const geom2 = flat_geometry(s.n_points());
    REQUIRE_THAT(s.area(geom2), Catch::Matchers::WithinRel(area_before, 1e-6));
}

TEST_CASE("recentre refuses an origin outside the surface", "[strahlkoerper]") {
    // Outside, the surface is not star-shaped about the new origin and the
    // re-expression has no meaning; the surface must be left untouched rather
    // than silently mangled.
    strahlkoerper_t s(8, {0.0, 0.0, 0.0}, 1.0);
    auto const before = s.coefficients();

    REQUIRE_FALSE(s.recentre({5.0, 0.0, 0.0}));
    REQUIRE(s.coefficients() == before);
    REQUIRE(s.centre()[0] == 0.0);

    // Just inside is fine.
    REQUIRE(s.recentre({0.9, 0.0, 0.0}));
}

TEST_CASE("recentre round-trips", "[strahlkoerper]") {
    // Moving out and back must return the original shape, which is a strong
    // check that the ray-surface solve is consistent in both directions.
    strahlkoerper_t s(16, {0.0, 0.0, 0.0}, 1.5);
    s.coefficients()[spherical_harmonic_basis_t::coeff_index(2, 0, false)] = 0.18;
    auto const before = s.coefficients();

    REQUIRE(s.recentre({0.35, -0.2, 0.15}));
    REQUIRE(s.recentre({0.0, 0.0, 0.0}));

    for (std::size_t c = 0; c < before.size(); ++c) {
        INFO("coefficient " << c);
        REQUIRE_THAT(s.coefficients()[c],
                     Catch::Matchers::WithinAbs(before[c], 1e-6));
    }
}
