/**
 * @file test_horizon_fluxes.cpp
 * @brief Flux integrals over a horizon surface, against closed forms.
 *
 * Every case here has an answer that is known independently of the
 * implementation: uniform radial infall through a sphere, the flux of a
 * monopole field through any surface enclosing it, and the two identities that
 * must hold whatever the matter does -- a static fluid accretes nothing, and
 * the signed magnetic flux through a closed surface vanishes.
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

#include <grace/geometry/horizon_fluxes.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

/// Flat space at every collocation point.
std::vector<adm_point_t> flat_geometry(int npoints) {
    adm_point_t f{};
    f.g[ADM_XX] = f.g[ADM_YY] = f.g[ADM_ZZ] = 1.0;
    f.ginv[ADM_XX] = f.ginv[ADM_YY] = f.ginv[ADM_ZZ] = 1.0;
    f.sqrt_det_g = 1.0;
    return std::vector<adm_point_t>(static_cast<std::size_t>(npoints), f);
}

/// Matter built pointwise from the Cartesian position of each collocation point.
template <typename F>
std::vector<matter_point_t> matter_from(strahlkoerper_t const& s, F&& fill) {
    auto const pts = s.collocation_points();
    std::vector<matter_point_t> m(pts.size());
    for (std::size_t p = 0; p < pts.size(); ++p) {
        m[p].alp   = 1.0;
        m[p].valid = true;
        fill(pts[p], m[p]);
    }
    return m;
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Uniform radial infall gives the closed-form accretion rate",
          "[horizon_fluxes]") {
    // rho = rho0, fluid falling inward at Valencia speed v through a sphere of
    // radius R in flat space:  Mdot = 4 pi R^2 rho0 W v, with z = W v.
    double const R    = 2.0;
    double const rho0 = 0.125;
    double const v    = 0.4;
    double const W    = 1.0 / std::sqrt(1.0 - v * v);

    strahlkoerper_t s(8, {0.0, 0.0, 0.0}, R);
    auto const geom = flat_geometry(s.n_points());

    auto const matter = matter_from(s, [&](std::array<double,3> const& x,
                                           matter_point_t& m) {
        double const r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
        m.rho = rho0;
        for (int d = 0; d < 3; ++d) m.z[d] = -(W * v) * x[d] / r;   // inward
    });

    auto const f = horizon_fluxes(s, geom, matter, {0.0, 0.0, 0.0});
    REQUIRE(f.valid);

    double const expect = 4.0 * M_PI * R * R * rho0 * W * v;
    INFO("mdot = " << f.mdot << ", expected " << expect);
    REQUIRE_THAT(f.mdot, Catch::Matchers::WithinRel(expect, 1e-10));
    REQUIRE(f.mdot > 0.0);   // inflow is positive, by convention
}

TEST_CASE("A static fluid accretes nothing", "[horizon_fluxes]") {
    strahlkoerper_t s(6, {0.0, 0.0, 0.0}, 1.7);
    auto const geom = flat_geometry(s.n_points());

    auto const matter = matter_from(s, [](std::array<double,3> const&,
                                          matter_point_t& m) {
        m.rho = 1.0;   // dense, but going nowhere
    });

    auto const f = horizon_fluxes(s, geom, matter, {0.0, 0.0, 0.0});
    REQUIRE(f.valid);
    REQUIRE_THAT(f.mdot, Catch::Matchers::WithinAbs(0.0, 1e-12));
}

TEST_CASE("Magnetic flux through a closed surface obeys its identities",
          "[horizon_fluxes]") {
    strahlkoerper_t s(8, {0.0, 0.0, 0.0}, 1.5);
    auto const geom = flat_geometry(s.n_points());
    double const C = 0.7;

    SECTION("a monopole gives 4 pi C through any enclosing surface") {
        // B^i = C x^i / r^3 is divergence-free away from the origin, so the
        // signed flux is the same through every surface enclosing it. This
        // tests the normal and the proper area element together.
        auto const matter = matter_from(s, [&](std::array<double,3> const& x,
                                               matter_point_t& m) {
            double const r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
            double const r3 = r * r * r;
            for (int d = 0; d < 3; ++d) m.B[d] = C * x[d] / r3;
        });

        auto const f = horizon_fluxes(s, geom, matter, {0.0, 0.0, 0.0});
        REQUIRE(f.valid);
        REQUIRE_THAT(f.phi_B_signed,
                     Catch::Matchers::WithinRel(4.0 * M_PI * C, 1e-10));
    }

    SECTION("an odd field cancels, and the absolute flux does not") {
        // Separating the signed flux from the absolute one needs a field whose
        // normal component is odd across the equator. A hard sign flip will not
        // do: with an odd theta count the Gauss-Legendre grid puts a node
        // *exactly* on the equator, so a discontinuity there is resolved by
        // whichever side the tie-break picks and one whole ring fails to
        // cancel. Weighting by cos(theta) instead is smooth and vanishes on the
        // node itself.
        //
        //   B^i = C cos(theta) x^i / r^4   ->   B^i s_i = C cos(theta) / R^3
        //   signed:   0 exactly, by symmetry of the nodes about mu = 0
        //   absolute: (C/R^3) * R^2 * 2 pi * int |mu| dmu = 2 pi C / R
        double const R = 1.5;
        auto const matter = matter_from(s, [&](std::array<double,3> const& x,
                                               matter_point_t& m) {
            double const r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
            double const r4 = r * r * r * r;
            double const mu = x[2] / r;
            for (int d = 0; d < 3; ++d) m.B[d] = mu * C * x[d] / r4;
        });

        auto const f = horizon_fluxes(s, geom, matter, {0.0, 0.0, 0.0});
        REQUIRE(f.valid);
        REQUIRE_THAT(f.phi_B_signed, Catch::Matchers::WithinAbs(0.0, 1e-10));
        // In a fluid at rest b^i reduces to B^i and b^0 vanishes, so
        // phi_B = (1/2) integral |B.s| dA.
        //
        // Loose on purpose. |cos(theta)| has a kink at the equator, and Gauss-
        // Legendre is only algebraically accurate on a non-smooth integrand --
        // ~2% at nine nodes, falling as the node count squared. That is a
        // property of the absolute value in the Blandford-Znajek definition,
        // not of this code, and it is worth having on record: phi_B on a real
        // horizon is quadrature-limited wherever B changes sign across the
        // surface. The *signed* flux above, which is the identity that has to
        // hold, is exact to 1e-10.
        REQUIRE_THAT(f.phi_B,
                     Catch::Matchers::WithinRel(M_PI * C / R, 0.03));
    }
}

TEST_CASE("Matter parity carries B as a pseudovector", "[horizon_fluxes]") {
    // Under a reflection of axis a, a true vector flips its a-component and
    // keeps the others; a pseudovector does exactly the opposite. Getting this
    // backwards would corrupt the magnetic flux on every equatorially
    // symmetric run and leave mdot looking perfectly healthy.
    matter_point_t m{};
    m.rho = 2.0; m.eps = 0.5; m.press = 0.25; m.alp = 0.8;
    m.z[0] = 1.0; m.z[1] = 2.0; m.z[2] = 3.0;
    m.beta[0] = 4.0; m.beta[1] = 5.0; m.beta[2] = 6.0;
    m.B[0] = 7.0; m.B[1] = 8.0; m.B[2] = 9.0;

    SECTION("single z reflection") {
        auto q = m;
        detail::reflect_matter_point(q, 1u << 2);

        // Scalars untouched.
        REQUIRE(q.rho == m.rho);
        REQUIRE(q.eps == m.eps);
        REQUIRE(q.press == m.press);
        REQUIRE(q.alp == m.alp);
        // Vectors: only the z component flips.
        REQUIRE(q.z[0] ==  m.z[0]);
        REQUIRE(q.z[1] ==  m.z[1]);
        REQUIRE(q.z[2] == -m.z[2]);
        REQUIRE(q.beta[2] == -m.beta[2]);
        // Pseudovector: z survives, x and y flip.
        REQUIRE(q.B[0] == -m.B[0]);
        REQUIRE(q.B[1] == -m.B[1]);
        REQUIRE(q.B[2] ==  m.B[2]);
    }

    SECTION("two reflections restore the pseudovector to the vector rule") {
        // The extra factor a pseudovector picks up is the determinant of the
        // reflection, (-1)^(number of folded axes), so an even number of folds
        // makes B transform like z again.
        auto q = m;
        detail::reflect_matter_point(q, (1u << 0) | (1u << 2));
        for (int d = 0; d < 3; ++d) {
            INFO("component " << d);
            REQUIRE(q.B[d] == q.z[d] / m.z[d] * m.B[d]);
        }
    }

    SECTION("a zero mask is the identity") {
        auto q = m;
        detail::reflect_matter_point(q, 0u);
        for (int d = 0; d < 3; ++d) {
            REQUIRE(q.z[d] == m.z[d]);
            REQUIRE(q.B[d] == m.B[d]);
        }
    }
}

TEST_CASE("The factored integrand reproduces the original bh_diagnostics algebra",
          "[horizon_fluxes]") {
    // flux_integrand() was lifted out of bh_diagnostics::compute_local_fluxes
    // so the MHD stress algebra exists once. This pins the lift: the expressions
    // below are the originals, transcribed, and must agree with the shared
    // function on arbitrary input.
    //
    // Not asserted bitwise. Pulling `rho` inside the integrand regroups the
    // accretion product from ((-d*s)*rho)*u^r to (d*s)*(-(rho*u^r)), and
    // floating-point multiplication is not associative, so the two can differ
    // in the last place. Everything else is transcribed unchanged.
    metric_array_t const metric{ {1.1, 0.05, -0.02, 1.3, 0.07, 0.9},
                                 {0.10, -0.20, 0.05}, 0.85 };

    double const rho = 3.0, eps = 0.4, press = 0.9;
    double const vx = 0.10, vy = -0.25, vz = 0.30;    // Valencia z^i
    double const bx = 0.7,  by = 0.2,  bz = -0.5;
    double const dir[3]     = { 0.3, -0.6, 0.74 };
    double const phi_dir[3] = { -1.2, 0.8, 0.0 };

    // --- the original expressions ---------------------------------------
    double const one_over_alp = 1. / metric.alp();
    double const W  = std::sqrt(1 + metric.square_vec({vx, vy, vz}));
    double const u0 = one_over_alp * W;

    std::array<double,4> uU{{ u0,
                              vx - one_over_alp * metric.beta(0),
                              vy - one_over_alp * metric.beta(1),
                              vz - one_over_alp * metric.beta(2) }};
    auto uD = metric.lower_4vec(uU);

    double b0 = uD[1] * bx + uD[2] * by + uD[3] * bz;
    double b1 = (bx + b0 * uU[1]) / u0;
    double b2 = (by + b0 * uU[2]) / u0;
    double b3 = (bz + b0 * uU[3]) / u0;
    auto bD = metric.lower_4vec({b0, b1, b2, b3});
    double bsq = b0 * bD[0] + b1 * bD[1] + b2 * bD[2] + b3 * bD[3];

    double ur = dir[0] * uU[1] + dir[1] * uU[2] + dir[2] * uU[3];
    double br = dir[0] * b1    + dir[1] * b2    + dir[2] * b3;
    double u_phi = phi_dir[0] * uD[1] + phi_dir[1] * uD[2];
    double b_phi = phi_dir[0] * bD[1] + phi_dir[1] * bD[2];

    double rhoh  = rho + rho * eps + press;
    double tr_0  = (rhoh + bsq) * ur * uD[0] - br * bD[0];
    double tr_3  = (rhoh + bsq) * ur * u_phi - br * b_phi;
    double phi_l = 0.5 * std::fabs(br * u0 - b0 * ur);

    // --- the shared function ---------------------------------------------
    auto const fi = flux_integrand(
        metric, mhd_primitives_t{rho, eps, press, {vx, vy, vz}, {bx, by, bz}},
        dir, phi_dir);

    REQUIRE_THAT(fi.mdot,  Catch::Matchers::WithinRel(-rho * ur, 1e-15));
    REQUIRE_THAT(fi.edot,  Catch::Matchers::WithinRel(-tr_0,     1e-15));
    REQUIRE_THAT(fi.ldot,  Catch::Matchers::WithinRel(tr_3,      1e-15));
    REQUIRE_THAT(fi.phi_B, Catch::Matchers::WithinRel(phi_l,     1e-15));
    REQUIRE_THAT(fi.b_n,   Catch::Matchers::WithinRel(br,        1e-15));
}
