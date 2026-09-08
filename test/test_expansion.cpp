/**
 * @file test_expansion.cpp
 * @brief Tests for the expansion of the outgoing null normal.
 *
 * Checked against two situations with exact answers: a sphere in flat space,
 * where Theta = 2/r identically, and the Kerr-Schild horizon, where Theta = 0
 * by definition. Between them these pin down the divergence term, the
 * extrinsic-curvature term, and the coordinate-transformation derivatives.
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
#include <grace/physics/expansion.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

adm_point_t flat_point() {
    adm_point_t f{};
    f.g[ADM_XX] = f.g[ADM_YY] = f.g[ADM_ZZ] = 1.0;
    f.ginv[ADM_XX] = f.ginv[ADM_YY] = f.ginv[ADM_ZZ] = 1.0;
    f.sqrt_det_g = 1.0;
    return f;
}

/// Evaluate the expansion at every collocation point of a surface.
std::vector<expansion_result_t> expansion_on(strahlkoerper_t const& s,
                                             std::vector<adm_point_t> const& geom) {
    auto const shape = s.evaluate();
    auto const& b = s.basis();
    std::vector<expansion_result_t> out(b.n_points());

    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t sp{ shape.h[p], shape.dth[p], shape.dph[p],
                              shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            out[p] = expansion_at(geom[p], sp, b.theta(it), b.phi(ip));
        }
    return out;
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Flat-space sphere has expansion 2/r", "[expansion]") {
    // A sphere of radius r in flat space has mean curvature 2/r, and K = 0, so
    // Theta_(l) = 2/r exactly.  This isolates the divergence term together with
    // the analytic coordinate-transformation derivatives: any error in
    // d_j d_i theta or d_j d_i phi shows up immediately as an angle-dependent
    // deviation.
    for (double r : {0.5, 1.0, 3.0, 10.0}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);
        std::vector<adm_point_t> geom(s.n_points(), flat_point());

        for (auto const& e : expansion_on(s, geom)) {
            INFO("r=" << r);
            REQUIRE_THAT(e.theta_out,      Catch::Matchers::WithinRel(2.0 / r, 1e-11));
            REQUIRE_THAT(e.mean_curvature, Catch::Matchers::WithinRel(2.0 / r, 1e-11));
            // K = 0, so the ingoing expansion is exactly minus the outgoing.
            REQUIRE_THAT(e.theta_in,       Catch::Matchers::WithinRel(-2.0 / r, 1e-11));
            REQUIRE(e.theta_d > 0.0);
        }
    }
}

TEST_CASE("Flat-space sphere is unaffected by a shifted origin", "[expansion]") {
    // Same surface, described from an off-centre origin, is no longer a
    // coordinate sphere: h varies with angle and the dh terms in s_i and
    // d_j s_i all switch on.  The expansion must still be 2/R everywhere.
    double const R = 2.0;
    std::array<double,3> const off{0.4, -0.3, 0.25};

    int const l_max = 20;
    strahlkoerper_t s(l_max, off, 1.0);

    // Radius, from the offset origin, of the sphere of radius R centred at 0:
    // solves |off + h n|^2 = R^2.
    std::vector<double> h(s.n_points());
    for (int it = 0; it < s.basis().n_theta(); ++it) {
        double const st = s.basis().sin_theta(it);
        double const ct = s.basis().cos_theta(it);
        for (int ip = 0; ip < s.basis().n_phi(); ++ip) {
            double const cp = std::cos(s.basis().phi(ip));
            double const sp = std::sin(s.basis().phi(ip));
            double const n[3]{ st * cp, st * sp, ct };
            double const b = off[0]*n[0] + off[1]*n[1] + off[2]*n[2];
            double const c = off[0]*off[0] + off[1]*off[1] + off[2]*off[2] - R*R;
            h[s.basis().point_index(it, ip)] = -b + std::sqrt(b*b - c);
        }
    }
    s.coefficients() = s.project(h);

    std::vector<adm_point_t> geom(s.n_points(), flat_point());
    for (auto const& e : expansion_on(s, geom)) {
        REQUIRE_THAT(e.theta_out, Catch::Matchers::WithinRel(2.0 / R, 1e-8));
    }
}

TEST_CASE("Expansion vanishes on the Kerr-Schild horizon", "[expansion]") {
    // The defining property of an apparent horizon.  In Kerr-Schild slicing the
    // apparent horizon coincides with the event horizon, so the exact surface is
    // known and Theta must vanish on it -- a genuine test of the K_ij n^i n^j
    // and -K terms, which the flat-space checks cannot reach.
    for (double spin : {0.0, 0.4, 0.7}) {
        kerr_schild_geometry_t ks(1.0, spin);
        int const l_max = 16;
        strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, ks.horizon_radius());

        std::vector<double> h(s.n_points());
        for (int it = 0; it < s.basis().n_theta(); ++it)
            for (int ip = 0; ip < s.basis().n_phi(); ++ip)
                h[s.basis().point_index(it, ip)] =
                    ks.horizon_coordinate_radius(s.basis().theta(it), s.basis().phi(ip));
        s.coefficients() = s.project(h);

        auto const geom = ks.at(s.collocation_points());

        double worst = 0.0;
        for (auto const& e : expansion_on(s, geom))
            worst = std::max(worst, std::abs(e.theta_out));

        INFO("spin=" << spin << " max|Theta| on the horizon = " << worst);
        REQUIRE(worst < 1e-6);
    }
}

TEST_CASE("Expansion is positive outside and negative inside the horizon",
          "[expansion]") {
    // Sign structure the Newton solve relies on to bracket the horizon: Theta > 0
    // on a surface outside it (trapped-surface condition not yet met) and Theta < 0
    // strictly inside.
    kerr_schild_geometry_t ks(1.0, 0.0);
    double const rp = ks.horizon_radius();

    for (double factor : {1.5, 2.0, 4.0}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, factor * rp);
        auto const geom = ks.at(s.collocation_points());
        for (auto const& e : expansion_on(s, geom)) {
            INFO("outside, r = " << factor * rp);
            REQUIRE(e.theta_out > 0.0);
        }
    }

    for (double factor : {0.6, 0.8}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, factor * rp);
        auto const geom = ks.at(s.collocation_points());
        for (auto const& e : expansion_on(s, geom)) {
            INFO("inside, r = " << factor * rp);
            REQUIRE(e.theta_out < 0.0);
        }
    }
}
