/**
 * @file test_horizon_finder.cpp
 * @brief Tests for the apparent-horizon Newton solve.
 *
 * Run against analytic Kerr-Schild geometry, where the answer is known in
 * closed form: the solve must converge to r+ = M + sqrt(M^2 - a^2) from a
 * deliberately wrong initial guess, and the resulting proper area must be
 * 8 pi M r+. Using an analytic spacetime keeps interpolation error out of the
 * picture, so a failure here is a failure of the solver.
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

#include <grace/geometry/horizon_finder.hh>
#include <grace/physics/analytic_geometry.hh>

#include <cmath>
#include <memory>
#include <vector>

using namespace grace;

namespace {

/// Geometry provider backed by an analytic spacetime.
geometry_provider_t analytic_provider(kerr_schild_geometry_t const& ks) {
    return [ks](std::vector<std::array<double,3>> const& pts,
                std::vector<adm_point_t>& out) {
        out = ks.at(pts);
        return true;
    };
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Newton solve finds the Schwarzschild horizon", "[horizon_finder]") {
    kerr_schild_geometry_t ks(1.0, 0.0);
    double const rp = ks.horizon_radius();  // = 2M

    // Start well away from the answer, on both sides.
    for (double guess : {1.2, 1.6, 2.6, 3.5}) {
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, guess);

        horizon_config_t cfg;
        // 1e-11 would sit below the floor set by the analytic provider's own
        // finite-differenced metric gradients: the solve reaches ~1e-11 and then
        // bounces on noise, never tripping the convergence test.
        cfg.tolerance = 1e-9;
        cfg.max_iterations = 60;

        auto const r = find_horizon(s, analytic_provider(ks), cfg);

        INFO("initial guess r=" << guess << " -> status " << to_string(r.status)
             << " after " << r.iterations << " iterations, residual " << r.residual);
        REQUIRE(r.found);
        REQUIRE(r.status == horizon_status_t::converged);

        // The horizon is a coordinate sphere at r = 2M for a = 0.
        REQUIRE_THAT(r.mean_radius, Catch::Matchers::WithinRel(rp, 1e-8));
        REQUIRE_THAT(r.min_radius,  Catch::Matchers::WithinRel(rp, 1e-8));
        REQUIRE_THAT(r.max_radius,  Catch::Matchers::WithinRel(rp, 1e-8));

        REQUIRE_THAT(r.area,             Catch::Matchers::WithinRel(ks.horizon_area(), 1e-8));
        REQUIRE_THAT(r.irreducible_mass, Catch::Matchers::WithinRel(1.0, 1e-8));
        REQUIRE_THAT(r.areal_radius,     Catch::Matchers::WithinRel(rp, 1e-8));
    }
}

TEST_CASE("Newton solve finds the spinning Kerr horizon", "[horizon_finder]") {
    // For a != 0 the horizon is an ellipsoid in Cartesian Kerr-Schild
    // coordinates, so the solve has to recover a genuinely non-spherical shape
    // rather than just a radius.
    for (double spin : {0.3, 0.6, 0.8}) {
        kerr_schild_geometry_t ks(1.0, spin);

        strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 1.6 * ks.horizon_radius());

        horizon_config_t cfg;
        cfg.tolerance = 1e-9;
        cfg.max_iterations = 80;

        auto const r = find_horizon(s, analytic_provider(ks), cfg);

        INFO("spin=" << spin << " status " << to_string(r.status)
             << " iters " << r.iterations << " residual " << r.residual);
        REQUIRE(r.found);

        // Area is coordinate-independent; the shape is not, so this is the
        // meaningful check.
        REQUIRE_THAT(r.area, Catch::Matchers::WithinRel(ks.horizon_area(), 1e-6));
        REQUIRE_THAT(r.irreducible_mass,
                     Catch::Matchers::WithinRel(ks.irreducible_mass(), 1e-6));

        // The recovered surface must match the analytic horizon pointwise.
        for (int it = 0; it < s.basis().n_theta(); ++it) {
            double const th = s.basis().theta(it);
            double const want = ks.horizon_coordinate_radius(th, 0.0);
            INFO("theta=" << th);
            REQUIRE_THAT(s.radius_at(th, 0.0), Catch::Matchers::WithinRel(want, 1e-5));
        }

        // Oblate: the horizon is widest in the equatorial plane.
        REQUIRE(r.max_radius > r.min_radius);
    }
}

TEST_CASE("Newton solve converges from an off-centre origin", "[horizon_finder]") {
    // The Strahlkoerper origin need not sit at the centre of the hole; the
    // surface just has to stay single-valued as seen from it.
    kerr_schild_geometry_t ks(1.0, 0.0);
    double const rp = ks.horizon_radius();

    strahlkoerper_t s(12, {0.35, -0.2, 0.15}, rp);

    horizon_config_t cfg;
    cfg.tolerance = 1e-9;
    cfg.max_iterations = 80;

    auto const r = find_horizon(s, analytic_provider(ks), cfg);

    INFO("status " << to_string(r.status) << " iters " << r.iterations
         << " residual " << r.residual);
    REQUIRE(r.found);
    REQUIRE_THAT(r.area, Catch::Matchers::WithinRel(ks.horizon_area(), 1e-6));

    // The area-weighted centroid must come back to the hole, not the origin
    // the search started from.
    for (int d = 0; d < 3; ++d) {
        INFO("centroid component " << d);
        REQUIRE_THAT(r.centroid[d], Catch::Matchers::WithinAbs(0.0, 1e-3));
    }
}

TEST_CASE("Converged surface has vanishing expansion", "[horizon_finder]") {
    kerr_schild_geometry_t ks(1.0, 0.5);
    strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 1.3 * ks.horizon_radius());

    horizon_config_t cfg;
    cfg.tolerance = 1e-9;
    cfg.max_iterations = 80;

    auto const r = find_horizon(s, analytic_provider(ks), cfg);
    REQUIRE(r.found);
    REQUIRE(r.residual < cfg.tolerance);
}

TEST_CASE("Failures are reported rather than returned as answers",
          "[horizon_finder]") {
    SECTION("no horizon to find") {
        // Flat space contains no trapped surface: Theta = 2/r > 0 everywhere,
        // so the solve must fail rather than converge on something.
        geometry_provider_t flat =
            [](std::vector<std::array<double,3>> const& pts,
               std::vector<adm_point_t>& out) {
                adm_point_t f{};
                f.g[ADM_XX] = f.g[ADM_YY] = f.g[ADM_ZZ] = 1.0;
                f.ginv[ADM_XX] = f.ginv[ADM_YY] = f.ginv[ADM_ZZ] = 1.0;
                f.sqrt_det_g = 1.0;
                out.assign(pts.size(), f);
                return true;
            };

        strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 1.0);
        horizon_config_t cfg;
        cfg.max_iterations = 30;

        auto const r = find_horizon(s, flat, cfg);
        INFO("status " << to_string(r.status));
        REQUIRE_FALSE(r.found);
        REQUIRE(r.area == 0.0);
    }

    SECTION("geometry unavailable") {
        geometry_provider_t refuses =
            [](std::vector<std::array<double,3>> const&,
               std::vector<adm_point_t>&) { return false; };

        strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 1.0);
        auto const r = find_horizon(s, refuses);

        REQUIRE_FALSE(r.found);
        REQUIRE(r.status == horizon_status_t::geometry_unavailable);
    }

    SECTION("iteration budget exhausted is distinguishable from divergence") {
        kerr_schild_geometry_t ks(1.0, 0.0);
        strahlkoerper_t s(6, {0.0, 0.0, 0.0}, 8.0);

        horizon_config_t cfg;
        cfg.tolerance = 1e-12;
        cfg.max_iterations = 1;   // far too few

        auto const r = find_horizon(s, analytic_provider(ks), cfg);
        REQUIRE_FALSE(r.found);
        REQUIRE(r.status == horizon_status_t::max_iterations);
    }
}

TEST_CASE("A converged horizon is a cheap starting point for the next solve",
          "[horizon_finder]") {
    // How the finder is actually used in an evolution: the previous shape
    // warm-starts the next search, which should then converge in very few
    // iterations.
    kerr_schild_geometry_t ks(1.0, 0.6);
    strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 1.5 * ks.horizon_radius());

    horizon_config_t cfg;
    cfg.tolerance = 1e-9;
    cfg.max_iterations = 80;

    auto const first = find_horizon(s, analytic_provider(ks), cfg);
    REQUIRE(first.found);

    auto const second = find_horizon(s, analytic_provider(ks), cfg);
    REQUIRE(second.found);

    INFO("cold start " << first.iterations << " iterations, warm start "
         << second.iterations);
    REQUIRE(second.iterations < first.iterations);
    REQUIRE(second.iterations <= 2);
}

//---------------------------------------------------------------------------
// Termination guards.
//
// These four exits -- runaway, stalled-and-accepted, stalled-and-rejected, and
// diverged -- are the difference between a finder that admits it has no answer
// and one that returns a meaningless surface. Each has been wrong at least once
// and none was covered.
//---------------------------------------------------------------------------

TEST_CASE("Runaway growth is caught before the surface leaves the domain",
          "[horizon_finder]") {
    // Theta -> 2/r in flat space, so a search started outside a horizon that
    // does not exist can always reduce its residual by drifting outwards, and
    // will eventually satisfy any fixed tolerance. Without a growth guard that
    // is reported as a find.
    geometry_provider_t flat =
        [](std::vector<std::array<double,3>> const& pts,
           std::vector<adm_point_t>& out) {
            adm_point_t f{};
            f.g[ADM_XX] = f.g[ADM_YY] = f.g[ADM_ZZ] = 1.0;
            f.ginv[ADM_XX] = f.ginv[ADM_YY] = f.ginv[ADM_ZZ] = 1.0;
            f.sqrt_det_g = 1.0;
            out.assign(pts.size(), f);
            return true;
        };

    strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 1.0);
    horizon_config_t cfg;
    cfg.tolerance         = 1e-3;   // reachable simply by growing
    cfg.max_iterations    = 200;
    cfg.max_radius_growth = 3.0;

    auto const r = find_horizon(s, flat, cfg);
    INFO("status " << to_string(r.status) << ", mean radius " << r.mean_radius);
    REQUIRE_FALSE(r.found);
    REQUIRE(r.status == horizon_status_t::runaway);

    // And the guard must not fire on a legitimate search that has to travel:
    // the Schwarzschild solve below starts at 1.2M and ends at 2M.
    kerr_schild_geometry_t ks(1.0, 0.0);
    strahlkoerper_t good(6, {0.0, 0.0, 0.0}, 1.2);
    horizon_config_t gcfg;
    gcfg.tolerance = 1e-9;
    auto const g = find_horizon(good, analytic_provider(ks), gcfg);
    REQUIRE(g.found);
    REQUIRE(g.status != horizon_status_t::runaway);
}

TEST_CASE("A stall is accepted only close to a horizon", "[horizon_finder]") {
    // A solve whose tolerance sits below what its l_max can represent stops
    // improving and must terminate as `stalled` rather than spend its whole
    // budget. Whether that stall counts as a find is a separate question, and
    // the rule is scale-free: Theta has dimensions of 1/length and a generic
    // surface has Theta ~ 2/R, so what matters is residual * R.
    kerr_schild_geometry_t ks(1.0, 0.9);

    auto solve = [&](double stall_accept) {
        strahlkoerper_t s(2, {0.0, 0.0, 0.0}, 1.3 * ks.horizon_radius());
        horizon_config_t cfg;
        cfg.tolerance      = 1e-14;   // unreachable at l_max = 2
        cfg.max_iterations = 60;
        cfg.stall_accept   = stall_accept;
        return find_horizon(s, analytic_provider(ks), cfg);
    };

    // Accept anything, to establish that the solve does stall and to measure
    // where. A stall is what the resolution floor looks like from inside.
    auto const loose = solve(1.0);
    INFO("status " << to_string(loose.status) << ", residual " << loose.residual);
    REQUIRE(loose.status == horizon_status_t::stalled);
    REQUIRE(loose.found);
    REQUIRE(loose.iterations < 60);
    REQUIRE(loose.mean_radius > 0.0);

    // The identical solve, with the acceptance threshold moved below what this
    // surface achieves, must refuse the very same surface -- and report no
    // diagnostics, rather than numbers that look like measurements.
    double const achieved = loose.residual * loose.mean_radius;
    auto const tight = solve(0.5 * achieved);
    REQUIRE_FALSE(tight.found);
    REQUIRE(tight.area == 0.0);
}

TEST_CASE("A growing residual is diagnosed as divergence, not as a stall",
          "[horizon_finder]") {
    // The two streak tests overlap by construction: any residual that grows
    // also fails the stall threshold. Testing them independently made the
    // smaller stall count fire first every time, so `diverged` was unreachable
    // and -- worse -- a *growing* residual was handed to the stall-acceptance
    // branch, which could declare it found.
    //
    // Provoked with a horizon that runs away faster than the step cap lets the
    // surface follow: the mass halves every iteration while `max_relative_step`
    // holds the surface to 10% per step, so the residual can only grow.
    auto calls = std::make_shared<int>(0);
    geometry_provider_t shrinking =
        [calls](std::vector<std::array<double,3>> const& pts,
                std::vector<adm_point_t>& out) {
            // Three samples per Newton iteration (the surface and the two
            // displaced ones); all three must see the same spacetime or the
            // Jacobian is measuring the wrong thing.
            int const iter = (*calls)++ / 3;
            kerr_schild_geometry_t ks(std::pow(0.5, iter), 0.0);
            out = ks.at(pts);
            return true;
        };

    strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 2.0);   // on the initial horizon

    horizon_config_t cfg;
    cfg.tolerance             = 1e-12;
    cfg.max_iterations        = 60;
    cfg.max_growth_iterations = 3;

    auto const r = find_horizon(s, shrinking, cfg);
    INFO("status " << to_string(r.status) << ", residual " << r.residual);
    REQUIRE_FALSE(r.found);
    REQUIRE(r.status == horizon_status_t::diverged);
}

//---------------------------------------------------------------------------
// Constant-expansion surfaces, and why the search needs a scaled expansion.
//
// Thornburg, Living Rev. Relativity 10 (2007) 3, figure 3.4: in an
// Eddington-Finkelstein slice of Schwarzschild -- which is Kerr-Schild at zero
// spin -- Theta vanishes at the horizon *and again at infinity*, with a maximum
// in between. R*Theta and R^2*Theta instead increase monotonically. That is the
// whole justification for scaling the expansion during pretracking, so it is
// checked against our own expansion code rather than taken on trust.
//---------------------------------------------------------------------------

namespace {

/// max |Theta| ... on a coordinate sphere of radius r, and its mean.
double mean_expansion_on_sphere(kerr_schild_geometry_t const& ks, double r) {
    strahlkoerper_t s(4, {0.0, 0.0, 0.0}, r);
    auto const geom  = ks.at(s.collocation_points());
    auto const shape = s.evaluate();
    auto const& b    = s.basis();

    double acc = 0.0;
    int    n   = 0;
    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            acc += expansion_at(geom[p], sp, b.theta(it), b.phi(ip)).theta_out;
            ++n;
        }
    return acc / n;
}

}  // namespace

TEST_CASE("Theta turns over but R^2 Theta does not", "[horizon_finder]") {
    kerr_schild_geometry_t ks(1.0, 0.0);   // m = 1, horizon at r = 2

    // Zero at the horizon.
    REQUIRE_THAT(mean_expansion_on_sphere(ks, 2.0),
                 Catch::Matchers::WithinAbs(0.0, 1e-8));

    std::vector<double> const radii{ 2.5, 3.0, 3.5, 4.0, 4.4, 5.0, 6.0, 8.0,
                                     12.0, 20.0, 40.0, 80.0 };
    std::vector<double> theta, r2theta;
    for (double r : radii) {
        double const th = mean_expansion_on_sphere(ks, r);
        theta.push_back(th);
        r2theta.push_back(r * r * th);
    }

    // Theta is positive outside and has an interior maximum, so it is *not*
    // monotonic -- which is exactly what makes a search on Theta ambiguous.
    for (double v : theta) REQUIRE(v > 0.0);
    std::size_t imax = 0;
    for (std::size_t i = 1; i < theta.size(); ++i)
        if (theta[i] > theta[imax]) imax = i;
    INFO("max of Theta at r = " << radii[imax]);
    REQUIRE(imax > 0);                       // not at the innermost sample
    REQUIRE(imax + 1 < theta.size());        // nor at the outermost
    REQUIRE_THAT(radii[imax], Catch::Matchers::WithinAbs(4.4, 1.5));
    REQUIRE(theta.back() < theta[imax]);     // decays again far out

    // R^2 Theta increases monotonically over the same range, so it has the one
    // zero the pretracking search needs.
    for (std::size_t i = 1; i < r2theta.size(); ++i) {
        INFO("r = " << radii[i] << ": r^2 Theta = " << r2theta[i]
             << " vs " << r2theta[i-1] << " at r = " << radii[i-1]);
        REQUIRE(r2theta[i] > r2theta[i-1]);
    }
}

TEST_CASE("A constant-expansion search follows the outer surface without scaling",
          "[horizon_finder]") {
    // The failure figure 3.4 predicts, demonstrated: for a small positive
    // target there are two surfaces, and a search started far out on the plain
    // expansion converges to the one that never becomes the horizon.
    kerr_schild_geometry_t ks(1.0, 0.0);
    double const E = 0.05;

    horizon_config_t cfg;
    cfg.tolerance        = 1e-9;
    cfg.max_iterations   = 80;
    cfg.target_expansion = E;

    strahlkoerper_t outer(4, {0.0, 0.0, 0.0}, 30.0);
    auto const r_outer = find_horizon(outer, analytic_provider(ks), cfg);
    REQUIRE(r_outer.found);

    strahlkoerper_t inner(4, {0.0, 0.0, 0.0}, 2.4);
    auto const r_inner = find_horizon(inner, analytic_provider(ks), cfg);
    REQUIRE(r_inner.found);

    INFO("inner CE surface at " << r_inner.mean_radius
         << ", outer at " << r_outer.mean_radius);
    // Two genuinely different surfaces at the same expansion -- the ambiguity.
    REQUIRE(r_outer.mean_radius > 3.0 * r_inner.mean_radius);
    // And it is the inner one that sits just outside the horizon.
    REQUIRE(r_inner.mean_radius > 2.0);
    REQUIRE(r_inner.mean_radius < 4.0);
}
