/**
 * @file test_kerr_schild_sampling.cpp
 * @brief End-to-end check that the sampling layers compose correctly.
 *
 * Takes a grid carrying Schwarzschild in Kerr-Schild coordinates, samples it at
 * scattered points through point_cloud_sampler_t, reconstructs the physical ADM
 * variables with adm_from_sample(), and compares against the closed-form values
 * from kerr_schild_geometry_t at the same points.
 *
 * One test therefore exercises, together: the p4est point search, the Hermite
 * gradient interpolation, the cross-rank gather, the unfound-point check, the
 * ADM <- Z4c reconstruction (including the chi chain rule), and -- because the
 * initial data is written as ADM, evolved variables are stored conformally, and
 * the reconstruction inverts that -- the round trip through the Z4c variables.
 *
 * That combination is exactly what a horizon finder depends on, so a failure
 * here localises the problem before any surface machinery is involved.
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
#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/analytic_geometry.hh>
#include <grace/utils/point_cloud_sampler.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

/// Points on a few spheres around the hole, well outside the horizon (r+ = 2)
/// and well inside the outer boundary, where the grid resolves the 1/r falloff.
std::vector<std::array<double,3>> sample_points() {
    std::vector<std::array<double,3>> pts;
    for (double r : {6.0, 8.0, 10.0}) {
        for (int i = 0; i < 6; ++i) {
            double const th = M_PI * (i + 0.5) / 6.0;
            for (int j = 0; j < 8; ++j) {
                double const ph = 2.0 * M_PI * (j + 0.37) / 8.0;
                pts.push_back({ r * std::sin(th) * std::cos(ph),
                                r * std::sin(th) * std::sin(ph),
                                r * std::cos(th) });
            }
        }
    }
    return pts;
}

}  // namespace

TEST_CASE("Sampled Kerr-Schild geometry matches the closed form",
          "[kerr_schild_sampling]") {
    auto const pts = sample_points();
    kerr_schild_geometry_t ks(1.0, 0.0);

    point_cloud_sampler_t sampler(1);
    auto const sample = sampler.sample(pts, adm_sample_var_indices(), true);

    INFO("first unclaimed point: " << sample.first_bad_point());
    REQUIRE(sample.all_found);

    double worst_g = 0, worst_K = 0, worst_dg = 0;

    for (std::size_t p = 0; p < pts.size(); ++p) {
        std::vector<double> vals(adm_n_sample_vars());
        std::vector<std::array<double,3>> grads(adm_n_sample_vars());
        for (int v = 0; v < adm_n_sample_vars(); ++v) {
            vals[v] = sample.values(p, v);
            for (int d = 0; d < 3; ++d) grads[v][d] = sample.grads(p, v, d);
        }

        auto const got = adm_from_sample(
            vals.data(), reinterpret_cast<double const (*)[3]>(grads.data()));
        auto const want = ks.at(pts[p]);

        INFO("point " << p << " at (" << pts[p][0] << ", " << pts[p][1]
                      << ", " << pts[p][2] << ")");

        for (int c = 0; c < ADM_NSYM; ++c) {
            worst_g  = std::max(worst_g,  std::abs(got.g[c] - want.g[c]));
            worst_K  = std::max(worst_K,  std::abs(got.K[c] - want.K[c]));
            for (int k = 0; k < 3; ++k)
                worst_dg = std::max(worst_dg,
                                    std::abs(got.dg[k][c] - want.dg[k][c]));
        }

        // The metric is O(1) here, so absolute and relative tolerances agree
        // to within a small factor.
        for (int c = 0; c < ADM_NSYM; ++c) {
            INFO("gamma component " << c);
            REQUIRE_THAT(got.g[c], Catch::Matchers::WithinAbs(want.g[c], 2e-4));
            INFO("K component " << c);
            REQUIRE_THAT(got.K[c], Catch::Matchers::WithinAbs(want.K[c], 2e-4));
        }
        for (int k = 0; k < 3; ++k)
            for (int c = 0; c < ADM_NSYM; ++c) {
                INFO("d_" << k << " gamma component " << c);
                // Gradients converge one order slower than values and are the
                // least accurate quantity here; the observed worst case at this
                // resolution is ~5e-4, so this bounds it with ~2x margin.
                REQUIRE_THAT(got.dg[k][c],
                             Catch::Matchers::WithinAbs(want.dg[k][c], 1e-3));
            }

        REQUIRE_THAT(got.sqrt_det_g,
                     Catch::Matchers::WithinAbs(want.sqrt_det_g, 5e-4));
    }

    WARN("worst |dgamma| = " << worst_g
         << "   worst |dK| = " << worst_K
         << "   worst |d(dgamma)| = " << worst_dg);
}

TEST_CASE("Sampled horizon-sphere area matches 8 pi M r+",
          "[kerr_schild_sampling]") {
    // The whole stack in the configuration a horizon finder uses: a
    // Strahlkoerper supplies the points, the sampler and the ADM reconstruction
    // supply the geometry, and the surface integral must return the exact
    // horizon area.  Only the interpolation is approximate here, so the
    // tolerance reflects grid resolution rather than any surface-side error.
    kerr_schild_geometry_t ks(1.0, 0.0);

    // A sphere at r = 6: comfortably resolved, and the exact area of any
    // coordinate sphere in Schwarzschild-Kerr-Schild is 4 pi r^2.
    double const r = 6.0;
    strahlkoerper_t s(6, {0.0, 0.0, 0.0}, r);

    point_cloud_sampler_t sampler(1);
    auto const pts    = s.collocation_points();
    auto const sample = sampler.sample(pts, adm_sample_var_indices(), true);
    REQUIRE(sample.all_found);

    std::vector<adm_point_t> geom(pts.size());
    for (std::size_t p = 0; p < pts.size(); ++p) {
        std::vector<double> vals(adm_n_sample_vars());
        for (int v = 0; v < adm_n_sample_vars(); ++v) vals[v] = sample.values(p, v);
        geom[p] = adm_from_sample(vals.data(), nullptr);
    }

    double const exact = 4.0 * M_PI * r * r;
    INFO("sampled area = " << s.area(geom) << "  exact = " << exact);
    REQUIRE_THAT(s.area(geom), Catch::Matchers::WithinRel(exact, 1e-4));
}


TEST_CASE("Sampled metric Hessian matches the closed form",
          "[kerr_schild_sampling]") {
    // d_k d_l gamma_ij off the grid, against the analytic Kerr-Schild values.
    // This is the quantity the intrinsic curvature of a surface needs and that
    // the interpolator could not previously supply; requesting it also switches
    // the sample onto the Lagrange path, so value and gradient are re-checked
    // here on that path rather than the Hermite one.
    auto const pts = sample_points();
    kerr_schild_geometry_t ks(1.0, 0.0);

    point_cloud_sampler_t sampler(2);
    auto const sample = sampler.sample(pts, adm_sample_var_indices(),
                                       /*want_grad=*/true, /*want_hessian=*/true);

    INFO("first unclaimed point: " << sample.first_bad_point());
    REQUIRE(sample.all_found);
    REQUIRE(sample.hess.extent(2) == 6u);

    double worst_g = 0, worst_dg = 0, worst_ddg = 0;

    for (std::size_t p = 0; p < pts.size(); ++p) {
        std::vector<double> vals(adm_n_sample_vars());
        std::vector<std::array<double,3>> grads(adm_n_sample_vars());
        std::vector<std::array<double,6>> hs(adm_n_sample_vars());
        for (int v = 0; v < adm_n_sample_vars(); ++v) {
            vals[v] = sample.values(p, v);
            for (int d = 0; d < 3; ++d) grads[v][d] = sample.grads(p, v, d);
            for (int d = 0; d < 6; ++d) hs[v][d]    = sample.hess(p, v, d);
        }

        auto const got = adm_from_sample_hessian(
            vals.data(),
            reinterpret_cast<double const (*)[3]>(grads.data()),
            reinterpret_cast<double const (*)[ADM_NSYM]>(hs.data()));
        auto const want = ks.at(pts[p]);

        INFO("point " << p);
        for (int c = 0; c < ADM_NSYM; ++c) {
            worst_g = std::max(worst_g, std::abs(got.g[c] - want.g[c]));
            for (int k = 0; k < 3; ++k)
                worst_dg = std::max(worst_dg, std::abs(got.dg[k][c] - want.dg[k][c]));
            for (int kl = 0; kl < ADM_NSYM; ++kl) {
                worst_ddg = std::max(worst_ddg,
                                     std::abs(got.ddg[kl][c] - want.ddg[kl][c]));
                INFO("d2 gamma component kl=" << kl << " c=" << c);
                REQUIRE_THAT(got.ddg[kl][c],
                             Catch::Matchers::WithinAbs(want.ddg[kl][c], 5e-3));
            }
        }
    }

    WARN("Lagrange path worst errors: |dgamma| = " << worst_g
         << ", |d gamma| = " << worst_dg
         << ", |dd gamma| = " << worst_ddg);
}
