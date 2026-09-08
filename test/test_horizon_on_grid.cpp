/**
 * @file test_horizon_on_grid.cpp
 * @brief Apparent-horizon solve driven by interpolated grid data.
 *
 * The full stack end to end, against a spacetime whose horizon is known
 * exactly: point search, Hermite gradients, cross-rank gather, ADM
 * reconstruction, expansion, Jacobian and Newton step all have to be right for
 * the solve to land on r+ = 2M.
 *
 * Accuracy here is set by the grid, not the solver -- see the resolution note
 * in configs/kerr_schild_horizon.yaml. The analytic-geometry counterpart in
 * test_horizon_finder.cpp is what pins down the solver itself.
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

#include <grace/geometry/grid_geometry_provider.hh>
#include <grace/geometry/horizon_finder.hh>
#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/analytic_geometry.hh>
#include <grace/physics/expansion.hh>

#include <cmath>
#include <vector>

using namespace grace;

TEST_CASE("Apparent horizon is found in grid data", "[horizon_on_grid]") {
    kerr_schild_geometry_t ks(1.0, 0.0);
    double const rp = ks.horizon_radius();   // 2M

    grid_geometry_provider_t provider(1);
    geometry_provider_t geom = [&provider](auto const& pts, auto& out) {
        return provider(pts, out);
    };

    // Start deliberately off the answer, outside the horizon.
    strahlkoerper_t s(6, {0.0, 0.0, 0.0}, 2.4);

    horizon_config_t cfg;
    cfg.tolerance      = 2e-2;   // grid-resolution floor; see the config comment
    cfg.max_iterations = 40;

    auto const r = find_horizon(s, geom, cfg);

    INFO("status " << to_string(r.status) << " after " << r.iterations
         << " iterations, residual " << r.residual
         << ", mean radius " << r.mean_radius << " (exact " << rp << ")");
    REQUIRE(r.found);

    REQUIRE_THAT(r.mean_radius,      Catch::Matchers::WithinRel(rp, 5e-2));
    REQUIRE_THAT(r.area,             Catch::Matchers::WithinRel(ks.horizon_area(), 1e-1));
    REQUIRE_THAT(r.irreducible_mass, Catch::Matchers::WithinRel(1.0, 5e-2));

    // The horizon of a non-spinning hole is a coordinate sphere, so the
    // recovered surface should be very nearly round.
    REQUIRE((r.max_radius - r.min_radius) / r.mean_radius < 2e-2);

    // Quasi-local spin, computed from interpolated K_ij rather than analytic
    // values.  Schwarzschild has none, so this catches an error in the spin
    // integral that the analytic tests would not reach.
    INFO("|J| = " << r.spin.magnitude
         << ", m_christodoulou = " << r.spin.christodoulou_mass);
    REQUIRE_THAT(r.spin.magnitude, Catch::Matchers::WithinAbs(0.0, 3e-2));
    REQUIRE_THAT(r.spin.christodoulou_mass, Catch::Matchers::WithinRel(1.0, 5e-2));
}

TEST_CASE("A surface that leaves the grid is reported, not guessed at",
          "[horizon_on_grid]") {
    grid_geometry_provider_t provider(1);
    geometry_provider_t geom = [&provider](auto const& pts, auto& out) {
        return provider(pts, out);
    };

    // Far outside the +/-8 domain.
    strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 500.0);

    auto const r = find_horizon(s, geom);
    REQUIRE_FALSE(r.found);
    REQUIRE(r.status == horizon_status_t::geometry_unavailable);
}

TEST_CASE("Grid resolution sets the achievable expansion floor",
          "[horizon_on_grid]") {
    // Put the surface exactly on the analytic horizon and measure |Theta| from
    // interpolated data.  Whatever this returns is the floor the solve can
    // reach: it is the error the grid introduces, with no solver involved.
    // Keeping it as a test documents that floor and catches a regression in
    // interpolation accuracy that would otherwise show up only as a mysterious
    // convergence failure.
    kerr_schild_geometry_t ks(1.0, 0.0);

    grid_geometry_provider_t provider(1);
    strahlkoerper_t s(6, {0.0, 0.0, 0.0}, ks.horizon_radius());

    std::vector<adm_point_t> geom;
    REQUIRE(provider(s.collocation_points(), geom));

    auto const shape = s.evaluate();
    auto const& b = s.basis();

    double worst_grid = 0.0, worst_exact = 0.0;
    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t sp{ shape.h[p], shape.dth[p], shape.dph[p],
                              shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            worst_grid = std::max(worst_grid,
                std::abs(expansion_at(geom[p], sp, b.theta(it), b.phi(ip)).theta_out));
        }

    auto const exact_geom = ks.at(s.collocation_points());
    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t sp{ shape.h[p], shape.dth[p], shape.dph[p],
                              shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            worst_exact = std::max(worst_exact,
                std::abs(expansion_at(exact_geom[p], sp, b.theta(it), b.phi(ip)).theta_out));
        }

    WARN("max|Theta| on the exact horizon: from grid = " << worst_grid
         << ", from analytic geometry = " << worst_exact);

    // Analytic geometry must give essentially zero; the grid must be close
    // enough that a solve can bracket the horizon at all.
    REQUIRE(worst_exact < 1e-8);
    REQUIRE(worst_grid  < 5e-2);
}
