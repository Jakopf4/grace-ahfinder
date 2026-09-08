/**
 * @file test_horizon_spin.cpp
 * @brief Tests for the quasi-local (isolated-horizon) spin.
 *
 * Kerr is the case where the answer is known exactly and the coordinate
 * rotational generator genuinely is the horizon's Killing vector, so the
 * integral must reproduce J = a M, the Christodoulou mass must come back as the
 * ADM mass, and the dimensionless spin as a/M.
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
#include <grace/geometry/horizon_spin.hh>
#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/analytic_geometry.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

/// Set a surface to the exact Kerr-Schild horizon.
strahlkoerper_t exact_horizon(kerr_schild_geometry_t const& ks, int l_max = 16) {
    strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, ks.horizon_radius());
    std::vector<double> h(s.n_points());
    for (int it = 0; it < s.basis().n_theta(); ++it)
        for (int ip = 0; ip < s.basis().n_phi(); ++ip)
            h[s.basis().point_index(it, ip)] =
                ks.horizon_coordinate_radius(s.basis().theta(it), s.basis().phi(ip));
    s.coefficients() = s.project(h);
    return s;
}

geometry_provider_t analytic_provider(kerr_schild_geometry_t const& ks) {
    return [ks](std::vector<std::array<double,3>> const& pts,
                std::vector<adm_point_t>& out) {
        out = ks.at(pts);
        return true;
    };
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Schwarzschild horizon has no spin", "[horizon_spin]") {
    kerr_schild_geometry_t ks(1.0, 0.0);
    auto const s    = exact_horizon(ks, 8);
    auto const geom = ks.at(s.collocation_points());

    auto const spin = horizon_spin(s, geom, s.irreducible_mass(geom));

    for (int a = 0; a < 3; ++a) {
        INFO("component " << a);
        REQUIRE_THAT(spin.J[a], Catch::Matchers::WithinAbs(0.0, 1e-8));
    }
    REQUIRE_THAT(spin.magnitude, Catch::Matchers::WithinAbs(0.0, 1e-8));

    // With J = 0 the Christodoulou mass reduces to the irreducible mass, which
    // for Schwarzschild is M.
    REQUIRE_THAT(spin.christodoulou_mass, Catch::Matchers::WithinRel(1.0, 1e-8));
    REQUIRE_THAT(spin.dimensionless, Catch::Matchers::WithinAbs(0.0, 1e-8));
}

TEST_CASE("Kerr horizon spin reproduces J = a M", "[horizon_spin]") {
    for (double a : {0.2, 0.4, 0.6, 0.8}) {
        kerr_schild_geometry_t ks(1.0, a);
        auto const s    = exact_horizon(ks);
        auto const geom = ks.at(s.collocation_points());

        auto const spin = horizon_spin(s, geom, s.irreducible_mass(geom));

        INFO("a=" << a << "  J = (" << spin.J[0] << ", " << spin.J[1] << ", "
             << spin.J[2] << ")");

        // Spin is along z in these coordinates.
        REQUIRE_THAT(spin.J[0], Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(spin.J[1], Catch::Matchers::WithinAbs(0.0, 1e-6));
        REQUIRE_THAT(spin.J[2], Catch::Matchers::WithinRel(a, 1e-5));

        REQUIRE_THAT(spin.magnitude, Catch::Matchers::WithinRel(a, 1e-5));
        REQUIRE_THAT(spin.axis[2],   Catch::Matchers::WithinAbs(1.0, 1e-5));

        // Christodoulou mass must recover the ADM mass, M = 1.  This is a
        // genuine consistency check across the two independent integrals:
        // M^2 = M_irr^2 + J^2/(4 M_irr^2) ties the area to the spin.
        REQUIRE_THAT(spin.christodoulou_mass, Catch::Matchers::WithinRel(1.0, 1e-5));
        REQUIRE_THAT(spin.dimensionless,      Catch::Matchers::WithinRel(a, 1e-5));
    }
}

TEST_CASE("Mass scaling of the spin is correct", "[horizon_spin]") {
    // J has dimensions of M^2, so a hole of mass M with dimensionless spin chi
    // must return J = chi M^2 while chi itself is unchanged.
    double const M = 2.5, chi = 0.5;
    kerr_schild_geometry_t ks(M, chi);

    auto const s    = exact_horizon(ks);
    auto const geom = ks.at(s.collocation_points());
    auto const spin = horizon_spin(s, geom, s.irreducible_mass(geom));

    INFO("J_z = " << spin.J[2] << " expected " << chi * M * M);
    REQUIRE_THAT(spin.J[2], Catch::Matchers::WithinRel(chi * M * M, 1e-5));
    REQUIRE_THAT(spin.christodoulou_mass, Catch::Matchers::WithinRel(M, 1e-5));
    REQUIRE_THAT(spin.dimensionless,      Catch::Matchers::WithinRel(chi, 1e-5));
}

TEST_CASE("Spin is insensitive to the parameterisation origin", "[horizon_spin]") {
    // The generators are taken about the centroid, not about the surface's
    // parameterisation origin, so describing the same horizon from an offset
    // origin must give the same J.  This is what makes the result meaningful
    // for a hole whose Strahlkoerper origin has drifted.
    kerr_schild_geometry_t ks(1.0, 0.6);

    auto const centred = exact_horizon(ks);
    auto const geom_c  = ks.at(centred.collocation_points());
    auto const spin_c  = horizon_spin(centred, geom_c, centred.irreducible_mass(geom_c));

    // Re-find the same horizon from a displaced origin.
    strahlkoerper_t offset(16, {0.3, -0.2, 0.15}, ks.horizon_radius());
    horizon_config_t cfg;
    cfg.tolerance = 1e-9;
    cfg.max_iterations = 80;
    auto const r = find_horizon(offset, analytic_provider(ks), cfg);
    REQUIRE(r.found);

    auto const geom_o = ks.at(offset.collocation_points());
    auto const spin_o = horizon_spin(offset, geom_o, offset.irreducible_mass(geom_o));

    INFO("centred J_z = " << spin_c.J[2] << ", offset J_z = " << spin_o.J[2]);
    REQUIRE_THAT(spin_o.magnitude, Catch::Matchers::WithinRel(spin_c.magnitude, 1e-4));
    REQUIRE_THAT(spin_o.J[2],      Catch::Matchers::WithinRel(spin_c.J[2], 1e-4));
}

TEST_CASE("Spin is reported through the solver result", "[horizon_spin]") {
    kerr_schild_geometry_t ks(1.0, 0.7);
    strahlkoerper_t s(16, {0.0, 0.0, 0.0}, 1.4 * ks.horizon_radius());

    horizon_config_t cfg;
    cfg.tolerance = 1e-9;
    cfg.max_iterations = 80;

    auto const r = find_horizon(s, analytic_provider(ks), cfg);
    REQUIRE(r.found);

    INFO("spin from solve: |J| = " << r.spin.magnitude
         << ", chi = " << r.spin.dimensionless);
    REQUIRE_THAT(r.spin.magnitude,          Catch::Matchers::WithinRel(0.7, 1e-4));
    REQUIRE_THAT(r.spin.dimensionless,      Catch::Matchers::WithinRel(0.7, 1e-4));
    REQUIRE_THAT(r.spin.christodoulou_mass, Catch::Matchers::WithinRel(1.0, 1e-4));
    REQUIRE_THAT(r.spin.axis[2],            Catch::Matchers::WithinAbs(1.0, 1e-4));
}

//---------------------------------------------------------------------------
// Integration: find_horizon must populate every quasi-local field
//---------------------------------------------------------------------------

TEST_CASE("find_horizon populates the full quasi-local result",
          "[horizon_spin][integration]") {
    // Regression guard.  Every quantity below has its own unit test that calls
    // the computing function directly, and those all passed while the block
    // that calls them from find_horizon had been dropped -- leaving twelve of
    // the manager's output columns silently zero.  Unit coverage of the parts
    // does not cover the wiring.
    kerr_schild_geometry_t ks(1.0, 0.6);
    strahlkoerper_t s(16, {0.0, 0.0, 0.0}, 1.3 * ks.horizon_radius());

    horizon_config_t cfg;
    cfg.tolerance = 1e-9;
    cfg.max_iterations = 80;

    auto const r = find_horizon(s, analytic_provider(ks), cfg);
    REQUIRE(r.found);

    // Coordinate-generator spin.
    REQUIRE_THAT(r.spin.magnitude, Catch::Matchers::WithinRel(0.6, 1e-3));

    // Approximate-Killing-vector spin.
    INFO("AKV: valid=" << r.spin_akv.valid << " J=" << r.spin_akv.J
         << " defect=" << r.spin_akv.killing_defect);
    REQUIRE(r.spin_akv.valid);
    REQUIRE_THAT(r.spin_akv.J, Catch::Matchers::WithinAbs(0.6, 5e-3));
    REQUIRE_THAT(r.spin_akv.christodoulou_mass,
                 Catch::Matchers::WithinRel(1.0, 5e-3));
    REQUIRE(r.spin_akv.killing_defect < 1e-2);

    // Multipoles: M_0 is the mass, J_1 the spin.
    INFO("multipoles: valid=" << r.multipoles.valid
         << " M0=" << r.multipoles.mass[0] << " J1=" << r.multipoles.spin[1]);
    REQUIRE(r.multipoles.valid);
    REQUIRE_THAT(r.multipoles.mass[0],
                 Catch::Matchers::WithinRel(r.spin_akv.christodoulou_mass, 1e-4));
    REQUIRE_THAT(r.multipoles.spin[1],
                 Catch::Matchers::WithinAbs(r.spin_akv.J, 1e-6));
    REQUIRE(r.multipoles.mass[2] < 0.0);        // oblate

    // Isolated-horizon validity: expansion and shear both vanish.
    INFO("max_shear = " << r.max_shear);
    REQUIRE(r.max_shear < 1e-5);

    // Circumferences, and the oblateness they encode.
    INFO("C_eq = " << r.circumferences.equatorial
         << " ratio = " << r.circumferences.ratio_xz_eq);
    REQUIRE(r.circumferences.equatorial > 0.0);
    REQUIRE(r.circumferences.polar_xz > 0.0);
    REQUIRE(r.circumferences.ratio_xz_eq < 1.0);
}

TEST_CASE("Quasi-local fields stay at their defaults without a Hessian",
          "[horizon_spin][integration]") {
    // A geometry provider that supplies no second derivatives must leave the
    // curvature-dependent results unset rather than reporting zeros that look
    // like measurements.
    kerr_schild_geometry_t ks(1.0, 0.0);
    geometry_provider_t no_hessian =
        [ks](std::vector<std::array<double,3>> const& pts,
             std::vector<adm_point_t>& out) {
            out = ks.at(pts);
            for (auto& g : out) {
                g.has_hessian = false;
                for (int kl = 0; kl < ADM_NSYM; ++kl)
                    for (int c = 0; c < ADM_NSYM; ++c) g.ddg[kl][c] = 0.0;
            }
            return true;
        };

    strahlkoerper_t s(8, {0.0, 0.0, 0.0}, 2.2);
    horizon_config_t cfg;
    cfg.tolerance = 1e-9;

    auto const r = find_horizon(s, no_hessian, cfg);
    REQUIRE(r.found);

    // The parts that do not need curvature still work.
    REQUIRE_THAT(r.irreducible_mass, Catch::Matchers::WithinRel(1.0, 1e-6));

    // The parts that do are explicitly unavailable.
    REQUIRE_FALSE(r.spin_akv.valid);
    REQUIRE_FALSE(r.multipoles.valid);
    REQUIRE(std::isnan(r.circumferences.equatorial));
}
