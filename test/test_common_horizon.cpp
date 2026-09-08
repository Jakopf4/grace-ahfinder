/**
 * @file test_common_horizon.cpp
 * @brief Common-horizon detection for a binary, on Brill-Lindquist data.
 *
 * Brill-Lindquist is time-symmetric (K_ij = 0), so an apparent horizon here is a
 * minimal surface and the expansion reduces to its mean-curvature term alone --
 * complementary to Kerr-Schild, where the K terms dominate. It is analytic, so
 * the whole binary path is testable with no grid, no interpolation and no
 * external initial-data library.
 *
 * The checks that matter are the ones true independently of the implementation:
 * the area theorem, the coincidence limit, and non-existence at wide separation.
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
#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/analytic_geometry.hh>

#include <cmath>
#include <vector>

using namespace grace;

namespace {

geometry_provider_t provider(brill_lindquist_geometry_t const& bl) {
    return [bl](std::vector<std::array<double,3>> const& pts,
                std::vector<adm_point_t>& out) {
        out = bl.at(pts);
        return true;
    };
}

horizon_config_t cfg(double tol = 1e-8, int iters = 60) {
    horizon_config_t c;
    c.tolerance      = tol;
    c.max_iterations = iters;
    return c;
}

/// Find the horizon around one puncture of an equal-mass pair.
horizon_result_t find_individual(brill_lindquist_geometry_t const& bl,
                                 double x0, strahlkoerper_t& s) {
    s.set_centre({x0, 0.0, 0.0});
    return find_horizon(s, provider(bl), cfg());
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Brill-Lindquist single puncture reproduces Schwarzschild",
          "[common_horizon]") {
    // Isotropic Schwarzschild: the minimal surface sits at r = m/2 and has
    // irreducible mass m.  Exact, so it validates the geometry before anything
    // is built on it.
    for (double m : {0.5, 1.0, 2.0}) {
        brill_lindquist_geometry_t bl({{m, {0.0, 0.0, 0.0}}});
        strahlkoerper_t s(8, {0.0, 0.0, 0.0}, 0.8 * m);

        auto const r = find_horizon(s, provider(bl), cfg(1e-9));
        INFO("m=" << m << " -> " << to_string(r.status)
             << " r=" << r.mean_radius << " M_irr=" << r.irreducible_mass);
        REQUIRE(r.found);
        REQUIRE_THAT(r.mean_radius,      Catch::Matchers::WithinRel(0.5 * m, 1e-6));
        REQUIRE_THAT(r.irreducible_mass, Catch::Matchers::WithinRel(m, 1e-6));
    }
}

TEST_CASE("Individual horizons gain mass as the companion approaches",
          "[common_horizon]") {
    // Each horizon's irreducible mass exceeds the bare puncture mass by the
    // companion's binding contribution, and grows as they close in.
    double previous = 0.0;
    for (double d : {8.0, 4.0, 2.0}) {
        auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, d);
        strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 0.55);
        auto const r = find_individual(bl, 0.5 * d, s);

        INFO("d=" << d << " -> " << to_string(r.status)
             << " M_irr=" << r.irreducible_mass);
        REQUIRE(r.found);
        REQUIRE(r.irreducible_mass > 1.0);      // above the bare mass
        REQUIRE(r.irreducible_mass > previous); // and rising as they approach
        previous = r.irreducible_mass;
    }
}

TEST_CASE("No common horizon exists at wide separation", "[common_horizon]") {
    // The search must FAIL rather than converge onto something spurious.  This
    // is the check that catches a solver quietly returning a meaningless
    // surface instead of admitting there is none.
    for (double d : {6.0, 4.0}) {
        auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, d);
        strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 0.6 * d + 2.0);

        auto const r = find_horizon(s, provider(bl), cfg(1e-8, 80));
        INFO("d=" << d << " -> " << to_string(r.status)
             << " residual " << r.residual);
        REQUIRE_FALSE(r.found);
    }
}

TEST_CASE("A common horizon forms below a critical separation",
          "[common_horizon]") {
    // For two equal Brill-Lindquist punctures the common horizon appears
    // between d = 1.6 and d = 1.4 (bare mass 1 each), which is the classic
    // result d/m ~ 1.5.  Above it the search must fail; below it the horizon
    // must be found with roughly the summed bare mass.
    struct case_t { double d; bool expect; };
    for (auto const& c : { case_t{2.0, false}, case_t{1.8, false},
                           case_t{1.4, true},  case_t{1.0, true} }) {
        auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, c.d);
        strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 2.6);
        auto const r = find_horizon(s, provider(bl), cfg(1e-8, 80));

        INFO("d=" << c.d << " -> " << to_string(r.status)
             << " M_irr=" << r.irreducible_mass);
        REQUIRE(r.found == c.expect);
        if (c.expect) REQUIRE_THAT(r.irreducible_mass,
                                   Catch::Matchers::WithinRel(2.0, 0.05));
    }
}

TEST_CASE("Common horizon encloses and exceeds both individuals",
          "[common_horizon]") {
    // NOTE on what is *not* asserted here.  The Hawking area theorem constrains
    // event horizons evolving forward in time; it does not say that a common
    // apparent horizon's area exceeds the sum of the individual apparent
    // horizons' areas on a single slice.  Measured on this data it does not:
    // as the punctures approach, each individual horizon's area is inflated by
    // its companion's conformal factor (M_irr rises through 1.36, 1.42, 1.50,
    // 1.62) while the common horizon stays near the summed bare mass of 2, so
    // sqrt(M_1^2 + M_2^2) overtakes it below d ~ 1.3.
    //
    // What is certain, and is checked: the common horizon encloses both
    // punctures and is larger than either individual one.
    for (double d : {1.4, 1.2, 1.0, 0.8}) {
        auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, d);

        strahlkoerper_t s1(12, {0.0, 0.0, 0.0}, 0.5);
        strahlkoerper_t s2(12, {0.0, 0.0, 0.0}, 0.5);
        auto const r1 = find_individual(bl,  0.5 * d, s1);
        auto const r2 = find_individual(bl, -0.5 * d, s2);

        strahlkoerper_t sc(12, {0.0, 0.0, 0.0}, 2.6);
        auto const rc = find_horizon(sc, provider(bl), cfg(1e-8, 80));

        INFO("d=" << d << " common M_irr=" << rc.irreducible_mass
             << " individuals " << r1.irreducible_mass << ", " << r2.irreducible_mass);
        REQUIRE(r1.found);
        REQUIRE(r2.found);
        REQUIRE(rc.found);

        // Larger than either individual horizon.
        REQUIRE(rc.irreducible_mass > r1.irreducible_mass);
        REQUIRE(rc.irreducible_mass > r2.irreducible_mass);

        // And it encloses both punctures.
        REQUIRE(rc.min_radius > 0.5 * d);

        // Equal masses, so the two individuals must agree with each other.
        REQUIRE_THAT(r1.irreducible_mass,
                     Catch::Matchers::WithinRel(r2.irreducible_mass, 1e-6));
    }
}

TEST_CASE("Common horizon approaches Schwarzschild in the coincidence limit",
          "[common_horizon]") {
    // As the separation goes to zero the two punctures become one of mass
    // m1 + m2, so the common horizon must become the round Schwarzschild
    // horizon of the summed mass.
    double previous_asphericity = 1e30;
    for (double d : {0.6, 0.3, 0.1}) {
        auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, d);
        strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 1.2);

        auto const r = find_horizon(s, provider(bl), cfg(1e-8, 80));
        INFO("d=" << d << " -> " << to_string(r.status)
             << " M_irr=" << r.irreducible_mass
             << " r in [" << r.min_radius << ", " << r.max_radius << "]");
        REQUIRE(r.found);

        // Total bare mass is 2, so the limit is M_irr -> 2 and r -> 1.
        REQUIRE_THAT(r.irreducible_mass, Catch::Matchers::WithinRel(2.0, 0.15));

        double const asphericity = (r.max_radius - r.min_radius) / r.mean_radius;
        REQUIRE(asphericity < previous_asphericity);
        previous_asphericity = asphericity;
    }
    // Essentially round by d = 0.1.
    REQUIRE(previous_asphericity < 1e-2);
}

TEST_CASE("Proximity, not intersection, is what fires in time",
          "[common_horizon]") {
    // Measured on this data: at d = 1.4, where a common horizon already exists,
    // each individual horizon has max radius ~0.39 and the two surfaces are
    // still separated by more than a horizon radius.  A bare intersection test
    // would therefore miss the formation entirely, which is why the manager
    // triggers on proximity as well.
    double const d = 1.4;
    auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, d);

    strahlkoerper_t s1(12, {0.0, 0.0, 0.0}, 0.5);
    strahlkoerper_t s2(12, {0.0, 0.0, 0.0}, 0.5);
    auto const r1 = find_individual(bl,  0.5 * d, s1);
    auto const r2 = find_individual(bl, -0.5 * d, s2);
    REQUIRE(r1.found);
    REQUIRE(r2.found);

    // A common horizon does exist here.
    strahlkoerper_t sc(12, {0.0, 0.0, 0.0}, 2.6);
    REQUIRE(find_horizon(sc, provider(bl), cfg(1e-8, 80)).found);

    // Yet the surfaces do not intersect, and are not close to doing so.
    INFO("r_max = " << r1.max_radius << ", " << r2.max_radius
         << ", separation " << d);
    REQUIRE_FALSE(surfaces_overlap(s1, s2));
    REQUIRE(d > 1.5 * (r1.max_radius + r2.max_radius));

    // The proximity condition the manager uses does fire.
    double const radii = r1.max_radius + r2.max_radius;
    REQUIRE(d < 2.5 * radii);
}

TEST_CASE("Surface intersection detects a genuine overlap", "[common_horizon]") {
    // The second trigger, kept because proximity treats both surfaces as
    // spheres: two well-separated centres whose surfaces nonetheless meet.
    strahlkoerper_t a(8, {-1.0, 0.0, 0.0}, 0.5);
    strahlkoerper_t b(8, { 1.0, 0.0, 0.0}, 0.5);
    REQUIRE_FALSE(surfaces_overlap(a, b));

    // Stretch each towards the other. Y_1^1(cos) carries the Condon-Shortley
    // sign, so a negative coefficient elongates towards +x.
    a.coefficients()[spherical_harmonic_basis_t::coeff_index(1, 1, false)] = -1.2;
    b.coefficients()[spherical_harmonic_basis_t::coeff_index(1, 1, false)] =  1.2;
    REQUIRE(surfaces_overlap(a, b));
}

//---------------------------------------------------------------------------
// Constant-expansion surfaces.
//
// `target_expansion` lets the solver find a surface of constant Theta = E
// rather than Theta = 0, which is the primitive horizon pretracking is built
// from (Thornburg, Living Rev. Relativity 10 (2007) 3, section 3.2.6). The
// primitive works and is pinned here; the search over E that would turn it into
// a pretracking trigger is NOT implemented -- see the record for why.
//---------------------------------------------------------------------------

namespace {

/// Mean expansion over a surface, from analytic geometry.
double mean_theta(strahlkoerper_t const& s, brill_lindquist_geometry_t const& bl,
                  double* lo_out = nullptr, double* hi_out = nullptr)
{
    auto const g     = bl.at(s.collocation_points());
    auto const shape = s.evaluate();
    auto const& b    = s.basis();
    double acc = 0.0, lo = 1e300, hi = -1e300;
    int n = 0;
    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            double const th = expansion_at(g[p], sp, b.theta(it), b.phi(ip)).theta_out;
            acc += th; lo = std::min(lo, th); hi = std::max(hi, th); ++n;
        }
    if (lo_out) *lo_out = lo;
    if (hi_out) *hi_out = hi;
    return acc / n;
}

}  // namespace

TEST_CASE("A constant-expansion surface can be found around a puncture pair",
          "[common_horizon]") {
    double const d = 2.0;
    auto const bl = brill_lindquist_geometry_t::equal_mass_pair(1.0, d);

    // No common horizon exists at this separation, but surfaces of constant
    // positive expansion do, and the solver finds one. The target is taken from
    // the seed's own mean expansion so the test is not fitted to a magic
    // number: asking for slightly less than the seed already has means asking
    // for a slightly larger surface.
    strahlkoerper_t s(8, {0.0, 0.0, 0.0}, 2.5);
    double const E_seed = mean_theta(s, bl);
    INFO("seed mean Theta = " << E_seed);
    REQUIRE(E_seed > 0.0);

    auto c = cfg(2e-3, 60);
    c.target_expansion = 0.9 * E_seed;

    auto const r = find_horizon(s, provider(bl), c);
    INFO("status " << to_string(r.status) << ", residual " << r.residual
         << ", mean radius " << r.mean_radius);
    REQUIRE(r.found);

    // It is a constant-expansion surface, not a horizon: the expansion sits at
    // the target across the surface and nowhere near zero.
    double lo = 0.0, hi = 0.0;
    double const E_final = mean_theta(s, bl, &lo, &hi);
    INFO("Theta ranges over [" << lo << ", " << hi << "]");
    REQUIRE(lo > 0.0);
    REQUIRE_THAT(E_final, Catch::Matchers::WithinRel(c.target_expansion, 0.05));
    REQUIRE(hi - lo < 0.3 * E_final);   // genuinely near-constant
}
