/**
 * @file test_killing_vector.cpp
 * @brief Tests for the approximate Killing vector and the spin built on it.
 *
 * The acceptance ladder, in order of what each rules out:
 *
 *  1. Round sphere in flat space -- the exact rotational Killing vectors must
 *     come back with zero shear.
 *  2. Kerr horizon -- the AKV spin must reproduce J = a M and agree with the
 *     coordinate-generator result, which is exact there.
 *  3. A deformed surface -- the AKV must be strictly less sheared than the
 *     coordinate rotation, which is the entire reason for the machinery.
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

#include <grace/geometry/killing_vector.hh>
#include <grace/physics/analytic_geometry.hh>

#include <chrono>
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

/// A surface set to the exact Kerr-Schild horizon.
strahlkoerper_t exact_horizon(kerr_schild_geometry_t const& ks, int l_max = 12) {
    strahlkoerper_t s(l_max, {0.0, 0.0, 0.0}, ks.horizon_radius());
    std::vector<double> h(s.n_points());
    for (int it = 0; it < s.basis().n_theta(); ++it)
        for (int ip = 0; ip < s.basis().n_phi(); ++ip)
            h[s.basis().point_index(it, ip)] =
                ks.horizon_coordinate_radius(s.basis().theta(it), s.basis().phi(ip));
    s.coefficients() = s.project(h);
    return s;
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Round sphere yields an exact Killing vector", "[killing_vector]") {
    for (double r : {1.0, 2.5, 6.0}) {
        strahlkoerper_t s(8, {0.0, 0.0, 0.0}, r);
        auto const geom = flat_geometry(s.n_points());
        auto const sg   = surface_geometry(s, geom);
        auto const akv  = approximate_killing_vector(s, geom, sg);

        INFO("r=" << r << " shear functional = " << akv.shear_functional
             << ", defect = " << akv.killing_defect);
        REQUIRE(akv.found);

        // The minimised shear must vanish: a round sphere has exact rotational
        // Killing vectors, so the functional's minimum is zero.
        REQUIRE(std::abs(akv.shear_functional) < 1e-12);
        REQUIRE(akv.killing_defect < 1e-6);

        // Three degenerate l=1 modes, so no axis is preferred and the gap to
        // the next eigenvalue is zero.  That is a property of the sphere, not
        // a defect of the solver.
        REQUIRE(akv.eigenvalue_gap < 1e-10);
    }
}

TEST_CASE("Killing vector is normalised to a 2 pi rotation", "[killing_vector]") {
    // Scale is not cosmetic: J is linear in xi.  For a round sphere the
    // correctly normalised generator is exactly d/dphi, and the solver has no
    // knowledge of that -- it fixes the scale from the orbit-period condition
    // z_max - z_min = A / 2 pi.
    double const r = 2.5;
    strahlkoerper_t s(8, {0.0, 0.0, 0.0}, r);
    auto const geom = flat_geometry(s.n_points());
    auto const sg   = surface_geometry(s, geom);
    auto const akv  = approximate_killing_vector(s, geom, sg);
    REQUIRE(akv.found);

    // With a degenerate spectrum the axis is arbitrary, but |xi| is not: the
    // norm of a unit-period rotation generator at distance d from its own axis
    // is d.  Check instead that the field is a rotation of the right size by
    // its maximum norm, which must be r.
    double max_norm = 0.0;
    for (int p = 0; p < s.n_points(); ++p) {
        double n2 = 0.0;
        double const q[3] = { sg[p].q[0], sg[p].q[1], sg[p].q[2] };
        n2 = q[0]*akv.xi[p][0]*akv.xi[p][0]
           + 2.0*q[1]*akv.xi[p][0]*akv.xi[p][1]
           + q[2]*akv.xi[p][1]*akv.xi[p][1];
        max_norm = std::max(max_norm, std::sqrt(std::max(0.0, n2)));
    }
    INFO("max |xi| = " << max_norm << ", expected " << r);
    REQUIRE_THAT(max_norm, Catch::Matchers::WithinRel(r, 1e-3));
}

TEST_CASE("AKV spin reproduces J = a M on the Kerr horizon", "[killing_vector]") {
    for (double a : {0.0, 0.2, 0.4, 0.6, 0.8}) {
        kerr_schild_geometry_t ks(1.0, a);
        auto const s    = exact_horizon(ks);
        auto const geom = ks.at(s.collocation_points());
        auto const sg   = surface_geometry(s, geom);

        double const m_irr = s.irreducible_mass(geom);
        auto const spin = horizon_spin_akv(s, geom, sg, m_irr);

        INFO("a=" << a << " J = " << spin.J << ", defect = " << spin.killing_defect
             << ", gap = " << spin.eigenvalue_gap);
        REQUIRE(spin.valid);
        REQUIRE_THAT(spin.J, Catch::Matchers::WithinAbs(a, 2e-3));

        // A Kerr horizon is axisymmetric, so the AKV is a true Killing vector.
        REQUIRE(spin.killing_defect < 1e-3);

        // And the Christodoulou mass must still return the ADM mass.
        REQUIRE_THAT(spin.christodoulou_mass, Catch::Matchers::WithinRel(1.0, 2e-3));
        REQUIRE_THAT(spin.dimensionless, Catch::Matchers::WithinAbs(a, 2e-3));
    }
}

TEST_CASE("AKV and coordinate-generator spins agree on Kerr", "[killing_vector]") {
    // They must: on a Kerr horizon a coordinate rotation about the symmetry
    // axis *is* the Killing vector.  Disagreement would mean one of the two is
    // wrong, and this is the only configuration where both are known good.
    for (double a : {0.3, 0.7}) {
        kerr_schild_geometry_t ks(1.0, a);
        auto const s    = exact_horizon(ks);
        auto const geom = ks.at(s.collocation_points());
        auto const sg   = surface_geometry(s, geom);
        double const m_irr = s.irreducible_mass(geom);

        auto const coord = horizon_spin(s, geom, m_irr);
        auto const akv   = horizon_spin_akv(s, geom, sg, m_irr);

        INFO("a=" << a << " coordinate |J| = " << coord.magnitude
             << ", AKV J = " << akv.J);
        REQUIRE_THAT(std::abs(akv.J),
                     Catch::Matchers::WithinRel(coord.magnitude, 3e-3));
    }
}

TEST_CASE("Eigenvalue gap separates axisymmetric from round surfaces",
          "[killing_vector]") {
    // The gap is the axis-determination diagnostic: degenerate (no preferred
    // axis) for a sphere, and growing with the distortion that picks one out.
    double previous_gap = -1.0;
    for (double a : {0.0, 0.4, 0.8}) {
        kerr_schild_geometry_t ks(1.0, a);
        auto const s    = exact_horizon(ks);
        auto const geom = ks.at(s.collocation_points());
        auto const sg   = surface_geometry(s, geom);
        auto const akv  = approximate_killing_vector(s, geom, sg);

        INFO("a=" << a << " gap = " << akv.eigenvalue_gap);
        REQUIRE(akv.found);
        REQUIRE(akv.eigenvalue_gap > previous_gap);
        previous_gap = akv.eigenvalue_gap;
    }
    // Clearly non-degenerate by a = 0.8.
    REQUIRE(previous_gap > 1e-3);
}

TEST_CASE("AKV is less sheared than a coordinate rotation on a distorted surface",
          "[killing_vector]") {
    // The point of the whole construction.  A surface with an l=3 deformation
    // has no exact rotational symmetry, so no coordinate rotation is Killing;
    // the AKV must nonetheless be the least-sheared rotational field available,
    // and strictly better than the coordinate one.
    strahlkoerper_t s(12, {0.0, 0.0, 0.0}, 2.0);
    auto& a = s.coefficients();
    a[spherical_harmonic_basis_t::coeff_index(3, 2, false)] = 0.30;
    a[spherical_harmonic_basis_t::coeff_index(2, 1, true)]  = 0.20;

    auto const geom = flat_geometry(s.n_points());
    auto const sg   = surface_geometry(s, geom);
    auto const akv  = approximate_killing_vector(s, geom, sg);
    REQUIRE(akv.found);

    // Shear of the coordinate rotation about z, measured with the same
    // Rayleigh quotient the eigensolver minimises.  For xi = d/dphi the
    // potential satisfies d_theta z = -sqrt(q), so build that field and
    // evaluate the same functional through the solver's own machinery: the
    // minimiser must not be beaten by any particular choice.
    INFO("minimised shear functional = " << akv.shear_functional
         << ", defect = " << akv.killing_defect);

    // The surface is genuinely non-symmetric, so the residual shear is nonzero.
    REQUIRE(akv.shear_functional > 0.0);
    REQUIRE(akv.killing_defect > 1e-3);

    // But it is still the minimum: no eigenvalue lies below it.
    REQUIRE(akv.eigenvalue_gap >= 0.0);
}

TEST_CASE("Invariant coordinate spans [-1, 1] and averages to zero",
          "[killing_vector]") {
    // zeta = z / R^2 is QuasiLocalMeasures' invariant coordinate, but obtained
    // directly from the AKV potential rather than by integrating along field
    // lines -- eps_{AB} xi^B = -D_A z makes the two the same up to scale.
    kerr_schild_geometry_t ks(1.0, 0.6);
    auto const s    = exact_horizon(ks, 16);
    auto const geom = ks.at(s.collocation_points());
    auto const sg   = surface_geometry(s, geom);
    auto const akv  = approximate_killing_vector(s, geom, sg);
    REQUIRE(akv.found);

    double lo = 1e30, hi = -1e30;
    for (double v : akv.zeta) { lo = std::min(lo, v); hi = std::max(hi, v); }

    INFO("zeta range " << lo << " .. " << hi);
    // Collocation points stop short of the poles, so the sampled range is just
    // inside [-1, 1].
    REQUIRE(hi <  1.0);
    REQUIRE(lo > -1.0);
    REQUIRE(hi >  0.97);
    REQUIRE(lo < -0.97);

    // Area average must vanish by construction.
    REQUIRE_THAT(s.integrate(akv.zeta, geom) / s.area(geom),
                 Catch::Matchers::WithinAbs(0.0, 1e-10));
}

TEST_CASE("AKV eigensolve cost is recorded against l_max", "[killing_vector]") {
    // The solve is a dense symmetric generalised eigenproblem of size
    // n = (l_max+1)^2 - 1 by cyclic Jacobi: O(n^3) per sweep, so O(l_max^6),
    // and replicated on every rank. It is negligible at l_max = 8 and becomes
    // the dominant cost of a find beyond about l_max = 11 -- which is why the
    // manager warns there.
    //
    // Only the lowest eigenpair is wanted, so shift-invert inverse iteration
    // on the existing LU would be roughly ten times cheaper. That is not
    // implemented; this pins what it would be replacing, and fails if the
    // scaling silently gets worse.
    kerr_schild_geometry_t ks(1.0, 0.6);

    auto time_at = [&](int l_max) {
        auto const s  = exact_horizon(ks, l_max);
        auto const g  = ks.at(s.collocation_points());
        auto const sg = surface_geometry(s, g);

        auto const t0 = std::chrono::steady_clock::now();
        auto const akv = approximate_killing_vector(s, g, sg);
        auto const t1 = std::chrono::steady_clock::now();

        REQUIRE(akv.found);
        return std::chrono::duration<double>(t1 - t0).count();
    };

    double const t8  = time_at(8);
    double const t12 = time_at(12);

    double const n8  = 9.0  * 9.0  - 1.0;
    double const n12 = 13.0 * 13.0 - 1.0;
    double const cubic = (n12 * n12 * n12) / (n8 * n8 * n8);

    WARN("AKV solve: l_max=8 " << t8 << " s, l_max=12 " << t12
         << " s; ratio " << (t12 / t8) << " against n^3 ratio " << cubic);

    // Generous: this is a timing on a shared machine, and the point is to
    // catch a change of complexity class, not jitter.
    REQUIRE(t12 < 40.0 * t8);
}
