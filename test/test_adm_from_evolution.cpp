/**
 * @file test_adm_from_evolution.cpp
 * @brief Tests for the evolution-variable -> physical-ADM conversion.
 *
 * The conversion is exercised against independently-computed reference values
 * (not a second copy of the same algebra), and the chi chain rule in the metric
 * gradient is checked by differencing the conversion itself.
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

#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/analytic_geometry.hh>

#include <array>
#include <cmath>
#include <vector>

using namespace grace;

namespace {

/// A smooth, invertible analytic "spacetime" used as ground truth.
/// gammat_ij is unimodular-ish but need not be exactly so for these tests;
/// what matters is that the values and their derivatives are known exactly.
struct analytic_field_t {
    static double chi(std::array<double,3> const& x) {
        return 0.8 + 0.05 * x[0] + 0.03 * x[1] * x[1] - 0.02 * x[2];
    }
    static std::array<double,3> dchi(std::array<double,3> const& x) {
        return { 0.05, 0.06 * x[1], -0.02 };
    }
    static std::array<double,6> gt(std::array<double,3> const& x) {
        return { 1.0 + 0.10 * x[0],
                 0.05 * x[1],
                 0.02 * x[2],
                 1.0 + 0.07 * x[1],
                 0.03 * x[0],
                 1.0 - 0.04 * x[2] };
    }
    static std::array<std::array<double,3>,6> dgt(std::array<double,3> const&) {
        return {{ {0.10, 0.0,  0.0 },
                  {0.0,  0.05, 0.0 },
                  {0.0,  0.0,  0.02},
                  {0.0,  0.07, 0.0 },
                  {0.03, 0.0,  0.0 },
                  {0.0,  0.0, -0.04} }};
    }
    /// Trace-free (w.r.t. gammat_ij) conformal curvature.
    ///
    /// Z4c carries the algebraic constraint gammat^ij At_ij = 0, which GRACE
    /// enforces each step via enforce_algebraic_constraints(). K_ij built from
    /// an At that violates it would not reproduce K = Khat + 2 Theta, so the
    /// fixture projects the trace out rather than pretending it is absent.
    static std::array<double,6> At(std::array<double,3> const& x) {
        std::array<double,6> raw = { 0.01 * x[0], 0.02 * x[1], -0.01 * x[2],
                                     0.03 * x[1],  0.01 * x[0],  -0.04 * x[1] };
        auto const g = gt(x);

        double const det = g[0]*(g[3]*g[5] - g[4]*g[4])
                         - g[1]*(g[1]*g[5] - g[2]*g[4])
                         + g[2]*(g[1]*g[4] - g[2]*g[3]);
        std::array<double,6> gi{
            (g[3]*g[5] - g[4]*g[4]) / det,
            (g[2]*g[4] - g[1]*g[5]) / det,
            (g[1]*g[4] - g[2]*g[3]) / det,
            (g[0]*g[5] - g[2]*g[2]) / det,
            (g[1]*g[2] - g[0]*g[4]) / det,
            (g[0]*g[3] - g[1]*g[1]) / det };

        double const tr = gi[0]*raw[0] + gi[3]*raw[3] + gi[5]*raw[5]
                        + 2.0 * (gi[1]*raw[1] + gi[2]*raw[2] + gi[4]*raw[4]);
        for (int c = 0; c < 6; ++c) raw[c] -= tr * g[c] / 3.0;
        return raw;
    }
    static constexpr double khat  = 0.11;
    static constexpr double theta = 0.02;
};

/// Pack the analytic field into the sample layout adm_from_sample() expects.
std::vector<double> make_vals(std::array<double,3> const& x) {
    std::vector<double> v(adm_n_sample_vars(), 0.0);
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    auto const gt = analytic_field_t::gt(x);
    auto const At = analytic_field_t::At(x);
    for (int c = 0; c < 6; ++c) {
        v[ADM_SLOT_GTXX + c] = gt[c];
        v[ADM_SLOT_ATXX + c] = At[c];
    }
    v[ADM_SLOT_CHI]   = analytic_field_t::chi(x);
    v[ADM_SLOT_KHAT]  = analytic_field_t::khat;
    v[ADM_SLOT_THETA] = analytic_field_t::theta;
#else
    // COWLING stores physical variables; reuse the same shapes directly.
    auto const g = analytic_field_t::gt(x);
    auto const K = analytic_field_t::At(x);
    for (int c = 0; c < 6; ++c) {
        v[ADM_SLOT_GXX + c] = g[c];
        v[ADM_SLOT_KXX + c] = K[c];
    }
#endif
    return v;
}

std::vector<std::array<double,3>> make_grads(std::array<double,3> const& x) {
    std::vector<std::array<double,3>> gr(adm_n_sample_vars(), {0.0, 0.0, 0.0});
    auto const dgt = analytic_field_t::dgt(x);
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    for (int c = 0; c < 6; ++c) gr[ADM_SLOT_GTXX + c] = dgt[c];
    gr[ADM_SLOT_CHI] = analytic_field_t::dchi(x);
#else
    for (int c = 0; c < 6; ++c) gr[ADM_SLOT_GXX + c] = dgt[c];
#endif
    return gr;
}

adm_point_t convert(std::array<double,3> const& x, bool with_grad = true) {
    auto const vals  = make_vals(x);
    auto const grads = make_grads(x);
    return adm_from_sample(vals.data(),
                           with_grad ? reinterpret_cast<double const (*)[3]>(grads.data())
                                     : nullptr);
}

std::array<double,3> const kTestPoint{0.3, -0.4, 0.25};

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("adm_sym maps index pairs onto the packed symmetric layout",
          "[adm_from_evolution]") {
    REQUIRE(adm_sym(0, 0) == ADM_XX);
    REQUIRE(adm_sym(0, 1) == ADM_XY);
    REQUIRE(adm_sym(0, 2) == ADM_XZ);
    REQUIRE(adm_sym(1, 1) == ADM_YY);
    REQUIRE(adm_sym(1, 2) == ADM_YZ);
    REQUIRE(adm_sym(2, 2) == ADM_ZZ);
    // Symmetry under exchange.
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(adm_sym(i, j) == adm_sym(j, i));
}

TEST_CASE("adm_sample_var_indices matches the declared slot count",
          "[adm_from_evolution]") {
    auto const idx = adm_sample_var_indices();
    REQUIRE(static_cast<int>(idx.size()) == adm_n_sample_vars());
    REQUIRE(static_cast<int>(idx.size()) == ADM_N_SLOTS);
    // Indices must be distinct: a duplicate would silently alias two slots.
    for (size_t i = 0; i < idx.size(); ++i)
        for (size_t j = i + 1; j < idx.size(); ++j)
            REQUIRE(idx[i] != idx[j]);
}

TEST_CASE("Inverse metric satisfies gamma^ik gamma_kj = delta^i_j",
          "[adm_from_evolution]") {
    auto const adm = convert(kTestPoint);

    auto g    = [&](int i, int j) { return adm.g[adm_sym(i, j)]; };
    auto ginv = [&](int i, int j) { return adm.ginv[adm_sym(i, j)]; };

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0;
            for (int k = 0; k < 3; ++k) sum += ginv(i, k) * g(k, j);
            INFO("i=" << i << " j=" << j);
            REQUIRE_THAT(sum, Catch::Matchers::WithinAbs(i == j ? 1.0 : 0.0, 1e-13));
        }
    }
}

TEST_CASE("sqrt_det_g matches the determinant of gamma_ij", "[adm_from_evolution]") {
    auto const adm = convert(kTestPoint);
    auto g = [&](int i, int j) { return adm.g[adm_sym(i, j)]; };

    double const det = g(0,0) * (g(1,1)*g(2,2) - g(1,2)*g(1,2))
                     - g(0,1) * (g(0,1)*g(2,2) - g(0,2)*g(1,2))
                     + g(0,2) * (g(0,1)*g(1,2) - g(0,2)*g(1,1));

    REQUIRE(det > 0.0);
    REQUIRE_THAT(adm.sqrt_det_g, Catch::Matchers::WithinRel(std::sqrt(det), 1e-13));
}

TEST_CASE("K_trace is consistent with gamma^ij K_ij", "[adm_from_evolution]") {
    auto const adm = convert(kTestPoint);

    double trace = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            trace += adm.ginv[adm_sym(i, j)] * adm.K[adm_sym(i, j)];

    // Z4c defines K = Khat + 2 Theta and builds K_ij to reproduce it, so this
    // is a genuine cross-check of the reconstruction rather than a tautology.
    REQUIRE_THAT(adm.K_trace, Catch::Matchers::WithinAbs(trace, 1e-12));
}

TEST_CASE("Metric gradient matches a finite difference of the conversion",
          "[adm_from_evolution]") {
    // This is the chi chain rule under test:
    //   d_k gamma_ij = d_k gammat_ij / chi^2 - 2 gammat_ij d_k chi / chi^3
    double const e = 1e-6;

    for (int k = 0; k < 3; ++k) {
        auto xp = kTestPoint; xp[k] += e;
        auto xm = kTestPoint; xm[k] -= e;

        auto const ap = convert(xp, false);
        auto const am = convert(xm, false);
        auto const a0 = convert(kTestPoint, true);

        for (int c = 0; c < ADM_NSYM; ++c) {
            double const fd = (ap.g[c] - am.g[c]) / (2 * e);
            INFO("direction k=" << k << " component c=" << c);
            REQUIRE_THAT(a0.dg[k][c], Catch::Matchers::WithinAbs(fd, 1e-7));
        }
    }
}

TEST_CASE("Null gradient pointer zeroes dg rather than reading garbage",
          "[adm_from_evolution]") {
    auto const vals = make_vals(kTestPoint);
    auto const adm  = adm_from_sample(vals.data(), nullptr);

    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < ADM_NSYM; ++c) {
            INFO("k=" << k << " c=" << c);
            REQUIRE(adm.dg[k][c] == 0.0);
        }
}

TEST_CASE("Flat space converts to the identity metric", "[adm_from_evolution]") {
    std::vector<double> vals(adm_n_sample_vars(), 0.0);
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    vals[ADM_SLOT_GTXX] = 1.0;
    vals[ADM_SLOT_GTYY] = 1.0;
    vals[ADM_SLOT_GTZZ] = 1.0;
    vals[ADM_SLOT_CHI]  = 1.0;
#else
    vals[ADM_SLOT_GXX] = 1.0;
    vals[ADM_SLOT_GYY] = 1.0;
    vals[ADM_SLOT_GZZ] = 1.0;
#endif
    auto const adm = adm_from_sample(vals.data(), nullptr);

    REQUIRE_THAT(adm.g[ADM_XX], Catch::Matchers::WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(adm.g[ADM_YY], Catch::Matchers::WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(adm.g[ADM_ZZ], Catch::Matchers::WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(adm.g[ADM_XY], Catch::Matchers::WithinAbs(0.0, 1e-14));
    REQUIRE_THAT(adm.sqrt_det_g, Catch::Matchers::WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(adm.ginv[ADM_XX], Catch::Matchers::WithinAbs(1.0, 1e-14));
    REQUIRE_THAT(adm.K_trace, Catch::Matchers::WithinAbs(0.0, 1e-14));
}

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
TEST_CASE("Conformal rescaling reproduces gamma_ij = gammat_ij / chi^2",
          "[adm_from_evolution]") {
    auto const vals = make_vals(kTestPoint);
    auto const adm  = convert(kTestPoint, false);

    double const chi = analytic_field_t::chi(kTestPoint);
    auto const gt    = analytic_field_t::gt(kTestPoint);

    for (int c = 0; c < ADM_NSYM; ++c) {
        INFO("component " << c);
        REQUIRE_THAT(adm.g[c], Catch::Matchers::WithinRel(gt[c] / (chi * chi), 1e-13));
    }

    // K_ij = At_ij/chi^2 + (1/3) gamma_ij K, with K = Khat + 2 Theta.
    auto const At = analytic_field_t::At(kTestPoint);  // trace-free by construction
    double const K = analytic_field_t::khat + 2.0 * analytic_field_t::theta;
    for (int c = 0; c < ADM_NSYM; ++c) {
        double const expected = At[c] / (chi * chi) + adm.g[c] * K / 3.0;
        INFO("component " << c);
        REQUIRE_THAT(adm.K[c], Catch::Matchers::WithinAbs(expected, 1e-13));
    }
}

TEST_CASE("chi below the floor does not produce non-finite output",
          "[adm_from_evolution]") {
    // Near a puncture chi -> 0. The conversion must degrade to large-but-finite
    // values rather than inf/NaN, so callers can detect and reject the point.
    auto vals = make_vals(kTestPoint);
    vals[ADM_SLOT_CHI] = 0.0;
    auto const adm = adm_from_sample(vals.data(), nullptr);

    for (int c = 0; c < ADM_NSYM; ++c) {
        INFO("component " << c);
        REQUIRE(std::isfinite(adm.g[c]));
        REQUIRE(std::isfinite(adm.K[c]));
        REQUIRE(std::isfinite(adm.ginv[c]));
    }
    REQUIRE(std::isfinite(adm.sqrt_det_g));
}
#endif

//---------------------------------------------------------------------------
// Reflection symmetry.
//
// A run with amr.reflection_symmetries.z holds only z >= zmin, so half of every
// horizon's collocation points lie outside the grid. The geometry provider
// folds them back and undoes the fold on the sampled tensors. That is exact
// under an exact symmetry, and this pins the parity bookkeeping against an
// analytic spacetime that genuinely has the symmetry.
//---------------------------------------------------------------------------

TEST_CASE("Reflecting a sample reproduces the mirrored geometry",
          "[adm_from_evolution]") {
    // Which reflections are isometries is a property of the spacetime, not of
    // the bookkeeping: Kerr has a preferred sense of rotation, so z -> -z is a
    // symmetry at any spin while x -> -x alone is one only at zero spin. That
    // is exactly the assumption `amr.reflection_symmetries` encodes, so the
    // test respects it.
    std::array<double,3> const x{ 1.3, -0.8, 0.9 };

    struct case_t {
        double               spin;
        unsigned             flip;
        std::array<double,3> mirrored;
    };
    std::vector<case_t> const cases{
        { 0.7, 1u << 2,               { x[0],  x[1], -x[2] } },   // z, spinning
        { 0.0, 1u << 2,               { x[0],  x[1], -x[2] } },   // z
        { 0.0, 1u << 0,               { -x[0], x[1],  x[2] } },   // x
        { 0.0, 1u << 1,               { x[0], -x[1],  x[2] } },   // y
        { 0.0, (1u << 0) | (1u << 2), { -x[0], x[1], -x[2] } },   // x and z
    };

    for (auto const& c : cases) {
        INFO("spin " << c.spin << ", flip mask " << c.flip);
        kerr_schild_geometry_t ks(1.0, c.spin);

        auto const truth = ks.at(c.mirrored);   // geometry where we want it
        auto folded      = ks.at(x);            // geometry where we sampled
        detail::reflect_adm_point(folded, c.flip);

        // The analytic provider differences its own metric for dg and ddg, so
        // the second derivatives carry the coarsest floor here.
        auto close = [](double a, double b, double tol = 1e-9) {
            REQUIRE_THAT(a, Catch::Matchers::WithinAbs(b, tol));
        };

        for (int i = 0; i < ADM_NSYM; ++i) {
            INFO("component " << i);
            close(folded.g[i],    truth.g[i]);
            close(folded.ginv[i], truth.ginv[i]);
            close(folded.K[i],    truth.K[i]);
            for (int d = 0; d < 3; ++d) {
                close(folded.dg[d][i], truth.dg[d][i]);
                close(folded.dK[d][i], truth.dK[d][i]);
            }
            for (int c2 = 0; c2 < ADM_NSYM; ++c2)
                close(folded.ddg[c2][i], truth.ddg[c2][i], 1e-5);
        }
        for (int d = 0; d < 3; ++d) close(folded.dK_trace[d], truth.dK_trace[d]);

        // Scalars carry no indices and must be left alone.
        close(folded.K_trace,    truth.K_trace);
        close(folded.sqrt_det_g, truth.sqrt_det_g);
    }

    // A zero mask is the identity.
    kerr_schild_geometry_t ks(1.0, 0.7);
    auto a = ks.at(x);
    auto const b = a;
    detail::reflect_adm_point(a, 0u);
    for (int i = 0; i < ADM_NSYM; ++i) REQUIRE(a.g[i] == b.g[i]);
}
