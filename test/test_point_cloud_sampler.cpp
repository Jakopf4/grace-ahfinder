/**
 * @file test_point_cloud_sampler.cpp
 * @brief Tests for the scattered-point sampler.
 *
 * Covers the three things the sampler adds over calling the interpolator
 * directly: results indexed by global point id, identical contents on every
 * rank, and detection of points that no rank owns.
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

#include <grace/utils/point_cloud_sampler.hh>

#include <grace/amr/amr_functions.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/data_structures/variables.hh>
#include <grace/data_structures/variable_indices.hh>
#include <grace/config/config_parser.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cmath>
#include <vector>

using namespace grace;

namespace {

/// Trilinear-plus-quadratic field, reproduced exactly by a Hermite cubic.
double poly_a(double x, double y, double z) {
    return 1.0 + 2.0*x - 0.5*y + 0.25*z + 0.1*x*y - 0.2*y*z + 0.3*x*x;
}
std::array<double,3> grad_poly_a(double x, double y, double z) {
    return { 2.0 + 0.1*y + 0.6*x,
            -0.5 + 0.1*x - 0.2*z,
             0.25 - 0.2*y };
}

double poly_b(double x, double y, double z) {
    return -3.0 + 0.7*z + 0.4*y*y - 0.15*x*z;
}
std::array<double,3> grad_poly_b(double x, double y, double z) {
    return { -0.15*z, 0.8*y, 0.7 - 0.15*x };
}

/// Write an analytic field into two slots of the state array, ghosts included.
void fill_state(int slot_a, int slot_b) {
    DECLARE_GRID_EXTENTS;
    auto& state  = variable_list::get().getstate();
    auto  state_h = Kokkos::create_mirror_view(state);
    auto& coords = coordinate_system::get();

    for (size_t q = 0; q < static_cast<size_t>(nq); ++q)
    for (size_t i = 0; i < nx + 2*ngz; ++i)
    for (size_t j = 0; j < ny + 2*ngz; ++j)
    for (size_t k = 0; k < nz + 2*ngz; ++k) {
        auto const pc = coords.get_physical_coordinates(
            {i, j, k}, q, {0.5, 0.5, 0.5}, /*use_ghostzones=*/true);
        state_h(i, j, k, slot_a, q) = poly_a(pc[0], pc[1], pc[2]);
        state_h(i, j, k, slot_b, q) = poly_b(pc[0], pc[1], pc[2]);
    }
    Kokkos::deep_copy(state, state_h);
}

/// A lattice of points spread across the interior of the physical domain.
///
/// fill_state() writes the exact analytic field into every cell including
/// ghosts, so interpolation is exact regardless of refinement level and the
/// points may be scattered freely. They are placed at irrational-ish offsets so
/// none coincides with a cell centre or a stencil knot, where the interpolation
/// weights would degenerate to something trivially correct.
std::vector<std::array<double,3>> interior_points() {
    auto const& cfg = grace::config_parser::get().get()["amr"];
    double const xmin = cfg["xmin"].as<double>();
    double const xmax = cfg["xmax"].as<double>();

    // Stay well inside the outer boundary so the stencil has room on all sides.
    double const lo = 0.65 * xmin;
    double const hi = 0.65 * xmax;
    int    const n  = 5;

    std::vector<std::array<double,3>> pts;
    pts.reserve(n * n * n);
    for (int a = 0; a < n; ++a)
    for (int b = 0; b < n; ++b)
    for (int c = 0; c < n; ++c) {
        auto coord = [&](int idx, double jitter) {
            return lo + (hi - lo) * (idx + jitter) / (n - 1 + 2 * jitter);
        };
        pts.push_back({ coord(a, 0.137), coord(b, 0.411), coord(c, 0.283) });
    }
    return pts;
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Sampler reproduces a polynomial and its gradient",
          "[point_cloud_sampler]") {
    int const slot_a = 0;
    int const slot_b = 1;
    fill_state(slot_a, slot_b);

    auto const pts = interior_points();
    REQUIRE(pts.size() > 0);

    point_cloud_sampler_t sampler(1);
    auto const s = sampler.sample(pts, {slot_a, slot_b}, /*want_grad=*/true);

    INFO("first unclaimed point index: " << s.first_bad_point());
    REQUIRE(s.all_found);
    REQUIRE(s.npoints() == pts.size());
    REQUIRE(s.values.extent(0) == pts.size());
    REQUIRE(s.values.extent(1) == 2u);
    REQUIRE(s.grads.extent(2) == 3u);

    for (size_t p = 0; p < pts.size(); ++p) {
        double const x = pts[p][0], y = pts[p][1], z = pts[p][2];
        INFO("point " << p << " at (" << x << ", " << y << ", " << z << ")");

        REQUIRE_THAT(s.values(p, 0), Catch::Matchers::WithinAbs(poly_a(x, y, z), 1e-9));
        REQUIRE_THAT(s.values(p, 1), Catch::Matchers::WithinAbs(poly_b(x, y, z), 1e-9));

        auto const ga = grad_poly_a(x, y, z);
        auto const gb = grad_poly_b(x, y, z);
        for (int d = 0; d < 3; ++d) {
            INFO("gradient component " << d);
            REQUIRE_THAT(s.grads(p, 0, d), Catch::Matchers::WithinAbs(ga[d], 1e-7));
            REQUIRE_THAT(s.grads(p, 1, d), Catch::Matchers::WithinAbs(gb[d], 1e-7));
        }
    }
}

TEST_CASE("Every rank receives identical sampled data", "[point_cloud_sampler]") {
    // The point of the global gather: a replicated Newton solve on a surface
    // must not diverge between ranks.
    fill_state(0, 1);
    auto const pts = interior_points();

    point_cloud_sampler_t sampler(1);
    auto const s = sampler.sample(pts, {0, 1}, true);
    REQUIRE(s.all_found);

    // Reduce max and min of every entry across ranks; they must coincide.
    std::vector<double> flat;
    flat.reserve(s.npoints() * 2);
    for (size_t p = 0; p < s.npoints(); ++p)
        for (int v = 0; v < 2; ++v) flat.push_back(s.values(p, v));

    std::vector<double> vmax(flat.size()), vmin(flat.size());
    parallel::mpi_allreduce(flat.data(), vmax.data(),
                            static_cast<int>(flat.size()), sc_MPI_MAX);
    parallel::mpi_allreduce(flat.data(), vmin.data(),
                            static_cast<int>(flat.size()), sc_MPI_MIN);

    for (size_t i = 0; i < flat.size(); ++i) {
        INFO("entry " << i);
        REQUIRE(vmax[i] == vmin[i]);
    }
}

TEST_CASE("Points outside the domain are reported, not silently zeroed",
          "[point_cloud_sampler]") {
    // This is the failure mode the sampler exists to close: under a plain
    // sum-reduction an unowned point reads back as exactly 0.0, which looks
    // like valid data downstream.
    fill_state(0, 1);

    auto pts = interior_points();
    size_t const n_good = pts.size();
    REQUIRE(n_good > 0);

    double const far = 1.0e6;
    pts.push_back({ far,  far,  far});
    pts.push_back({-far, -far, -far});

    point_cloud_sampler_t sampler(1);
    auto const s = sampler.sample(pts, {0, 1}, true);

    REQUIRE_FALSE(s.all_found);
    REQUIRE(s.owner_count[n_good]     == 0);
    REQUIRE(s.owner_count[n_good + 1] == 0);
    REQUIRE(s.first_bad_point() == static_cast<long>(n_good));

    // The in-domain points must be unaffected by the presence of bad ones.
    for (size_t p = 0; p < n_good; ++p) {
        double const x = pts[p][0], y = pts[p][1], z = pts[p][2];
        INFO("point " << p);
        REQUIRE(s.owner_count[p] == 1);
        REQUIRE_THAT(s.values(p, 0), Catch::Matchers::WithinAbs(poly_a(x, y, z), 1e-9));
    }
}

TEST_CASE("Sampling an empty point set is a well-defined no-op",
          "[point_cloud_sampler]") {
    point_cloud_sampler_t sampler(1);
    auto const s = sampler.sample({}, {0}, true);
    REQUIRE(s.all_found);
    REQUIRE(s.npoints() == 0u);
    REQUIRE(s.first_bad_point() == -1);
}

TEST_CASE("Gradients can be skipped", "[point_cloud_sampler]") {
    fill_state(0, 1);
    auto const pts = interior_points();

    point_cloud_sampler_t sampler(1);
    auto const s = sampler.sample(pts, {0}, /*want_grad=*/false);

    REQUIRE(s.all_found);
    REQUIRE(s.grads.extent(0) == 0u);
    for (size_t p = 0; p < pts.size(); ++p) {
        double const x = pts[p][0], y = pts[p][1], z = pts[p][2];
        INFO("point " << p);
        REQUIRE_THAT(s.values(p, 0), Catch::Matchers::WithinAbs(poly_a(x, y, z), 1e-9));
    }
}
