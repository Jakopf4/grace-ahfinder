/**
 * @file test_surface_shape_output.cpp
 * @brief Tests for surface shape and time-series output.
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

#include <grace/IO/surface_shape_output.hh>
#include <grace/geometry/strahlkoerper.hh>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace grace;

namespace {

std::filesystem::path scratch_dir() {
    auto d = std::filesystem::temp_directory_path() / "grace_surface_output_test";
    std::filesystem::create_directories(d);
    return d;
}

std::vector<std::string> read_lines(std::filesystem::path const& p) {
    std::ifstream f(p);
    std::vector<std::string> out;
    for (std::string line; std::getline(f, line); ) out.push_back(line);
    return out;
}

}  // namespace

TEST_CASE("Shape output is parseable and geometrically correct",
          "[surface_output]") {
    double const r = 2.5;
    strahlkoerper_t s(4, {0.1, -0.2, 0.3}, r);

    auto const path = scratch_dir() / "shape.dat";
    std::filesystem::remove(path);
    REQUIRE(IO::write_surface_shape(path, s, 42, 1.25));
    REQUIRE(std::filesystem::exists(path));

    auto const lines = read_lines(path);
    REQUIRE(lines.size() > 0);

    // Metadata is present and correct.
    bool saw_iteration = false, saw_lmax = false;
    for (auto const& l : lines) {
        if (l.rfind("# iteration", 0) == 0) {
            REQUIRE(l.find("42") != std::string::npos);
            saw_iteration = true;
        }
        if (l.rfind("# l_max", 0) == 0) {
            REQUIRE(l.find("4") != std::string::npos);
            saw_lmax = true;
        }
    }
    REQUIRE(saw_iteration);
    REQUIRE(saw_lmax);

    // Every data row must place (x,y,z) at distance h from the centre, which
    // is the property any downstream plot depends on.
    int data_rows = 0;
    for (auto const& l : lines) {
        if (l.empty() || l[0] == '#') continue;
        std::istringstream is(l);
        double th, ph, h, x, y, z;
        REQUIRE(static_cast<bool>(is >> th >> ph >> h >> x >> y >> z));

        double const d = std::sqrt((x-0.1)*(x-0.1) + (y+0.2)*(y+0.2) + (z-0.3)*(z-0.3));
        REQUIRE_THAT(d, Catch::Matchers::WithinAbs(h, 1e-10));
        REQUIRE_THAT(h, Catch::Matchers::WithinAbs(r, 1e-10));
        ++data_rows;
    }
    // n_theta rows of (n_phi + 1) points -- the +1 closes the ring in phi.
    REQUIRE(data_rows == s.basis().n_theta() * (s.basis().n_phi() + 1));
}

TEST_CASE("Coefficients written to the shape file reconstruct the surface",
          "[surface_output]") {
    strahlkoerper_t s(4, {0.0, 0.0, 0.0}, 1.0);
    s.coefficients()[spherical_harmonic_basis_t::coeff_index(2, 0, false)] = 0.15;

    auto const path = scratch_dir() / "shape_coeffs.dat";
    std::filesystem::remove(path);
    REQUIRE(IO::write_surface_shape(path, s, 0, 0.0));

    std::vector<double> parsed;
    for (auto const& l : read_lines(path)) {
        if (l.rfind("# a ", 0) != 0) continue;
        std::istringstream is(l.substr(4));
        double v; is >> v;
        parsed.push_back(v);
    }

    REQUIRE(parsed.size() == static_cast<std::size_t>(s.n_coeffs()));
    for (int c = 0; c < s.n_coeffs(); ++c) {
        INFO("coefficient " << c);
        REQUIRE_THAT(parsed[c],
                     Catch::Matchers::WithinAbs(s.coefficients()[c], 1e-12));
    }
}

TEST_CASE("Time series writes a header and appends rows", "[surface_output]") {
    auto const path = scratch_dir() / "series.dat";
    std::filesystem::remove(path);

    REQUIRE(IO::initialize_surface_series(path, {"area", "m_irr"}));
    REQUIRE(IO::append_surface_series(path, 0,  0.0, {12.5, 0.99}));
    REQUIRE(IO::append_surface_series(path, 10, 1.5, {12.6, 1.00}));

    auto const lines = read_lines(path);
    REQUIRE(lines.size() == 3);
    REQUIRE(lines[0].rfind("#", 0) == 0);
    REQUIRE(lines[0].find("3:area")  != std::string::npos);
    REQUIRE(lines[0].find("4:m_irr") != std::string::npos);

    std::istringstream is(lines[2]);
    std::size_t iter; double t, area, m;
    REQUIRE(static_cast<bool>(is >> iter >> t >> area >> m));
    REQUIRE(iter == 10u);
    REQUIRE_THAT(t,    Catch::Matchers::WithinAbs(1.5,  1e-12));
    REQUIRE_THAT(area, Catch::Matchers::WithinAbs(12.6, 1e-12));

    // Re-initialising truncates, so a fresh run does not append to stale data.
    REQUIRE(IO::initialize_surface_series(path, {"area", "m_irr"}));
    REQUIRE(read_lines(path).size() == 1);
}
