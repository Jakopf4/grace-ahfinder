/**
 * @file surface_shape_output.hh
 * @brief Write the shape of a Strahlkoerper surface to disk.
 *
 * Two products per surface:
 *
 *   - a shape file, one per output time, holding the surface as sampled points
 *     plus the spectral coefficients that generated them;
 *   - a scalar time series, one row appended per output, for quantities like
 *     area and irreducible mass.
 *
 * The shape layout follows AHFinderDirect's ASCII convention closely enough to
 * be plotted the same way (`splot 'file' using 4:5:6 with lines`): one row per
 * surface point, blank line between successive theta rows so the surface reads
 * as a mesh rather than a point cloud. Keeping the format familiar matters more
 * than inventing a tidier one, since the people reading these files already
 * have tooling for it.
 *
 * Rank-agnostic by design: these take plain values, so the caller decides which
 * rank writes. Nothing here communicates.
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
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef GRACE_IO_SURFACE_SHAPE_OUTPUT_HH
#define GRACE_IO_SURFACE_SHAPE_OUTPUT_HH

#include <grace_config.h>

#include <grace/geometry/strahlkoerper.hh>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

namespace grace::IO {

/**
 * @brief Write one surface shape file.
 *
 * @param path      Destination; parent directories are created if needed.
 * @param surface   Surface to write.
 * @param iteration Iteration this shape belongs to.
 * @param time      Simulation time this shape belongs to.
 * @return false if the file could not be opened.
 */
inline bool write_surface_shape(std::filesystem::path const& path,
                                strahlkoerper_t const& surface,
                                std::size_t iteration,
                                double time)
{
    std::error_code ec;
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream f(path);
    if (!f) return false;

    f << std::scientific << std::setprecision(16);

    auto const& b     = surface.basis();
    auto const  shape = surface.evaluate();
    auto const  pts   = surface.collocation_points();
    auto const& c     = surface.centre();

    f << "# GRACE surface shape\n"
      << "# iteration " << iteration << "\n"
      << "# time      " << time << "\n"
      << "# centre    " << c[0] << ' ' << c[1] << ' ' << c[2] << "\n"
      << "# l_max     " << b.l_max() << "\n"
      << "# n_theta   " << b.n_theta() << "\n"
      << "# n_phi     " << b.n_phi() << "\n"
      << "#\n"
      << "# Spectral coefficients (real spherical harmonics), one per line,\n"
      << "# indexed as in spherical_harmonic_basis_t::coeff_index:\n";
    for (double a : surface.coefficients()) f << "# a " << a << "\n";

    f << "#\n"
      << "# columns: theta  phi  h  x  y  z\n"
      << "# blank line between theta rows, so `splot ... using 4:5:6 with lines`\n"
      << "# renders the surface as a mesh.\n";

    for (int it = 0; it < b.n_theta(); ++it) {
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            f << b.theta(it) << ' ' << b.phi(ip) << ' ' << shape.h[p] << ' '
              << pts[p][0] << ' ' << pts[p][1] << ' ' << pts[p][2] << '\n';
        }
        // Close the ring in phi so the mesh renders without a seam, then break
        // the row.
        int const p0 = b.point_index(it, 0);
        f << b.theta(it) << ' ' << (b.phi(0) + 2.0 * M_PI) << ' ' << shape.h[p0]
          << ' ' << pts[p0][0] << ' ' << pts[p0][1] << ' ' << pts[p0][2] << "\n\n";
    }

    return static_cast<bool>(f);
}

/**
 * @brief Create (or truncate) a scalar time-series file and write its header.
 *
 * @param columns Column names after the leading `iteration` and `time`.
 * @return false if the file could not be opened.
 */
inline bool initialize_surface_series(std::filesystem::path const& path,
                                      std::vector<std::string> const& columns)
{
    std::error_code ec;
    if (path.has_parent_path())
        std::filesystem::create_directories(path.parent_path(), ec);

    std::ofstream f(path, std::ios::trunc);
    if (!f) return false;

    f << "# 1:iteration 2:time";
    for (std::size_t i = 0; i < columns.size(); ++i)
        f << ' ' << (i + 3) << ':' << columns[i];
    f << '\n';
    return static_cast<bool>(f);
}

/**
 * @brief Append one row to a scalar time-series file.
 *
 * The column count is not checked against the header: the file may have been
 * created by a previous run. Keep the caller's column list in one place.
 */
inline bool append_surface_series(std::filesystem::path const& path,
                                  std::size_t iteration,
                                  double time,
                                  std::vector<double> const& values)
{
    std::ofstream f(path, std::ios::app);
    if (!f) return false;

    f << iteration << ' ' << std::scientific << std::setprecision(16) << time;
    for (double v : values) f << ' ' << v;
    f << '\n';
    return static_cast<bool>(f);
}

}  // namespace grace::IO

#endif  // GRACE_IO_SURFACE_SHAPE_OUTPUT_HH
