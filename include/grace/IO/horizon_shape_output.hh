/**
 * @file horizon_shape_output.hh
 * @brief HDF5 output of a located apparent horizon.
 *
 * One self-contained file per horizon per output: the spectral coefficients
 * (so the surface can be resampled at any resolution), the collocation grid and
 * its Cartesian points, every per-point diagnostic field the solve produced,
 * and the full scalar row -- carrying its own column names, so the file is
 * readable without the companion `.dat`.
 *
 * Deliberately not a mesh format. No connectivity, no polar caps, no sidecar:
 * GRACEPy owns the visualisation, and what it wants from here is the data.
 *
 * Serial and rank-0 only. The surface is a few hundred points on one rank, so
 * the collective machinery in hdf5_output.hh -- built around global offsets and
 * per-quadrant hyperslabs -- does not fit and is not used.
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

#ifndef GRACE_IO_HORIZON_SHAPE_OUTPUT_HH
#define GRACE_IO_HORIZON_SHAPE_OUTPUT_HH

#include <grace_config.h>

#include <grace/geometry/horizon_finder.hh>
#include <grace/geometry/strahlkoerper.hh>

#include <filesystem>
#include <string>
#include <vector>

namespace grace::IO {

/**
 * @brief Write one horizon to HDF5.
 *
 * @param path      Destination; parent directories are created if needed.
 * @param name      Horizon name, stored as an attribute.
 * @param surface   The located surface.
 * @param result    Its result, including the retained per-point fields.
 * @param columns   Names of the scalar row, stored alongside it so the file
 *                  needs no external key. Pass the same list the `.dat` uses.
 * @param row       The scalar row itself, same order as `columns`.
 * @param iteration Iteration this shape belongs to.
 * @param time      Simulation time this shape belongs to.
 * @return false if the file could not be written.
 */
bool write_horizon_shape_hdf5(std::filesystem::path const& path,
                              std::string const& name,
                              strahlkoerper_t const& surface,
                              horizon_result_t const& result,
                              std::vector<std::string> const& columns,
                              std::vector<double> const& row,
                              std::size_t iteration,
                              double time);

}  // namespace grace::IO

#endif  // GRACE_IO_HORIZON_SHAPE_OUTPUT_HH
