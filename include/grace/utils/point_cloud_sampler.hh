/**
 * @file point_cloud_sampler.hh
 * @brief Sample grid variables (values and Cartesian gradients) at an arbitrary
 *        set of points, globally gathered so every rank sees the whole cloud.
 *
 * The underlying sequence — build p4est point descriptors, run
 * p4est_search_local, hand the hits to an interpolator, pull to host, and
 * combine partial results across ranks — is needed by every diagnostic that
 * evaluates the solution somewhere other than a cell centre, and had been
 * open-coded separately at each such site. This class is the one
 * implementation.
 *
 * The behaviour it adds over those copies is unfound-point detection. Under a
 * plain sum-reduction over a zero-filled array, a point that no rank owns —
 * because it left the domain, or fell in a hole between refinement patches —
 * silently reads back as exactly zero, which downstream looks like a
 * degenerate metric rather than like missing data. Every sample therefore
 * carries a per-point owner count, and `all_found` is false unless every point
 * was claimed exactly once. Callers are expected to check it.
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

#ifndef GRACE_UTILS_POINT_CLOUD_SAMPLER_HH
#define GRACE_UTILS_POINT_CLOUD_SAMPLER_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/grid_interpolator.hh>
#include <grace/data_structures/variable_properties.hh>

#include <Kokkos_Core.hpp>

#include <array>
#include <cstddef>
#include <vector>

namespace grace {

/**
 * @brief Result of sampling a point cloud.
 *
 * Values and gradients are host views indexed by *global* point id, i.e. the
 * position of the point in the array handed to sample(). Every rank receives
 * identical contents.
 */
struct point_sample_t {
    /// (npoints, nvars)
    Kokkos::View<double**, Kokkos::HostSpace> values;
    /// (npoints, nvars, 3) with last index (d/dx, d/dy, d/dz). Extent 0 if
    /// gradients were not requested.
    Kokkos::View<double***, Kokkos::HostSpace> grads;

    /// (npoints, nvars, 6), packed symmetric (xx, xy, xz, yy, yz, zz).
    /// Extent 0 unless Hessians were requested.
    Kokkos::View<double***, Kokkos::HostSpace> hess;

    /// How many ranks claimed each point. Exactly 1 is correct; 0 means the
    /// point is outside the domain; >1 means overlapping ownership, which
    /// indicates a bug in the search rather than in the caller.
    std::vector<int> owner_count;

    /// True iff every owner_count entry is exactly 1.
    bool all_found = false;

    std::size_t npoints() const { return owner_count.size(); }

    /// Index of the first point that was not claimed exactly once, or -1.
    long first_bad_point() const {
        for (std::size_t i = 0; i < owner_count.size(); ++i)
            if (owner_count[i] != 1) return static_cast<long>(i);
        return -1;
    }
};

/**
 * @brief Samples grid variables at arbitrary points using Hermite-cubic
 *        interpolation, with optional analytic Cartesian gradients.
 *
 * Stateless between calls apart from the interpolator's scratch: the point set
 * is supplied afresh each time, which is what a moving surface needs.
 */
class point_cloud_sampler_t {
public:
    /**
     * @param valid_gz Number of ghost cells the interpolation stencil may
     *                 reach into. Must not exceed the driver's ghost width.
     */
    explicit point_cloud_sampler_t(int valid_gz = 1)
        : _interp(valid_gz)
        , _interp_lag(valid_gz)
    {}

    /**
     * @brief Sample `var_idx` at `xyz`.
     *
     * @param xyz          Query points, in Cartesian coordinates.
     * @param var_idx      Evolution-variable indices to sample.
     * @param want_grad    Whether to also return Cartesian gradients.
     * @param want_hessian Whether to also return second derivatives.
     *
     * Requesting the Hessian switches the whole sample onto the Lagrange
     * interpolator, so value, gradient and Hessian all come from *one*
     * polynomial. Mixing a Hermite value/gradient with a Lagrange Hessian would
     * leave the second derivative inconsistent with the first, which the
     * intrinsic-curvature algebra downstream would turn into a spurious
     * curvature. The Lagrange path is also more accurate on values and
     * gradients; it pays for that with a 5^3 rather than 4^3 stencil.
     *
     * Collective: must be called on every rank with identical `xyz` and
     * `var_idx`, because the reduction that assembles the global result is.
     */
    point_sample_t sample(std::vector<std::array<double,3>> const& xyz,
                          std::vector<int> const& var_idx,
                          bool want_grad = true,
                          bool want_hessian = false);

    /// Sample from an explicit state array rather than the active one.
    point_sample_t sample(var_array_t data,
                          std::vector<std::array<double,3>> const& xyz,
                          std::vector<int> const& var_idx,
                          bool want_grad = true,
                          bool want_hessian = false);

private:
    grid_interpolator_t<poly_kind::hermite, 3>              _interp;
    grid_interpolator_t<poly_kind::lagrange, LAGRANGE_INTERP_ORDER> _interp_lag;
};

}  // namespace grace

#endif  // GRACE_UTILS_POINT_CLOUD_SAMPLER_HH
