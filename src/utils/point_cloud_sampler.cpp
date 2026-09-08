/**
 * @file point_cloud_sampler.cpp
 * @brief Implementation of the scattered-point sampler.
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

#include <grace_config.h>

#include <grace/utils/point_cloud_sampler.hh>

#include <grace/amr/forest.hh>
#include <grace/amr/amr_functions.hh>
#include <grace/data_structures/variables.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/print.hh>

#include <Kokkos_Core.hpp>

namespace grace {

point_sample_t point_cloud_sampler_t::sample(
    std::vector<std::array<double,3>> const& xyz,
    std::vector<int> const& var_idx,
    bool want_grad,
    bool want_hessian)
{
    return sample(variable_list::get().getstate(), xyz, var_idx,
                  want_grad, want_hessian);
}

point_sample_t point_cloud_sampler_t::sample(
    var_array_t data,
    std::vector<std::array<double,3>> const& xyz,
    std::vector<int> const& var_idx,
    bool want_grad,
    bool want_hessian)
{
    int const npoints = static_cast<int>(xyz.size());
    int const nvars   = static_cast<int>(var_idx.size());

    // The Hessian path needs the gradient from the same polynomial.
    if (want_hessian) want_grad = true;

    point_sample_t result;
    result.owner_count.assign(npoints, 0);
    Kokkos::realloc(result.values, npoints, nvars);
    Kokkos::realloc(result.grads, want_grad ? npoints : 0,
                                  want_grad ? nvars   : 0,
                                  want_grad ? 3       : 0);
    Kokkos::realloc(result.hess, want_hessian ? npoints : 0,
                                 want_hessian ? nvars   : 0,
                                 want_hessian ? 6       : 0);

    if (npoints == 0 || nvars == 0) {
        result.all_found = true;
        return result;
    }

    //-----------------------------------------------------------------------
    // 1. Locate the points in the local part of the forest.
    //-----------------------------------------------------------------------
    std::vector<point_host_t> points(npoints);
    for (int i = 0; i < npoints; ++i)
        points[i] = { static_cast<std::size_t>(i), xyz[i] };

    std::vector<intersected_cell_descriptor_t> cells;
    std::vector<std::size_t>                   hit_points;
    intersected_cell_set_t set{ &cells, &hit_points };

    auto* p4est = grace::amr::forest::get().get();

    auto points_array = sc_array_new_data(
        const_cast<point_host_t*>(points.data()),
        sizeof(point_host_t),
        points.size());

    // grace_search_points communicates through the forest's user_pointer.
    // Restore whatever was there: the pointer is shared mutable state and
    // other subsystems reach for it too.
    void* saved_user_pointer = p4est->user_pointer;
    p4est->user_pointer = static_cast<void*>(&set);
    p4est_search_local(p4est, false, nullptr, &grace_search_points, points_array);
    p4est->user_pointer = saved_user_pointer;

    sc_array_destroy(points_array);

    //-----------------------------------------------------------------------
    // 2. Interpolate at the locally-owned points.
    //-----------------------------------------------------------------------
    std::vector<double> local_vals(static_cast<std::size_t>(npoints) * nvars, 0.0);
    std::vector<double> local_grads(
        want_grad ? static_cast<std::size_t>(npoints) * nvars * 3 : 0, 0.0);
    std::vector<double> local_hess(
        want_hessian ? static_cast<std::size_t>(npoints) * nvars * 6 : 0, 0.0);

    if (!hit_points.empty()) {
        Kokkos::View<double**,  default_space> vals_d("pcs_vals", 0, 0);
        Kokkos::View<double***, default_space> grad_d("pcs_grad", 0, 0, 0);
        Kokkos::View<double***, default_space> hess_d("pcs_hess", 0, 0, 0);

        if (want_hessian) {
            _interp_lag.compute_weights(points, hit_points, cells);
            _interp_lag.interpolate_with_hessian(data, var_idx,
                                                 vals_d, grad_d, hess_d);
        } else {
            _interp.compute_weights(points, hit_points, cells);
            _interp.interpolate_with_grad(data, var_idx, vals_d, grad_d, want_grad);
        }

        auto vals_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), vals_d);

        // Scatter local rows into their global slots.
        for (std::size_t loc = 0; loc < hit_points.size(); ++loc) {
            std::size_t const g = hit_points[loc];
            for (int v = 0; v < nvars; ++v)
                local_vals[g * nvars + v] = vals_h(loc, v);
        }

        if (want_grad) {
            auto grad_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), grad_d);
            for (std::size_t loc = 0; loc < hit_points.size(); ++loc) {
                std::size_t const g = hit_points[loc];
                for (int v = 0; v < nvars; ++v)
                    for (int d = 0; d < 3; ++d)
                        local_grads[(g * nvars + v) * 3 + d] = grad_h(loc, v, d);
            }
        }

        if (want_hessian) {
            auto hess_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), hess_d);
            for (std::size_t loc = 0; loc < hit_points.size(); ++loc) {
                std::size_t const g = hit_points[loc];
                for (int v = 0; v < nvars; ++v)
                    for (int d = 0; d < 6; ++d)
                        local_hess[(g * nvars + v) * 6 + d] = hess_h(loc, v, d);
            }
        }
    }

    // Ownership tally, built before the reduction so a duplicated claim shows
    // up as a count > 1 rather than as silently doubled data.
    std::vector<int> local_count(npoints, 0);
    for (std::size_t g : hit_points) local_count[g] += 1;

    //-----------------------------------------------------------------------
    // 3. Combine across ranks. Each point is owned by exactly one rank, so a
    //    sum reconstructs the global array; the parallel count reduction is
    //    what makes that assumption checkable rather than assumed.
    //-----------------------------------------------------------------------
    std::vector<double> global_vals(local_vals.size(), 0.0);
    parallel::mpi_allreduce(local_vals.data(), global_vals.data(),
                            static_cast<int>(local_vals.size()), mpi_sum);

    parallel::mpi_allreduce(local_count.data(), result.owner_count.data(),
                            npoints, mpi_sum);

    for (int i = 0; i < npoints; ++i)
        for (int v = 0; v < nvars; ++v)
            result.values(i, v) = global_vals[static_cast<std::size_t>(i) * nvars + v];

    if (want_grad) {
        std::vector<double> global_grads(local_grads.size(), 0.0);
        parallel::mpi_allreduce(local_grads.data(), global_grads.data(),
                                static_cast<int>(local_grads.size()), mpi_sum);
        for (int i = 0; i < npoints; ++i)
            for (int v = 0; v < nvars; ++v)
                for (int d = 0; d < 3; ++d)
                    result.grads(i, v, d) =
                        global_grads[(static_cast<std::size_t>(i) * nvars + v) * 3 + d];
    }

    if (want_hessian) {
        std::vector<double> global_hess(local_hess.size(), 0.0);
        parallel::mpi_allreduce(local_hess.data(), global_hess.data(),
                                static_cast<int>(local_hess.size()), mpi_sum);
        for (int i = 0; i < npoints; ++i)
            for (int v = 0; v < nvars; ++v)
                for (int d = 0; d < 6; ++d)
                    result.hess(i, v, d) =
                        global_hess[(static_cast<std::size_t>(i) * nvars + v) * 6 + d];
    }

    //-----------------------------------------------------------------------
    // 4. Report coverage.
    //-----------------------------------------------------------------------
    int n_missing = 0, n_duplicated = 0;
    for (int i = 0; i < npoints; ++i) {
        if (result.owner_count[i] == 0) ++n_missing;
        else if (result.owner_count[i] > 1) ++n_duplicated;
    }
    result.all_found = (n_missing == 0 && n_duplicated == 0);

    if (n_missing > 0) {
        auto const bad = result.first_bad_point();
        GRACE_WARN("point_cloud_sampler: {} of {} points lie outside the domain "
                   "(first at index {}: {:.6e} {:.6e} {:.6e}); their sampled "
                   "values are zero, not physical.",
                   n_missing, npoints, bad,
                   xyz[bad][0], xyz[bad][1], xyz[bad][2]);
    }
    if (n_duplicated > 0) {
        GRACE_WARN("point_cloud_sampler: {} of {} points were claimed by more "
                   "than one rank; sampled values are summed duplicates.",
                   n_duplicated, npoints);
    }

    return result;
}

}  // namespace grace
