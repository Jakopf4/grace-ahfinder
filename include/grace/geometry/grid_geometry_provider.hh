/**
 * @file grid_geometry_provider.hh
 * @brief Geometry provider backed by the evolution grid.
 *
 * Adapts point_cloud_sampler_t + adm_from_sample() to the callback a horizon
 * solve expects, so the same solver runs against evolved data or against an
 * analytic spacetime.
 *
 * Collective: every call reduces across ranks, so all ranks must run the solve
 * in lockstep on an identical surface. That holds naturally because the Newton
 * iteration is replicated -- every rank sees the same globally-gathered
 * geometry and therefore takes the same step.
 *
 * A point that no rank owns makes the whole call fail rather than returning
 * zeros: a surface that has wandered off the grid must stop the solve, not
 * quietly produce a degenerate metric there.
 *
 * Reflection symmetries are folded, not refused. A grid run with
 * `amr.reflection_symmetries.z: true` holds only z >= zmin, so half of every
 * horizon's collocation points lie outside it and the solve would fail on every
 * step. The surface itself stays a full sphere -- the Newton system is posed
 * over all of it -- and only the *sampling* is folded into the computed
 * octant, with the ADM tensors carried back across the plane by their parity.
 * That is exact, not an approximation: the reflection is an isometry of the
 * spacetime, so a component picks up (-1) per index lying along a folded axis,
 * and each derivative counts as one more such index.
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

#ifndef GRACE_GEOMETRY_GRID_GEOMETRY_PROVIDER_HH
#define GRACE_GEOMETRY_GRID_GEOMETRY_PROVIDER_HH

#include <grace_config.h>

#include <grace/physics/adm_from_evolution.hh>
#include <grace/config/config_parser.hh>
#include <grace/utils/point_cloud_sampler.hh>

#include <array>
#include <vector>

namespace grace {

class grid_geometry_provider_t {
public:
    /// @param valid_gz Ghost-cell budget for the interpolation stencil.
    /// @param valid_gz    Ghost-cell budget for the interpolation stencil.
    /// @param want_hessian Also sample d_k d_l gamma_ij, which the quasi-local
    ///                     measures (intrinsic curvature, Killing vector,
    ///                     multipoles) require. Costs a wider stencil.
    explicit grid_geometry_provider_t(int valid_gz = 1, bool want_hessian = true)
        : _sampler(valid_gz)
        , _var_idx(adm_sample_var_indices())
        , _want_hessian(want_hessian)
    {
        char const* axis[3] = { "x", "y", "z" };
        char const* lo[3]   = { "xmin", "ymin", "zmin" };
        for (int d = 0; d < 3; ++d) {
            _reflect[d] = get_param<bool>("amr", "reflection_symmetries", axis[d]);
            _plane[d]   = get_param<double>("amr", lo[d]);
            _any_reflect = _any_reflect || _reflect[d];
        }
    }

    /// Sample the geometry at `pts`.  Returns false if any point is outside the
    /// domain, in which case `out` is left unspecified.
    bool operator()(std::vector<std::array<double,3>> const& pts,
                    std::vector<adm_point_t>& out)
    {
        // Fold each point into the computed domain, remembering which planes
        // it was carried across so the sample can be carried back.
        std::vector<std::array<double,3>> const* query = &pts;
        std::vector<std::array<double,3>> folded;
        std::vector<unsigned>             flip;

        if (_any_reflect) {
            folded = pts;
            flip.assign(pts.size(), 0u);
            for (std::size_t p = 0; p < pts.size(); ++p)
                for (int d = 0; d < 3; ++d) {
                    if (!_reflect[d]) continue;
                    if (folded[p][d] < _plane[d]) {
                        folded[p][d] = 2.0 * _plane[d] - folded[p][d];
                        flip[p] |= (1u << d);
                    }
                    // A point sitting exactly on the plane is on the boundary
                    // of the first cell and may fail the owning-quadrant test.
                    // With an odd theta count the Gauss-Legendre grid puts a
                    // node exactly on the equator, so this is the common case,
                    // not a corner one.
                    if (!(folded[p][d] > _plane[d]))
                        folded[p][d] = _plane[d] + 1e-12;
                }
            query = &folded;
        }

        auto const sample = _sampler.sample(*query, _var_idx, /*want_grad=*/true,
                                            _want_hessian);
        if (!sample.all_found) return false;

        int const nvars = adm_n_sample_vars();
        out.resize(pts.size());

        std::vector<double>                    vals(nvars);
        std::vector<std::array<double,3>>      grads(nvars);
        std::vector<std::array<double,ADM_NSYM>> hs(nvars);

        for (std::size_t p = 0; p < pts.size(); ++p) {
            for (int v = 0; v < nvars; ++v) {
                vals[v] = sample.values(p, v);
                for (int d = 0; d < 3; ++d) grads[v][d] = sample.grads(p, v, d);
                if (_want_hessian)
                    for (int d = 0; d < ADM_NSYM; ++d) hs[v][d] = sample.hess(p, v, d);
            }
            out[p] = _want_hessian
                ? adm_from_sample_hessian(
                      vals.data(),
                      reinterpret_cast<double const (*)[3]>(grads.data()),
                      reinterpret_cast<double const (*)[ADM_NSYM]>(hs.data()))
                : adm_from_sample(
                      vals.data(),
                      reinterpret_cast<double const (*)[3]>(grads.data()));

            if (_any_reflect) detail::reflect_adm_point(out[p], flip[p]);
        }
        return true;
    }

private:
    point_cloud_sampler_t _sampler;
    std::vector<int>      _var_idx;
    bool                  _want_hessian;

    bool   _reflect[3]  = { false, false, false };
    double _plane[3]    = { 0.0, 0.0, 0.0 };
    bool   _any_reflect = false;
};

}  // namespace grace

#endif  // GRACE_GEOMETRY_GRID_GEOMETRY_PROVIDER_HH
