/**
 * @file grid_matter_provider.hh
 * @brief Matter primitives at the horizon collocation points.
 *
 * The companion to grid_geometry_provider_t. Two samples are needed because
 * GRACE keeps the primitives in the auxiliary array and the lapse and shift in
 * the evolved one. Neither needs gradients, and both are taken once per *find*
 * rather than once per Newton iteration, so the cost is a rounding error next
 * to the geometry the solve already moves.
 *
 * Reflection symmetries are folded exactly as the geometry provider folds them,
 * and with the same care: the sampling is mirrored into the computed domain and
 * the tensors are carried back by their parity. B^i is a pseudovector and does
 * not share the parity of z^i and beta^i -- see reflect_matter_point().
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

#ifndef GRACE_GEOMETRY_GRID_MATTER_PROVIDER_HH
#define GRACE_GEOMETRY_GRID_MATTER_PROVIDER_HH

#include <grace_config.h>

#include <grace/config/config_parser.hh>
#include <grace/data_structures/variables.hh>
#include <grace/physics/matter_from_evolution.hh>
#include <grace/utils/point_cloud_sampler.hh>

#include <array>
#include <vector>

namespace grace {

class grid_matter_provider_t {
public:
    /// @param want_gauge also sample the lapse and shift. They enter nothing
    ///        but the flux integrand, so the constraint monitor alone does not
    ///        need the second collective.
    explicit grid_matter_provider_t(int valid_gz = 1, bool want_gauge = true)
        : _sampler(valid_gz)
        , _aux_idx(matter_aux_var_indices())
        , _evol_idx(matter_evol_var_indices())
        , _want_gauge(want_gauge)
    {
        char const* axis[3] = { "x", "y", "z" };
        char const* lo[3]   = { "xmin", "ymin", "zmin" };
        for (int d = 0; d < 3; ++d) {
            _reflect[d] = get_param<bool>("amr", "reflection_symmetries", axis[d]);
            _plane[d]   = get_param<double>("amr", lo[d]);
            _any_reflect = _any_reflect || _reflect[d];
        }
    }

    /// Sample the matter at `pts`. Returns false if any point is outside the
    /// domain, in which case `out` is left unspecified.
    bool operator()(std::vector<std::array<double,3>> const& pts,
                    std::vector<matter_point_t>& out)
    {
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
                    if (!(folded[p][d] > _plane[d]))
                        folded[p][d] = _plane[d] + 1e-12;
                }
            query = &folded;
        }

        auto& vars = variable_list::get();

        auto const aux = _sampler.sample(vars.getaux(), *query, _aux_idx,
                                         /*want_grad=*/false, /*want_hessian=*/false);
        if (!aux.all_found) return false;

        point_sample_t evol;
        if (_want_gauge) {
            evol = _sampler.sample(vars.getstate(), *query, _evol_idx,
                                   /*want_grad=*/false, /*want_hessian=*/false);
            if (!evol.all_found) return false;
        }

        out.resize(pts.size());
        std::vector<double> a(static_cast<std::size_t>(matter_n_aux_vars()));
        std::vector<double> e(static_cast<std::size_t>(matter_n_evol_vars()));

        for (std::size_t p = 0; p < pts.size(); ++p) {
            for (int v = 0; v < matter_n_aux_vars();  ++v) a[static_cast<std::size_t>(v)] = aux.values(p, v);
            if (_want_gauge)
                for (int v = 0; v < matter_n_evol_vars(); ++v)
                    e[static_cast<std::size_t>(v)] = evol.values(p, v);
            out[p] = matter_from_sample(a.data(), _want_gauge ? e.data() : nullptr);
            if (_any_reflect) detail::reflect_matter_point(out[p], flip[p]);
        }
        return true;
    }

private:
    point_cloud_sampler_t _sampler;
    std::vector<int>      _aux_idx;
    std::vector<int>      _evol_idx;
    bool                  _want_gauge = true;

    bool   _reflect[3]  = { false, false, false };
    double _plane[3]    = { 0.0, 0.0, 0.0 };
    bool   _any_reflect = false;
};

}  // namespace grace

#endif  // GRACE_GEOMETRY_GRID_MATTER_PROVIDER_HH
