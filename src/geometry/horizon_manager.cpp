/**
 * @file horizon_manager.cpp
 * @brief Implementation of the apparent-horizon manager.
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

#include <grace/geometry/horizon_manager.hh>
#include <grace/physics/grmhd_helpers.hh>

#include <grace/IO/spherical_surface_helpers.hh>
#include <grace/IO/horizon_shape_output.hh>
#include <grace/IO/surface_shape_output.hh>
#include <grace/config/config_parser.hh>
#include <grace/errors/assert.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/system/print.hh>

#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace grace {

namespace {

/// Column layout of the per-horizon scalar time series.
std::vector<std::string> const kSeriesColumns{
    // `state` distinguishes dormant / searching / found / abandoned, which
    // `found = 0` alone cannot.
    "found", "state", "status", "iterations", "residual",
    "area", "m_irr", "areal_radius",
    "centroid_x", "centroid_y", "centroid_z",
    "min_radius", "max_radius", "mean_radius",
    "Jx", "Jy", "Jz", "J", "chi", "m_christodoulou",
    "spin_axis_x", "spin_axis_y", "spin_axis_z",
    // Isolated-horizon (approximate-Killing-vector) quantities.
    "J_akv", "chi_akv", "m_akv", "killing_defect", "eigenvalue_gap",
    // Quality and shape.
    "max_shear", "C_equatorial", "C_polar_xz", "C_polar_yz", "ratio_polar_eq",
    // Horizon multipoles.
    "M0", "M2", "M4", "J1", "J3",
    // Fluxes through the horizon. Zero unless compute_fluxes is on.
    "mdot", "edot", "ldot", "edot_em", "ldot_em", "phi_B", "phi_B_signed",
    // Constraint violation on the surface itself. Z4 only.
    "ham_norm", "mom_norm"
};

}  // namespace

//---------------------------------------------------------------------------
// Construction
//---------------------------------------------------------------------------

horizon_manager_impl_t::horizon_manager_impl_t() {
    int const n  = get_param<int>("apparent_horizon", "n_horizons");
    _find_every  = get_param<int>("apparent_horizon", "find_every");
    _verbose     = get_param<bool>("apparent_horizon", "verbose");
    _output_shape = get_param<bool>("apparent_horizon", "output_shape");
    _shape_format = get_param<std::string>("apparent_horizon", "shape_format");
    _shape_every  = get_param<int>("apparent_horizon", "shape_output_every");

    _active = (n > 0);
    if (!_active) return;

    auto list = grace::config_parser::get().get()["apparent_horizon"]["horizons"];
    ASSERT(static_cast<int>(list.size()) == n,
           "apparent_horizon: n_horizons does not match the number of entries "
           "in the horizons list");

    _horizons.reserve(n);
    std::set<std::string> seen;
    std::map<int, std::vector<std::string>> parent_names;

    for (int i = 0; i < n; ++i) {
        auto h = list[i];

        auto const name = h["name"].as<std::string>();
        if (!seen.insert(name).second)
            ERROR("apparent_horizon: duplicate horizon name " << name);

        std::array<double,3> const centre{ h["center_x"].as<double>(),
                                           h["center_y"].as<double>(),
                                           h["center_z"].as<double>() };

        auto entry = std::make_unique<horizon_entry_t>(
            name, h["l_max"].as<int>(), centre, h["initial_radius"].as<double>());

        entry->solver.tolerance         = h["tolerance"].as<double>();
        entry->solver.max_iterations    = h["max_iterations"].as<int>();
        entry->solver.max_relative_step = h["max_relative_step"].as<double>();
        entry->solver.verbose           = _verbose;

        int const per = h["find_every"].as<int>();
        entry->find_every   = (per >= 0) ? per : _find_every;
        entry->track_centre = h["track_center"].as<bool>();
        entry->enabled      = h["enabled"].as<bool>();

        auto const role = h["role"].as<std::string>();
        if (role == "common") {
            entry->role  = horizon_role_t::common;
            entry->state = horizon_state_t::dormant;
            if (h["parents"]) {
                for (auto const& pn : h["parents"])
                    parent_names[i].push_back(pn.as<std::string>());
            }
            entry->overlap_margin = h["overlap_margin"].as<double>();
            entry->trigger_separation_factor =
                h["trigger_separation_factor"].as<double>();
            entry->parent_staleness_limit = static_cast<std::size_t>(
                std::max(0, h["parent_staleness_limit"].as<int>()));
        }
        entry->stop_after_failures = h["stop_after_failures"].as<int>();
        {
            int const re = h["retry_every"].as<int>();
            // Negative means "derive it": a slow multiple of this horizon's own
            // search cadence, so one knob does the right thing at any find_every.
            entry->retry_every = (re >= 0) ? re : 20 * std::max(1, entry->find_every);
        }

        // The approximate-Killing-vector solve is a dense symmetric generalised
        // eigenproblem of size n = (l_max+1)^2 - 1, solved by cyclic Jacobi:
        // O(n^3) per sweep, and replicated on every rank. The constant below is
        // measured, not estimated -- 3.5 ms at l_max = 8 and 21 ms at l_max = 12
        // on an M1, which is 4.4e-9 * n^3 (see the scaling test in
        // test_killing_vector.cpp). It is negligible at l_max = 8 and becomes
        // the dominant cost of a find around l_max = 16, so make it visible
        // rather than surprising.
        {
            int const    lm  = entry->surface.basis().l_max();
            double const n   = static_cast<double>((lm + 1) * (lm + 1) - 1);
            double const est = 4.4e-9 * n * n * n;   // seconds, measured fit
            if (est > 0.1 && parallel::mpi_comm_rank() == 0) {
                GRACE_WARN("AH [{}]: l_max = {} makes the quasi-local Killing-"
                           "vector eigensolve about {:.2f} s per find, which "
                           "will dominate the search. Lower l_max if the "
                           "multipoles are not needed.",
                           entry->name, lm, est);
            }
        }

        // Theta has dimensions of 1/length, so a tolerance is only meaningful
        // relative to the horizon's size: a generic surface has Theta ~ 2/R, and
        // a residual of eps corresponds to a radius error of roughly eps*R/2.
        // A tolerance chosen without that in mind can be satisfied by the
        // initial guess itself -- the search then "converges" in zero
        // iterations on a surface that is not the horizon.
        if (entry->solver.tolerance * entry->initial_radius > 1e-3
            && parallel::mpi_comm_rank() == 0) {
            GRACE_WARN("AH [{}]: tolerance {:.2e} on a horizon of radius ~{:.2e} "
                       "permits a radius error of order {:.1e}. Consider a "
                       "tolerance below {:.2e}.",
                       entry->name, entry->solver.tolerance, entry->initial_radius,
                       0.5 * entry->solver.tolerance * entry->initial_radius,
                       1e-3 / entry->initial_radius);
        }

        _horizons.push_back(std::move(entry));
    }

    // Resolve parent names now that every entry exists.
    {
        std::vector<std::string>    all_names;
        std::vector<horizon_role_t> all_roles;
        for (auto const& e : _horizons) {
            all_names.push_back(e->name);
            all_roles.push_back(e->role);
        }
        for (auto const& [idx, names] : parent_names) {
            auto& e = *_horizons[idx];
            auto const why = detail::resolve_parents(e.name, names, all_names,
                                                     all_roles, e.parents);
            if (!why.empty()) ERROR("apparent_horizon: " << why);
        }
    }

    // Built lazily so that constructing the manager costs nothing when no
    // horizon is ever due.
    _provider = std::make_unique<grid_geometry_provider_t>(1);

    // compute_fluxes is off by default: a vacuum run has no matter to
    // integrate, and turning it on is the user's assertion that the hydro
    // variables are live. monitor_constraints is on by default instead --
    // z4c_H and z4c_M are refilled once per RK step regardless
    // (compute_constraint_violations_fast(), evolve.cpp:438), so reading them
    // on the surface costs one small collective and answers the question every
    // other column raises.
    // Excision is configured on the grmhd side, not here: this only supplies
    // the geometry when that criterion asks for it.
    _excise = (get_param<std::string>("grmhd", "excision", "excision_criterion")
               == "horizon");
    _excise_factor = get_param<double>("grmhd", "excision",
                                       "excision_horizon_factor");
    if (_excise && parallel::mpi_comm_rank() == 0) {
        GRACE_INFO("apparent_horizon: matter excision will follow the located "
                   "horizons at {:.2f} of their inscribed radius, in union with "
                   "the lapse test.", _excise_factor);

        // The excised region is only refreshed when a horizon is actually
        // searched for, so between finds it is a stale snapshot of a moving
        // hole. The margin that makes this safe is exactly the (1 - factor) the
        // sphere was shrunk by: once the horizon has travelled further than
        // that, the stale sphere can reach outside the current surface, which
        // is the one thing excision must never do.
        if (_find_every > 8) {
            GRACE_WARN("apparent_horizon: excision follows the horizon but "
                       "find_every is {}. The excised region is only refreshed "
                       "on a find, so a hole moving further than {:.0f}% of its "
                       "horizon radius between finds would be excised outside "
                       "its own surface. Use a small find_every when excising.",
                       _find_every, 100.0 * (1.0 - _excise_factor));
        }
    }

    _predict_origin = get_param<bool>("apparent_horizon", "predict_origin_movement");
    _want_fluxes      = get_param<bool>("apparent_horizon", "compute_fluxes");
    _want_constraints = get_param<bool>("apparent_horizon", "monitor_constraints");
    if (_want_fluxes || _want_constraints)
        _matter = std::make_unique<grid_matter_provider_t>(1, _want_fluxes);

    if (parallel::mpi_comm_rank() == 0) {
        for (auto const& e : _horizons) {
            auto const path = series_path(*e);
            // Header only for a file that does not exist yet.  The manager is
            // constructed before a checkpoint is loaded, so truncating here
            // would discard the history of every restarted run -- the same
            // reason diagnostic_base_t opens its files in append mode.
            if (!std::filesystem::exists(path))
                IO::initialize_surface_series(path, kSeriesColumns);
        }

        // Create the shape directory here rather than leaving it to the first
        // write. This constructor runs on both the fresh and the restart path,
        // so the directory exists from step one either way; left to the writer
        // it appears only when the cadence first fires, which on a restart can
        // be hundreds of iterations in -- a run that looks like it has shape
        // output enabled and silently has none.
        if (_output_shape) {
            std::error_code ec;
            std::filesystem::create_directories(shape_dir(), ec);
            if (ec)
                GRACE_WARN("AH: could not create shape output directory {}: {}",
                           shape_dir().string(), ec.message());
        }

    }
}

//---------------------------------------------------------------------------
// Lookup
//---------------------------------------------------------------------------

int horizon_manager_impl_t::index_of(std::string const& name) const {
    for (int i = 0; i < n_horizons(); ++i)
        if (_horizons[i]->name == name) return i;
    return -1;
}

//---------------------------------------------------------------------------
// Searching
//---------------------------------------------------------------------------

//---------------------------------------------------------------------------
// Common-horizon trigger
//---------------------------------------------------------------------------

bool horizon_manager_impl_t::parent_reference(int p, std::size_t staleness_limit,
                                              horizon_result_t& out) const {
    auto const& e = *_horizons[p];
    if (!e.has_last_good) return false;

    // A result from long ago describes a configuration the binary has left. Let
    // it go stale rather than trigger or seed from it. The limit is the common
    // horizon's, not the parent's: it is the consumer that knows how far back
    // is still meaningful.
    if (staleness_limit > 0) {
        auto const iter = grace::runtime::get().iteration();
        if (iter > e.last_good_iteration
            && iter - e.last_good_iteration > staleness_limit)
            return false;
    }
    out = e.last_good;
    return true;
}

void horizon_manager_impl_t::seed_common_from_parents(horizon_entry_t& e) {
    // last_good, not last: at merger the individual horizons distort and the
    // finder starts failing on them, which is precisely when this runs. Seeding
    // from a failed result would give m_irr = 0 and a zero-radius guess.
    horizon_result_t g1, g2;
    if (!parent_reference(e.parents[0], e.parent_staleness_limit, g1)
        || !parent_reference(e.parents[1], e.parent_staleness_limit, g2)) {
        e.reseed();
        return;
    }

    double const m1 = std::max(g1.irreducible_mass, 0.0);
    double const m2 = std::max(g2.irreducible_mass, 0.0);
    double const mt = m1 + m2;

    auto const& c1 = g1.centroid;
    auto const& c2 = g2.centroid;

    std::array<double,3> centre{};
    for (int d = 0; d < 3; ++d)
        centre[d] = (mt > 0.0) ? (m1 * c1[d] + m2 * c2[d]) / mt
                               : 0.5 * (c1[d] + c2[d]);

    double const dx = c1[0] - c2[0], dy = c1[1] - c2[1], dz = c1[2] - c2[2];
    double const sep = std::sqrt(dx*dx + dy*dy + dz*dz);

    // Start OUTSIDE the common horizon: the expansion is positive there and the
    // Newton solve descends inward. Two lower bounds, whichever is larger --
    // the Schwarzschild radius of the combined mass, and a sphere that provably
    // encloses both parents.
    double const enclosing = 0.5 * sep + std::max(g1.max_radius, g2.max_radius);
    double const r0 = 1.2 * std::max(2.0 * mt, enclosing);

    e.surface.set_centre(centre);
    e.surface.set_sphere(r0);
    e.ever_found = false;   // this is a fresh guess, not a warm start
}

bool horizon_manager_impl_t::maybe_trigger_common(int i) {
    auto& e = *_horizons[i];
    if (e.role != horizon_role_t::common) return false;
    if (e.state != horizon_state_t::dormant) return false;

    auto const& p1 = *_horizons[e.parents[0]];
    auto const& p2 = *_horizons[e.parents[1]];

    horizon_result_t g1, g2;
    if (!parent_reference(e.parents[0], e.parent_staleness_limit, g1)
        || !parent_reference(e.parents[1], e.parent_staleness_limit, g2))
        return false;


    // Fire on EITHER of two conditions, whichever comes first.
    //
    // 1. Proximity. A common horizon forms well before the individual horizons
    //    touch -- measured on Brill-Lindquist data, two equal punctures acquire
    //    a common horizon at a separation where the individual surfaces still
    //    have a gap of more than one horizon radius between them. Waiting for
    //    them to intersect would miss the formation entirely. The transition
    //    sits near d ~ 2 (r1 + r2), so the default factor is set above that:
    //    triggering early costs only a few failed searches, and those now
    //    terminate in a handful of iterations via stall detection, whereas
    //    triggering late loses the horizon for good.
    //
    // 2. Actual intersection. Proximity compares centre separation against
    //    radii, which treats both surfaces as spheres. That is fine while they
    //    are round, and wrong for a tidally stretched pair whose near tips meet
    //    long before their centres are close -- so the true overlap test is
    //    kept as a second, independent way in.
    auto const& c1 = g1.centroid;
    auto const& c2 = g2.centroid;
    double const dx = c1[0] - c2[0], dy = c1[1] - c2[1], dz = c1[2] - c2[2];
    double const sep = std::sqrt(dx*dx + dy*dy + dz*dz);
    double const radii = g1.max_radius + g2.max_radius;

    bool const near    = (radii > 0.0) && (sep < e.trigger_separation_factor * radii);
    bool const touching = surfaces_overlap(p1.surface, p2.surface, e.overlap_margin);

    // 3. Both parents existed and have now both stopped being findable. That
    //    pattern is a merger signature: two horizons do not vanish together by
    //    coincidence. Without it the feature can miss its own reason for
    //    existing, since the individual searches are at their least reliable
    //    exactly when the common horizon forms.
    //
    //    Gated on the parents having been in the same neighbourhood when last
    //    seen, at twice the proximity factor. Two well-separated horizons can
    //    also both fail for unrelated numerical reasons, and a common horizon
    //    triggered by that never returns to dormant -- it would keep searching
    //    for a merger that is not happening.
    bool const both_lost = p1.has_existed && p2.has_existed
                        && !p1.last.found && !p2.last.found
                        && (radii > 0.0)
                        && (sep < 2.0 * e.trigger_separation_factor * radii);

    if (!near && !touching && !both_lost) return false;

    seed_common_from_parents(e);
    e.state = horizon_state_t::searching;
    e.consecutive_failures = 0;

    if (parallel::mpi_comm_rank() == 0) {
        GRACE_INFO("AH [{}]: parents {} and {} are close enough ({}); beginning "
                   "common-horizon search",
                   e.name, p1.name, p2.name,
                   both_lost ? "both parents lost"
                             : (touching ? "surfaces intersect" : "proximity"));
    }
    return true;
}

void horizon_manager_impl_t::record_outcome(horizon_entry_t& e, bool found) {
    if (found) {
        if (e.state == horizon_state_t::abandoned
            && parallel::mpi_comm_rank() == 0) {
            GRACE_INFO("AH [{}]: found again after being abandoned; resuming the "
                       "normal search cadence.", e.name);
        }
        e.consecutive_failures = 0;
        e.state = horizon_state_t::found;
        return;
    }
    // An abandoned horizon that fails its retry simply waits for the next one;
    // the failure count is already saturated and must not keep growing.
    if (e.state == horizon_state_t::abandoned) return;

    ++e.consecutive_failures;

    // Give up only on a horizon that existed and has stopped existing. One that
    // has never been found may simply not have formed yet -- a collapsing star,
    // or a merger that has not happened -- and must keep being looked for.
    //
    // has_existed, not ever_found: the latter is cleared by the very failure
    // being counted here, so testing it would prevent this from ever firing.
    if (e.has_existed && e.consecutive_failures >= e.stop_after_failures) {
        e.state = horizon_state_t::abandoned;
        if (parallel::mpi_comm_rank() == 0)
            GRACE_WARN("AH [{}]: not found in {} consecutive attempts after "
                       "previously existing; no longer searching for it.",
                       e.name, e.consecutive_failures);
    } else {
        e.state = horizon_state_t::searching;
    }
}

//---------------------------------------------------------------------------
// Searching
//---------------------------------------------------------------------------

horizon_result_t horizon_manager_impl_t::find_now(int i) {
    auto& e = *_horizons[i];

    // A horizon that has never been found starts from its configured guess;
    // afterwards the previous shape is a far better starting point and usually
    // converges in a couple of iterations. A common horizon that has not yet
    // been found is re-seeded from its parents instead, since they have moved.
    if (!e.ever_found) {
        if (e.role == horizon_role_t::common && e.parents.size() == 2) {
            seed_common_from_parents(e);
        } else {
            e.reseed();
        }
    }

    // Extrapolate the origin one find-interval forward before searching. The
    // warm start is only worth anything if the surface still sits near the
    // horizon, and for a hole moving a good fraction of its own radius between
    // finds it does not. AHFinderDirect calls this predict_origin_movement and
    // leaves it off by default; so do we, because a bad prediction costs
    // iterations rather than saving them.
    if (_predict_origin && e.ever_found && e.has_prev_centroid) {
        auto const& c = e.last_good.centroid;
        double step2 = 0.0;
        std::array<double,3> pred{};
        for (int d = 0; d < 3; ++d) {
            double const v = c[d] - e.prev_centroid[d];
            step2 += v * v;
            pred[d] = c[d] + v;
        }
        // A displacement comparable to the horizon itself is a jump -- a
        // regrid, a re-seed, or a bad find -- not motion. Following it would
        // throw the surface off the hole entirely.
        double const step = std::sqrt(step2);
        if (step > 0.0 && step < 0.5 * std::max(e.last_good.min_radius, 1e-30)) {
            if (!e.surface.recentre(pred))
                GRACE_WARN("AH [{}]: predicted origin lies outside the surface; "
                           "searching from the current one.", e.name);
        }
    }

    geometry_provider_t geom = [this](auto const& pts, auto& out) {
        return (*_provider)(pts, out);
    };

    std::function<void(std::string const&)> log;
    if (_verbose) {
        log = [&e](std::string const& msg) {
            GRACE_INFO("AH [{}]{}", e.name, msg);
        };
    }

    e.last = find_horizon(e.surface, geom, e.solver, log);

    if (e.last.found) {
        e.ever_found  = true;
        e.has_existed = true;
        if (e.has_last_good) {
            e.prev_centroid     = e.last_good.centroid;
            e.has_prev_centroid = true;
        }
        e.last_good   = e.last;
        e.last_good_iteration = grace::runtime::get().iteration();
        e.has_last_good = true;

        compute_fluxes(e);
        // recentre(), not set_centre(): the latter keeps h as the radius about
        // the old origin and so translates the whole surface, throwing away the
        // converged shape the next warm start depends on.
        if (e.track_centre && !e.surface.recentre(e.last.centroid)) {
            GRACE_WARN("AH [{}]: centroid lies outside the surface; leaving the "
                       "origin where it is.", e.name);
        }
    } else {
        // Do not carry a failed iterate forward: a surface left mid-divergence
        // is a worse starting point than the original guess, and would make one
        // bad step poison every later attempt.
        e.reseed();
        e.ever_found = false;
    }

    record_outcome(e, e.last.found);
    return e.last;
}

void horizon_manager_impl_t::publish_excision_region() {
    if (!_excise) return;

    // A sphere inscribed in each horizon that is currently found.
    //
    // The radius is the *smallest* distance from the centroid to a collocation
    // point, so the sphere lies inside the surface however distorted it is.
    // `min_radius` is not that number: it is the minimum of h about the
    // parameterisation origin, which only coincides with the centroid when
    // track_center is on. Measuring it directly costs one pass and is correct
    // either way.
    std::vector<std::array<double,4>> spheres;

    for (auto const& p : _horizons) {
        auto const& e = *p;
        if (!e.enabled || !e.last.found) continue;

        auto const pts = e.surface.collocation_points();
        double r_in = std::numeric_limits<double>::infinity();
        for (auto const& x : pts) {
            double d2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                double const dx = x[d] - e.last.centroid[d];
                d2 += dx * dx;
            }
            r_in = std::min(r_in, std::sqrt(d2));
        }
        if (!(r_in > 0.0) || !std::isfinite(r_in)) continue;

        double r_ex = _excise_factor * r_in;

        // The excised region is refreshed only on a find, so between finds it
        // is a stale snapshot of a moving hole. Shrink it by how far the hole
        // actually travelled over the last interval, with a safety factor --
        // measured rather than derived from dt * find_every, so it calibrates
        // itself at any cadence and needs no timestep.
        //
        // The surface may be *seeded* at a predicted position, but the excised
        // region is never predicted forward: excising where the horizon is
        // expected to be is precisely the failure this must not have.
        if (e.has_prev_centroid) {
            double step2 = 0.0;
            for (int d = 0; d < 3; ++d) {
                double const v = e.last.centroid[d] - e.prev_centroid[d];
                step2 += v * v;
            }
            r_ex -= 1.5 * std::sqrt(step2);
        }

        // A hole that outran its own margin gets no excision at all this step.
        // A degenerate or negative sphere would be worse than none.
        if (!(r_ex > 0.0)) {
            if (parallel::mpi_comm_rank() == 0)
                GRACE_WARN("AH [{}]: moved further between finds than the "
                           "excision margin allows; not excising this step. "
                           "Lower find_every or excision_horizon_factor.",
                           e.name);
            continue;
        }

        spheres.push_back({ e.last.centroid[0], e.last.centroid[1],
                            e.last.centroid[2], r_ex });
    }

    if (static_cast<int>(spheres.size()) > GRACE_MAX_EXCISION_SPHERES
        && parallel::mpi_comm_rank() == 0) {
        GRACE_WARN("apparent_horizon: {} horizons found but only {} excision "
                   "spheres are supported; the remainder will not be excised.",
                   spheres.size(), GRACE_MAX_EXCISION_SPHERES);
    }

    // Every rank publishes the same list -- the finds are collective and
    // bitwise reproducible -- so the excised region is identical everywhere
    // without any further communication.
    set_excision_spheres(spheres);
}

bool horizon_manager_impl_t::due(horizon_entry_t const& e, std::size_t iter) const {
    if (!e.enabled || e.find_every <= 0) return false;
    if (e.state == horizon_state_t::abandoned) {
        // Not "never again": a horizon abandoned through a merger, a regrid or
        // a gauge transient must be able to come back. A search for something
        // that is genuinely not there terminates in a handful of iterations via
        // the stall and runaway guards, so the retry is cheap.
        if (e.retry_every <= 0) return false;
        return iter % static_cast<std::size_t>(e.retry_every) == 0;
    }
    return iter % static_cast<std::size_t>(e.find_every) == 0;
}

void horizon_manager_impl_t::compute_fluxes(horizon_entry_t& e) {
    if (!_matter) return;

    // The surface was recentred only *after* this point in find_now(), so the
    // collocation points here are the ones the solve converged on.
    std::vector<matter_point_t> matter;
    auto const pts = e.surface.collocation_points();
    if (!(*_matter)(pts, matter)) {
        GRACE_WARN("AH [{}]: matter sampling failed on the horizon; fluxes not "
                   "computed this step.", e.name);
        return;
    }

    std::vector<adm_point_t> geom;
    if (!(*_provider)(pts, geom)) return;

    auto const f = horizon_fluxes(e.surface, geom, matter, e.last.centroid);

    // The two blocks are independently gated, so copy across only what was
    // asked for: reporting a flux computed without the lapse and shift would be
    // worse than reporting nothing.
    if (_want_fluxes) {
        e.last.fluxes = f;
    } else {
        e.last.fluxes.valid = true;
    }
    if (_want_constraints) {
        e.last.fluxes.ham_norm         = f.ham_norm;
        e.last.fluxes.mom_norm         = f.mom_norm;
        e.last.fluxes.have_constraints = f.have_constraints;
    }
}

void horizon_manager_impl_t::find_and_write() {
    if (!_active) return;

    auto const iter = grace::runtime::get().iteration();

    // The cadence test itself cannot be hoisted -- find_every is per horizon --
    // but "is anything due at all?" can be, and on a run with a cadence of, say,
    // 32 that skips both passes and the whole entry sweep on 31 steps out of 32.
    bool any_due = false;
    for (int i = 0; i < n_horizons() && !any_due; ++i)
        any_due = due(*_horizons[i], iter);
    if (!any_due) return;

    // Two passes: individuals, then commons. A common horizon's trigger must
    // see this step's parent shapes, not last step's. No cycles are possible
    // because a common horizon may not itself be a parent.
    for (int pass = 0; pass < 2; ++pass) {
        auto const want = (pass == 0) ? horizon_role_t::individual
                                      : horizon_role_t::common;

        for (int i = 0; i < n_horizons(); ++i) {
            auto& e = *_horizons[i];
            if (e.role != want) continue;
            if (!due(e, iter)) continue;

            // Dormant commons cost nothing but the trigger test until their
            // parents meet. Every rank evaluates it identically, so the
            // collective search below stays in lockstep.
            if (e.state == horizon_state_t::dormant) {
                if (!maybe_trigger_common(i)) { write_output(e); continue; }
            }

            auto const res = find_now(i);

            if (_verbose && parallel::mpi_comm_rank() == 0) {
                GRACE_INFO("AH [{}] {}: residual {:.3e} after {} iterations"
                           "{}",
                           e.name, to_string(res.status), res.residual,
                           res.iterations,
                           res.found ? fmt::format(", area {:.6e}, m_irr {:.6e}",
                                                   res.area, res.irreducible_mass)
                                     : std::string{});
            }

            write_output(e);
        }
    }

    publish_excision_region();
}

//---------------------------------------------------------------------------
// Output
//---------------------------------------------------------------------------

std::filesystem::path
horizon_manager_impl_t::series_path(horizon_entry_t const& e) const {
    return std::filesystem::path(grace::runtime::get().scalar_io_basepath())
         / ("horizon_" + e.name + ".dat");
}

std::filesystem::path
horizon_manager_impl_t::shape_dir() const {
    return std::filesystem::path(grace::runtime::get().scalar_io_basepath())
         / "horizons";
}

std::filesystem::path
horizon_manager_impl_t::shape_path(horizon_entry_t const& e, std::size_t iter,
                                   char const* extension) const {
    std::ostringstream ss;
    ss << "horizon_" << e.name << ".it" << iter << extension;
    return shape_dir() / ss.str();
}

void horizon_manager_impl_t::write_output(horizon_entry_t& e) {
    if (parallel::mpi_comm_rank() != 0) return;

    auto const& r    = e.last;
    auto const  iter = grace::runtime::get().iteration();
    auto const  t    = grace::runtime::get().time();

    // A row is written whether or not the horizon was found, so a gap in the
    // series is unambiguous: it means the search did not run, not that it
    // failed silently.
    std::vector<double> const row{
        r.found ? 1.0 : 0.0,
        static_cast<double>(static_cast<int>(e.state)),
        static_cast<double>(static_cast<int>(r.status)),
        static_cast<double>(r.iterations),
        r.residual,
        r.area, r.irreducible_mass, r.areal_radius,
        r.centroid[0], r.centroid[1], r.centroid[2],
        r.min_radius, r.max_radius, r.mean_radius,
        r.spin.J[0], r.spin.J[1], r.spin.J[2],
        r.spin.magnitude, r.spin.dimensionless, r.spin.christodoulou_mass,
        r.spin.axis[0], r.spin.axis[1], r.spin.axis[2],

        r.spin_akv.J, r.spin_akv.dimensionless, r.spin_akv.christodoulou_mass,
        r.spin_akv.killing_defect, r.spin_akv.eigenvalue_gap,

        r.max_shear,
        r.circumferences.equatorial,
        r.circumferences.polar_xz,
        r.circumferences.polar_yz,
        r.circumferences.ratio_xz_eq,

        r.multipoles.mass[0], r.multipoles.mass[2], r.multipoles.mass[4],
        r.multipoles.spin[1], r.multipoles.spin[3],

        r.fluxes.mdot, r.fluxes.edot, r.fluxes.ldot,
        r.fluxes.edot_em, r.fluxes.ldot_em,
        r.fluxes.phi_B, r.fluxes.phi_B_signed,
        r.fluxes.ham_norm, r.fluxes.mom_norm
    };
    // Quantities that were not computed are NaN in the result struct, which is
    // the right in-memory marker but a poor one in a data file: it is
    // inconsistent with the zeros every other field carries on a row where the
    // horizon was not found, and not every reader parses it. A `found = 0` row
    // therefore carries zeros throughout.
    std::vector<double> clean(row.size());
    for (std::size_t i = 0; i < row.size(); ++i)
        clean[i] = std::isfinite(row[i]) ? row[i] : 0.0;

    IO::append_surface_series(series_path(e), iter, t, clean);

    // The shape cadence is separate from the search cadence. A run that finds
    // every step does not want a surface file every step, and before this knob
    // existed it had no choice.
    //
    // The first successful find of a process is always due, whatever the
    // cadence. A fresh start is unaffected -- iteration 0 already satisfies the
    // modulo, so this writes the same single file. A restart is: the pre-loop
    // find is skipped there (src/grace.cpp:107), so the cadence alone would
    // leave the run with no shape output until the next multiple of
    // `shape_output_every`, and the state actually restarted from -- a
    // perfectly good horizon carried in from the checkpoint -- would never be
    // recorded at all.
    bool const shape_due = (_shape_every <= 0)
                         || (iter % static_cast<std::size_t>(_shape_every) == 0)
                         || !e.shape_written_this_run;

    if (_output_shape && r.found && shape_due) {
        e.shape_written_this_run = true;
        if (_shape_format != "ascii")
            IO::write_horizon_shape_hdf5(shape_path(e, iter, ".h5"), e.name,
                                         e.surface, r, kSeriesColumns, row,
                                         iter, t);
        if (_shape_format != "hdf5")
            IO::write_surface_shape(shape_path(e, iter, ".dat"), e.surface,
                                    iter, t);
    }
}

//---------------------------------------------------------------------------
// Checkpoint support
//---------------------------------------------------------------------------

std::vector<double> horizon_manager_impl_t::serialize_horizon(int i) const {
    return _horizons[i]->surface.serialize();
}

std::vector<double> horizon_manager_impl_t::serialize_last_good(int i) const {
    auto const& e = *_horizons[i];
    auto const& g = e.last_good;
    return { e.has_last_good ? 1.0 : 0.0,
             static_cast<double>(e.last_good_iteration),
             g.centroid[0], g.centroid[1], g.centroid[2],
             g.min_radius, g.max_radius, g.mean_radius,
             g.irreducible_mass,
             e.has_prev_centroid ? 1.0 : 0.0,
             e.prev_centroid[0], e.prev_centroid[1], e.prev_centroid[2] };
}

void horizon_manager_impl_t::deserialize_last_good(int i,
                                                   std::vector<double> const& b)
{
    auto& e = *_horizons[i];
    auto at = [&b](std::size_t k, double dflt) {
        return (k < b.size()) ? b[k] : dflt;
    };

    e.has_last_good       = at(0, 0.0) != 0.0;
    e.last_good_iteration = static_cast<std::size_t>(std::max(0.0, at(1, 0.0)));

    e.last_good = horizon_result_t{};
    e.last_good.found            = e.has_last_good;
    e.last_good.centroid         = { at(2, 0.0), at(3, 0.0), at(4, 0.0) };
    e.last_good.min_radius       = at(5, 0.0);
    e.last_good.max_radius       = at(6, 0.0);
    e.last_good.mean_radius      = at(7, 0.0);
    e.last_good.irreducible_mass = at(8, 0.0);

    e.has_prev_centroid = at(9, 0.0) != 0.0;
    e.prev_centroid     = { at(10, 0.0), at(11, 0.0), at(12, 0.0) };
}

bool horizon_manager_impl_t::deserialize_horizon(int i,
                                                 std::vector<double> const& buf,
                                                 bool ever_found)
{
    auto& e = *_horizons[i];
    if (!e.surface.deserialize(buf)) {
        // Most likely the parfile's l_max changed since the checkpoint was
        // written.  Falling back to the initial guess costs a few extra Newton
        // iterations; misreading the buffer would silently produce a wrong
        // surface, so refuse.
        GRACE_WARN("AH [{}]: checkpointed shape does not match the configured "
                   "l_max={}; restarting this horizon from its initial guess.",
                   e.name, e.surface.l_max());
        e.reseed();
        e.ever_found = false;
        return false;
    }
    e.ever_found = ever_found;
    return true;
}

//---------------------------------------------------------------------------
// Tracking policy: a detector surface that follows a horizon.
//
// Defined here rather than in spherical_surface_helpers.hh so that the surface
// machinery does not have to depend on the horizon stack.
//---------------------------------------------------------------------------

ah_tracking_policy_t::ah_tracking_policy_t(std::string const& detector_name)
    : _target(detector_name)
{
    // Find this detector's own configuration to learn what it should follow.
    auto spheres = grace::config_parser::get().get()["spherical_surfaces"]
                                              ["spherical_detectors"];
    for (std::size_t i = 0; i < spheres.size(); ++i) {
        if (spheres[i]["name"].as<std::string>() != detector_name) continue;
        if (spheres[i]["tracking_target"])
            _target = spheres[i]["tracking_target"].as<std::string>();
        if (spheres[i]["tracking_radius_factor"])
            _radius_factor = spheres[i]["tracking_radius_factor"].as<double>();
        break;
    }
    if (_target.empty()) _target = detector_name;
}

bool ah_tracking_policy_t::track(double& radius, std::array<double,3>& center)
{
    auto& mgr = grace::horizon_manager::get();
    if (!mgr.is_active()) return false;

    if (_index < 0) {
        _index = mgr.index_of(_target);
        if (_index < 0) {
            GRACE_WARN("Spherical surface tracks apparent horizon '{}', which is "
                       "not configured; the surface will not move.", _target);
            return false;
        }
    }

    auto const& e = mgr.get(_index);
    // Nothing to follow until the horizon has actually been located; leave the
    // surface at its configured geometry rather than collapsing it onto an
    // unset centroid.
    if (!e.ever_found || !e.last.found) return false;

    center = e.last.centroid;
    radius = _radius_factor * e.last.mean_radius;
    return true;
}

}  // namespace grace
