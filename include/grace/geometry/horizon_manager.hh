/**
 * @file horizon_manager.hh
 * @brief Owns the apparent horizons configured in the parfile and runs them.
 *
 * Turns the solver in horizon_finder.hh into something a simulation uses:
 * constructs surfaces from configuration, searches on a cadence, warm-starts
 * each search from the previous shape, optionally follows a moving hole, writes
 * output, and exposes state for checkpointing and for surfaces that track a
 * horizon.
 *
 * Horizons are independent: each has its own origin, resolution and cadence,
 * and none waits on another. Common-envelope searches for a binary (the
 * "look for a third horizon once the two approach" logic that AHFinderDirect
 * spells with find_after_individual / depends_on) are deliberately not here;
 * they can be added without changing this interface.
 *
 * @par Collective behaviour
 * find_horizon() reduces across ranks through its geometry provider, so every
 * rank must perform every search, in the same order. The cadence test therefore
 * depends only on the iteration counter and never on rank-local state -- a
 * search skipped on one rank alone would deadlock the others. Only rank 0
 * writes files.
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

#ifndef GRACE_GEOMETRY_HORIZON_MANAGER_HH
#define GRACE_GEOMETRY_HORIZON_MANAGER_HH

#include <grace_config.h>

#include <grace/geometry/grid_geometry_provider.hh>
#include <grace/geometry/grid_matter_provider.hh>
#include <grace/geometry/horizon_finder.hh>
#include <grace/geometry/strahlkoerper.hh>
#include <grace/utils/lifetime_tracker.hh>
#include <grace/utils/singleton_holder.hh>

#include <array>
#include <filesystem>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace grace {

class checkpoint_handler_impl_t;

/// What a horizon is: one of a binary's individual holes, or the common
/// horizon that forms around them at merger.
enum class horizon_role_t { individual, common };

/// Where a horizon is in its life cycle. `found = 0` in the output alone cannot
/// distinguish these, which is why the state is reported as its own column.
enum class horizon_state_t {
    dormant,    ///< Not searched yet: a common horizon whose parents have not
                ///< met, or a horizon waiting for the lapse to fall below its
                ///< `activation_lapse`.
    searching,  ///< Being searched for; may or may not be found each time.
    found,      ///< Located at the most recent attempt.
    abandoned   ///< Existed once, has failed repeatedly, no longer searched.
};

inline char const* to_string(horizon_state_t s) {
    switch (s) {
        case horizon_state_t::dormant:   return "dormant";
        case horizon_state_t::searching: return "searching";
        case horizon_state_t::found:     return "found";
        case horizon_state_t::abandoned: return "abandoned";
    }
    return "unknown";
}

/// One configured horizon and everything remembered about it between searches.
struct horizon_entry_t {
    std::string      name;
    strahlkoerper_t  surface;
    horizon_config_t solver;

    int    find_every   = -1;
    bool   track_centre = true;
    bool   enabled      = true;

    /// The configured guess, kept so a failed search can be re-seeded from it.
    double               initial_radius = 1.0;
    std::array<double,3> initial_centre{};

    horizon_result_t last;
    /// The most recent *successful* result, and the iteration it came from.
    /// `last` is overwritten by every attempt including failures, so it cannot
    /// be used to reason about a horizon that has momentarily gone missing --
    /// which is precisely the state the individual horizons of a binary are in
    /// while the common one is forming.
    horizon_result_t last_good;
    std::size_t      last_good_iteration = 0;
    bool             has_last_good = false;
    /// Centroid of the find *before* last_good, and whether there was one.
    /// One interval of history is enough for both of its uses -- extrapolating
    /// the origin forward, and measuring how far the hole travels between finds
    /// so the excised region can be shrunk by that much. A longer history would
    /// buy quadratic extrapolation, which the jump guard makes pointless.
    std::array<double,3> prev_centroid{};
    bool                 has_prev_centroid = false;
    /// Whether the *previous* attempt succeeded, i.e. whether the current shape
    /// is worth warm-starting from. Reset on every failure.
    bool             ever_found = false;
    /// Whether this horizon has been found at any point in the run. Latches
    /// true and is never reset -- it answers "did this horizon ever exist?",
    /// which is a different question from "can I warm-start?" and must not
    /// share a flag with it.
    bool             has_existed = false;
    /// Whether this horizon has written a shape file yet in this process. The
    /// first successful find writes one whatever the cadence says. On a fresh
    /// start iteration 0 is due anyway, so nothing changes; on a restart the
    /// pre-loop find at src/grace.cpp:107 is skipped, so without this there is
    /// no shape output -- and no `horizons/` directory -- until the run reaches
    /// the next multiple of `shape_output_every`, which can be hundreds of
    /// iterations away or never. Per horizon, not per manager: otherwise the
    /// first entry to write would suppress the others at the same iteration.
    /// Not checkpointed -- it is about this process, not about the horizon.
    bool             shape_written_this_run = false;

    //-- binary / merger bookkeeping ---------------------------------------
    horizon_role_t   role = horizon_role_t::individual;
    /// Indices of the two parents, for a common horizon. Resolved from names
    /// once every entry exists.
    std::vector<int> parents;
    /// Fractional enlargement applied when testing whether the parents overlap.
    double           overlap_margin = 0.0;
    /// How many iterations a parent's last successful result may be stale
    /// before it stops being usable as a trigger. Zero disables the limit.
    std::size_t      parent_staleness_limit = 0;
    /// Trigger when the parents' centre separation falls below this multiple of
    /// the sum of their radii. The common horizon forms before the individual
    /// surfaces touch, so this, not intersection, is the condition that fires
    /// at the right time.
    double           trigger_separation_factor = 2.5;
    /// Consecutive failures tolerated after a horizon has been found before it
    /// is presumed gone.
    int              stop_after_failures = 5;
    /// How often an abandoned horizon is looked for again. Abandonment is a
    /// slow cadence, not a death sentence: a few bad steps through a merger, a
    /// regrid or a gauge transient must not lose a horizon for the rest of the
    /// run. Zero makes abandonment permanent.
    int              retry_every = 0;

    /// Stay dormant until the global minimum of the lapse falls below this,
    /// then search on the normal cadence for the rest of the run. Zero
    /// disables the gate. Meant for a single collapsing star, where every
    /// search before the horizon forms ends at the runaway guard; a binary's
    /// common horizon has its own trigger and ignores this. The latch lives in
    /// `state`, which the checkpoint carries, so a restart neither re-arms it
    /// nor forgets that it fired.
    double           activation_lapse = 0.0;

    horizon_state_t  state = horizon_state_t::searching;
    int              consecutive_failures = 0;

    horizon_entry_t(std::string n, int l_max,
                    std::array<double,3> const& centre, double radius)
        : name(std::move(n))
        , surface(l_max, centre, radius)
        , initial_radius(radius)
        , initial_centre(centre)
    {}

    /// Return the surface to its configured guess.
    void reseed() {
        surface.set_centre(initial_centre);
        surface.set_sphere(initial_radius);
    }
};

namespace detail {

/// Resolve a common horizon's parent names against the full roster.
///
/// Split out of the manager's constructor so the rejection rules can be tested
/// without standing up a manager from a deliberately broken parfile -- the
/// manager is a singleton built from the one global config, so a bad-config
/// test is otherwise impossible to write.
///
/// @return an empty string on success, with the resolved indices in `out`;
///         otherwise the reason for rejection.
inline std::string resolve_parents(std::string const& name,
                                   std::vector<std::string> const& parents,
                                   std::vector<std::string> const& all_names,
                                   std::vector<horizon_role_t> const& all_roles,
                                   std::vector<int>& out)
{
    out.clear();
    if (parents.size() != 2)
        return "common horizon " + name + " needs exactly two parents, got "
             + std::to_string(parents.size());

    for (auto const& n : parents) {
        int j = -1;
        for (std::size_t k = 0; k < all_names.size(); ++k)
            if (all_names[k] == n) { j = static_cast<int>(k); break; }

        if (j < 0)
            return "common horizon " + name + " names unknown parent " + n;
        if (all_roles[static_cast<std::size_t>(j)] != horizon_role_t::individual)
            return "parent " + n + " of " + name
                 + " is itself a common horizon; chains are not supported";
        // A repeated parent would make the trigger compare a surface with
        // itself, where the strict inequality in surfaces_overlap() is decided
        // by round-off.
        for (int prev : out)
            if (prev == j)
                return "common horizon " + name + " lists parent " + n + " twice";

        out.push_back(j);
    }
    return {};
}

}  // namespace detail

class horizon_manager_impl_t {
public:
    /// Search for every horizon due at the current iteration, and write output.
    /// Safe (and cheap) to call unconditionally every step.
    void find_and_write();

    bool is_active()  const { return _active; }
    int  n_horizons() const { return static_cast<int>(_horizons.size()); }

    horizon_entry_t const& get(int i) const { return *_horizons[i]; }
    horizon_entry_t&       get(int i)       { return *_horizons[i]; }

    /// Index of the horizon with this name, or -1.
    int index_of(std::string const& name) const;

    /// Search a single horizon now, regardless of cadence. Collective.
    /// Exposed for tests and for callers that drive the schedule themselves.
    horizon_result_t find_now(int i);

    /// Whether horizon `e` is scheduled to be searched at this iteration.
    /// Depends only on the iteration counter and on state every rank agrees
    /// on -- the search below it is collective, so a rank-local answer here
    /// would deadlock.
    bool due(horizon_entry_t const& e, std::size_t iter) const;

    /// Test a dormant common horizon's trigger and, if its parents now overlap,
    /// seed it from them and move it to `searching`. Returns true if it fired.
    /// Exposed so a test can drive the transition without a full step.
    bool maybe_trigger_common(int i);

    /// Wake a dormant individual horizon once `alp_min`, the global minimum
    /// of the lapse, has fallen below its `activation_lapse`. Returns true if
    /// it fired. Every rank passes the same reduced value, so every rank makes
    /// the same decision and the collective search below stays in lockstep.
    /// Exposed so a test can drive the transition with a chosen lapse.
    bool maybe_activate_on_lapse(int i, double alp_min);

    /// Hand the located horizons to the matter excision as a set of inscribed
    /// spheres. A no-op unless grmhd.excision.excision_criterion is "horizon".
    /// Called at the end of find_and_write(); exposed so a test can check the
    /// published region against the surface it came from.
    void publish_excision_region();

    /// Flat buffer of the last *successful* result, plus the previous
    /// centroid.
    ///
    /// Only the fields the manager actually reads back: the common-horizon
    /// trigger and seed, and the motion estimate the excision margin uses.
    /// Kept next to the code that consumes them rather than spelled out in the
    /// checkpoint writer, so the two cannot drift apart. Read is
    /// length-tolerant, so a checkpoint written before a field was added loads
    /// with that field defaulted rather than aborting.
    std::vector<double> serialize_last_good(int i) const;
    void deserialize_last_good(int i, std::vector<double> const& buf);


private:
    /// Seed a common horizon's surface from its parents' current shapes.
    void seed_common_from_parents(horizon_entry_t& e);

    /// The most recent successful result of horizon `p`, if it exists and is
    /// not older than that horizon's `parent_staleness_limit`.
    bool parent_reference(int p, std::size_t staleness_limit,
                          horizon_result_t& out) const;
    /// Fold one search outcome into the entry's state machine.
    void record_outcome(horizon_entry_t& e, bool found);

    /// Sample the matter on a just-found horizon and fill `e.last.fluxes`.
    /// A no-op unless `apparent_horizon.compute_fluxes` is set. Collective.
    void compute_fluxes(horizon_entry_t& e);

    /// Non-const: latches `shape_written_this_run` on the first shape write.
    void write_output(horizon_entry_t& e);
    std::filesystem::path series_path(horizon_entry_t const& e) const;
    /// Directory holding the per-find shape files. Created eagerly at
    /// construction rather than left to the writer, so it exists on a restart
    /// too -- see `_shape_written_this_run`.
    std::filesystem::path shape_dir() const;
    std::filesystem::path shape_path(horizon_entry_t const& e, std::size_t iter,
                                     char const* extension) const;

    std::vector<std::unique_ptr<horizon_entry_t>> _horizons;
    std::unique_ptr<grid_geometry_provider_t>     _provider;
    std::unique_ptr<grid_matter_provider_t>       _matter;
    bool _want_fluxes      = false;
    bool _want_constraints = false;
    bool   _predict_origin = false;
    bool   _excise        = false;
    double _excise_factor = 0.8;

    int  _find_every   = -1;
    bool _active       = false;
    bool _verbose      = false;
    bool        _output_shape = true;
    std::string _shape_format = "hdf5";
    int         _shape_every  = -1;

protected:
    horizon_manager_impl_t();
    ~horizon_manager_impl_t() = default;

    /// Checkpoint support: flat coefficient buffer for horizon i.
    std::vector<double> serialize_horizon(int i) const;
    /// Restore horizon i. False if the buffer does not match this l_max.
    bool deserialize_horizon(int i, std::vector<double> const& buf, bool ever_found);

    bool horizon_ever_found(int i) const { return _horizons[i]->ever_found; }
    bool horizon_has_existed(int i) const { return _horizons[i]->has_existed; }

    /// Life-cycle state, packed for the checkpoint. Without it a restart after
    /// merger would send a common horizon back to dormant and re-trigger it,
    /// and would forget that a vanished horizon had been abandoned.
    int  horizon_state_code(int i) const {
        return static_cast<int>(_horizons[i]->state);
    }
    int  horizon_failure_count(int i) const {
        return _horizons[i]->consecutive_failures;
    }
    void set_horizon_state(int i, int state_code, int failures, int has_existed) {
        _horizons[i]->state = static_cast<horizon_state_t>(state_code);
        _horizons[i]->consecutive_failures = failures;
        _horizons[i]->has_existed = (has_existed != 0);
    }

    static constexpr unsigned long longevity =
        unique_objects_lifetimes::GRACE_SPHERICAL_SURFACES;

    friend class utils::singleton_holder<horizon_manager_impl_t>;
    friend class memory::new_delete_creator<horizon_manager_impl_t,
                                           memory::new_delete_allocator>;
    friend class grace::checkpoint_handler_impl_t;
};

using horizon_manager = utils::singleton_holder<horizon_manager_impl_t>;

}  // namespace grace

#endif  // GRACE_GEOMETRY_HORIZON_MANAGER_HH
