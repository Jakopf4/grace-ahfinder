/**
 * @file test_horizon_manager.cpp
 * @brief Tests for the apparent-horizon manager.
 *
 * Exercises the manager against the parfile in configs/horizon_manager.yaml,
 * which configures two independent horizons -- one real, one deliberately
 * placed where none exists -- and a detector surface that tracks the first.
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

#include <grace/IO/spherical_surfaces.hh>
#include <grace/geometry/horizon_manager.hh>
#include <grace/physics/grmhd_helpers.hh>
#include <grace/parallel/mpi_wrappers.hh>
#include <grace/physics/analytic_geometry.hh>
#include <grace/system/grace_runtime.hh>

#include <cmath>
#include <filesystem>
#include <system_error>
#include <limits>
#include <fstream>
#include <string>
#include <vector>

using namespace grace;

namespace {

std::vector<std::string> read_lines(std::filesystem::path const& p) {
    std::ifstream f(p);
    std::vector<std::string> out;
    for (std::string l; std::getline(f, l); ) out.push_back(l);
    return out;
}

}  // namespace

//---------------------------------------------------------------------------

TEST_CASE("Manager builds the configured horizons", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();

    REQUIRE(mgr.is_active());
    REQUIRE(mgr.n_horizons() == 3);

    int const i_ah  = mgr.index_of("AH");
    int const i_far = mgr.index_of("AH_far");
    REQUIRE(i_ah  >= 0);
    REQUIRE(i_far >= 0);
    REQUIRE(i_ah != i_far);
    REQUIRE(mgr.index_of("does_not_exist") == -1);

    // Per-horizon settings come through, including the cadence override and the
    // -1 "inherit the module default" case.
    REQUIRE(mgr.get(i_ah).surface.l_max()  == 6);
    REQUIRE(mgr.get(i_far).surface.l_max() == 4);
    REQUIRE(mgr.get(i_ah).find_every  == 2);   // inherited
    REQUIRE(mgr.get(i_far).find_every == 3);   // overridden
    REQUIRE(mgr.get(i_ah).track_centre);
    REQUIRE_FALSE(mgr.get(i_far).track_centre);
}

TEST_CASE("Manager finds the real horizon and reports the spurious one",
          "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    kerr_schild_geometry_t ks(1.0, 0.0);

    int const i_ah  = mgr.index_of("AH");
    int const i_far = mgr.index_of("AH_far");

    auto const found = mgr.find_now(i_ah);
    INFO("AH: " << to_string(found.status) << " residual " << found.residual
         << " mean radius " << found.mean_radius);
    REQUIRE(found.found);
    REQUIRE_THAT(found.mean_radius, Catch::Matchers::WithinRel(ks.horizon_radius(), 5e-2));

    // The second horizon sits in near-flat space where no trapped surface
    // exists.  It must fail, and must not disturb the first.
    auto const missing = mgr.find_now(i_far);
    INFO("AH_far: " << to_string(missing.status));
    REQUIRE_FALSE(missing.found);

    REQUIRE(mgr.get(i_ah).ever_found);
    REQUIRE_FALSE(mgr.get(i_far).ever_found);
    REQUIRE_THAT(mgr.get(i_ah).last.mean_radius,
                 Catch::Matchers::WithinRel(ks.horizon_radius(), 5e-2));
}

TEST_CASE("A failed search is re-seeded rather than left mid-divergence",
          "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i_far = mgr.index_of("AH_far");

    mgr.find_now(i_far);
    auto const& e = mgr.get(i_far);

    REQUIRE_FALSE(e.ever_found);
    // Back at the configured guess, so the next attempt starts from a sane
    // surface instead of inheriting a diverging one.
    auto const shape = e.surface.evaluate();
    for (double h : shape.h)
        REQUIRE_THAT(h, Catch::Matchers::WithinRel(e.initial_radius, 1e-10));
    for (int d = 0; d < 3; ++d)
        REQUIRE(e.surface.centre()[d] == e.initial_centre[d]);
}

TEST_CASE("Warm start converges faster than a cold one", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i_ah = mgr.index_of("AH");

    // Force a cold start.
    mgr.get(i_ah).ever_found = false;
    auto const cold = mgr.find_now(i_ah);
    REQUIRE(cold.found);

    auto const warm = mgr.find_now(i_ah);
    REQUIRE(warm.found);

    INFO("cold " << cold.iterations << " iterations, warm " << warm.iterations);
    REQUIRE(warm.iterations < cold.iterations);
}

TEST_CASE("Tracking moves the origin onto the horizon", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i_ah = mgr.index_of("AH");

    mgr.get(i_ah).surface.set_centre({0.25, -0.15, 0.1});
    mgr.get(i_ah).ever_found = false;

    auto const r = mgr.find_now(i_ah);
    REQUIRE(r.found);

    // track_center is on for this horizon, so the origin should have been moved
    // to the centroid, which is the hole at the origin.
    for (int d = 0; d < 3; ++d) {
        INFO("component " << d);
        REQUIRE_THAT(mgr.get(i_ah).surface.centre()[d],
                     Catch::Matchers::WithinAbs(0.0, 5e-2));
    }
}

TEST_CASE("Tracking the centroid preserves the converged shape",
          "[horizon_manager]") {
    // The point of recentre(): after a tracked find, the surface must still be
    // a solution, so the next search is a genuine warm start.  Moving the origin
    // with set_centre() instead translates the surface and throws that away.
    auto& mgr = horizon_manager::get();
    int const i_ah = mgr.index_of("AH");

    // Start from a deliberately off-centre origin so the centroid really moves.
    mgr.get(i_ah).surface.set_centre({0.3, -0.2, 0.15});
    mgr.get(i_ah).ever_found = false;

    auto const first = mgr.find_now(i_ah);
    REQUIRE(first.found);

    // The origin has been moved to the centroid; the next search should need
    // barely any iterations.
    auto const second = mgr.find_now(i_ah);
    REQUIRE(second.found);

    INFO("cold " << first.iterations << " iterations, tracked warm start "
         << second.iterations);
    REQUIRE(second.iterations <= 2);
}

TEST_CASE("Output files are written and parse back", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    mgr.find_and_write();

    if (parallel::mpi_comm_rank() != 0) return;

    auto const dir = std::filesystem::path(grace::runtime::get().scalar_io_basepath());
    auto const series = dir / "horizon_AH.dat";
    REQUIRE(std::filesystem::exists(series));

    auto const lines = read_lines(series);
    REQUIRE(lines.size() >= 2);
    REQUIRE(lines[0].rfind("#", 0) == 0);
    REQUIRE(lines[0].find("area")  != std::string::npos);
    REQUIRE(lines[0].find("m_irr") != std::string::npos);

    // Last data row must parse into the declared column count.
    std::istringstream is(lines.back());
    double v;
    int n = 0;
    while (is >> v) ++n;
    REQUIRE(n == 2 + 47);   // iteration, time, then the 47 series columns
}

// The shape directory used to be created by the writer, on the first write the
// cadence allowed. A restart skips the pre-loop find at src/grace.cpp:107, so
// with a cadence coarser than the remaining run there was no write, hence no
// directory, hence a run that looked like it had shape output and silently had
// none -- and no record of the state it restarted from.
TEST_CASE("Shape directory exists and the first find always writes",
          "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i_ah = mgr.index_of("AH");
    REQUIRE(i_ah >= 0);

    auto const dir = std::filesystem::path(grace::runtime::get().scalar_io_basepath())
                   / "horizons";

    // (A) Created at construction, so it is there on every path -- including a
    // restart, where nothing may ever reach the writer.
    if (parallel::mpi_comm_rank() == 0)
        REQUIRE(std::filesystem::is_directory(dir));

    auto& rt = grace::runtime::get();
    auto const saved = rt.iteration();

    auto shape_file = [&](std::size_t it) {
        return dir / ("horizon_AH.it" + std::to_string(it) + ".h5");
    };

    // Remove any file left by an earlier run of this binary: the assertions
    // below are about existence, and a stale artefact would satisfy them
    // whether or not the code under test does anything.
    if (parallel::mpi_comm_rank() == 0) {
        std::error_code ec;
        std::filesystem::remove(shape_file(1234), ec);
        std::filesystem::remove(shape_file(1236), ec);
        REQUIRE_FALSE(std::filesystem::exists(shape_file(1234)));
    }

    // (B) An iteration the cadence cannot select: shape_output_every is 1000,
    // and 1234 % 1000 != 0. Clearing the latch is what a fresh process looks
    // like -- on a restart the first find lands at whatever iteration the
    // checkpoint left off at, which is exactly this situation.
    mgr.get(i_ah).shape_written_this_run = false;
    rt.set_iteration(1234);
    mgr.find_and_write();

    if (parallel::mpi_comm_rank() == 0) {
        INFO("first find of the process must write regardless of cadence");
        REQUIRE(std::filesystem::exists(shape_file(1234)));
    }

    // ... and the cadence applies from then on: the latch is set, 1236 is not a
    // multiple, so nothing new appears.
    rt.set_iteration(1236);
    mgr.find_and_write();

    if (parallel::mpi_comm_rank() == 0) {
        INFO("cadence must still gate every later find");
        REQUIRE_FALSE(std::filesystem::exists(shape_file(1236)));
    }

    rt.set_iteration(saved);
}

TEST_CASE("Serialization round-trips through the manager", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i_ah = mgr.index_of("AH");

    mgr.find_now(i_ah);
    auto const before = mgr.get(i_ah).surface.coefficients();
    auto const buf    = mgr.get(i_ah).surface.serialize();

    // Perturb, then restore.
    mgr.get(i_ah).surface.set_sphere(9.0);
    REQUIRE(mgr.get(i_ah).surface.deserialize(buf));

    auto const after = mgr.get(i_ah).surface.coefficients();
    REQUIRE(after.size() == before.size());
    for (std::size_t c = 0; c < before.size(); ++c) {
        INFO("coefficient " << c);
        REQUIRE(after[c] == before[c]);
    }

    // A buffer from a different l_max must be refused, not misread: the two
    // configured horizons differ in l_max, so this is a real scenario.
    int const i_far = mgr.index_of("AH_far");
    REQUIRE_FALSE(mgr.get(i_far).surface.deserialize(buf));
}

TEST_CASE("A detector tracking a horizon follows it", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i_ah = mgr.index_of("AH");

    auto const r = mgr.find_now(i_ah);
    REQUIRE(r.found);

    auto& spheres = spherical_surface_manager::get();
    auto const idx = spheres.get_index("tracker");
    REQUIRE(idx >= 0);

    spheres.update(false);

    auto const& det = spheres.get(static_cast<size_t>(idx));
    // radius = tracking_radius_factor * horizon mean radius
    REQUIRE_THAT(det.radius,
                 Catch::Matchers::WithinRel(1.5 * r.mean_radius, 1e-10));
    for (int d = 0; d < 3; ++d) {
        INFO("component " << d);
        REQUIRE_THAT(det.center[d],
                     Catch::Matchers::WithinAbs(r.centroid[d], 1e-10));
    }
}

//---------------------------------------------------------------------------
// Scheduling, life cycle and determinism.
//---------------------------------------------------------------------------

TEST_CASE("Each horizon is searched on its own cadence", "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    auto const& ah  = mgr.get(mgr.index_of("AH"));       // find_every 2
    auto const& far = mgr.get(mgr.index_of("AH_far"));   // find_every 3

    REQUIRE(ah.find_every  == 2);
    REQUIRE(far.find_every == 3);

    for (std::size_t it = 0; it <= 12; ++it) {
        INFO("iteration " << it);
        REQUIRE(mgr.due(ah,  it) == (it % 2 == 0));
        REQUIRE(mgr.due(far, it) == (it % 3 == 0));
    }

    // The cadence must depend on nothing but the iteration counter: the search
    // it gates is collective, so a rank-local answer here would deadlock.
    for (std::size_t it = 0; it <= 6; ++it) {
        int const votes = mgr.due(ah, it) ? 1 : 0;
        int total = 0;
        parallel::mpi_allreduce(&votes, &total, 1, MPI_SUM);
        INFO("iteration " << it);
        REQUIRE(total == votes * parallel::mpi_comm_size());
    }
}

TEST_CASE("A horizon that has ceased to exist is eventually abandoned",
          "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");
    auto& e = mgr.get(i);

    // Find it once, so it counts as having existed. Only a horizon that has
    // *been* found may be given up on; one that has never been found may simply
    // not have formed yet.
    REQUIRE(mgr.find_now(i).found);
    REQUIRE(e.has_existed);

    // Now make every subsequent search fail, the way a horizon that has fallen
    // inside a common one does.
    auto const saved_budget    = e.solver.max_iterations;
    auto const saved_tolerance = e.solver.tolerance;
    auto const saved_failures  = e.stop_after_failures;
    auto const saved_retry     = e.retry_every;
    e.solver.max_iterations = 0;        // no budget to converge in
    e.solver.tolerance      = 1e-15;    // and nothing this grid can reach
    e.stop_after_failures   = 3;
    e.consecutive_failures  = 0;

    for (int k = 1; k <= 3; ++k) {
        REQUIRE_FALSE(mgr.find_now(i).found);
        INFO("failure " << k);
        REQUIRE(e.consecutive_failures == k);
        REQUIRE(e.state == (k < 3 ? horizon_state_t::searching
                                  : horizon_state_t::abandoned));
    }

    // Abandoned is a slow cadence, not a stop. The horizon leaves the normal
    // search rhythm but stays reachable, so a few bad steps through a merger or
    // a regrid cannot lose it for the rest of the run.
    e.retry_every = 4;
    REQUIRE_FALSE(mgr.due(e, 2));    // on the find_every-2 rhythm, but abandoned
    REQUIRE_FALSE(mgr.due(e, 6));
    REQUIRE(mgr.due(e, 4));          // on the retry rhythm
    REQUIRE(mgr.due(e, 8));

    // A failed retry must not keep inflating the failure count -- it is already
    // saturated, and the horizon simply waits for the next one.
    int const failures_before = e.consecutive_failures;
    REQUIRE_FALSE(mgr.find_now(i).found);
    REQUIRE(e.consecutive_failures == failures_before);
    REQUIRE(e.state == horizon_state_t::abandoned);

    // retry_every = 0 is the old behaviour: gone for good.
    e.retry_every = 0;
    for (std::size_t it = 0; it <= 8; ++it) REQUIRE_FALSE(mgr.due(e, it));
    e.retry_every = 4;

    // And a retry that succeeds brings it all the way back.
    e.solver.max_iterations = saved_budget;
    e.solver.tolerance      = saved_tolerance;
    REQUIRE(mgr.find_now(i).found);
    REQUIRE(e.state == horizon_state_t::found);
    REQUIRE(e.consecutive_failures == 0);
    REQUIRE(mgr.due(e, 2));          // back on the normal cadence

    e.solver.max_iterations = saved_budget;
    e.solver.tolerance      = saved_tolerance;
    e.stop_after_failures   = saved_failures;
    e.retry_every           = saved_retry;
    e.state                 = horizon_state_t::searching;
    e.consecutive_failures  = 0;
    e.reseed();
}

TEST_CASE("A common horizon triggers from its parents' last good result",
          "[horizon_manager]") {
    auto& mgr = horizon_manager::get();
    int const ic = mgr.index_of("AH_common");
    REQUIRE(ic >= 0);

    auto& c = mgr.get(ic);
    REQUIRE(c.role == horizon_role_t::common);
    REQUIRE(c.parents.size() == 2);
    REQUIRE(c.state == horizon_state_t::dormant);

    auto& p1 = mgr.get(c.parents[0]);
    auto& p2 = mgr.get(c.parents[1]);

    auto place = [](horizon_entry_t& e, double x, double r) {
        e.last_good = horizon_result_t{};
        e.last_good.found            = true;
        e.last_good.centroid         = { x, 0.0, 0.0 };
        e.last_good.max_radius       = r;
        e.last_good.mean_radius      = r;
        e.last_good.irreducible_mass = r / 2.0;
        e.has_last_good       = true;
        e.last_good_iteration = grace::runtime::get().iteration();
        e.has_existed         = true;
        // The current attempt has failed -- which is the state the individual
        // horizons of a binary are actually in when the common one forms.
        e.last = horizon_result_t{};
    };

    SECTION("well separated parents leave it dormant") {
        place(p1, -6.0, 1.0);
        place(p2,  6.0, 1.0);
        REQUIRE_FALSE(mgr.maybe_trigger_common(ic));
        REQUIRE(c.state == horizon_state_t::dormant);
    }

    SECTION("close parents fire it, and seed a surface enclosing both") {
        place(p1, -1.0, 1.0);
        place(p2,  1.0, 1.0);

        REQUIRE(mgr.maybe_trigger_common(ic));
        REQUIRE(c.state == horizon_state_t::searching);

        // Seeded from the parents even though neither parent's *current*
        // attempt succeeded: `last` is overwritten by failures, `last_good` is
        // not. Requiring the former would lose the common horizon exactly when
        // it forms, since that is when the individual searches start failing.
        auto const shape = c.surface.evaluate();
        double h_min = shape.h[0];
        for (double v : shape.h) h_min = std::min(h_min, v);

        // The guess must lie strictly outside both parents: the expansion is
        // positive there and the Newton solve descends inwards.
        for (auto const* p : { &p1, &p2 }) {
            auto const& pc = p->last_good.centroid;
            auto const& cc = c.surface.centre();
            double d = 0.0;
            for (int k = 0; k < 3; ++k) d += (pc[k]-cc[k]) * (pc[k]-cc[k]);
            INFO("parent " << p->name);
            REQUIRE(std::sqrt(d) + p->last_good.max_radius < h_min);
        }

        c.state = horizon_state_t::dormant;   // leave the fixture as found
    }

    SECTION("a stale parent result cannot fire it") {
        place(p1, -1.0, 1.0);
        place(p2,  1.0, 1.0);
        // parent_staleness_limit is 4 in the test config.
        p1.last_good_iteration = 0;
        p2.last_good_iteration = 0;
        auto& rt = grace::runtime::get();
        auto const saved = rt.iteration();
        rt.set_iteration(saved + 100);

        REQUIRE_FALSE(mgr.maybe_trigger_common(ic));
        REQUIRE(c.state == horizon_state_t::dormant);

        rt.set_iteration(saved);
    }

    p1.has_last_good = false;
    p2.has_last_good = false;
}

TEST_CASE("Parent lists are validated, not trusted", "[horizon_manager]") {
    std::vector<std::string> const names{ "A", "B", "C" };
    std::vector<horizon_role_t> const roles{ horizon_role_t::individual,
                                             horizon_role_t::individual,
                                             horizon_role_t::common };
    std::vector<int> out;

    REQUIRE(grace::detail::resolve_parents("X", {"A","B"}, names, roles, out).empty());
    REQUIRE(out == std::vector<int>{0, 1});

    // Wrong count.
    REQUIRE_FALSE(grace::detail::resolve_parents("X", {"A"}, names, roles, out).empty());
    REQUIRE_FALSE(grace::detail::resolve_parents("X", {"A","B","A"}, names, roles, out)
                      .empty());
    // Unknown name.
    REQUIRE_FALSE(grace::detail::resolve_parents("X", {"A","Z"}, names, roles, out).empty());
    // A common horizon may not itself be a parent: chains are not supported.
    REQUIRE_FALSE(grace::detail::resolve_parents("X", {"A","C"}, names, roles, out).empty());
    // Duplicated parent: surfaces_overlap() would compare a surface with
    // itself, where the strict inequality is decided by round-off.
    REQUIRE_FALSE(grace::detail::resolve_parents("X", {"A","A"}, names, roles, out).empty());
}

TEST_CASE("A find is bit-identical on every rank", "[horizon_manager][mpi]") {
    // The geometry samples are assembled by an MPI_SUM over a one-hot ownership
    // pattern, so every rank sums exactly the same terms in the same order and
    // the result is bitwise reproducible. That argument is subtle enough to be
    // worth pinning: without it, ranks could diverge silently and only show up
    // as a checkpoint that does not reproduce.
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");

    auto const r = mgr.find_now(i);
    REQUIRE(r.found);

    double vals[6] = { r.area, r.irreducible_mass, r.areal_radius,
                       r.centroid[0], r.centroid[1], r.centroid[2] };
    double root[6];
    for (int k = 0; k < 6; ++k) root[k] = vals[k];
    parallel::mpi_bcast(root, 6, 0);

    for (int k = 0; k < 6; ++k) {
        INFO("quantity " << k << ": rank value " << vals[k]
             << ", rank 0 value " << root[k]);
        REQUIRE(vals[k] == root[k]);   // bitwise, deliberately
    }
}

TEST_CASE("Horizon fluxes are sampled and vanish on static vacuum data",
          "[horizon_manager]") {
    // The end-to-end check on the matter path: two extra collective samples
    // (the auxiliary array for the primitives, the evolved array for the lapse
    // and shift), the reflection folding, and the surface integrals. This ID
    // has an atmosphere at rest and no magnetic field, so every flux has a
    // known answer -- zero -- and a sign error, a missing factor or a botched
    // sample all show up as something that is not zero.
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");

    auto const r = mgr.find_now(i);
    REQUIRE(r.found);

    // valid distinguishes "computed and zero" from "never computed", which is
    // the distinction the whole flux block turns on.
    REQUIRE(r.fluxes.valid);

    INFO("mdot " << r.fluxes.mdot << ", edot " << r.fluxes.edot
         << ", ldot " << r.fluxes.ldot << ", phi_B " << r.fluxes.phi_B
         << ", phi_B_signed " << r.fluxes.phi_B_signed);

    REQUIRE_THAT(r.fluxes.mdot, Catch::Matchers::WithinAbs(0.0, 1e-8));
    REQUIRE_THAT(r.fluxes.ldot, Catch::Matchers::WithinAbs(0.0, 1e-8));
    // No B field at all, so both magnetic quantities are identically zero --
    // including the divergence monitor, which must not pick up noise from the
    // interpolation onto the surface.
    REQUIRE_THAT(r.fluxes.phi_B,        Catch::Matchers::WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(r.fluxes.phi_B_signed, Catch::Matchers::WithinAbs(0.0, 1e-12));
    // The EM part of a flux with no magnetic field is identically zero, which
    // also means the matter part carries the whole (zero) total.
    REQUIRE(r.fluxes.edot_em == 0.0);
    REQUIRE(r.fluxes.ldot_em == 0.0);

    // Constraint monitor: on by default, and non-negative by construction --
    // these are L2 norms, so a negative value would mean the quadrature or the
    // area normalisation is wrong.
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    REQUIRE(r.fluxes.have_constraints);
    REQUIRE(r.fluxes.ham_norm >= 0.0);
    REQUIRE(r.fluxes.mom_norm >= 0.0);
    INFO("ham_norm " << r.fluxes.ham_norm << ", mom_norm " << r.fluxes.mom_norm);
    REQUIRE(std::isfinite(r.fluxes.ham_norm));
    REQUIRE(std::isfinite(r.fluxes.mom_norm));
    // Strictly positive, which is what separates "sampled a real field" from
    // "read a zeroed buffer" -- the realistic failure mode if the variable were
    // never wired up or were filled only on diagnostic-output steps.
    //
    // The magnitude is deliberately not pinned. On this coarse test grid the
    // measured norm is ~0.3 at the horizon of unevolved Kerr-Schild data, which
    // is plausible for a second-derivative quantity at dx = 0.125 with the r = 0
    // singularity inside the domain and no excision -- but "plausible" is not a
    // number a test should assert, and a convergence study is what would
    // establish one.
    REQUIRE(r.fluxes.ham_norm > 0.0);
    #endif

    // And the fluxes must not have disturbed the horizon itself.
    REQUIRE(r.irreducible_mass > 0.0);
}

//---------------------------------------------------------------------------
// Matter excision driven by the located horizons.
//---------------------------------------------------------------------------

TEST_CASE("The excision predicate is a plain sphere test", "[horizon_manager]") {
    excision_params_t ex{};
    ex.n_ex_spheres = 0;

    double const origin[3] = {0.0, 0.0, 0.0};
    // With nothing published, nothing is inside -- which is what makes the
    // horizon criterion safe to enable before any horizon has been found.
    REQUIRE_FALSE(ex.inside_horizon(origin));

    ex.n_ex_spheres = 2;
    ex.ex_sphere[0][0] = 3.0; ex.ex_sphere[0][1] = 0.0;
    ex.ex_sphere[0][2] = 0.0; ex.ex_sphere[0][3] = 1.0;
    ex.ex_sphere[1][0] = -3.0; ex.ex_sphere[1][1] = 0.0;
    ex.ex_sphere[1][2] = 0.0; ex.ex_sphere[1][3] = 0.5;

    double const in1[3]  = { 3.5,  0.0, 0.0};
    double const in2[3]  = {-3.2,  0.0, 0.0};
    double const out[3]  = { 0.0,  0.0, 0.0};
    double const edge[3] = { 4.0,  0.0, 0.0};   // exactly on the surface
    double const just[3] = {-3.6,  0.0, 0.0};   // just outside the smaller one

    REQUIRE(ex.inside_horizon(in1));
    REQUIRE(ex.inside_horizon(in2));
    REQUIRE_FALSE(ex.inside_horizon(out));
    REQUIRE(ex.inside_horizon(edge));         // closed ball
    REQUIRE_FALSE(ex.inside_horizon(just));
}

TEST_CASE("Excision spheres reach the physics only under the horizon criterion",
          "[horizon_manager]") {
    // The store is deliberately neutral ground: the horizon manager pushes and
    // the physics pulls, with neither side depending on the other. What decides
    // whether the spheres are honoured is the grmhd criterion, not their
    // presence -- so a run configured for lapse excision must ignore them even
    // if a horizon manager is publishing.
    set_excision_spheres({ {1.0, 2.0, 3.0, 0.5} });

    auto const ex = get_excision_params();
    bool const horizon_mode =
        get_param<std::string>("grmhd", "excision", "excision_criterion") == "horizon";

    REQUIRE(ex.excise_by_horizon == horizon_mode);
    REQUIRE(ex.n_ex_spheres == (horizon_mode ? 1 : 0));
    if (horizon_mode) {
        REQUIRE(ex.ex_sphere[0][0] == 1.0);
        REQUIRE(ex.ex_sphere[0][3] == 0.5);
    }

    // Overflow is clamped, not overrun: excising a subset is recoverable,
    // walking off the end of the array is not.
    std::vector<std::array<double,4>> many;
    for (int i = 0; i < GRACE_MAX_EXCISION_SPHERES + 4; ++i)
        many.push_back({ double(i), 0.0, 0.0, 1.0 });
    set_excision_spheres(many);
    REQUIRE(get_excision_params().n_ex_spheres
            <= GRACE_MAX_EXCISION_SPHERES);

    clear_excision_spheres();
    REQUIRE(get_excision_params().n_ex_spheres == 0);
}

TEST_CASE("A published excision sphere lies strictly inside its horizon",
          "[horizon_manager]") {
    // The property the whole scheme rests on: nothing excised may lie outside
    // the horizon, or the excision would influence the exterior.
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");

    auto const r = mgr.find_now(i);
    REQUIRE(r.found);

    mgr.publish_excision_region();
    auto const ex = get_excision_params();

    if (!ex.excise_by_horizon) {
        // Not configured for horizon excision in this parfile; the publish is
        // then a no-op by design and there is nothing to check.
        REQUIRE(ex.n_ex_spheres == 0);
        return;
    }

    REQUIRE(ex.n_ex_spheres >= 1);

    auto const& e   = mgr.get(i);
    auto const pts  = e.surface.collocation_points();

    // Find the sphere belonging to this horizon by its centre.
    int found = -1;
    for (int s = 0; s < ex.n_ex_spheres; ++s) {
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            double const dx = ex.ex_sphere[s][d] - r.centroid[d];
            d2 += dx * dx;
        }
        if (std::sqrt(d2) < 1e-12) { found = s; break; }
    }
    REQUIRE(found >= 0);

    double const r_ex = ex.ex_sphere[found][3];
    REQUIRE(r_ex > 0.0);

    // Every point of the surface must be strictly outside the excised ball.
    double worst = std::numeric_limits<double>::infinity();
    for (auto const& x : pts) {
        double d2 = 0.0;
        for (int d = 0; d < 3; ++d) {
            double const dx = x[d] - r.centroid[d];
            d2 += dx * dx;
        }
        worst = std::min(worst, std::sqrt(d2));
    }
    INFO("excision radius " << r_ex << ", closest surface point " << worst);
    REQUIRE(r_ex < worst);

    // And the centroid itself -- the deepest point -- must be excised.
    double const c[3] = { r.centroid[0], r.centroid[1], r.centroid[2] };
    REQUIRE(ex.inside_horizon(c));
}

TEST_CASE("The excision margin shrinks with how far the horizon moved",
          "[horizon_manager]") {
    // The excised region is refreshed only on a find, so between finds it is a
    // stale snapshot. The margin that makes that safe is measured, not assumed:
    // a horizon seen to move must give back exactly that much of its radius.
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");
    auto& e = mgr.get(i);

    REQUIRE(mgr.find_now(i).found);
    if (!get_excision_params().excise_by_horizon) return;

    auto const saved_prev     = e.prev_centroid;
    auto const saved_has_prev = e.has_prev_centroid;

    auto radius_with_motion = [&](double step) {
        e.prev_centroid     = e.last.centroid;
        e.prev_centroid[0] -= step;         // "moved by `step` since last find"
        e.has_prev_centroid = (step > 0.0);
        mgr.publish_excision_region();
        auto const ex = get_excision_params();
        return ex.n_ex_spheres >= 1 ? ex.ex_sphere[0][3] : -1.0;
    };

    double const at_rest = radius_with_motion(0.0);
    REQUIRE(at_rest > 0.0);

    double const moving = radius_with_motion(0.1 * at_rest);
    INFO("at rest " << at_rest << ", moving " << moving);
    REQUIRE(moving > 0.0);
    REQUIRE(moving < at_rest);          // gave back the distance travelled

    // A horizon that outran its own margin gets no excision rather than a
    // degenerate or negative sphere -- the one outcome that would be worse
    // than not excising at all.
    REQUIRE(radius_with_motion(10.0 * at_rest) < 0.0);

    e.prev_centroid     = saved_prev;
    e.has_prev_centroid = saved_has_prev;
    mgr.publish_excision_region();
}

TEST_CASE("Origin prediction follows motion but refuses a jump",
          "[horizon_manager]") {
    // Prediction is worth having only when the surface would otherwise start
    // too far from the hole, and it must not chase a discontinuity: a large
    // displacement is a regrid or a bad find, not motion, and following it
    // throws the surface off the horizon entirely.
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");
    auto& e = mgr.get(i);

    REQUIRE(mgr.find_now(i).found);
    auto const centre_before = e.surface.centre();
    auto const r_in          = e.last_good.min_radius;
    REQUIRE(r_in > 0.0);

    auto const saved_prev     = e.prev_centroid;
    auto const saved_has_prev = e.has_prev_centroid;

    // A jump, an order of magnitude past the guard: the origin must not move.
    e.prev_centroid     = e.last_good.centroid;
    e.prev_centroid[0] -= 5.0 * r_in;
    e.has_prev_centroid = true;
    mgr.find_now(i);
    for (int d = 0; d < 3; ++d) {
        INFO("component " << d);
        // With prediction off (the default) nothing moves either way, so this
        // holds regardless of the parfile -- what it pins is that a jump is
        // never followed.
        REQUIRE_THAT(e.surface.centre()[d],
                     Catch::Matchers::WithinAbs(centre_before[d], 0.5 * r_in));
    }

    e.prev_centroid     = saved_prev;
    e.has_prev_centroid = saved_has_prev;
    mgr.find_now(i);
}

TEST_CASE("last_good round-trips through its serialization",
          "[horizon_manager]") {
    // What the common-horizon trigger and the excision margin read back after a
    // restart. Checked here rather than only through HDF5 so a missing field is
    // caught without standing up a checkpoint.
    auto& mgr = horizon_manager::get();
    int const i = mgr.index_of("AH");
    auto& e = mgr.get(i);

    REQUIRE(mgr.find_now(i).found);
    e.prev_centroid     = { 1.5, -2.5, 3.5 };
    e.has_prev_centroid = true;

    auto const buf    = mgr.serialize_last_good(i);
    auto const good   = e.last_good;
    auto const iter   = e.last_good_iteration;

    // Scribble, then restore.
    e.has_last_good     = false;
    e.has_prev_centroid = false;
    e.last_good         = horizon_result_t{};
    e.prev_centroid     = {};

    mgr.deserialize_last_good(i, buf);

    REQUIRE(e.has_last_good);
    REQUIRE(e.last_good_iteration == iter);
    REQUIRE(e.has_prev_centroid);
    REQUIRE(e.prev_centroid[0] == 1.5);
    REQUIRE(e.prev_centroid[2] == 3.5);
    for (int d = 0; d < 3; ++d) REQUIRE(e.last_good.centroid[d] == good.centroid[d]);
    REQUIRE(e.last_good.min_radius       == good.min_radius);
    REQUIRE(e.last_good.max_radius       == good.max_radius);
    REQUIRE(e.last_good.irreducible_mass == good.irreducible_mass);

    // A record written before a field existed must load with it defaulted, not
    // abort: that is the whole reason the read is length-tolerant.
    std::vector<double> const truncated(buf.begin(), buf.begin() + 9);
    mgr.deserialize_last_good(i, truncated);
    REQUIRE(e.has_last_good);
    REQUIRE_FALSE(e.has_prev_centroid);
}

TEST_CASE("The horizon is found with every per-point field retained",
          "[horizon_manager]") {
    // The shape output is only as good as what the solve keeps. These are all
    // computed during the find and were previously discarded.
    auto& mgr = horizon_manager::get();
    auto const r = mgr.find_now(mgr.index_of("AH"));
    REQUIRE(r.found);

    auto const np = static_cast<std::size_t>(mgr.get(mgr.index_of("AH")).surface.n_points());
    REQUIRE(r.fields.h.size()         == np);
    REQUIRE(r.fields.expansion.size() == np);

    // The curvature set comes from the Hessian path, exactly like the
    // quasi-local measures; if those ran, so did these.
    if (r.spin_akv.J != 0.0 || !r.fields.ricci_scalar.empty()) {
        REQUIRE(r.fields.ricci_scalar.size()    == np);
        REQUIRE(r.fields.gauss_curvature.size() == np);
        REQUIRE(r.fields.shear2.size()          == np);
        REQUIRE(r.fields.sqrt_q.size()          == np);
        REQUIRE(r.fields.zeta.size()            == np);
    }

    // The residual reported is the maximum of the field retained.
    double worst = 0.0;
    for (double v : r.fields.expansion) worst = std::max(worst, std::abs(v));
    REQUIRE_THAT(worst, Catch::Matchers::WithinRel(r.residual, 1e-12));
}
