/**
 * @file horizon_finder.hh
 * @brief Newton solve for an apparent horizon: Theta[h] = 0.
 *
 * The surface is a spectral Strahlkoerper r = h(theta,phi); the unknowns are
 * its (l_max+1)^2 spherical-harmonic coefficients. Each iteration evaluates the
 * expansion at the collocation points, projects it onto the basis to form a
 * square residual, solves for a coefficient correction, and applies it with a
 * step limiter.
 *
 * Two choices differ from AHFinderDirect and are worth stating:
 *
 * 1. **Galerkin rather than collocation.** The residual is the projection
 *    r_c = <Theta, Y_c>, which makes the system square ((l_max+1)^2 on a side)
 *    and lets it be solved by a plain pivoted LU. Posing it at the collocation
 *    points instead gives an overdetermined system whose usual fix -- normal
 *    equations -- squares the condition number, and reports "not positive
 *    definite" when it really means "lost precision". The deleted GRACE finder
 *    did exactly that.
 *
 * 2. **The radial Jacobian term needs displaced geometry.** Moving the surface
 *    outward changes both the shape derivatives and the geometry sampled at the
 *    point, and Theta depends on d_k gamma_ij as well as on gamma_ij itself.
 *    Advecting the sample along the normal with the first derivatives it
 *    carries therefore gets the gamma_ij and K_ij parts right but leaves
 *    d_k gamma_ij unmoved, which is not a small correction -- for a
 *    Schwarzschild coordinate sphere it is the *dominant* term and flips the
 *    sign of dTheta/dr. Correcting it would need second derivatives of the
 *    metric, which no interpolator here supplies.
 *
 *    So the geometry is re-evaluated on two displaced surfaces, h +/- delta,
 *    giving three geometry evaluations per Newton step. This is what
 *    AHFinderDirect does and why. The cost is per *iteration*, not per
 *    coefficient: perturbing each of the (l_max+1)^2 unknowns separately would
 *    be that many evaluations, and is what makes a naive numerical Jacobian
 *    unaffordable. The five angular sensitivities do not move the surface
 *    point, so they stay local and free.
 *
 * The geometry arrives through a callback, so the same solver runs against grid
 * data or against an analytic spacetime. That is what makes it testable without
 * an evolution.
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

#ifndef GRACE_GEOMETRY_HORIZON_FINDER_HH
#define GRACE_GEOMETRY_HORIZON_FINDER_HH

#include <grace_config.h>

#include <grace/geometry/horizon_multipoles.hh>
#include <grace/geometry/killing_vector.hh>
#include <grace/geometry/surface_geometry.hh>
#include <grace/geometry/horizon_spin.hh>
#include <grace/geometry/horizon_fluxes.hh>
#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/expansion.hh>
#include <grace/utils/dense_solve.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <functional>
#include <string>
#include <vector>

namespace grace {

/// Why a search stopped.
enum class horizon_status_t {
    converged,           ///< ||Theta||_inf fell below the tolerance.
    /// The residual stopped improving above the tolerance. The surface is as
    /// converged as the angular resolution allows: the floor is set by how well
    /// l_max can represent this shape, not by the solver. Treated as found,
    /// with the status distinguishing it from a clean convergence.
    stalled,
    max_iterations,      ///< Ran out of iterations while still converging.
    geometry_unavailable,///< The surface left the region where geometry can be sampled.
    singular_jacobian,   ///< The linearised system could not be solved.
    degenerate_surface,  ///< gamma^ij s_i s_j <= 0: the surface stopped being spacelike.
    diverged,            ///< The residual grew persistently.
    /// The surface ran away outwards. In an asymptotically flat spacetime the
    /// expansion tends to 2/r, so a large enough sphere satisfies *any*
    /// tolerance: without a bound the solve can "converge" on a meaningless
    /// surface far outside the region of interest.
    runaway
};

inline char const* to_string(horizon_status_t s) {
    switch (s) {
        case horizon_status_t::converged:            return "converged";
        case horizon_status_t::stalled:              return "stalled at the resolution floor";
        case horizon_status_t::max_iterations:       return "max iterations reached";
        case horizon_status_t::geometry_unavailable: return "surface left the domain";
        case horizon_status_t::singular_jacobian:    return "singular Jacobian";
        case horizon_status_t::degenerate_surface:   return "degenerate surface";
        case horizon_status_t::diverged:             return "residual diverged";
        case horizon_status_t::runaway:              return "surface ran away outwards";
    }
    return "unknown";
}

struct horizon_config_t {
    double tolerance         = 1e-10;  ///< Convergence threshold on max|Theta|.
    /// Value the expansion is driven to. Zero is an apparent horizon; a
    /// positive value is a constant-expansion surface, which is what horizon
    /// pretracking searches over. Zero reproduces the plain solve exactly.
    ///
    /// Thornburg's review (section 3.2.6, figure 3.4) warns that Theta vanishes
    /// at the horizon *and again at infinity*, so for a positive target there
    /// are two solutions and a search can converge on the outer, weak-field one
    /// that never becomes the horizon. Schnetter et al. defend against that by
    /// solving on a scaled expansion such as R^2*Theta, which is monotonic.
    ///
    /// That defence is deliberately not used here, because this solver cannot
    /// support it cheaply: d(R^2 Theta)/dh is *not* R^2 dTheta/dh. The areal
    /// radius responds to the surface as a whole, adding a rank-one term that
    /// carries the entire sign change -- and dropping it, as a per-point
    /// finite-differenced Jacobian must, leaves a Jacobian with the wrong sign
    /// on the uniform mode. Measured: the solve diverges from the first step.
    ///
    /// The two defences used instead are already present and already tested:
    /// seed the search from a sphere around the parents rather than far out, so
    /// the nearby root is the inner one; and let `max_radius_growth` stop a
    /// surface that drifts outwards anyway.
    double target_expansion  = 0.0;
    int    max_iterations    = 50;
    double max_relative_step = 0.1;    ///< Cap on |dh|_inf / mean(h) per step.
    bool   verbose           = false;
    /// Iterations of non-decreasing residual tolerated before declaring divergence.
    int    max_growth_iterations = 6;

    /// Iterations improving by less than `stall_factor` before declaring the
    /// solve stalled. Without this a solve that reaches its resolution floor in
    /// three steps burns every remaining iteration bouncing there and then
    /// reports failure -- which is both wasteful and misleading.
    ///
    /// Two rather than four: with a tolerance below the achievable floor -- the
    /// normal case once the grid, not the solver, is the limit -- *every* solve
    /// ends by stalling, so this count is paid on every find, warm starts
    /// included. A genuine descent improves far faster than `stall_factor`, so
    /// there is no risk of cutting one short.
    int    max_stall_iterations = 2;
    /// Residual ratio above which an iteration counts as no improvement.
    double stall_factor = 0.99;
    /// A stalled solve counts as found only if `residual * mean_radius` falls
    /// below this. Theta has dimensions of 1/length and a generic surface has
    /// Theta ~ 2/R, so the product is the scale-free measure of how close to a
    /// horizon the surface actually is. Without it, a search that stalls far
    /// from any horizon -- on a surface cutting through a puncture, say --
    /// would be reported as a find.
    double stall_accept = 1e-3;

    /// Largest mean radius the surface may reach, as a multiple of its starting
    /// mean radius. Guards the Theta -> 2/r trap described under `runaway`:
    /// without it a search started outside a horizon that does not exist will
    /// happily drift outwards until the residual falls below tolerance.
    /// Expressed as a ratio so it needs no absolute length scale.
    double max_radius_growth = 8.0;
};

struct horizon_result_t {
    bool   found     = false;
    horizon_status_t status = horizon_status_t::max_iterations;
    int    iterations = 0;
    double residual   = 0.0;   ///< Final max|Theta| over the collocation points.

    double area             = 0.0;
    /// Per-point fields over the collocation grid, retained for output.
    ///
    /// Every one of these is computed during the solve and was previously
    /// discarded at the end of the diagnostics block. Keeping them costs a few
    /// hundred doubles and no extra sampling, and they are what turns a shape
    /// file from an outline into a diagnostic: `expansion` shows where the
    /// solve is worst, `ricci_scalar` the oblateness, `shear2` how far the
    /// surface is from a non-expanding horizon, and `zeta` is the coordinate
    /// the multipoles are taken against.
    ///
    /// Empty when the horizon was not found; the curvature fields are also
    /// empty without the Hessian sampling path, exactly like the quasi-local
    /// measures they come from.
    struct surface_fields_t {
        /// The surface these fields were sampled on.
        ///
        /// Carried alongside the fields rather than read back from the live
        /// surface, because the two can disagree: the manager may `recentre()`
        /// after a successful find, which leaves the surface geometrically
        /// identical but re-expressed about a new origin. Fields captured
        /// before that point belong to the old parameterisation, and pairing
        /// them with the new coefficients puts every per-point value on
        /// slightly the wrong collocation direction.
        std::vector<double>  coefficients;
        std::array<double,3> centre{};
        std::vector<double> h;                ///< radius at each collocation point
        std::vector<double> expansion;        ///< Theta, the Newton residual field
        std::vector<double> ricci_scalar;     ///< Rtilde
        std::vector<double> gauss_curvature;
        std::vector<double> shear2;
        std::vector<double> sqrt_q;
        std::vector<double> zeta;             ///< invariant coordinate (AKV potential)
    };
    surface_fields_t fields;

    /// Matter and EM fluxes through the surface. Filled by the manager, not by
    /// find_horizon(): they need matter variables, and the solver must stay
    /// runnable against analytic spacetimes that have none. `valid` is false
    /// when they were not computed.
    horizon_flux_t fluxes;

    double irreducible_mass = 0.0;
    double areal_radius     = 0.0;
    std::array<double,3> centroid{};
    double min_radius = 0.0, max_radius = 0.0, mean_radius = 0.0;

    /// Quasi-local angular momentum from coordinate rotational generators.
    /// Exact for Kerr at any orientation; approximate on a distorted horizon.
    horizon_spin_t spin{};

    /// The same from an approximate Killing vector -- the isolated-horizon
    /// formalism proper. Check `spin_akv.killing_defect` before trusting it on
    /// a distorted horizon.
    horizon_spin_akv_t spin_akv{};

    /// Horizon mass and angular-momentum multipoles.
    horizon_multipoles_t multipoles{};

    /// max |sigma_AB| of the outgoing null normal. Vanishes, along with the
    /// expansion, on a genuine isolated horizon.
    double max_shear = 0.0;

    /// Proper circumferences and their ratios.
    surface_circumferences_t circumferences{};
};

/**
 * @brief Supplies ADM geometry at a set of points.
 * @return false if the geometry is not available there (outside the grid, say).
 */
using geometry_provider_t =
    std::function<bool(std::vector<std::array<double,3>> const&,
                       std::vector<adm_point_t>&)>;

namespace detail {

/// Expansion at every collocation point of `surface`.
inline void expansion_field(strahlkoerper_t const& surface,
                            strahlkoerper_t::shape_t const& shape,
                            std::vector<adm_point_t> const& geom,
                            std::vector<expansion_result_t>& out)
{
    auto const& b = surface.basis();
    out.resize(b.n_points());

    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            out[p] = expansion_at(geom[p], sp, b.theta(it), b.phi(ip));
        }
}

/// Convenience overload for callers without a precomputed shape.
inline void expansion_field(strahlkoerper_t const& surface,
                            std::vector<adm_point_t> const& geom,
                            std::vector<expansion_result_t>& out)
{
    expansion_field(surface, surface.evaluate(), geom, out);
}

}  // namespace detail

/**
 * @brief Solve Theta[h] = 0, starting from the surface's current shape.
 *
 * @param surface  In/out: the initial guess on entry, the horizon on success.
 *                 Left at the last iterate on failure, so a caller can inspect
 *                 or reset it.
 * @param geometry Callback supplying ADM data at the collocation points.
 * @param cfg      Convergence and step-control settings.
 * @param log      Optional sink for per-iteration progress lines.
 */
inline horizon_result_t find_horizon(
    strahlkoerper_t& surface,
    geometry_provider_t const& geometry,
    horizon_config_t const& cfg = {},
    std::function<void(std::string const&)> const& log = {})
{
    auto const& b  = surface.basis();
    int const   np = b.n_points();
    int const   nc = b.n_coeffs();

    horizon_result_t res;

    std::vector<adm_point_t>        geom;
    std::vector<expansion_result_t> theta(np);
    std::vector<double>             theta_vals(np);

    double previous_residual = std::numeric_limits<double>::infinity();
    int    no_progress_streak    = 0;
    double streak_start_residual = std::numeric_limits<double>::infinity();
    double initial_mean_h    = -1.0;

    double const* Ym      = b.Y_matrix();
    double const* dYth    = b.dY_dth_matrix();
    double const* dYph    = b.dY_dph_matrix();
    double const* d2Yth2  = b.d2Y_dth2_matrix();
    double const* d2Ythph = b.d2Y_dthdph_matrix();
    double const* d2Yph2  = b.d2Y_dph2_matrix();

    for (int iter = 0; iter <= cfg.max_iterations; ++iter) {

        // The surface does not change anywhere below until the coefficient
        // update at the very end of the iteration, so one spectral synthesis
        // serves the whole body: the sample points, the expansion, the runaway
        // guard, the stall test and the sensitivity block all read this.
        auto const shape = surface.evaluate();

        double mean_h = 0.0;
        for (int p = 0; p < np; ++p) mean_h += shape.h[p];
        mean_h /= np;

        auto const pts = surface.collocation_points(shape);
        if (!geometry(pts, geom) || static_cast<int>(geom.size()) != np) {
            res.status = horizon_status_t::geometry_unavailable;
            break;
        }

        detail::expansion_field(surface, shape, geom, theta);

        double max_abs = 0.0;
        bool   degenerate = false;
        for (int p = 0; p < np; ++p) {
            theta_vals[p] = theta[p].theta_out - cfg.target_expansion;
            max_abs = std::max(max_abs, std::abs(theta_vals[p]));
            if (!(theta[p].theta_d > 0.0) || !std::isfinite(theta_vals[p]))
                degenerate = true;
        }

        if (degenerate) {
            res.status = horizon_status_t::degenerate_surface;
            res.residual = max_abs;
            res.iterations = iter;
            break;
        }

        res.residual   = max_abs;
        res.iterations = iter;

        // Runaway guard.
        if (initial_mean_h < 0.0) initial_mean_h = mean_h;
        else if (cfg.max_radius_growth > 0.0
                 && mean_h > cfg.max_radius_growth * initial_mean_h) {
            res.status = horizon_status_t::runaway;
            break;
        }

        if (log && cfg.verbose) {
            log("  iter " + std::to_string(iter) +
                "  max|Theta| = " + std::to_string(max_abs));
        }

        if (max_abs < cfg.tolerance) {
            res.status = horizon_status_t::converged;
            res.found  = true;
            break;
        }
        if (iter == cfg.max_iterations) {
            res.status = horizon_status_t::max_iterations;
            break;
        }

        // One streak, two verdicts.
        //
        // These cannot be two independent tests: `max_abs >= previous` implies
        // `max_abs > stall_factor * previous`, so a growing residual increments
        // both, and the smaller stall threshold fires first every time. That
        // made `diverged` unreachable and -- worse -- routed a *growing*
        // residual into the stall branch, where `stall_accept` could declare it
        // found. Making them mutually exclusive instead breaks stall detection
        // the other way: a residual jittering on its noise floor alternates
        // between the two branches and resets both counters forever.
        //
        // So count non-improvement once, remembering where the run of it
        // started, and let *proximity to a horizon* decide what the run means.
        if (max_abs > cfg.stall_factor * previous_residual) {
            if (no_progress_streak == 0) streak_start_residual = previous_residual;
            ++no_progress_streak;

            // Theta has dimensions of 1/length and a generic surface has
            // Theta ~ 2/R, so the scale-free question is whether residual * R
            // is small -- not whether the residual is. A run of non-improvement
            // this close to a horizon is the resolution floor and nothing more:
            // stop promptly and call it found.
            if (max_abs * mean_h < cfg.stall_accept
                && no_progress_streak >= cfg.max_stall_iterations) {
                res.status = horizon_status_t::stalled;
                res.found  = true;
                break;
            }

            // Otherwise the surface is not near a horizon, so give it the
            // longer budget -- Newton with an approximate Jacobian is allowed
            // the occasional uphill step -- and then classify by which way the
            // residual actually went over the run.
            if (no_progress_streak >= cfg.max_growth_iterations) {
                res.status = (max_abs > streak_start_residual)
                           ? horizon_status_t::diverged
                           : horizon_status_t::stalled;
                break;
            }
        } else {
            no_progress_streak = 0;
        }

        previous_residual = max_abs;

        //-------------------------------------------------------------------
        // Per-point sensitivities of Theta to the six shape quantities.
        // All six have dimensions of length, so one step size serves.
        //
        // The h sensitivity moves the surface, so it needs geometry sampled on
        // the displaced surfaces; the five angular ones do not, and reuse the
        // geometry already in hand.
        //-------------------------------------------------------------------
        double const delta = 1e-6 * std::max(mean_h, 1e-8);

        std::vector<std::array<double,3>> pts_p(np), pts_m(np);
        {
            auto const& c = surface.centre();
            for (int it = 0; it < b.n_theta(); ++it) {
                double const st = b.sin_theta(it), ct = b.cos_theta(it);
                for (int ip = 0; ip < b.n_phi(); ++ip) {
                    int const p = b.point_index(it, ip);
                    double const cp = std::cos(b.phi(ip)), sp_ = std::sin(b.phi(ip));
                    double const nv[3] = { st * cp, st * sp_, ct };
                    for (int d = 0; d < 3; ++d) {
                        pts_p[p][d] = c[d] + (shape.h[p] + delta) * nv[d];
                        pts_m[p][d] = c[d] + (shape.h[p] - delta) * nv[d];
                    }
                }
            }
        }

        std::vector<adm_point_t> geom_p, geom_m;
        if (!geometry(pts_p, geom_p) || !geometry(pts_m, geom_m)
            || static_cast<int>(geom_p.size()) != np
            || static_cast<int>(geom_m.size()) != np) {
            res.status = horizon_status_t::geometry_unavailable;
            break;
        }

        std::vector<double> sens(static_cast<std::size_t>(np) * 6);

        for (int it = 0; it < b.n_theta(); ++it) {
            for (int ip = 0; ip < b.n_phi(); ++ip) {
                int const p  = b.point_index(it, ip);
                double const th = b.theta(it), ph = b.phi(ip);

                shape_point_t sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                  shape.dthth[p], shape.dthph[p], shape.dphph[p] };

                // Radial sensitivity: shape and geometry both displaced.
                {
                    shape_point_t up = sp; up.h += delta;
                    shape_point_t dn = sp; dn.h -= delta;
                    double const tp = expansion_at(geom_p[p], up, th, ph).theta_out;
                    double const tm = expansion_at(geom_m[p], dn, th, ph).theta_out;
                    sens[static_cast<std::size_t>(p) * 6 + 0] = (tp - tm) / (2.0 * delta);
                }

                // Angular sensitivities: the surface point does not move.
                double* field[6] = { &sp.h, &sp.dth, &sp.dph,
                                     &sp.dthth, &sp.dthph, &sp.dphph };
                for (int m = 1; m < 6; ++m) {
                    double const saved = *field[m];

                    *field[m] = saved + delta;
                    double const tp = expansion_at(geom[p], sp, th, ph).theta_out;

                    *field[m] = saved - delta;
                    double const tm = expansion_at(geom[p], sp, th, ph).theta_out;

                    *field[m] = saved;
                    sens[static_cast<std::size_t>(p) * 6 + m] = (tp - tm) / (2.0 * delta);
                }
            }
        }

        //-------------------------------------------------------------------
        // Chain onto the coefficients and project:
        //     M_{p,c}   = sum_m  sens_{p,m}  dY^{(m)}_{p,c}
        //     J_{c,c'}  = sum_p  w_p Y_{p,c} M_{p,c'}
        //     r_c       = sum_p  w_p Y_{p,c} Theta_p
        //-------------------------------------------------------------------
        std::vector<double> M(static_cast<std::size_t>(np) * nc);
        for (int p = 0; p < np; ++p) {
            std::size_t const row = static_cast<std::size_t>(p) * nc;
            double const* Sp = &sens[static_cast<std::size_t>(p) * 6];
            for (int c = 0; c < nc; ++c) {
                M[row + c] = Sp[0] * Ym     [row + c]
                           + Sp[1] * dYth   [row + c]
                           + Sp[2] * dYph   [row + c]
                           + Sp[3] * d2Yth2 [row + c]
                           + Sp[4] * d2Ythph[row + c]
                           + Sp[5] * d2Yph2 [row + c];
            }
        }

        std::vector<double> J(static_cast<std::size_t>(nc) * nc, 0.0);
        std::vector<double> rhs(nc, 0.0);

        for (int it = 0; it < b.n_theta(); ++it)
            for (int ip = 0; ip < b.n_phi(); ++ip) {
                int const p = b.point_index(it, ip);
                double const w = b.weight(it, ip);
                std::size_t const row = static_cast<std::size_t>(p) * nc;
                for (int c = 0; c < nc; ++c) {
                    double const wY = w * Ym[row + c];
                    rhs[c] -= wY * theta_vals[p];       // solve J da = -r
                    for (int cp = 0; cp < nc; ++cp)
                        J[static_cast<std::size_t>(c) * nc + cp] += wY * M[row + cp];
                }
            }

        if (!lu_solve_in_place(J, nc, rhs)) {
            res.status = horizon_status_t::singular_jacobian;
            break;
        }

        //-------------------------------------------------------------------
        // Limit the step. An early iteration on a poor guess can otherwise
        // throw the surface clean out of the grid.
        //-------------------------------------------------------------------
        double max_dh = 0.0;
        for (int p = 0; p < np; ++p) {
            std::size_t const row = static_cast<std::size_t>(p) * nc;
            double dh = 0.0;
            for (int c = 0; c < nc; ++c) dh += Ym[row + c] * rhs[c];
            max_dh = std::max(max_dh, std::abs(dh));
        }

        double scale = 1.0;
        double const allowed = cfg.max_relative_step * mean_h;
        if (max_dh > allowed && max_dh > 0.0) scale = allowed / max_dh;

        auto& a = surface.coefficients();
        for (int c = 0; c < nc; ++c) a[c] += scale * rhs[c];
    }

    //-----------------------------------------------------------------------
    // Diagnostics, from the geometry already in hand.
    //-----------------------------------------------------------------------
    if (res.found && static_cast<int>(geom.size()) == np) {
        // The expansion field as the solve last saw it. `theta_vals` is already
        // the converged residual per point.
        res.fields.h.resize(static_cast<std::size_t>(np));
        res.fields.expansion.assign(theta_vals.begin(), theta_vals.end());
        {
            auto const sh = surface.evaluate();
            res.fields.h.assign(sh.h.begin(), sh.h.end());
        }
        res.fields.coefficients = surface.coefficients();
        res.fields.centre       = surface.centre();

        auto const diag      = surface.diagnostics(geom);
        res.area             = diag.area;
        res.irreducible_mass = diag.irreducible_mass;
        res.areal_radius     = diag.areal_radius;
        res.centroid         = diag.centroid;
        res.min_radius       = diag.radius.min;
        res.max_radius       = diag.radius.max;
        res.mean_radius      = diag.radius.mean;

        // Taken about the centroid rather than the parameterisation origin, so
        // the answer does not depend on where the search happened to start.
        res.spin = horizon_spin(surface, geom, res.irreducible_mass, &res.centroid);

        // The quasi-local measures all rest on the intrinsic curvature, which
        // needs d_k d_l gamma_ij. Only the Hessian sampling path supplies it;
        // without it these are left at their defaults rather than reported as
        // zero, which would be indistinguishable from a genuine answer.
        if (!geom.empty() && geom.front().has_hessian) {
            auto const sg = surface_geometry(surface, geom);

            auto const nsz = static_cast<std::size_t>(np);
            res.fields.ricci_scalar.resize(nsz);
            res.fields.gauss_curvature.resize(nsz);
            res.fields.shear2.resize(nsz);
            res.fields.sqrt_q.resize(nsz);
            for (std::size_t p = 0; p < nsz; ++p) {
                res.fields.ricci_scalar[p]    = sg[p].ricci_scalar;
                res.fields.gauss_curvature[p] = sg[p].gauss_curvature;
                res.fields.shear2[p]          = sg[p].shear2;
                res.fields.sqrt_q[p]          = sg[p].sqrt_q;
            }

            res.max_shear = 0.0;
            for (auto const& g : sg)
                res.max_shear = std::max(res.max_shear,
                                         std::sqrt(std::abs(g.shear2)));

            res.circumferences = surface_circumferences(surface, sg);

            auto const akv = approximate_killing_vector(surface, geom, sg);
            if (akv.found) {
                res.fields.zeta.assign(akv.zeta.begin(), akv.zeta.end());
                res.spin_akv = horizon_spin_akv(surface, geom, sg,
                                                res.irreducible_mass);
                res.multipoles = horizon_multipoles(surface, geom, sg, akv,
                                                    res.spin_akv.christodoulou_mass);
            }
        }
    }

    return res;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_HORIZON_FINDER_HH
