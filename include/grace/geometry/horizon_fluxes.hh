/**
 * @file horizon_fluxes.hh
 * @brief Accretion, energy, angular-momentum and magnetic flux through a horizon.
 *
 * The integrands are shared with bh_diagnostics (horizon_flux_integrand.hh);
 * what differs is the surface. bh_diagnostics evaluates them on a coordinate
 * sphere, where "outward" is d/dr and the measure is r^2 sqrt(gamma) dOmega.
 * A horizon is neither r = const nor radially normal, so here the direction is
 * the surface's own unit normal s^i and the measure is the proper area element
 * that strahlkoerper_t::integrate() already carries.
 *
 * Sign convention: `mdot` is positive for *inflow*, so a black hole accreting
 * reports a positive accretion rate. `edot` and `ldot` follow the same
 * convention as bh_diagnostics.
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

#ifndef GRACE_GEOMETRY_HORIZON_FLUXES_HH
#define GRACE_GEOMETRY_HORIZON_FLUXES_HH

#include <grace_config.h>

#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/expansion.hh>
#include <grace/physics/horizon_flux_integrand.hh>
#include <grace/physics/matter_from_evolution.hh>

#include <array>
#include <cmath>
#include <vector>

namespace grace {

struct horizon_flux_t {
    double mdot  = 0.0;   ///< rest-mass accretion rate, positive for inflow
    double edot  = 0.0;   ///< energy flux
    double ldot  = 0.0;   ///< angular-momentum flux about the z axis
    /// The electromagnetic part of `edot` and `ldot`. The matter part is the
    /// difference, so two columns carry the whole split. Worth separating at a
    /// horizon: the Blandford-Znajek signature is an EM energy flux that is
    /// *negative*, i.e. energy leaving the hole, while matter still falls in.
    double edot_em = 0.0;
    double ldot_em = 0.0;
    double phi_B = 0.0;   ///< (1/2) integral |b^n u^t - b^t u^n| dA
    /// integral B^i s_i dA, which vanishes identically by div B = 0. Kept
    /// precisely *because* it must vanish: it is a live check on the
    /// divergence constraint and on the interpolation onto the surface, in the
    /// same spirit as the Gauss-Bonnet test on the intrinsic curvature.
    double phi_B_signed = 0.0;
    /// Area-averaged L2 norms of the Hamiltonian and momentum constraint
    /// violation on the surface. Z4 only; zero elsewhere. These answer the
    /// question every other column raises -- whether the data the horizon sits
    /// in is trustworthy -- and a global norm cannot, being dominated by the
    /// puncture and the outer boundary rather than by the horizon.
    double ham_norm = 0.0;
    double mom_norm = 0.0;
    bool   have_constraints = false;
    bool   valid = false;
};

/**
 * @brief Flux integrals over a horizon surface.
 *
 * @param surface the Strahlkoerper.
 * @param geom    ADM data at its collocation points (for the normal).
 * @param matter  matter primitives at the same points.
 * @param centre  origin of the azimuthal generator; pass the horizon centroid
 *                so `ldot` does not depend on where the search started.
 *
 * Note against bh_diagnostics: that routine multiplies by
 * `scalar_symmetry_multiplier()` because its quadrature only covers the
 * computed octant. This surface is a **full sphere** with the *sampling*
 * folded, so no such factor applies here -- applying one would double every
 * flux.
 */
inline horizon_flux_t horizon_fluxes(strahlkoerper_t const& surface,
                                     std::vector<adm_point_t> const& geom,
                                     std::vector<matter_point_t> const& matter,
                                     std::array<double,3> const& centre)
{
    horizon_flux_t out;

    auto const& b  = surface.basis();
    int const   np = surface.n_points();
    if (static_cast<int>(geom.size()) != np ||
        static_cast<int>(matter.size()) != np) return out;

    auto const shape = surface.evaluate();
    auto const pts   = surface.collocation_points(shape);

    std::vector<double> f_mdot(np), f_edot(np), f_ldot(np),
                        f_edot_em(np), f_ldot_em(np),
                        f_phi(np), f_phi_signed(np),
                        f_ham(np), f_mom(np);

    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);

            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            auto const n = surface_normal_at(geom[p], sp, b.theta(it), b.phi(ip));
            if (!(n.norm2 > 0.0)) return out;   // surface is not spacelike here

            // Azimuthal generator about the centroid, phi^i = (-dy, dx, 0),
            // with a lower index for the momentum contraction -- the same
            // convention bh_diagnostics uses for its r sin(theta) factors, and
            // taken about the centroid so the answer does not depend on the
            // parameterisation origin.
            double const dx = pts[p][0] - centre[0];
            double const dy = pts[p][1] - centre[1];
            double const phi_dir[3] = { -dy, dx, 0.0 };

            auto const& m = matter[p];
            #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
            metric_array_t const metric{
                { geom[p].g[ADM_XX], geom[p].g[ADM_XY], geom[p].g[ADM_XZ],
                  geom[p].g[ADM_YY], geom[p].g[ADM_YZ], geom[p].g[ADM_ZZ] },
                { m.beta[0], m.beta[1], m.beta[2] }, m.alp };
            #else
            metric_array_t const metric{
                { geom[p].g[ADM_XX], geom[p].g[ADM_XY], geom[p].g[ADM_XZ],
                  geom[p].g[ADM_YY], geom[p].g[ADM_YZ], geom[p].g[ADM_ZZ] },
                { m.beta[0], m.beta[1], m.beta[2] }, m.alp };
            #endif

            auto const fi = flux_integrand(
                metric,
                mhd_primitives_t{ m.rho, m.eps, m.press,
                                  { m.z[0], m.z[1], m.z[2] },
                                  { m.B[0], m.B[1], m.B[2] } },
                n.upper, phi_dir);

            f_mdot[p]       = fi.mdot;
            f_edot[p]       = fi.edot;
            f_ldot[p]       = fi.ldot;
            f_edot_em[p]    = fi.edot_em;
            f_ldot_em[p]    = fi.ldot_em;

            // Squared, so the surface integral below gives an L2 norm rather
            // than a signed average that a sign change could hide.
            f_ham[p] = m.ham * m.ham;
            double mom2 = 0.0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    mom2 += geom[p].ginv[adm_sym(i, j)] * m.mom[i] * m.mom[j];
            f_mom[p] = mom2;
            f_phi[p]        = fi.phi_B;
            // B^i s_i, the signed normal component -- not the same object as
            // the b^mu-based phi_B, and zero by the divergence constraint.
            f_phi_signed[p] = m.B[0]*n.lower[0] + m.B[1]*n.lower[1] + m.B[2]*n.lower[2];
        }

    out.mdot         = surface.integrate(f_mdot,       geom);
    out.edot         = surface.integrate(f_edot,       geom);
    out.ldot         = surface.integrate(f_ldot,       geom);
    out.edot_em      = surface.integrate(f_edot_em,    geom);
    out.ldot_em      = surface.integrate(f_ldot_em,    geom);
    out.phi_B        = surface.integrate(f_phi,        geom);
    out.phi_B_signed = surface.integrate(f_phi_signed, geom);

    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    double const area = surface.integrate(std::vector<double>(static_cast<std::size_t>(np), 1.0), geom);
    if (area > 0.0) {
        out.ham_norm = std::sqrt(std::max(0.0, surface.integrate(f_ham, geom) / area));
        out.mom_norm = std::sqrt(std::max(0.0, surface.integrate(f_mom, geom) / area));
        out.have_constraints = true;
    }
    #endif

    out.valid        = true;
    return out;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_HORIZON_FLUXES_HH
