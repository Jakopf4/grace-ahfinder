/**
 * @file horizon_flux_integrand.hh
 * @brief Pointwise MHD flux integrands on a closed surface.
 *
 * The accretion rate, energy flux, angular-momentum flux and magnetic flux all
 * come from the same handful of quantities: the fluid four-velocity, the
 * magnetic four-vector b^mu, and the stress tensor built from them. This
 * factors that algebra out of bh_diagnostics so it exists once, and generalises
 * it from a coordinate sphere to an arbitrary direction.
 *
 * The only difference between the two callers is what "outward" means and what
 * the surface element is:
 *
 *  - bh_diagnostics integrates over a coordinate sphere, where the outward
 *    direction is d/dr and the element carries r^2 sqrt(gamma) dOmega;
 *  - the apparent-horizon finder integrates over a Strahlkoerper, where the
 *    outward direction is the unit normal s^i of a surface that is not r=const
 *    and the element is the proper area element sqrt(det q).
 *
 * Both reduce to contracting the same tensors with a direction, so the caller
 * supplies the direction and applies its own measure. Nothing here knows about
 * either surface.
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

#ifndef GRACE_PHYSICS_HORIZON_FLUX_INTEGRAND_HH
#define GRACE_PHYSICS_HORIZON_FLUX_INTEGRAND_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/utils/metric_utils.hh>

#include <array>

namespace grace {

/// Primitive state at one point, in the form the flux integrands want.
struct mhd_primitives_t {
    double rho, eps, press;
    double z[3];   ///< Valencia z^i = W v^i, as stored in ZVEC*
    double B[3];   ///< cell-centred B^i
};

/// The four pointwise integrands, already contracted with the supplied
/// direction. Multiply by the caller's surface measure and sum.
struct flux_integrand_t {
    double mdot;    ///< -rho u^n : positive for inflow once integrated
    double edot;    ///< -T^n_0
    double ldot;    ///< +T^n_phi
    double edot_em; ///< the electromagnetic part of edot
    double ldot_em; ///< the electromagnetic part of ldot
    double phi_B;   ///< (1/2)|b^n u^t - b^t u^n| : the Blandford-Znajek flux
    double b_n;     ///< b^n itself, signed -- for the div B monitor
};

/**
 * @brief Flux integrands along an arbitrary direction.
 *
 * @param metric  metric, shift and lapse at the point.
 * @param p       primitives at the point.
 * @param dir     the outward direction with *upper* index, `d x^i / d lambda`.
 *                For a coordinate sphere this is `dr/dx^i`'s dual, i.e. the
 *                gradient row that turns a vector into its radial component;
 *                for a Strahlkoerper it is the unit normal `s^i`. Nothing is
 *                normalised here -- whatever scaling `dir` carries ends up in
 *                the result, which is what lets the sphere caller keep its
 *                `dr/dx^i` convention unchanged.
 * @param phi_dir the azimuthal direction with *lower* index, used only for
 *                `ldot` (the generator of the rotation whose momentum is
 *                being measured).
 */
GRACE_HOST_DEVICE inline
flux_integrand_t flux_integrand(metric_array_t const& metric,
                                mhd_primitives_t const& p,
                                double const dir[3],
                                double const phi_dir[3])
{
    double const one_over_alp = 1.0 / metric.alp();

    // W^2 = 1 + z^2
    double const W  = Kokkos::sqrt(1.0 + metric.square_vec({p.z[0], p.z[1], p.z[2]}));
    double const u0 = one_over_alp * W;

    std::array<double,4> const uU{{
        u0,
        p.z[0] - one_over_alp * metric.beta(0),
        p.z[1] - one_over_alp * metric.beta(1),
        p.z[2] - one_over_alp * metric.beta(2) }};
    auto const uD = metric.lower_4vec(uU);

    double const b0 = uD[1] * p.B[0] + uD[2] * p.B[1] + uD[3] * p.B[2];
    double const b1 = (p.B[0] + b0 * uU[1]) / u0;
    double const b2 = (p.B[1] + b0 * uU[2]) / u0;
    double const b3 = (p.B[2] + b0 * uU[3]) / u0;

    auto const bD = metric.lower_4vec({b0, b1, b2, b3});

    double const bsq = b0 * bD[0] + b1 * bD[1] + b2 * bD[2] + b3 * bD[3];

    // Components along the requested direction.
    double const un = dir[0] * uU[1] + dir[1] * uU[2] + dir[2] * uU[3];
    double const bn = dir[0] * b1    + dir[1] * b2    + dir[2] * b3;

    double const u_phi = phi_dir[0] * uD[1] + phi_dir[1] * uD[2] + phi_dir[2] * uD[3];
    double const b_phi = phi_dir[0] * bD[1] + phi_dir[1] * bD[2] + phi_dir[2] * bD[3];

    double const rhoh = p.rho + p.rho * p.eps + p.press;

    flux_integrand_t out;
    out.mdot  = -p.rho * un;
    out.edot  = -((rhoh + bsq) * un * uD[0] - bn * bD[0]);
    out.ldot  =  ((rhoh + bsq) * un * u_phi - bn * b_phi);
    // The stress tensor splits additively, so the EM part is the same
    // expression with rhoh dropped. The matter part is the difference; it is
    // not reported separately because one subtraction recovers it.
    out.edot_em = -(bsq * un * uD[0]  - bn * bD[0]);
    out.ldot_em =  (bsq * un * u_phi  - bn * b_phi);
    out.phi_B =  0.5 * Kokkos::fabs(bn * u0 - b0 * un);
    out.b_n   =  bn;
    return out;
}

}  // namespace grace

#endif  // GRACE_PHYSICS_HORIZON_FLUX_INTEGRAND_HH
