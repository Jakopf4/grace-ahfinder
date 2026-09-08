/**
 * @file horizon_multipoles.hh
 * @brief Mass and angular-momentum multipole moments of a horizon.
 *
 * The Ashtekar-Engle-Pawlowski-Van Den Broeck horizon multipoles: the standard
 * way to say *how* a horizon is distorted, rather than only how massive and how
 * fast-spinning it is.
 *
 *     M_n = (M R^n / 2 pi) * (1/4) integral_S Rtilde  P_n(zeta)  dA
 *     J_n = (R^(n-1) / 8 pi)      integral_S (phi^a s^b K_ab) P'_n(zeta) dA
 *
 * with `Rtilde` the intrinsic 2-Ricci scalar, `R` the areal radius, `M` the
 * Christodoulou mass, and `zeta` the invariant coordinate.
 *
 * Two normalisation checks fall straight out and are enforced by the tests:
 * `M_0 = M` (because Gauss-Bonnet fixes the integral of Rtilde at 8 pi) and
 * `J_1 = J` (because `P'_1 = 1` reduces the second integral to the spin).
 *
 * @par Differences from QuasiLocalMeasures
 * QLM obtains `zeta` by integrating `d_a zeta = R^-2 xi^b eps_ba` along the
 * equator and then up every meridian -- serial and path-dependent. Here it
 * comes directly from the Killing-vector potential, since
 * `eps_{AB} xi^B = -D_A z` makes the two proportional. QLM also mixes
 * quadratures, using Driscoll-Healy weights for the area and spin but naive
 * `dtheta dphi` for the multipoles, which is why its `qlm_mp_m0` and
 * `qlm_mass` do not quite agree; a single spectral quadrature is used
 * throughout here, so `M_0` and `M` agree to round-off.
 *
 * Only `m = 0` moments exist: they require axisymmetry, and `zeta` is the
 * invariant coordinate along the Killing orbits. QLM sketches an invariant
 * azimuthal coordinate but leaves it disabled behind `#if 0`.
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

#ifndef GRACE_GEOMETRY_HORIZON_MULTIPOLES_HH
#define GRACE_GEOMETRY_HORIZON_MULTIPOLES_HH

#include <grace_config.h>

#include <grace/geometry/killing_vector.hh>
#include <grace/geometry/strahlkoerper.hh>
#include <grace/geometry/surface_geometry.hh>
#include <grace/physics/adm_from_evolution.hh>

#include <array>
#include <cmath>
#include <vector>

namespace grace {

/// Highest multipole order retained.
inline constexpr int HORIZON_MULTIPOLE_MAX = 8;

struct horizon_multipoles_t {
    bool   valid = false;
    /// Mass multipoles M_0 .. M_8; M_0 is the Christodoulou mass.
    std::array<double, HORIZON_MULTIPOLE_MAX + 1> mass{};
    /// Angular-momentum multipoles J_0 .. J_8; J_1 is the spin.
    std::array<double, HORIZON_MULTIPOLE_MAX + 1> spin{};
};

namespace detail {

/// Legendre P_n(x) and P'_n(x) for n = 0 .. N, by recurrence.
///
/// The derivative uses P'_{n+1} = (2n+1) P_n + P'_{n-1} rather than the closed
/// form n(x P_n - P_{n-1})/(x^2-1), which is singular at the poles -- exactly
/// where zeta reaches +/-1.
inline void legendre_and_derivative(double x, int N,
                                    std::vector<double>& P,
                                    std::vector<double>& dP)
{
    P.assign(N + 1, 0.0);
    dP.assign(N + 1, 0.0);
    P[0]  = 1.0;
    dP[0] = 0.0;
    if (N == 0) return;
    P[1]  = x;
    dP[1] = 1.0;
    for (int n = 1; n < N; ++n) {
        P[n+1]  = ((2*n + 1) * x * P[n] - n * P[n-1]) / (n + 1);
        dP[n+1] = (2*n + 1) * P[n] + dP[n-1];
    }
}

}  // namespace detail

/**
 * @brief Horizon multipole moments.
 *
 * @param surface Surface to integrate over.
 * @param geom    ADM data at its collocation points.
 * @param sg      Intrinsic geometry (supplies Rtilde).
 * @param akv     Approximate Killing vector (supplies zeta and phi^i).
 * @param mass    Christodoulou mass, used to normalise the mass moments.
 */
inline horizon_multipoles_t horizon_multipoles(
    strahlkoerper_t const& surface,
    std::vector<adm_point_t> const& geom,
    std::vector<surface_point_geometry_t> const& sg,
    akv_result_t const& akv,
    double mass)
{
    horizon_multipoles_t out;

    auto const& b  = surface.basis();
    int const   np = b.n_points();
    if (!akv.found || static_cast<int>(sg.size()) != np) return out;

    double const area = surface.area(geom);
    if (!(area > 0.0)) return out;
    double const R = std::sqrt(area / (4.0 * M_PI));

    // Spin density phi^a s^b K_ab: the same integrand J_1 must reduce to.
    auto const shape = surface.evaluate();
    std::vector<double> spin_density(np, 0.0);
    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            auto const nrm = surface_normal_at(geom[p], sp, b.theta(it), b.phi(ip));
            if (!(nrm.norm2 > 0.0)) continue;
            double acc = 0.0;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    acc += akv.phi[p][i] * nrm.upper[j] * geom[p].K[adm_sym(i, j)];
            spin_density[p] = acc;
        }

    std::vector<double> P, dP;
    std::vector<double> integrand_m(np), integrand_j(np);

    for (int n = 0; n <= HORIZON_MULTIPOLE_MAX; ++n) {
        for (int p = 0; p < np; ++p) {
            detail::legendre_and_derivative(akv.zeta[p], HORIZON_MULTIPOLE_MAX, P, dP);
            integrand_m[p] = 0.25 * sg[p].ricci_scalar * P[n];
            integrand_j[p] = spin_density[p] * dP[n];
        }
        out.mass[n] = surface.integrate(integrand_m, geom)
                    * mass * std::pow(R, n) / (2.0 * M_PI);
        out.spin[n] = surface.integrate(integrand_j, geom)
                    * std::pow(R, n - 1) / (8.0 * M_PI);
    }

    out.valid = true;
    return out;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_HORIZON_MULTIPOLES_HH
