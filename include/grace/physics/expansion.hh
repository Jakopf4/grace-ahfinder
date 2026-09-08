/**
 * @file expansion.hh
 * @brief Expansion of the outgoing null normal to a Strahlkoerper surface.
 *
 * For a surface r = h(theta, phi) about a coordinate origin, with unit outward
 * normal n^i in the slice,
 *
 *     Theta = D_i n^i  +  K_ij n^i n^j  -  K
 *
 * and an apparent horizon is a surface on which Theta vanishes.
 *
 * The evaluation follows the decomposition of Thornburg (1996), PRD 54, 4899,
 * eqs. (14) and (15a-d): writing s_i for the *unnormalised* normal covector and
 * Theta_D = gamma^ij s_i s_j,
 *
 *     Theta = Theta_A / Theta_D^{3/2}     (radial part of the divergence)
 *           + Theta_B / Theta_D^{1/2}     (rest of the divergence)
 *           + Theta_C / Theta_D           (K_ij n^i n^j)
 *           - K
 *
 * Splitting it this way keeps every term a polynomial in s_i and its Cartesian
 * derivative, so no normalisation appears inside the differentiated
 * expressions -- which is what makes the derivatives below tractable.
 *
 * Unlike AHFinderDirect, which finite-differences the angular coordinate
 * transformation on a six-patch grid, the transformation here is analytic: the
 * surface is parameterised by ordinary spherical angles about the origin and
 * the required first and second derivatives of (r, theta, phi) with respect to
 * Cartesian x^i are written out in closed form. That is only possible because
 * the spectral representation tolerates the polar coordinate singularity --
 * the Gauss-Legendre collocation points never land on the axis, where the
 * 1/sin(theta) factors below would blow up.
 *
 * All host/device, operating on plain arrays: no allocation, no grid access.
 * The geometry arrives as an adm_point_t, which the interpolation path and the
 * analytic path both produce.
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

#ifndef GRACE_PHYSICS_EXPANSION_HH
#define GRACE_PHYSICS_EXPANSION_HH

#include <grace_config.h>

#include <grace/physics/adm_from_evolution.hh>
#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <Kokkos_Core.hpp>

namespace grace {

/**
 * @brief The surface shape at one angle: h and its angular derivatives.
 *
 * These are exactly the six quantities spherical_harmonic_basis_t::evaluate()
 * produces, and the six the expansion depends on.
 */
struct shape_point_t {
    double h     = 0.0;
    double dth   = 0.0;
    double dph   = 0.0;
    double dthth = 0.0;
    double dthph = 0.0;
    double dphph = 0.0;
};

/**
 * @brief Pieces of the expansion, kept separately for diagnostics.
 */
struct expansion_result_t {
    double theta_out;        ///< Theta_(l): outgoing expansion; zero on an apparent horizon.
    double theta_in;         ///< Theta_(n): ingoing expansion; distinguishes horizon types.
    double mean_curvature;   ///< D_i n^i, the surface's mean curvature in the slice.
    double theta_d;          ///< gamma^ij s_i s_j; positive for a well-posed surface.
};

namespace detail {

/// Flat orthonormal spherical triad at (theta, phi).
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
void spherical_triad(double theta, double phi,
                     double (&n)[3], double (&eth)[3], double (&eph)[3])
{
    double const st = Kokkos::sin(theta), ct = Kokkos::cos(theta);
    double const sp = Kokkos::sin(phi),   cp = Kokkos::cos(phi);

    n[0]   = st * cp;  n[1]   = st * sp;  n[2]   =  ct;
    eth[0] = ct * cp;  eth[1] = ct * sp;  eth[2] = -st;
    eph[0] = -sp;      eph[1] =  cp;      eph[2] =  0.0;
}

/**
 * @brief Unnormalised outward normal covector of F = r - h(theta,phi).
 *
 *     s_i = d_i r - h_theta d_i theta - h_phi d_i phi
 *
 * Shared by the expansion and by anything else that needs the surface normal,
 * so the coordinate algebra exists in one place.
 */
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
void normal_covector(shape_point_t const& s, double theta, double phi,
                     double (&sd)[3])
{
    double n[3], eth[3], eph[3];
    spherical_triad(theta, phi, n, eth, eph);

    double const inv_r  = 1.0 / s.h;
    double const inv_st = 1.0 / Kokkos::sin(theta);

    for (int i = 0; i < 3; ++i)
        sd[i] = n[i]
              - s.dth * eth[i] * inv_r
              - s.dph * eph[i] * inv_r * inv_st;
}

}  // namespace detail

/**
 * @brief Unit outward normal to the surface within the slice.
 */
struct surface_normal_t {
    double lower[3];   ///< s_i, normalised so gamma^ij s_i s_j = 1
    double upper[3];   ///< s^i = gamma^ij s_j
    double norm2;      ///< gamma^ij s~_i s~_j of the unnormalised covector
};

/**
 * @brief Normalised outward normal at a surface point.
 *
 * `norm2` is returned so callers can reject a surface that has stopped being
 * spacelike (norm2 <= 0) rather than taking a square root of it blindly.
 */
GRACE_HOST_DEVICE
inline surface_normal_t surface_normal_at(adm_point_t const& geom,
                                          shape_point_t const& s,
                                          double theta, double phi)
{
    surface_normal_t out{};

    double sd[3];
    detail::normal_covector(s, theta, phi, sd);

    double norm2 = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            norm2 += geom.ginv[adm_sym(i, j)] * sd[i] * sd[j];

    out.norm2 = norm2;
    if (!(norm2 > 0.0)) return out;

    double const inv = 1.0 / Kokkos::sqrt(norm2);
    for (int i = 0; i < 3; ++i) out.lower[i] = sd[i] * inv;
    for (int i = 0; i < 3; ++i) {
        double acc = 0.0;
        for (int j = 0; j < 3; ++j) acc += geom.ginv[adm_sym(i, j)] * out.lower[j];
        out.upper[i] = acc;
    }
    return out;
}

/**
 * @brief Evaluate the expansion at a single surface point.
 *
 * @param geom  ADM geometry at the surface point (needs g, ginv, K, dg, K_trace).
 * @param s     Surface shape at this angle.
 * @param theta Polar angle; must not be 0 or pi (see the file comment).
 * @param phi   Azimuthal angle.
 */
GRACE_HOST_DEVICE
inline expansion_result_t expansion_at(adm_point_t const& geom,
                                       shape_point_t const& s,
                                       double theta, double phi)
{
    expansion_result_t out{};

    double n[3], eth[3], eph[3];
    detail::spherical_triad(theta, phi, n, eth, eph);

    double const st  = Kokkos::sin(theta);
    double const ct  = Kokkos::cos(theta);
    double const r   = s.h;
    double const inv_r  = 1.0 / r;
    double const inv_st = 1.0 / st;

    //-----------------------------------------------------------------------
    // Gradients of the spherical coordinates with respect to x^i, evaluated on
    // the surface (r = h):
    //     d_i r     = n_i
    //     d_i theta = eth_i / r
    //     d_i phi   = eph_i / (r sin(theta))
    //-----------------------------------------------------------------------
    double dr[3], dth[3], dph[3];
    for (int i = 0; i < 3; ++i) {
        dr [i] = n[i];
        dth[i] = eth[i] * inv_r;
        dph[i] = eph[i] * inv_r * inv_st;
    }

    //-----------------------------------------------------------------------
    // Second derivatives of the same, obtained by differentiating the triad
    // along the surface (d eth/d theta = -n, d eth/d phi = cos(theta) eph,
    // d eph/d phi = -(sin(theta) n + cos(theta) eth)):
    //
    //   d_j d_i r     = (delta_ij - n_i n_j) / r
    //   d_j d_i theta = [ -n_i eth_j - eth_i n_j + (ct/st) eph_i eph_j ] / r^2
    //   d_j d_i phi   = -[ (st n_i + ct eth_i) eph_j
    //                    +  eph_i (st n_j + ct eth_j) ] / (r^2 st^2)
    //
    // Each is symmetric in (i,j), as it must be.
    //-----------------------------------------------------------------------
    double ddr[3][3], ddth[3][3], ddph[3][3];
    double const inv_r2 = inv_r * inv_r;
    double const cot    = ct * inv_st;

    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double const delta = (i == j) ? 1.0 : 0.0;
            ddr[i][j] = (delta - n[i] * n[j]) * inv_r;

            ddth[i][j] = (-n[i] * eth[j] - eth[i] * n[j]
                          + cot * eph[i] * eph[j]) * inv_r2;

            double const wi = st * n[i] + ct * eth[i];
            double const wj = st * n[j] + ct * eth[j];
            ddph[i][j] = -(wi * eph[j] + eph[i] * wj) * inv_r2 * inv_st * inv_st;
        }
    }

    //-----------------------------------------------------------------------
    // Unnormalised outward normal covector of F = r - h(theta, phi), and its
    // Cartesian gradient.
    //-----------------------------------------------------------------------
    double sd[3];
    detail::normal_covector(s, theta, phi, sd);

    double dsd[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            dsd[i][j] = ddr[i][j]
                      - s.dth   * ddth[i][j]
                      - s.dph   * ddph[i][j]
                      - s.dthth * dth[i] * dth[j]
                      - s.dthph * (dth[i] * dph[j] + dph[i] * dth[j])
                      - s.dphph * dph[i] * dph[j];

    //-----------------------------------------------------------------------
    // Geometry combinations.
    //-----------------------------------------------------------------------
    auto gi = [&](int i, int j) { return geom.ginv[adm_sym(i, j)]; };

    // d_k gamma^ij = -gamma^ia gamma^jb d_k gamma_ab
    double dginv[3][3][3];
    for (int k = 0; k < 3; ++k)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        acc -= gi(i, a) * gi(j, b) * geom.dg[k][adm_sym(a, b)];
                dginv[k][i][j] = acc;
            }

    // d_k ln sqrt(gamma) = (1/2) gamma^ab d_k gamma_ab
    double dlnsq[3];
    for (int k = 0; k < 3; ++k) {
        double acc = 0.0;
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                acc += gi(a, b) * geom.dg[k][adm_sym(a, b)];
        dlnsq[k] = 0.5 * acc;
    }

    // K^ij = gamma^ia gamma^jb K_ab
    double Kuu[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    acc += gi(i, a) * gi(j, b) * geom.K[adm_sym(a, b)];
            Kuu[i][j] = acc;
        }

    //-----------------------------------------------------------------------
    // Theta_A .. Theta_D  (Thornburg 1996, eqs. 15a-d).
    //-----------------------------------------------------------------------
    double theta_A = 0.0, theta_B = 0.0, theta_C = 0.0, theta_D = 0.0;

    // s^i = gamma^ij s_j, needed repeatedly below.
    double su[3];
    for (int i = 0; i < 3; ++i) {
        double acc = 0.0;
        for (int j = 0; j < 3; ++j) acc += gi(i, j) * sd[j];
        su[i] = acc;
    }

    // Theta_D = gamma^ij s_i s_j ,  Theta_C = K^ij s_i s_j
    for (int i = 0; i < 3; ++i) {
        theta_D += su[i] * sd[i];
        for (int j = 0; j < 3; ++j)
            theta_C += Kuu[i][j] * sd[i] * sd[j];
    }

    // Theta_A = - s^i s^j d_i s_j  -  (1/2) s^i (d_i gamma^kl) s_k s_l
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j)
            theta_A -= su[i] * su[j] * dsd[i][j];

        double quad = 0.0;
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l)
                quad += dginv[i][k][l] * sd[k] * sd[l];
        theta_A -= 0.5 * su[i] * quad;
    }

    // Theta_B = (d_i gamma^ij) s_j  +  gamma^ij d_i s_j
    //           +  gamma^ij (d_i ln sqrt(gamma)) s_j
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            theta_B += dginv[i][i][j] * sd[j]
                     + gi(i, j) * dsd[i][j]
                     + gi(i, j) * dlnsq[i] * sd[j];

    out.theta_d = theta_D;

    double const sqrt_D = Kokkos::sqrt(theta_D);
    double const inv_D  = 1.0 / theta_D;

    // D_i n^i split into its two pieces.
    double const div_n = theta_A * inv_D / sqrt_D + theta_B / sqrt_D;
    double const K_nn  = theta_C * inv_D;

    out.mean_curvature = div_n;
    out.theta_out      =  div_n + K_nn - geom.K_trace;
    out.theta_in       = -div_n + K_nn - geom.K_trace;

    return out;
}

/// Convenience: just the outgoing expansion.
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
double expansion_theta(adm_point_t const& geom, shape_point_t const& s,
                       double theta, double phi)
{
    return expansion_at(geom, s, theta, phi).theta_out;
}

}  // namespace grace

#endif  // GRACE_PHYSICS_EXPANSION_HH
