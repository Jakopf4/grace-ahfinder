/**
 * @file adm_from_evolution.hh
 * @brief Physical ADM quantities (gamma_ij, K_ij, d_k gamma_ij) at a point,
 *        reconstructed from whatever the active metric-evolution scheme stores.
 *
 * Under GRACE_METRIC_EVOL == Z4 the evolved variables are conformal, so the
 * physical 3-metric and extrinsic curvature have to be rebuilt, and the metric
 * gradient picks up a chi chain-rule term. Under COWLING the physical ADM
 * variables are stored directly and the conversion is a repack.
 *
 * Every consumer that samples the metric at scattered points needs exactly this
 * conversion, and it has historically been open-coded at each call site with
 * subtly different conventions (notably different floors on chi). Centralising
 * it here keeps those consistent, and pairing the extraction with
 * adm_sample_var_indices() removes the hand-maintained agreement between a
 * variable-index list and a slot enum that the two must otherwise share.
 *
 * Usage:
 *
 *     auto const idx = grace::adm_sample_var_indices();
 *     auto sample    = sampler.sample(points, idx, true);
 *     auto adm       = grace::adm_from_sample(&sample.values(p,0), grads_of_point_p);
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

#ifndef GRACE_PHYSICS_ADM_FROM_EVOLUTION_HH
#define GRACE_PHYSICS_ADM_FROM_EVOLUTION_HH

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>
#include <grace/data_structures/variable_indices.hh>

#include <Kokkos_Core.hpp>

#include <vector>

namespace grace {

/**
 * @brief Symmetric-tensor component order used throughout this header.
 *        (XX, XY, XZ, YY, YZ, ZZ) — the same packing as metric_array_t.
 */
enum adm_sym_idx : int { ADM_XX = 0, ADM_XY, ADM_XZ, ADM_YY, ADM_YZ, ADM_ZZ, ADM_NSYM };

/// Map a pair of Cartesian indices onto the packed symmetric slot.
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
constexpr int adm_sym(int i, int j) {
    // (0,0)->0 (0,1)->1 (0,2)->2 (1,1)->3 (1,2)->4 (2,2)->5
    int const a = (i < j) ? i : j;
    int const b = (i < j) ? j : i;
    return (a == 0) ? b : (a == 1 ? 2 + b : 5);
}

/**
 * @brief Floor applied to the conformal factor chi before dividing by it.
 *
 * Single definition on purpose: the two pre-existing open-coded conversions
 * used different floors (max(chi,1e-100) in one, chi*chi + 1e-200 in the
 * other), which made their results differ near punctures.
 */
inline constexpr double ADM_CHI_FLOOR = 1e-12;

/**
 * @brief Physical ADM geometry at a single point.
 */
struct adm_point_t {
    double g[ADM_NSYM];        ///< gamma_ij
    double ginv[ADM_NSYM];     ///< gamma^ij
    double K[ADM_NSYM];        ///< K_ij
    double dg[3][ADM_NSYM];    ///< d_k gamma_ij
    double dK[3][ADM_NSYM];    ///< d_k K_ij
    double dK_trace[3];        ///< d_k K
    /// d_k d_l gamma_ij, with (k,l) packed symmetrically like (i,j).
    /// Only filled by adm_from_sample_hessian(); zero otherwise.
    double ddg[ADM_NSYM][ADM_NSYM];
    double K_trace;            ///< K = gamma^ij K_ij
    double sqrt_det_g;         ///< sqrt(det gamma_ij)
    /// Whether `ddg` was actually filled. Consumers that need the intrinsic
    /// curvature must check this rather than inspecting `ddg` for nonzero
    /// entries: a genuinely flat region has a zero Hessian legitimately, so
    /// that test cannot tell "not computed" from "computed and zero".
    bool has_hessian = false;
};

namespace detail {

/// Reflect an ADM sample back across the symmetry planes it was folded through.
///
/// `flip` is a bitmask over the three axes. Under a reflection of axis `a`,
/// every tensor component picks up a factor of (-1) for each index equal to
/// `a`; a derivative index counts the same as a tensor index. Scalars (the
/// trace, the volume element) are untouched.
inline void reflect_adm_point(adm_point_t& q, unsigned flip) {
    if (flip == 0u) return;

    double sd[3];
    for (int d = 0; d < 3; ++d) sd[d] = (flip & (1u << d)) ? -1.0 : 1.0;

    // Sign of a symmetric index pair, in the (XX,XY,XZ,YY,YZ,ZZ) packing.
    double ss[ADM_NSYM];
    ss[ADM_XX] = sd[0]*sd[0];  ss[ADM_XY] = sd[0]*sd[1];
    ss[ADM_XZ] = sd[0]*sd[2];  ss[ADM_YY] = sd[1]*sd[1];
    ss[ADM_YZ] = sd[1]*sd[2];  ss[ADM_ZZ] = sd[2]*sd[2];

    for (int a = 0; a < ADM_NSYM; ++a) {
        q.g[a]    *= ss[a];
        q.ginv[a] *= ss[a];
        q.K[a]    *= ss[a];
    }
    for (int d = 0; d < 3; ++d) {
        q.dK_trace[d] *= sd[d];
        for (int a = 0; a < ADM_NSYM; ++a) {
            q.dg[d][a] *= sd[d] * ss[a];
            q.dK[d][a] *= sd[d] * ss[a];
        }
    }
    // (k,l) of d_k d_l gamma_ij are packed symmetrically, exactly like (i,j),
    // so the pair sign applies to both slots.
    for (int c = 0; c < ADM_NSYM; ++c)
        for (int a = 0; a < ADM_NSYM; ++a)
            q.ddg[c][a] *= ss[c] * ss[a];
}

}  // namespace detail

//===========================================================================
// Slot layout of the interpolated sample buffer.
//===========================================================================

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4

/// Slots in the order returned by adm_sample_var_indices().
enum adm_slot_t : int {
    ADM_SLOT_GTXX = 0, ADM_SLOT_GTXY, ADM_SLOT_GTXZ,
    ADM_SLOT_GTYY, ADM_SLOT_GTYZ, ADM_SLOT_GTZZ,
    ADM_SLOT_CHI,
    ADM_SLOT_ATXX, ADM_SLOT_ATXY, ADM_SLOT_ATXZ,
    ADM_SLOT_ATYY, ADM_SLOT_ATYZ, ADM_SLOT_ATZZ,
    ADM_SLOT_KHAT, ADM_SLOT_THETA,
    ADM_N_SLOTS
};

#else  // COWLING

enum adm_slot_t : int {
    ADM_SLOT_GXX = 0, ADM_SLOT_GXY, ADM_SLOT_GXZ,
    ADM_SLOT_GYY, ADM_SLOT_GYZ, ADM_SLOT_GZZ,
    ADM_SLOT_KXX, ADM_SLOT_KXY, ADM_SLOT_KXZ,
    ADM_SLOT_KYY, ADM_SLOT_KYZ, ADM_SLOT_KZZ,
    ADM_N_SLOTS
};

#endif

/**
 * @brief Evolution-variable indices to sample, in the exact slot order that
 *        adm_from_sample() expects.
 *
 * Always obtain the index list from here rather than assembling it at the call
 * site: the ordering is part of this header's contract with adm_from_sample().
 */
inline std::vector<int> adm_sample_var_indices() {
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    return { GTXX_, GTXY_, GTXZ_, GTYY_, GTYZ_, GTZZ_,
             CHI_,
             ATXX_, ATXY_, ATXZ_, ATYY_, ATYZ_, ATZZ_,
             KHAT_, THETA_ };
#else
    return { GXX_, GXY_, GXZ_, GYY_, GYZ_, GZZ_,
             KXX_, KXY_, KXZ_, KYY_, KYZ_, KZZ_ };
#endif
}

/// Number of variables adm_sample_var_indices() returns.
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
constexpr int adm_n_sample_vars() { return ADM_N_SLOTS; }

//===========================================================================
// Conversion
//===========================================================================

namespace detail {

/// Inverse and sqrt(det) of a packed symmetric 3x3, shared by both branches.
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
void adm_invert_metric(double const (&g)[ADM_NSYM],
                       double (&ginv)[ADM_NSYM],
                       double& sqrt_det)
{
    double const det = g[ADM_XX] * (g[ADM_YY]*g[ADM_ZZ] - g[ADM_YZ]*g[ADM_YZ])
                     - g[ADM_XY] * (g[ADM_XY]*g[ADM_ZZ] - g[ADM_XZ]*g[ADM_YZ])
                     + g[ADM_XZ] * (g[ADM_XY]*g[ADM_YZ] - g[ADM_XZ]*g[ADM_YY]);

    double const safe_det = (det > 1e-300) ? det : 1e-300;
    double const inv_det  = 1.0 / safe_det;

    ginv[ADM_XX] = (g[ADM_YY]*g[ADM_ZZ] - g[ADM_YZ]*g[ADM_YZ]) * inv_det;
    ginv[ADM_XY] = (g[ADM_XZ]*g[ADM_YZ] - g[ADM_XY]*g[ADM_ZZ]) * inv_det;
    ginv[ADM_XZ] = (g[ADM_XY]*g[ADM_YZ] - g[ADM_XZ]*g[ADM_YY]) * inv_det;
    ginv[ADM_YY] = (g[ADM_XX]*g[ADM_ZZ] - g[ADM_XZ]*g[ADM_XZ]) * inv_det;
    ginv[ADM_YZ] = (g[ADM_XY]*g[ADM_XZ] - g[ADM_XX]*g[ADM_YZ]) * inv_det;
    ginv[ADM_ZZ] = (g[ADM_XX]*g[ADM_YY] - g[ADM_XY]*g[ADM_XY]) * inv_det;

    sqrt_det = (det > 0.0) ? Kokkos::sqrt(det) : 0.0;
}

/// K = gamma^ij K_ij for packed symmetric tensors (off-diagonals count twice).
GRACE_HOST_DEVICE GRACE_ALWAYS_INLINE
double adm_trace(double const (&ginv)[ADM_NSYM], double const (&K)[ADM_NSYM]) {
    return ginv[ADM_XX]*K[ADM_XX] + ginv[ADM_YY]*K[ADM_YY] + ginv[ADM_ZZ]*K[ADM_ZZ]
         + 2.0 * (ginv[ADM_XY]*K[ADM_XY] + ginv[ADM_XZ]*K[ADM_XZ] + ginv[ADM_YZ]*K[ADM_YZ]);
}

}  // namespace detail

/**
 * @brief Build the physical ADM geometry at one point from an interpolated sample.
 *
 * @param vals  Values, laid out per adm_slot_t; length adm_n_sample_vars().
 * @param grads Cartesian gradients, grads[slot][k] = d_k of that slot. May be
 *              null, in which case dg is zeroed (valid when the consumer needs
 *              only gamma_ij and K_ij, e.g. a surface-area integral).
 *
 * Z4c relations used:
 *     gamma_ij   = gammat_ij / chi^2
 *     d_k gamma_ij = d_k gammat_ij / chi^2  -  2 gammat_ij d_k chi / chi^3
 *     K          = Khat + 2 Theta
 *     K_ij       = At_ij / chi^2 + (1/3) gamma_ij K
 */
GRACE_HOST_DEVICE
inline adm_point_t adm_from_sample(double const* vals, double const (*grads)[3]) {
    adm_point_t out{};

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4

    double const chi      = (vals[ADM_SLOT_CHI] > ADM_CHI_FLOOR)
                          ? vals[ADM_SLOT_CHI] : ADM_CHI_FLOOR;
    double const inv_chi  = 1.0 / chi;
    double const inv_chi2 = inv_chi * inv_chi;
    double const inv_chi3 = inv_chi2 * inv_chi;

    for (int c = 0; c < ADM_NSYM; ++c)
        out.g[c] = vals[ADM_SLOT_GTXX + c] * inv_chi2;

    detail::adm_invert_metric(out.g, out.ginv, out.sqrt_det_g);

    double const K_trace = vals[ADM_SLOT_KHAT] + 2.0 * vals[ADM_SLOT_THETA];
    double const third_K = K_trace / 3.0;
    out.K_trace = K_trace;

    for (int c = 0; c < ADM_NSYM; ++c)
        out.K[c] = vals[ADM_SLOT_ATXX + c] * inv_chi2 + third_K * out.g[c];

    if (grads) {
        for (int k = 0; k < 3; ++k) {
            double const dchi = grads[ADM_SLOT_CHI][k];
            for (int c = 0; c < ADM_NSYM; ++c) {
                out.dg[k][c] = grads[ADM_SLOT_GTXX + c][k] * inv_chi2
                             - 2.0 * vals[ADM_SLOT_GTXX + c] * dchi * inv_chi3;
            }

            // d_k K = d_k Khat + 2 d_k Theta.
            double const dK_tr = grads[ADM_SLOT_KHAT][k]
                               + 2.0 * grads[ADM_SLOT_THETA][k];
            out.dK_trace[k] = dK_tr;

            // K_ij = At_ij/chi^2 + (1/3) gamma_ij K, differentiated.
            //
            // Having this analytically is what lets a horizon solve build the
            // radial part of its Jacobian without the two extra geometry
            // interpolations per Newton step that a numerical d/dr costs.
            for (int c = 0; c < ADM_NSYM; ++c) {
                out.dK[k][c] = grads[ADM_SLOT_ATXX + c][k] * inv_chi2
                             - 2.0 * vals[ADM_SLOT_ATXX + c] * dchi * inv_chi3
                             + (out.dg[k][c] * K_trace + out.g[c] * dK_tr) / 3.0;
            }
        }
    }

#else  // COWLING: the stored variables are already physical.

    for (int c = 0; c < ADM_NSYM; ++c) {
        out.g[c] = vals[ADM_SLOT_GXX + c];
        out.K[c] = vals[ADM_SLOT_KXX + c];
    }

    detail::adm_invert_metric(out.g, out.ginv, out.sqrt_det_g);
    out.K_trace = detail::adm_trace(out.ginv, out.K);

    if (grads) {
        for (int k = 0; k < 3; ++k) {
            for (int c = 0; c < ADM_NSYM; ++c) {
                out.dg[k][c] = grads[ADM_SLOT_GXX + c][k];
                out.dK[k][c] = grads[ADM_SLOT_KXX + c][k];
            }
            // d_k K = (d_k gamma^ij) K_ij + gamma^ij d_k K_ij, with
            // d_k gamma^ij = -gamma^ia gamma^jb d_k gamma_ab.
            double dginv[ADM_NSYM];
            for (int i = 0; i < 3; ++i)
                for (int j = i; j < 3; ++j) {
                    double acc = 0.0;
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            acc -= out.ginv[adm_sym(i,a)] * out.ginv[adm_sym(j,b)]
                                 * out.dg[k][adm_sym(a,b)];
                    dginv[adm_sym(i,j)] = acc;
                }
            out.dK_trace[k] = detail::adm_trace(dginv, out.K)
                            + detail::adm_trace(out.ginv, out.dK[k]);
        }
    }

#endif

    return out;
}

/**
 * @brief As adm_from_sample(), plus the second derivatives of gamma_ij.
 *
 * @param vals  Values, laid out per adm_slot_t.
 * @param grads grads[slot][k] = d_k of that slot.
 * @param hess  hess[slot][kl] = d_k d_l of that slot, (k,l) packed symmetric.
 *
 * The intrinsic curvature of a surface needs d_k d_l gamma_ij: the induced
 * 2-metric's second *angular* derivatives pull it in through the chain rule, and
 * no amount of angular resolution substitutes for it.
 *
 * Under Z4c this is the conformal rescaling differentiated twice:
 *
 *   gamma        = gt / chi^2
 *   d_k gamma    = d_k gt / chi^2  -  2 gt d_k chi / chi^3
 *   d_l d_k gamma= d_l d_k gt / chi^2
 *                  - 2 (d_k gt d_l chi + d_l gt d_k chi) / chi^3
 *                  - 2 gt d_l d_k chi / chi^3
 *                  + 6 gt d_k chi d_l chi / chi^4
 */
GRACE_HOST_DEVICE
inline adm_point_t adm_from_sample_hessian(double const* vals,
                                           double const (*grads)[3],
                                           double const (*hess)[ADM_NSYM])
{
    adm_point_t out = adm_from_sample(vals, grads);
    out.has_hessian = true;

#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4

    double const chi      = (vals[ADM_SLOT_CHI] > ADM_CHI_FLOOR)
                          ? vals[ADM_SLOT_CHI] : ADM_CHI_FLOOR;
    double const inv_chi  = 1.0 / chi;
    double const inv_chi2 = inv_chi * inv_chi;
    double const inv_chi3 = inv_chi2 * inv_chi;
    double const inv_chi4 = inv_chi2 * inv_chi2;

    for (int k = 0; k < 3; ++k) {
        double const dchi_k = grads[ADM_SLOT_CHI][k];
        for (int l = k; l < 3; ++l) {
            double const dchi_l  = grads[ADM_SLOT_CHI][l];
            double const ddchi   = hess[ADM_SLOT_CHI][adm_sym(k, l)];
            int    const kl      = adm_sym(k, l);

            for (int c = 0; c < ADM_NSYM; ++c) {
                double const gt    = vals[ADM_SLOT_GTXX + c];
                double const dgt_k = grads[ADM_SLOT_GTXX + c][k];
                double const dgt_l = grads[ADM_SLOT_GTXX + c][l];
                double const ddgt  = hess[ADM_SLOT_GTXX + c][kl];

                out.ddg[kl][c] = ddgt * inv_chi2
                               - 2.0 * (dgt_k * dchi_l + dgt_l * dchi_k) * inv_chi3
                               - 2.0 * gt * ddchi * inv_chi3
                               + 6.0 * gt * dchi_k * dchi_l * inv_chi4;
            }
        }
    }

#else  // COWLING: gamma_ij is stored directly.

    for (int kl = 0; kl < ADM_NSYM; ++kl)
        for (int c = 0; c < ADM_NSYM; ++c)
            out.ddg[kl][c] = hess[ADM_SLOT_GXX + c][kl];

#endif

    return out;
}

}  // namespace grace

#endif  // GRACE_PHYSICS_ADM_FROM_EVOLUTION_HH
