/**
 * @file surface_geometry.hh
 * @brief Intrinsic geometry of a Strahlkoerper: induced 2-metric, extrinsic
 *        curvature in the slice, and the intrinsic Ricci scalar.
 *
 * The quantity everything here exists to produce is `Rtilde`, the Ricci scalar
 * of the surface's own 2-geometry. It is what the horizon mass multipoles
 * integrate, and what the Gaussian curvature in an approximate-Killing-vector
 * solve is built from.
 *
 * @par Why the Gauss equation rather than differencing the 2-metric
 * Rtilde could be assembled from q_AB and its second *angular* derivatives, but
 * those pull in d_k d_l gamma_ij through the chain rule anyway, and the route is
 * badly conditioned. QuasiLocalMeasures uses the Gauss relation instead
 * (`src/qlm_weyl_scalars1.F90:294-321`) and so does this:
 *
 *     Rtilde = R3 - 2 R_ij s^i s^j + Theta^2 - k_AB k^AB
 *
 * with `R3`, `R_ij` the slice's Ricci tensor and scalar, `s^i` the unit outward
 * normal, `k_AB` the extrinsic curvature of the surface *within* the slice, and
 * `Theta = k^A_A` its trace. Every term is algebraic in quantities the sampler
 * already returns.
 *
 * @par The Gauss-Bonnet check
 * For any topological sphere, whatever its shape and whatever the metric,
 *
 *     integral_S Rtilde dA = 8 pi
 *
 * exactly. That is a complete, analytic-solution-free test of this entire file,
 * and it is the one to run first when anything here is changed.
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

#ifndef GRACE_GEOMETRY_SURFACE_GEOMETRY_HH
#define GRACE_GEOMETRY_SURFACE_GEOMETRY_HH

#include <grace_config.h>

#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/expansion.hh>

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace grace {

/// Intrinsic and extrinsic geometry of the surface at one point.
struct surface_point_geometry_t {
    double q[3];            ///< induced 2-metric (q_tt, q_tp, q_pp)
    double qinv[3];         ///< its inverse
    double sqrt_q;          ///< sqrt(det q)
    double theta_s;         ///< k^A_A: expansion of s^i within the slice
    double k_kk;            ///< k_AB k^AB
    double ricci_scalar;    ///< Rtilde, the intrinsic 2-Ricci scalar
    double gauss_curvature; ///< K_G = Rtilde / 2
    double expansion;       ///< Theta_(l): trace of (k_AB - K_AB)
    double shear2;          ///< sigma_AB sigma^AB of the outgoing null normal
};

namespace detail {

/// 3-Christoffels of gamma_ij from dg.
GRACE_HOST_DEVICE
inline void slice_christoffels(adm_point_t const& g, double (&G)[3][3][3]) {
    for (int a = 0; a < 3; ++a)
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int l = 0; l < 3; ++l)
                    acc += g.ginv[adm_sym(a,l)]
                         * ( g.dg[i][adm_sym(l,j)]
                           + g.dg[j][adm_sym(l,i)]
                           - g.dg[l][adm_sym(i,j)] );
                G[a][i][j] = 0.5 * acc;
            }
}

/// Ricci tensor and scalar of the slice; needs ddg.
GRACE_HOST_DEVICE
inline void slice_ricci(adm_point_t const& g,
                        double (&Ric)[3][3], double& R3)
{
    double G[3][3][3];
    slice_christoffels(g, G);

    // d_k Gamma^a_ij, from ddg via the same expression differentiated once.
    double dG[3][3][3][3];   // [k][a][i][j]
    for (int k = 0; k < 3; ++k) {
        // d_k gamma^{al} = -gamma^{am} gamma^{ln} d_k gamma_mn
        double dginv[ADM_NSYM];
        for (int a = 0; a < 3; ++a)
            for (int l = a; l < 3; ++l) {
                double acc = 0.0;
                for (int m = 0; m < 3; ++m)
                    for (int n = 0; n < 3; ++n)
                        acc -= g.ginv[adm_sym(a,m)] * g.ginv[adm_sym(l,n)]
                             * g.dg[k][adm_sym(m,n)];
                dginv[adm_sym(a,l)] = acc;
            }

        for (int a = 0; a < 3; ++a)
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j) {
                    double acc = 0.0;
                    for (int l = 0; l < 3; ++l) {
                        acc += dginv[adm_sym(a,l)]
                             * ( g.dg[i][adm_sym(l,j)]
                               + g.dg[j][adm_sym(l,i)]
                               - g.dg[l][adm_sym(i,j)] );
                        acc += g.ginv[adm_sym(a,l)]
                             * ( g.ddg[adm_sym(k,i)][adm_sym(l,j)]
                               + g.ddg[adm_sym(k,j)][adm_sym(l,i)]
                               - g.ddg[adm_sym(k,l)][adm_sym(i,j)] );
                    }
                    dG[k][a][i][j] = 0.5 * acc;
                }
    }

    // R_ij = d_a Gamma^a_ij - d_i Gamma^a_aj + Gamma^a_ab Gamma^b_ij
    //                                        - Gamma^a_ib Gamma^b_aj
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int a = 0; a < 3; ++a) {
                acc += dG[a][a][i][j] - dG[i][a][a][j];
                for (int b = 0; b < 3; ++b)
                    acc += G[a][a][b] * G[b][i][j] - G[a][i][b] * G[b][a][j];
            }
            Ric[i][j] = acc;
        }

    R3 = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R3 += g.ginv[adm_sym(i,j)] * Ric[i][j];
}

}  // namespace detail

/**
 * @brief Intrinsic geometry at one surface point.
 *
 * @param geom  ADM data there; `ddg` must be filled (see
 *              adm_from_sample_hessian) or `ricci_scalar` is meaningless.
 * @param s     Surface shape at this angle.
 * @param e     Tangent vectors e^i_A = dx^i/dy^A, A in {theta, phi}.
 */
GRACE_HOST_DEVICE
inline surface_point_geometry_t surface_geometry_at(adm_point_t const& geom,
                                                    shape_point_t const& s,
                                                    double const (&e)[2][3],
                                                    double theta, double phi)
{
    surface_point_geometry_t out{};

    //-- induced 2-metric -------------------------------------------------
    int const pack[3][2] = {{0,0},{0,1},{1,1}};
    for (int c = 0; c < 3; ++c) {
        int const A = pack[c][0], B = pack[c][1];
        double acc = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                acc += geom.g[adm_sym(i,j)] * e[A][i] * e[B][j];
        out.q[c] = acc;
    }

    double const detq = out.q[0]*out.q[2] - out.q[1]*out.q[1];
    double const safe = (detq > 1e-300) ? detq : 1e-300;
    out.sqrt_q  = (detq > 0.0) ? Kokkos::sqrt(detq) : 0.0;
    out.qinv[0] =  out.q[2] / safe;
    out.qinv[1] = -out.q[1] / safe;
    out.qinv[2] =  out.q[0] / safe;

    //-- unit normal and its covariant derivative -------------------------
    auto const nrm = surface_normal_at(geom, s, theta, phi);
    if (!(nrm.norm2 > 0.0)) return out;

    // d_j s_i of the *unnormalised* covector, then normalised: with
    // S_i = s~_i / N and N = sqrt(gamma^kl s~_k s~_l),
    //   d_j S_i = (d_j s~_i)/N - s~_i (d_j N)/N^2 .
    double sd[3];
    detail::normal_covector(s, theta, phi, sd);

    // d_j s~_i, rebuilt the same way expansion_at does.
    double n_[3], eth_[3], eph_[3];
    detail::spherical_triad(theta, phi, n_, eth_, eph_);
    double const st = Kokkos::sin(theta), ct = Kokkos::cos(theta);
    double const inv_r = 1.0 / s.h, inv_st = 1.0 / st;
    double const inv_r2 = inv_r * inv_r, cot = ct * inv_st;

    double dth_[3], dph_[3];
    for (int i = 0; i < 3; ++i) {
        dth_[i] = eth_[i] * inv_r;
        dph_[i] = eph_[i] * inv_r * inv_st;
    }

    double dsd[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double const delta = (i == j) ? 1.0 : 0.0;
            double const ddr  = (delta - n_[i]*n_[j]) * inv_r;
            double const ddth = (-n_[i]*eth_[j] - eth_[i]*n_[j]
                                 + cot * eph_[i]*eph_[j]) * inv_r2;
            double const wi = st*n_[i] + ct*eth_[i];
            double const wj = st*n_[j] + ct*eth_[j];
            double const ddph = -(wi*eph_[j] + eph_[i]*wj) * inv_r2 * inv_st * inv_st;

            dsd[i][j] = ddr - s.dth*ddth - s.dph*ddph
                      - s.dthth * dth_[i]*dth_[j]
                      - s.dthph * (dth_[i]*dph_[j] + dph_[i]*dth_[j])
                      - s.dphph * dph_[i]*dph_[j];
        }

    double const N = Kokkos::sqrt(nrm.norm2);

    // d_j N = (1/2N) d_j (gamma^kl s~_k s~_l)
    double dN[3];
    for (int j = 0; j < 3; ++j) {
        double acc = 0.0;
        for (int k = 0; k < 3; ++k)
            for (int l = 0; l < 3; ++l) {
                // d_j gamma^{kl} = -gamma^{km} gamma^{ln} d_j gamma_mn
                double dgi = 0.0;
                for (int m = 0; m < 3; ++m)
                    for (int nn = 0; nn < 3; ++nn)
                        dgi -= geom.ginv[adm_sym(k,m)] * geom.ginv[adm_sym(l,nn)]
                             * geom.dg[j][adm_sym(m,nn)];
                acc += dgi * sd[k] * sd[l]
                     + 2.0 * geom.ginv[adm_sym(k,l)] * dsd[k][j] * sd[l];
            }
        dN[j] = 0.5 * acc / N;
    }

    double G[3][3][3];
    detail::slice_christoffels(geom, G);

    // D_j S_i = d_j S_i - Gamma^a_ji S_a
    double DS[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double v = dsd[i][j] / N - sd[i] * dN[j] / (N * N);
            for (int a = 0; a < 3; ++a) v -= G[a][j][i] * nrm.lower[a];
            DS[i][j] = v;
        }

    //-- extrinsic curvature of S in the slice ----------------------------
    // DS was built as DS[i][j] = D_j S_i; transpose it so the index order
    // matches the formulae below.
    double DiSj[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) DiSj[i][j] = DS[j][i];   // = D_i S_j

    // k_ij = P_i^k P_j^l D_k S_l, with the projector carrying one index down
    // and one up: P_i^k = delta_i^k - S_i S^k.
    //
    // The transpose, delta - S^i S_k, is a *different* matrix unless S_i S^k
    // happens to be symmetric -- which it is exactly when S^i is parallel to
    // S_i, i.e. in spherical symmetry. Using the wrong one therefore passes
    // every non-spinning test and fails only once the normal stops being
    // radial.
    double Pdu[3][3];
    for (int i = 0; i < 3; ++i)
        for (int k = 0; k < 3; ++k)
            Pdu[i][k] = ((i==k)?1.0:0.0) - nrm.lower[i] * nrm.upper[k];

    double kk[3][3];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    acc += Pdu[i][k] * Pdu[j][l] * DiSj[k][l];
            kk[i][j] = acc;
        }

    // The trace is just gamma^ij D_i S_j: the normal-normal part vanishes
    // because S is a unit vector, so S^i S^j D_i S_j = (1/2) S^i d_i(S.S) = 0.
    out.theta_s = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            out.theta_s += geom.ginv[adm_sym(i,j)] * DiSj[i][j];

    out.k_kk = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                    out.k_kk += geom.ginv[adm_sym(i,k)] * geom.ginv[adm_sym(j,l)]
                              * kk[i][j] * kk[k][l];

    //-- shear of the outgoing null normal --------------------------------
    //
    // The expansion computed elsewhere is the trace of (k_AB - K_AB): with
    // Theta = D_i s^i + K_ij s^i s^j - K and q^ij K_ij = K - K_ss, the two
    // agree.  Its trace-free part is the shear,
    //
    //     sigma_AB = (k_AB - K_AB) - (1/2) q_AB Theta ,
    //
    // and a non-expanding (isolated) horizon has BOTH Theta and sigma zero.
    // Unlike the Weyl-scalar test QuasiLocalMeasures uses for the same purpose,
    // this needs no time derivatives of the metric.
    {
        double KAB[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        acc += Pdu[i][k] * Pdu[j][l] * geom.K[adm_sym(k,l)];
                KAB[i][j] = acc;
            }

        double M[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M[i][j] = kk[i][j] - KAB[i][j];

        double tr = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                tr += geom.ginv[adm_sym(i,j)] * M[i][j];
        out.expansion = tr;

        // Tangential metric with both indices down.
        double Pdd[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                Pdd[i][j] = geom.g[adm_sym(i,j)] - nrm.lower[i] * nrm.lower[j];

        double sig[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                sig[i][j] = M[i][j] - 0.5 * Pdd[i][j] * tr;

        out.shear2 = 0.0;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    for (int l = 0; l < 3; ++l)
                        out.shear2 += geom.ginv[adm_sym(i,k)] * geom.ginv[adm_sym(j,l)]
                                    * sig[i][j] * sig[k][l];
    }

    //-- Gauss equation ---------------------------------------------------
    double Ric[3][3], R3;
    detail::slice_ricci(geom, Ric, R3);

    double Rss = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            Rss += Ric[i][j] * nrm.upper[i] * nrm.upper[j];

    out.ricci_scalar    = R3 - 2.0 * Rss + out.theta_s * out.theta_s - out.k_kk;
    out.gauss_curvature = 0.5 * out.ricci_scalar;

    return out;
}

/**
 * @brief Tangent vectors e^i_A = dx^i/dy^A at every collocation point.
 *
 * Built from the spectral shape derivatives, so they are as accurate as the
 * surface representation itself rather than limited by an angular stencil.
 */
inline void surface_tangents(strahlkoerper_t const& surface,
                             std::vector<std::array<std::array<double,3>,2>>& out)
{
    auto const& b     = surface.basis();
    auto const  shape = surface.evaluate();
    out.resize(b.n_points());

    for (int it = 0; it < b.n_theta(); ++it) {
        double const st = b.sin_theta(it), ct = b.cos_theta(it);
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            double const cp = std::cos(b.phi(ip)), sp = std::sin(b.phi(ip));

            double const n  [3] = { st*cp, st*sp,  ct };
            double const eth[3] = { ct*cp, ct*sp, -st };
            double const eph[3] = {   -sp,    cp, 0.0 };

            double const h = shape.h[p];
            for (int i = 0; i < 3; ++i) {
                out[p][0][i] = shape.dth[p] * n[i] + h * eth[i];
                out[p][1][i] = shape.dph[p] * n[i] + h * st * eph[i];
            }
        }
    }
}

/// Intrinsic geometry at every collocation point of a surface.
inline std::vector<surface_point_geometry_t>
surface_geometry(strahlkoerper_t const& surface,
                 std::vector<adm_point_t> const& geom)
{
    auto const& b     = surface.basis();
    auto const  shape = surface.evaluate();

    std::vector<std::array<std::array<double,3>,2>> tang;
    surface_tangents(surface, tang);

    std::vector<surface_point_geometry_t> out(b.n_points());

    for (int it = 0; it < b.n_theta(); ++it)
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };
            double e[2][3];
            for (int A = 0; A < 2; ++A)
                for (int i = 0; i < 3; ++i) e[A][i] = tang[p][A][i];

            out[p] = surface_geometry_at(geom[p], sp, e, b.theta(it), b.phi(ip));
        }
    return out;
}

/**
 * @brief Proper circumferences of the surface and their ratios.
 *
 * The equatorial circumference is integrated along the collocation ring at
 * theta = pi/2, which exists only when `n_theta` is odd, i.e. `l_max` even.
 * AHFinderDirect and QuasiLocalMeasures both impose the same condition for the
 * same reason. With an odd `l_max` the equatorial entries come back as NaN.
 *
 * The polar circumferences integrate `sqrt(q_thth)` along a meridian and back
 * up the opposite one. Gauss-Legendre nodes cluster in `cos(theta)`, not
 * `theta`, so `sqrt(q_thth)` is first projected onto the spectral basis and
 * then evaluated on a quadrature adapted to `theta` -- rather than reweighting
 * a `cos(theta)` rule, which would have to cope with a 1/sin(theta) endpoint
 * singularity.
 *
 * These ratios were the standard spin proxy before quasi-local spins existed;
 * they are kept for comparison with AHFinderDirect's BH_diagnostics columns
 * 21-25, not because they are the best available spin measure here.
 */
struct surface_circumferences_t {
    double equatorial   = std::numeric_limits<double>::quiet_NaN();
    double polar_xz     = std::numeric_limits<double>::quiet_NaN();
    double polar_yz     = std::numeric_limits<double>::quiet_NaN();
    double ratio_xz_eq  = std::numeric_limits<double>::quiet_NaN();
    double ratio_yz_eq  = std::numeric_limits<double>::quiet_NaN();
};

inline surface_circumferences_t
surface_circumferences(strahlkoerper_t const& surface,
                       std::vector<surface_point_geometry_t> const& sg)
{
    surface_circumferences_t out;
    auto const& b = surface.basis();

    // Equatorial ring, if one exists.
    if (b.n_theta() % 2 == 1) {
        int const it = b.n_theta() / 2;
        double acc = 0.0;
        for (int ip = 0; ip < b.n_phi(); ++ip)
            acc += std::sqrt(std::max(0.0, sg[b.point_index(it, ip)].q[2]));
        out.equatorial = acc * 2.0 * M_PI / b.n_phi();
    }

    // Meridians: project sqrt(q_thth) onto the basis, then quadrature in theta.
    std::vector<double> f(b.n_points());
    for (int p = 0; p < b.n_points(); ++p)
        f[p] = std::sqrt(std::max(0.0, sg[p].q[0]));
    auto const coeffs = surface.project(f);

    int const nq = 64;
    std::vector<double> xq, wq;
    gauss_legendre(nq, xq, wq);

    auto meridian = [&](double phi0) {
        double acc = 0.0;
        for (int i = 0; i < nq; ++i) {
            // map [-1,1] -> [0,pi]
            double const th = 0.5 * M_PI * (xq[i] + 1.0);
            double const jac = 0.5 * M_PI;
            acc += wq[i] * jac * ( b.evaluate_single(coeffs.data(), th, phi0)
                                 + b.evaluate_single(coeffs.data(), th, phi0 + M_PI) );
        }
        return acc;
    };

    out.polar_xz = meridian(0.0);
    out.polar_yz = meridian(0.5 * M_PI);

    if (out.equatorial > 0.0) {
        out.ratio_xz_eq = out.polar_xz / out.equatorial;
        out.ratio_yz_eq = out.polar_yz / out.equatorial;
    }
    return out;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_SURFACE_GEOMETRY_HH
