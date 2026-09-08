/**
 * @file killing_vector.hh
 * @brief Approximate axial Killing vector of a 2-surface (Cook-Whiting).
 *
 * The isolated-horizon angular momentum
 *
 *     J = (1/8 pi) integral_S  phi^i s^j K_ij  dA
 *
 * is only as meaningful as the rotational field phi. On a Kerr horizon a
 * coordinate rotation about the right axis happens to be the exact Killing
 * vector; on a tidally distorted horizon in a binary, nothing coordinate-based
 * is, and the number quietly stops meaning what it should.
 *
 * @par Formulation
 * Write the rotational field as `xi^A = eps^{AB} D_B z`, which is divergence
 * free for any potential `z`, so only the shear can obstruct the Killing
 * equation. Minimising the total shear at fixed normalisation,
 *
 *     I[z] = integral_S sigma_AB sigma^AB dA
 *          = integral_S [ (1/2)(Delta z)^2  -  K (Dz).(Dz) ] dA
 *
 * (the second form follows from the 2D Bochner identity) subject to
 * `integral_S (Dz).(Dz) dA` fixed, gives a symmetric generalised eigenproblem.
 * Expanded in the surface's own spherical-harmonic basis it is dense and small
 * -- `(l_max+1)^2` on a side -- and entirely parallel.
 *
 * Sanity check the derivation reproduces: on a round sphere of radius r,
 * `z = Y_1m` gives `I = 0` exactly, while `l >= 2` gives
 * `l(l+1)[l(l+1)/2 - 1]/r^4 > 0`. So the three `l = 1` rotations are the unique
 * minimisers, as they must be.
 *
 * @par Relation to QuasiLocalMeasures
 * QLM's default is not this: it is Wald Killing *transport* around the
 * coordinate equator with a 3x3 monodromy eigensolve, which is serial, assumes
 * the symmetry axis is z, and which its own author records as "not ideal ... it
 * leads to large fluctuations in the phi direction". The variational form used
 * here has no preferred axis and no path integration.
 *
 * @par Normalisation
 * `xi` is fixed only up to scale by the eigenproblem, and J scales with it, so
 * the normalisation is not cosmetic. The generator of a 2 pi rotation must have
 * orbits of affine period 2 pi. For a Hamiltonian flow on a surface with
 * potential `z` and area form as the symplectic form, the orbit period is
 * `dA/dz`, so demanding `2 pi` is exactly
 *
 *     z_max - z_min = A / (2 pi).
 *
 * That replaces QLM's field-line geodesic integration (`killing_geodesic`,
 * which bails out with "does not enclose the pole" whenever the axis is not z)
 * with a condition read straight off the potential.
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

#ifndef GRACE_GEOMETRY_KILLING_VECTOR_HH
#define GRACE_GEOMETRY_KILLING_VECTOR_HH

#include <grace_config.h>

#include <grace/geometry/strahlkoerper.hh>
#include <grace/geometry/horizon_spin.hh>
#include <grace/geometry/surface_geometry.hh>
#include <grace/physics/adm_from_evolution.hh>
#include <grace/utils/dense_solve.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <vector>

namespace grace {

struct akv_result_t {
    bool found = false;

    /// Minimised shear functional. Zero for a genuine Killing vector, so this
    /// is the quality measure: a J computed from an xi with a large value here
    /// should not be trusted.
    double shear_functional = 0.0;

    /// Potential z at the collocation points, normalised.
    std::vector<double> z;
    /// xi^A = (xi^theta, xi^phi) at the collocation points.
    std::vector<std::array<double,2>> xi;
    /// The same field pushed into 3-space, phi^i = e^i_A xi^A (tangent to S).
    std::vector<std::array<double,3>> phi;
    /// Invariant coordinate zeta = z/R^2, centred so that its area integral
    /// vanishes. Spans [-1, 1] on a Kerr horizon.
    std::vector<double> zeta;

    /// Dimensionless measure of how far xi is from Killing:
    ///     sqrt( integral sigma^2 dA / integral xi.xi dA ) * R_areal
    /// Zero for a genuine Killing vector. This is the number to check before
    /// trusting a spin computed from this field.
    double killing_defect = 0.0;

    /// Gap to the next eigenvalue. A small gap means the rotation axis is
    /// poorly determined -- which is not a failure but a statement about the
    /// surface: a round sphere has three degenerate Killing vectors and no
    /// preferred axis at all.
    double eigenvalue_gap = 0.0;
};

namespace detail {

/// Angular derivatives of a collocation field, by projection onto the basis.
inline void angular_derivatives(strahlkoerper_t const& surface,
                                std::vector<double> const& f,
                                std::vector<double>& dth,
                                std::vector<double>& dph)
{
    auto const& b  = surface.basis();
    int const   np = b.n_points();
    int const   nc = b.n_coeffs();

    auto const a = surface.project(f);

    double const* dYt = b.dY_dth_matrix();
    double const* dYp = b.dY_dph_matrix();

    dth.assign(np, 0.0);
    dph.assign(np, 0.0);
    for (int p = 0; p < np; ++p) {
        std::size_t const row = static_cast<std::size_t>(p) * nc;
        double s1 = 0.0, s2 = 0.0;
        for (int c = 0; c < nc; ++c) {
            s1 += dYt[row + c] * a[c];
            s2 += dYp[row + c] * a[c];
        }
        dth[p] = s1;
        dph[p] = s2;
    }
}

}  // namespace detail

/**
 * @brief Solve for the approximate Killing vector of a surface.
 *
 * @param surface Surface to solve on.
 * @param geom    ADM data at its collocation points (needs `ddg`).
 * @param sg      Intrinsic geometry, from surface_geometry().
 */
inline akv_result_t approximate_killing_vector(
    strahlkoerper_t const& surface,
    std::vector<adm_point_t> const& geom,
    std::vector<surface_point_geometry_t> const& sg)
{
    akv_result_t out;

    auto const& b  = surface.basis();
    int const   np = b.n_points();
    int const   nc = b.n_coeffs();
    if (static_cast<int>(sg.size()) != np) return out;

    //-----------------------------------------------------------------------
    // Per-point surface quantities.
    //-----------------------------------------------------------------------
    std::vector<double> W(np);        // quadrature weight including dA
    std::vector<double> Ktilde(np);   // Gaussian curvature
    std::vector<double> qtt(np), qtp(np), qpp(np), sq(np);

    for (int it = 0; it < b.n_theta(); ++it) {
        double const inv_st = 1.0 / b.sin_theta(it);
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);
            qtt[p]    = sg[p].qinv[0];
            qtp[p]    = sg[p].qinv[1];
            qpp[p]    = sg[p].qinv[2];
            sq[p]     = sg[p].sqrt_q;
            Ktilde[p] = sg[p].gauss_curvature;
            // strahlkoerper_t::integrate uses weight * f * dA / sin(theta);
            // the area element here is sqrt(det q).
            W[p] = b.weight(it, ip) * sg[p].sqrt_q * inv_st;
        }
    }

    double area = 0.0;
    for (int p = 0; p < np; ++p) area += W[p];

    //-----------------------------------------------------------------------
    // Laplacian of the basis functions:
    //
    //     Delta_q Y = q^AB ( d_A d_B Y  -  Gamma^C_AB d_C Y )
    //
    // The contracted Christoffels are built from the *lower-index* 2-metric.
    // That matters: q_AB is smooth and band-limited (q_pp ~ sin^2 theta), so
    // projecting it onto the harmonic basis to differentiate is well posed,
    // whereas the inverse components carry 1/sin^2 theta and the combination
    // sqrt(q) q^pp is 1/sin(theta) -- singular at the poles, hopeless to
    // represent in a finite basis, and enough to contaminate the eigenvector
    // and drive the (necessarily non-negative) shear functional negative.
    // The inverse metric still appears, but only evaluated at collocation
    // points, which Gauss-Legendre keeps off the axis.
    //-----------------------------------------------------------------------
    std::vector<double> Qtt(np), Qtp(np), Qpp(np);
    for (int p = 0; p < np; ++p) {
        Qtt[p] = sg[p].q[0];
        Qtp[p] = sg[p].q[1];
        Qpp[p] = sg[p].q[2];
    }
    std::vector<double> dQtt_t, dQtt_p, dQtp_t, dQtp_p, dQpp_t, dQpp_p;
    detail::angular_derivatives(surface, Qtt, dQtt_t, dQtt_p);
    detail::angular_derivatives(surface, Qtp, dQtp_t, dQtp_p);
    detail::angular_derivatives(surface, Qpp, dQpp_t, dQpp_p);

    // Gt = q^AB Gamma^theta_AB ,  Gp = q^AB Gamma^phi_AB
    std::vector<double> Gt(np), Gp(np);
    for (int p = 0; p < np; ++p) {
        // dq[A][B][C] = d_A q_BC
        double dq[2][2][2];
        dq[0][0][0] = dQtt_t[p]; dq[0][0][1] = dQtp_t[p]; dq[0][1][1] = dQpp_t[p];
        dq[1][0][0] = dQtt_p[p]; dq[1][0][1] = dQtp_p[p]; dq[1][1][1] = dQpp_p[p];
        dq[0][1][0] = dq[0][0][1];
        dq[1][1][0] = dq[1][0][1];

        double const qu[2][2] = {{qtt[p], qtp[p]}, {qtp[p], qpp[p]}};

        double G[2] = {0.0, 0.0};
        for (int C = 0; C < 2; ++C)
            for (int A = 0; A < 2; ++A)
                for (int Bx = 0; Bx < 2; ++Bx) {
                    double gamma = 0.0;
                    for (int D = 0; D < 2; ++D)
                        gamma += qu[C][D] * ( dq[A][D][Bx] + dq[Bx][D][A]
                                            - dq[D][A][Bx] );
                    G[C] += qu[A][Bx] * 0.5 * gamma;
                }
        Gt[p] = G[0];
        Gp[p] = G[1];
    }

    double const* Ym   = b.Y_matrix();
    double const* dYt  = b.dY_dth_matrix();
    double const* dYp  = b.dY_dph_matrix();
    double const* d2tt = b.d2Y_dth2_matrix();
    double const* d2tp = b.d2Y_dthdph_matrix();
    double const* d2pp = b.d2Y_dph2_matrix();

    // L[p][c] = Delta_q Y_c at p.
    std::vector<double> L(static_cast<std::size_t>(np) * nc);
    for (int p = 0; p < np; ++p) {
        std::size_t const row = static_cast<std::size_t>(p) * nc;
        for (int c = 0; c < nc; ++c) {
            L[row + c] = qtt[p] * d2tt[row + c]
                       + 2.0 * qtp[p] * d2tp[row + c]
                       + qpp[p] * d2pp[row + c]
                       - Gt[p] * dYt[row + c]
                       - Gp[p] * dYp[row + c];
        }
    }

    //-----------------------------------------------------------------------
    // Assemble the eigenproblem, dropping the constant mode (c = 0): it has
    // Dz = 0, which would make B singular.
    //-----------------------------------------------------------------------
    int const n = nc - 1;
    if (n < 3) return out;

    std::vector<double> A(static_cast<std::size_t>(n) * n, 0.0);
    std::vector<double> B(static_cast<std::size_t>(n) * n, 0.0);

    for (int p = 0; p < np; ++p) {
        std::size_t const row = static_cast<std::size_t>(p) * nc;
        double const w = W[p];

        for (int i = 0; i < n; ++i) {
            int const ci = i + 1;
            double const Li  = L[row + ci];
            double const gti = dYt[row + ci];
            double const gpi = dYp[row + ci];

            for (int j = i; j < n; ++j) {
                int const cj = j + 1;
                double const Lj  = L[row + cj];
                double const gtj = dYt[row + cj];
                double const gpj = dYp[row + cj];

                // q^{AB} d_A Y_i d_B Y_j
                double const gg = qtt[p] * gti * gtj
                                + qtp[p] * (gti * gpj + gpi * gtj)
                                + qpp[p] * gpi * gpj;

                A[static_cast<std::size_t>(i)*n + j] +=
                    w * (0.5 * Li * Lj - Ktilde[p] * gg);
                B[static_cast<std::size_t>(i)*n + j] += w * gg;
            }
        }
    }
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) {
            A[static_cast<std::size_t>(j)*n + i] = A[static_cast<std::size_t>(i)*n + j];
            B[static_cast<std::size_t>(j)*n + i] = B[static_cast<std::size_t>(i)*n + j];
        }

    std::vector<double> eval, evec;
    if (!generalized_symmetric_eigen(A, B, n, eval, evec)) return out;

    // Smallest eigenvalue: the least-sheared rotational field available.
    int const best = 0;
    out.shear_functional = eval[best];

    std::vector<double> a(nc, 0.0);
    for (int i = 0; i < n; ++i) a[i + 1] = evec[static_cast<std::size_t>(i)*n + best];

    //-----------------------------------------------------------------------
    // Normalise so that the orbits have affine period 2 pi.
    //
    // z is evaluated on a grid several times finer than the collocation set:
    // Gauss-Legendre nodes never reach the poles, where z is extremal for an
    // axisymmetric surface, so taking the extrema over collocation points alone
    // would under-estimate the span by ~1% and bias J by the same amount.
    //-----------------------------------------------------------------------
    {
        int const nth = 4 * b.n_theta() + 1;
        int const nph = 4 * b.n_phi();
        double zmin =  1e300, zmax = -1e300;
        for (int i = 0; i < nth; ++i) {
            double const th = M_PI * (i + 0.5) / nth;
            for (int j = 0; j < nph; ++j) {
                double const ph = 2.0 * M_PI * j / nph;
                double const v = b.evaluate_single(a.data(), th, ph);
                zmin = std::min(zmin, v);
                zmax = std::max(zmax, v);
            }
        }
        double const span = zmax - zmin;
        if (!(span > 0.0)) return out;
        double scale = (area / (2.0 * M_PI)) / span;

        // Orientation: xi should circulate the same way as increasing phi.
        // For xi^phi = -d_theta z / sqrt(q), that means z decreasing with theta.
        {
            std::vector<double> ztmp(np);
            for (int p = 0; p < np; ++p) {
                std::size_t const row = static_cast<std::size_t>(p) * nc;
                double s = 0.0;
                for (int c = 0; c < nc; ++c) s += Ym[row + c] * a[c];
                ztmp[p] = s;
            }
            std::vector<double> dzt, dzp;
            detail::angular_derivatives(surface, ztmp, dzt, dzp);
            double acc = 0.0;
            for (int p = 0; p < np; ++p) acc += W[p] * (-dzt[p] / std::max(sq[p], 1e-300));
            if (acc < 0.0) scale = -scale;
        }

        for (int c = 0; c < nc; ++c) a[c] *= scale;
    }

    //-----------------------------------------------------------------------
    // Fields.
    //-----------------------------------------------------------------------
    out.z.assign(np, 0.0);
    for (int p = 0; p < np; ++p) {
        std::size_t const row = static_cast<std::size_t>(p) * nc;
        double s = 0.0;
        for (int c = 0; c < nc; ++c) s += Ym[row + c] * a[c];
        out.z[p] = s;
    }

    std::vector<double> dzt, dzp;
    detail::angular_derivatives(surface, out.z, dzt, dzp);

    // xi^A = eps^{AB} D_B z, with eps^{tp} = 1/sqrt(q).
    out.xi.assign(np, {0.0, 0.0});
    for (int p = 0; p < np; ++p) {
        double const inv_sq = (sq[p] > 0.0) ? 1.0 / sq[p] : 0.0;
        out.xi[p][0] =  dzp[p] * inv_sq;   // xi^theta
        out.xi[p][1] = -dzt[p] * inv_sq;   // xi^phi
    }

    // phi^i = e^i_A xi^A.
    std::vector<std::array<std::array<double,3>,2>> tang;
    surface_tangents(surface, tang);
    out.phi.assign(np, {0.0, 0.0, 0.0});
    for (int p = 0; p < np; ++p)
        for (int i = 0; i < 3; ++i)
            out.phi[p][i] = tang[p][0][i] * out.xi[p][0]
                          + tang[p][1][i] * out.xi[p][1];

    // Invariant coordinate.
    {
        double const R2 = area / (4.0 * M_PI);
        out.zeta.assign(np, 0.0);
        double mean = 0.0;
        for (int p = 0; p < np; ++p) mean += W[p] * out.z[p];
        mean /= area;
        for (int p = 0; p < np; ++p) out.zeta[p] = (out.z[p] - mean) / R2;
    }

    //-----------------------------------------------------------------------
    // Quality measure.
    //
    // The eigenvalue *is* the integrated shear per unit rotation,
    //     lambda = integral sigma_AB sigma^AB dA / integral xi.xi dA ,
    // since q_AB eps^{AC} eps^{BD} = q^{CD} makes xi.xi equal to (Dz).(Dz).
    // Rendered dimensionless with the areal radius it is directly comparable
    // between surfaces.
    //
    // Deliberately not the pointwise L_xi q_AB in coordinate components that
    // QuasiLocalMeasures reports: for a rotation about anything other than the
    // polar axis, xi^phi carries a cot(theta) and those components are singular
    // on the axis even when the vector field itself is perfectly regular.
    //-----------------------------------------------------------------------
    {
        double const R = std::sqrt(area / (4.0 * M_PI));
        double const lam = std::max(0.0, out.shear_functional);
        out.killing_defect = std::sqrt(lam) * R;
        out.eigenvalue_gap = (n > 1) ? (eval[1] - eval[0]) : 0.0;
    }

    out.found = true;
    return out;
}

/**
 * @brief Quasi-local spin using an approximate Killing vector.
 *
 * The isolated-horizon formalism proper: unlike the coordinate-generator
 * version in horizon_spin.hh, this makes no assumption that any coordinate
 * rotation is a symmetry of the surface. On a Kerr horizon the two agree,
 * because there a coordinate rotation *is* the Killing vector; on a distorted
 * horizon they part company and this one is the meaningful number -- provided
 * `killing_defect` is small.
 */
inline horizon_spin_akv_t horizon_spin_akv(
    strahlkoerper_t const& surface,
    std::vector<adm_point_t> const& geom,
    std::vector<surface_point_geometry_t> const& sg,
    double m_irr)
{
    horizon_spin_akv_t out;

    auto const akv = approximate_killing_vector(surface, geom, sg);
    if (!akv.found) return out;

    out.valid          = true;
    out.J              = spin_from_rotational_field(surface, geom, akv.phi);
    out.killing_defect = akv.killing_defect;
    out.eigenvalue_gap = akv.eigenvalue_gap;

    if (m_irr > 0.0) {
        double const m2 = m_irr * m_irr + out.J * out.J / (4.0 * m_irr * m_irr);
        out.christodoulou_mass = std::sqrt(m2);
        if (m2 > 0.0) out.dimensionless = std::abs(out.J) / m2;
    }
    return out;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_KILLING_VECTOR_HH
