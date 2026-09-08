/**
 * @file analytic_geometry.hh
 * @brief Closed-form ADM geometry at arbitrary points, bypassing the grid.
 *
 * Fills the same adm_point_t that the interpolation path produces, but from an
 * analytic spacetime rather than from evolved data. That separation is what
 * makes surface-side machinery (Strahlkoerper areas, expansions, a horizon
 * Newton solve) testable on its own: a disagreement against these values is a
 * bug in the surface code, whereas a disagreement between these values and the
 * sampled ones is a bug in the interpolation or the ADM reconstruction.
 *
 * The same idea as AHFinderDirect's src/gr/Schwarzschild_EF.cc, which exists
 * for exactly this reason.
 *
 * Kerr-Schild coordinates are horizon-penetrating, so unlike Schwarzschild or
 * Boyer-Lindquist the geometry is regular on and inside the horizon and a
 * surface search can cross it safely.
 *
 * Host-only: this is test and validation scaffolding, not an evolution path.
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

#ifndef GRACE_PHYSICS_ANALYTIC_GEOMETRY_HH
#define GRACE_PHYSICS_ANALYTIC_GEOMETRY_HH

#include <grace_config.h>

#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/id/kerr_schild_subexpressions.hh>

#include <array>
#include <cmath>
#include <vector>

namespace grace {

/**
 * @brief Kerr in Cartesian Kerr-Schild coordinates.
 *
 * The generated kernel in kerr_schild_subexpressions.hh hardcodes M = 1 (see
 * the `- 2*r` term in its Boyer-Lindquist radius). Other masses are obtained
 * from the exact scaling of the vacuum solution: with lengths measured in units
 * of M, the 3-metric is dimensionless and the extrinsic curvature carries one
 * inverse power of length, so
 *
 *     gamma_ij(x; M, a)   = gamma_ij(x/M; 1, a/M)
 *     K_ij(x; M, a)       = K_ij(x/M; 1, a/M) / M
 *     d_k gamma_ij(x;M,a) = d_k gamma_ij(x/M; 1, a/M) / M
 *
 * @param spin Dimensionless spin a/M, in [0, 1).
 */
struct kerr_schild_geometry_t {

    explicit kerr_schild_geometry_t(double mass = 1.0, double spin = 0.0)
        : _mass(mass), _spin(spin)
    {}

    /// Boyer-Lindquist radius of the outer horizon, r+ = M + sqrt(M^2 - a^2).
    double horizon_radius() const {
        double const a = _spin * _mass;
        return _mass + std::sqrt(std::max(0.0, _mass * _mass - a * a));
    }

    /// Proper area of the event horizon, A = 8 pi M r+.
    double horizon_area() const {
        return 8.0 * M_PI * _mass * horizon_radius();
    }

    /// Irreducible mass, sqrt(A / 16 pi).
    double irreducible_mass() const {
        return std::sqrt(horizon_area() / (16.0 * M_PI));
    }

    /**
     * @brief Coordinate radius of the horizon in the direction (theta, phi).
     *
     * Angles here are ordinary spherical-polar angles of the Cartesian point,
     * NOT Boyer-Lindquist angles -- the two differ for a != 0, and conflating
     * them yields a surface that is close to, but not, the horizon.
     *
     * In Cartesian Kerr-Schild coordinates the horizon is the level set
     * r_BL = r+, i.e. the ellipsoid
     *     (x^2 + y^2) / (r+^2 + a^2)  +  z^2 / r+^2  =  1.
     * Substituting x = R sin(theta) cos(phi) etc. and solving for R gives the
     * expression below. It reduces to R = r+ when a = 0.
     */
    double horizon_coordinate_radius(double theta, double /*phi*/) const {
        double const a  = _spin * _mass;
        double const rp = horizon_radius();
        double const s  = std::sin(theta);
        double const c  = std::cos(theta);
        return 1.0 / std::sqrt(s * s / (rp * rp + a * a) + c * c / (rp * rp));
    }

    /// Cartesian point on the horizon in the direction (theta, phi).
    std::array<double,3> horizon_point(double theta, double phi) const {
        double const R = horizon_coordinate_radius(theta, phi);
        double const s = std::sin(theta);
        return { R * s * std::cos(phi), R * s * std::sin(phi), R * std::cos(theta) };
    }

    /**
     * @brief ADM geometry at one point.
     *
     * Metric gradients come from centred differences of the analytic metric.
     * Differentiating the generated expressions symbolically would be exact,
     * but this is validation scaffolding: the finite-difference error (~1e-10
     * with the default step) is far below the accuracy any surface code needs
     * to be checked to, and it keeps this file independent of the code
     * generator.
     */
    adm_point_t at(std::array<double,3> const& xyz) const;

    /// ADM geometry at many points.
    std::vector<adm_point_t> at(std::vector<std::array<double,3>> const& xyz) const {
        std::vector<adm_point_t> out;
        out.reserve(xyz.size());
        for (auto const& p : xyz) out.push_back(at(p));
        return out;
    }

    double mass() const { return _mass; }
    double spin() const { return _spin; }

private:
    /// gamma_ij and K_ij only (no gradients), used by at() and its differences.
    void metric_at(std::array<double,3> const& xyz,
                   double (&g)[ADM_NSYM], double (&K)[ADM_NSYM]) const;

    double _mass;
    double _spin;

    /// Step for the centred metric differences, in units of M.  Second
    /// differences amplify round-off as 1/h^2, so this is looser than a
    /// first-derivative-only stencil would want: at 1e-4 the gradient carries
    /// ~1e-8 truncation and the Hessian ~1e-8 round-off, which is the balance
    /// point for double precision.
    static constexpr double _fd_step = 1e-4;
    /// Radius floor handed to the generated kernel, well inside the horizon.
    static constexpr double _r_eps = 1e-8;
};

//---------------------------------------------------------------------------

inline void kerr_schild_geometry_t::metric_at(
    std::array<double,3> const& xyz,
    double (&g)[ADM_NSYM], double (&K)[ADM_NSYM]) const
{
    // Evaluate the unit-mass solution at the rescaled point.
    double const inv_m = 1.0 / _mass;
    double const x[3]  = { xyz[0] * inv_m, xyz[1] * inv_m, xyz[2] * inv_m };

    double gxx, gxy, gxz, gyy, gyz, gzz;
    double gXX, gXY, gXZ, gYY, gYZ, gZZ;
    double alp, betaX, betaY, betaZ;
    double Kxx, Kxy, Kxz, Kyy, Kyz, Kzz;

    kerr_schild_adm_metric(x, _spin, _r_eps,
                           &gxx, &gxy, &gxz, &gyy, &gyz, &gzz,
                           &gXX, &gXY, &gXZ, &gYY, &gYZ, &gZZ,
                           &alp, &betaX, &betaY, &betaZ,
                           &Kxx, &Kxy, &Kxz, &Kyy, &Kyz, &Kzz);

    g[ADM_XX] = gxx; g[ADM_XY] = gxy; g[ADM_XZ] = gxz;
    g[ADM_YY] = gyy; g[ADM_YZ] = gyz; g[ADM_ZZ] = gzz;

    // K_ij carries one inverse power of length.
    K[ADM_XX] = Kxx * inv_m; K[ADM_XY] = Kxy * inv_m; K[ADM_XZ] = Kxz * inv_m;
    K[ADM_YY] = Kyy * inv_m; K[ADM_YZ] = Kyz * inv_m; K[ADM_ZZ] = Kzz * inv_m;
}

inline adm_point_t kerr_schild_geometry_t::at(std::array<double,3> const& xyz) const
{
    adm_point_t out{};

    metric_at(xyz, out.g, out.K);
    detail::adm_invert_metric(out.g, out.ginv, out.sqrt_det_g);
    out.K_trace     = detail::adm_trace(out.ginv, out.K);
    out.has_hessian = true;

    double const h = _fd_step * _mass;
    for (int k = 0; k < 3; ++k) {
        auto xp = xyz; xp[k] += h;
        auto xm = xyz; xm[k] -= h;

        double gp[ADM_NSYM], gm[ADM_NSYM], Kp[ADM_NSYM], Km[ADM_NSYM];
        metric_at(xp, gp, Kp);
        metric_at(xm, gm, Km);

        for (int c = 0; c < ADM_NSYM; ++c) {
            out.dg[k][c] = (gp[c] - gm[c]) / (2.0 * h);
            out.dK[k][c] = (Kp[c] - Km[c]) / (2.0 * h);
        }

        // d_k K, from the traces at the displaced points rather than by
        // differentiating the product rule -- fewer places to get it wrong.
        double gip[ADM_NSYM], gim[ADM_NSYM], sp_, sm_;
        detail::adm_invert_metric(gp, gip, sp_);
        detail::adm_invert_metric(gm, gim, sm_);
        out.dK_trace[k] = (detail::adm_trace(gip, Kp)
                         - detail::adm_trace(gim, Km)) / (2.0 * h);

        // d_k d_l gamma_ij.  Diagonal (l == k) from the three-point stencil
        // already in hand; mixed terms need the four corner evaluations.
        for (int c = 0; c < ADM_NSYM; ++c)
            out.ddg[adm_sym(k,k)][c] = (gp[c] - 2.0 * out.g[c] + gm[c]) / (h * h);

        for (int l = k + 1; l < 3; ++l) {
            auto xpp = xyz; xpp[k] += h; xpp[l] += h;
            auto xpm = xyz; xpm[k] += h; xpm[l] -= h;
            auto xmp = xyz; xmp[k] -= h; xmp[l] += h;
            auto xmm = xyz; xmm[k] -= h; xmm[l] -= h;

            double a[ADM_NSYM], b[ADM_NSYM], c2[ADM_NSYM], d[ADM_NSYM], Kd[ADM_NSYM];
            metric_at(xpp, a,  Kd);
            metric_at(xpm, b,  Kd);
            metric_at(xmp, c2, Kd);
            metric_at(xmm, d,  Kd);

            for (int c = 0; c < ADM_NSYM; ++c)
                out.ddg[adm_sym(k,l)][c] =
                    (a[c] - b[c] - c2[c] + d[c]) / (4.0 * h * h);
        }
    }

    return out;
}

/**
 * @brief Brill-Lindquist: two (or more) black holes on a time-symmetric slice.
 *
 *     psi = 1 + sum_i m_i / (2 r_i)
 *     gamma_ij = psi^4 delta_ij ,   K_ij = 0
 *
 * Analytic, needs no external initial-data library, and is the only
 * multi-black-hole geometry available in-tree: `physics/id/puncture.hh` is
 * single-puncture and `two_punctures` requires an optional external library.
 *
 * Being time-symmetric, `K_ij` vanishes, so an apparent horizon here is a
 * minimal surface and the expansion reduces to the mean curvature alone. That
 * complements Kerr-Schild, where the `K_ij n^i n^j - K` terms carry most of the
 * expansion: between them the two exercise both halves of Theta.
 *
 * The physics this exists to test: for two holes far apart there are two
 * separate horizons and no common one; brought close enough together a common
 * horizon appears enclosing both, and in the coincidence limit it becomes the
 * Schwarzschild horizon of the summed mass.
 *
 * Note `m_i` are the *bare* puncture masses, not the horizon masses. Only in
 * the well-separated limit does each horizon approach `M_irr = m_i`.
 */
class brill_lindquist_geometry_t {
public:
    struct puncture_t {
        double               mass = 1.0;
        std::array<double,3> position{};
    };

    brill_lindquist_geometry_t() = default;

    explicit brill_lindquist_geometry_t(std::vector<puncture_t> punctures)
        : _punctures(std::move(punctures))
    {}

    /// Two equal-mass punctures on the x axis at +/- separation/2.
    static brill_lindquist_geometry_t equal_mass_pair(double mass,
                                                      double separation) {
        return brill_lindquist_geometry_t({
            puncture_t{ mass, {-0.5 * separation, 0.0, 0.0} },
            puncture_t{ mass, { 0.5 * separation, 0.0, 0.0} } });
    }

    std::vector<puncture_t> const& punctures() const { return _punctures; }

    double total_bare_mass() const {
        double m = 0.0;
        for (auto const& p : _punctures) m += p.mass;
        return m;
    }

    /// Conformal factor. Diverges at each puncture, which is physical: the
    /// punctures are infinities of the conformal factor, not of the geometry.
    double conformal_factor(std::array<double,3> const& xyz) const {
        double psi = 1.0;
        for (auto const& p : _punctures) {
            double const dx = xyz[0] - p.position[0];
            double const dy = xyz[1] - p.position[1];
            double const dz = xyz[2] - p.position[2];
            double const r  = std::sqrt(dx*dx + dy*dy + dz*dz);
            psi += 0.5 * p.mass / std::max(r, _r_eps);
        }
        return psi;
    }

    adm_point_t at(std::array<double,3> const& xyz) const;

    std::vector<adm_point_t> at(std::vector<std::array<double,3>> const& xyz) const {
        std::vector<adm_point_t> out;
        out.reserve(xyz.size());
        for (auto const& p : xyz) out.push_back(at(p));
        return out;
    }

private:
    std::vector<puncture_t> _punctures;

    /// Differencing step; see the note on kerr_schild_geometry_t::_fd_step.
    static constexpr double _fd_step = 1e-4;
    /// Floor on the distance to a puncture.
    static constexpr double _r_eps = 1e-10;
};

inline adm_point_t
brill_lindquist_geometry_t::at(std::array<double,3> const& xyz) const
{
    adm_point_t out{};

    auto metric_at = [this](std::array<double,3> const& x, double (&g)[ADM_NSYM]) {
        double const psi  = conformal_factor(x);
        double const psi4 = psi * psi * psi * psi;
        for (int c = 0; c < ADM_NSYM; ++c) g[c] = 0.0;
        g[ADM_XX] = g[ADM_YY] = g[ADM_ZZ] = psi4;
    };

    metric_at(xyz, out.g);
    detail::adm_invert_metric(out.g, out.ginv, out.sqrt_det_g);

    // Time symmetric: K_ij = 0, hence K = 0 and all its derivatives vanish.
    out.K_trace     = 0.0;
    out.has_hessian = true;

    double const h = _fd_step;
    for (int k = 0; k < 3; ++k) {
        auto xp = xyz; xp[k] += h;
        auto xm = xyz; xm[k] -= h;

        double gp[ADM_NSYM], gm[ADM_NSYM];
        metric_at(xp, gp);
        metric_at(xm, gm);

        for (int c = 0; c < ADM_NSYM; ++c) {
            out.dg[k][c] = (gp[c] - gm[c]) / (2.0 * h);
            out.ddg[adm_sym(k,k)][c] = (gp[c] - 2.0 * out.g[c] + gm[c]) / (h * h);
        }

        for (int l = k + 1; l < 3; ++l) {
            auto xpp = xyz; xpp[k] += h; xpp[l] += h;
            auto xpm = xyz; xpm[k] += h; xpm[l] -= h;
            auto xmp = xyz; xmp[k] -= h; xmp[l] += h;
            auto xmm = xyz; xmm[k] -= h; xmm[l] -= h;

            double a[ADM_NSYM], b[ADM_NSYM], c2[ADM_NSYM], d[ADM_NSYM];
            metric_at(xpp, a);
            metric_at(xpm, b);
            metric_at(xmp, c2);
            metric_at(xmm, d);

            for (int c = 0; c < ADM_NSYM; ++c)
                out.ddg[adm_sym(k,l)][c] =
                    (a[c] - b[c] - c2[c] + d[c]) / (4.0 * h * h);
        }
    }

    return out;
}

}  // namespace grace

#endif  // GRACE_PHYSICS_ANALYTIC_GEOMETRY_HH
