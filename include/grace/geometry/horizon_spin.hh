/**
 * @file horizon_spin.hh
 * @brief Quasi-local angular momentum of a horizon, isolated-horizon formalism.
 *
 * For a rotational vector field phi^i tangent to the surface S,
 *
 *     J[phi] = (1/8 pi) * integral_S  phi^i s^j K_ij  dA
 *
 * with s^i the unit outward normal within the slice. This is the
 * Ashtekar-Krishnan / Dreyer-Krishnan-Shoemaker-Schnetter expression. Note that
 * only K_ij and the surface geometry enter -- lapse and shift do not, exactly as
 * for the expansion.
 *
 * @par Which phi
 * The formalism wants an axial *Killing* vector of the horizon's intrinsic
 * geometry. Solving for an approximate one (Cook-Whiting, or Killing transport)
 * means an eigenvalue problem on S^2; what is done here instead is to use the
 * three coordinate rotational generators about the horizon centroid,
 *
 *     phi^i_(a) = eps^{ijk} e_(a)j (x - c)_k ,
 *
 * projected tangent to S. That is not a shortcut to one component: J[phi] is
 * *linear* in the rotation axis, so the three integrals assemble a vector J^i,
 * and its magnitude and direction are the spin and the spin axis for a hole
 * whose spin points anywhere -- not only along z. For Kerr in Kerr-Schild
 * coordinates the coordinate rotation about the correct axis *is* the Killing
 * vector, so the result is exact there whatever the orientation.
 *
 * @par What this does not capture
 * For a horizon that is genuinely non-axisymmetric -- tidally distorted late in
 * a binary inspiral, say -- no coordinate rotation is a Killing vector and this
 * J is an approximation whose error is not controlled here. Recovering the
 * quasi-local spin in that regime needs an approximate-Killing-vector solve,
 * which would slot in behind the same integral by replacing the phi field.
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

#ifndef GRACE_GEOMETRY_HORIZON_SPIN_HH
#define GRACE_GEOMETRY_HORIZON_SPIN_HH

#include <grace_config.h>

#include <grace/geometry/strahlkoerper.hh>
#include <grace/physics/adm_from_evolution.hh>
#include <grace/physics/expansion.hh>

#include <array>
#include <cmath>
#include <vector>

namespace grace {

struct horizon_spin_t {
    std::array<double,3> J{};            ///< Angular-momentum vector.
    double magnitude          = 0.0;     ///< |J|.
    std::array<double,3> axis{0,0,1};    ///< J/|J|; z by default when J ~ 0.
    double christodoulou_mass = 0.0;     ///< sqrt(M_irr^2 + J^2/(4 M_irr^2)).
    double dimensionless      = 0.0;     ///< chi = |J| / M_christodoulou^2.
};

/**
 * @brief Quasi-local spin of a surface.
 *
 * @param surface Surface to integrate over.
 * @param geom    ADM geometry at its collocation points.
 * @param m_irr   Irreducible mass, sqrt(A/16 pi); pass
 *                surface.irreducible_mass(geom) unless it is already to hand.
 * @param centre  Point the rotational generators are taken about. Defaults to
 *                the area-weighted centroid, which is the right choice for a
 *                surface whose parameterisation origin has drifted off centre.
 */
/**
 * @brief J for an arbitrary rotational field phi^i already tangent to S.
 *
 * The isolated-horizon integral proper; everything else in this header is a
 * choice of phi. Splitting it out lets the coordinate generators and an
 * approximate Killing vector share one implementation.
 */
inline double spin_from_rotational_field(
    strahlkoerper_t const& surface,
    std::vector<adm_point_t> const& geom,
    std::vector<std::array<double,3>> const& phi)
{
    auto const& b  = surface.basis();
    int const   np = b.n_points();
    if (static_cast<int>(geom.size()) != np ||
        static_cast<int>(phi.size())  != np) return 0.0;

    auto const shape = surface.evaluate();
    std::vector<double> integrand(np, 0.0);

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
                    acc += phi[p][i] * nrm.upper[j] * geom[p].K[adm_sym(i, j)];
            integrand[p] = acc;
        }

    return surface.integrate(integrand, geom) / (8.0 * M_PI);
}

/// Spin from an approximate Killing vector, with its quality measures.
struct horizon_spin_akv_t {
    bool   valid              = false;
    double J                  = 0.0;
    double christodoulou_mass = 0.0;
    double dimensionless      = 0.0;
    /// sqrt(shear per unit rotation) x areal radius; 0 for a genuine Killing
    /// vector. Check this before trusting J on a distorted horizon.
    double killing_defect     = 0.0;
    /// Separation from the next eigenvalue; small means the rotation axis is
    /// not determined (a round surface has no preferred axis).
    double eigenvalue_gap     = 0.0;
};

inline horizon_spin_t horizon_spin(strahlkoerper_t const& surface,
                                   std::vector<adm_point_t> const& geom,
                                   double m_irr,
                                   std::array<double,3> const* centre = nullptr)
{
    horizon_spin_t out{};

    auto const& b  = surface.basis();
    int const   np = b.n_points();
    if (static_cast<int>(geom.size()) != np) return out;

    auto const shape = surface.evaluate();
    auto const pts   = surface.collocation_points();
    auto const c     = centre ? *centre : surface.centroid(geom);

    // One integrand per rotation axis.
    std::vector<double> integrand[3] = {
        std::vector<double>(np, 0.0),
        std::vector<double>(np, 0.0),
        std::vector<double>(np, 0.0)
    };

    for (int it = 0; it < b.n_theta(); ++it) {
        for (int ip = 0; ip < b.n_phi(); ++ip) {
            int const p = b.point_index(it, ip);

            shape_point_t const sp{ shape.h[p], shape.dth[p], shape.dph[p],
                                    shape.dthth[p], shape.dthph[p], shape.dphph[p] };

            auto const nrm = surface_normal_at(geom[p], sp, b.theta(it), b.phi(ip));
            if (!(nrm.norm2 > 0.0)) continue;   // degenerate point; contributes nothing

            double const dx[3] = { pts[p][0] - c[0],
                                   pts[p][1] - c[1],
                                   pts[p][2] - c[2] };

            // s^j K_ij, contracted once and reused for all three axes.
            double sK[3];
            for (int i = 0; i < 3; ++i) {
                double acc = 0.0;
                for (int j = 0; j < 3; ++j)
                    acc += nrm.upper[j] * geom[p].K[adm_sym(i, j)];
                sK[i] = acc;
            }

            for (int a = 0; a < 3; ++a) {
                // phi^i = eps^{ijk} e_(a)j dx_k, i.e. the generator of rotation
                // about axis a: e_(0) = x^, e_(1) = y^, e_(2) = z^.
                double phi[3]{};
                int const j = (a + 1) % 3;
                int const k = (a + 2) % 3;
                phi[j] = -dx[k];
                phi[k] =  dx[j];

                // Project out the normal part: a coordinate rotation about the
                // centroid is not exactly tangent to a deformed surface, and a
                // normal component would contribute a spurious s^i s^j K_ij
                // piece to the integral.
                double phi_dot_s = 0.0;
                for (int i = 0; i < 3; ++i) phi_dot_s += phi[i] * nrm.lower[i];
                for (int i = 0; i < 3; ++i) phi[i] -= phi_dot_s * nrm.upper[i];

                double acc = 0.0;
                for (int i = 0; i < 3; ++i) acc += phi[i] * sK[i];
                integrand[a][p] = acc;
            }
        }
    }

    for (int a = 0; a < 3; ++a)
        out.J[a] = surface.integrate(integrand[a], geom) / (8.0 * M_PI);

    out.magnitude = std::sqrt(out.J[0]*out.J[0] + out.J[1]*out.J[1] + out.J[2]*out.J[2]);

    if (out.magnitude > 0.0)
        for (int a = 0; a < 3; ++a) out.axis[a] = out.J[a] / out.magnitude;

    // Christodoulou mass: M^2 = M_irr^2 + J^2 / (4 M_irr^2).
    if (m_irr > 0.0) {
        double const m2 = m_irr * m_irr
                        + out.magnitude * out.magnitude / (4.0 * m_irr * m_irr);
        out.christodoulou_mass = std::sqrt(m2);
        if (m2 > 0.0) out.dimensionless = out.magnitude / m2;
    }

    return out;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_HORIZON_SPIN_HH
