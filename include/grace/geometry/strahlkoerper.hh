/**
 * @file strahlkoerper.hh
 * @brief A star-shaped surface r = h(theta, phi) expanded in real spherical
 *        harmonics, with the geometry it needs to be integrated over.
 *
 * "Strahlkoerper" is the standard term for the class of surfaces this can
 * represent: those which, seen from an interior origin, show exactly one point
 * in every direction, so the shape is a single-valued h : S^2 -> R+. An
 * apparent horizon must be one of these for a radial parameterisation to work
 * at all; the same restriction applies to AHFinderDirect.
 *
 * Choosing a spectral representation over an angular finite-difference grid
 * removes the coordinate singularity at the poles without the six-patch,
 * ghost-zoned, interpatch-interpolated machinery that dominates AHFinderDirect
 * (its patch_system.cc alone is larger than this entire directory). The price
 * is a dense rather than sparse Jacobian for any solve built on this, but at
 * (l_max+1)^2 unknowns that matrix is small enough not to matter.
 *
 * Host-only, and deliberately independent of where the geometry came from: the
 * integrals take adm_point_t values, which the interpolation path and the
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

#ifndef GRACE_GEOMETRY_STRAHLKOERPER_HH
#define GRACE_GEOMETRY_STRAHLKOERPER_HH

#include <grace_config.h>

#include <grace/physics/adm_from_evolution.hh>
#include <grace/utils/spherical_harmonics.hh>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace grace {

/**
 * @brief Star-shaped surface r = h(theta, phi) about a movable centre.
 */
class strahlkoerper_t {
public:
    /// h and its first and second angular derivatives at every collocation point.
    struct shape_t {
        std::vector<double> h;        ///< h
        std::vector<double> dth;      ///< dh/dtheta
        std::vector<double> dph;      ///< dh/dphi
        std::vector<double> dthth;    ///< d2h/dtheta2
        std::vector<double> dthph;    ///< d2h/dtheta dphi
        std::vector<double> dphph;    ///< d2h/dphi2
    };

    /**
     * @param l_max  Highest retained harmonic; the surface carries
     *               (l_max+1)^2 coefficients.
     * @param centre Coordinate origin the radius is measured from.
     * @param r0     Initial radius; the surface starts as a coordinate sphere.
     */
    strahlkoerper_t(int l_max, std::array<double,3> const& centre, double r0)
        : _basis(l_max)
        , _centre(centre)
        , _coeffs(static_cast<std::size_t>((l_max + 1) * (l_max + 1)), 0.0)
    {
        set_sphere(r0);
    }

    //-- configuration ------------------------------------------------------

    /// Reset to a coordinate sphere of the given radius.
    /// Y_00 = 1/sqrt(4 pi), so a_00 = r sqrt(4 pi) and all else vanishes.
    void set_sphere(double r) {
        std::fill(_coeffs.begin(), _coeffs.end(), 0.0);
        _coeffs[0] = r * std::sqrt(4.0 * M_PI);
    }

    std::vector<double>&       coefficients()       { return _coeffs; }
    std::vector<double> const& coefficients() const { return _coeffs; }

    std::array<double,3> const& centre() const { return _centre; }
    void set_centre(std::array<double,3> const& c) { _centre = c; }

    spherical_harmonic_basis_t const& basis() const { return _basis; }

    int l_max()    const { return _basis.l_max(); }
    int n_coeffs() const { return _basis.n_coeffs(); }
    int n_points() const { return _basis.n_points(); }

    //-- evaluation ---------------------------------------------------------

    /// Evaluate h and its angular derivatives at the collocation points.
    shape_t evaluate() const {
        int const np = _basis.n_points();
        shape_t s{ std::vector<double>(np), std::vector<double>(np),
                   std::vector<double>(np), std::vector<double>(np),
                   std::vector<double>(np), std::vector<double>(np) };
        _basis.evaluate(_coeffs.data(), s.h.data(), s.dth.data(), s.dph.data(),
                        s.dthth.data(), s.dthph.data(), s.dphph.data());
        return s;
    }

    /// Radius in an arbitrary direction, not just at collocation points.
    double radius_at(double theta, double phi) const {
        return _basis.evaluate_single(_coeffs.data(), theta, phi);
    }

    /// Cartesian collocation points; feeds a point sampler directly.
    std::vector<std::array<double,3>> collocation_points() const {
        return collocation_points(evaluate());
    }

    /// Overload for a caller that already has the evaluated shape in hand, so
    /// the spectral synthesis is not repeated.
    std::vector<std::array<double,3>> collocation_points(shape_t const& s) const {
        std::vector<std::array<double,3>> pts(_basis.n_points());
        for (int it = 0; it < _basis.n_theta(); ++it) {
            double const st = _basis.sin_theta(it);
            double const ct = _basis.cos_theta(it);
            for (int ip = 0; ip < _basis.n_phi(); ++ip) {
                int const p = _basis.point_index(it, ip);
                double const cp = std::cos(_basis.phi(ip));
                double const sp = std::sin(_basis.phi(ip));
                pts[p] = { _centre[0] + s.h[p] * st * cp,
                           _centre[1] + s.h[p] * st * sp,
                           _centre[2] + s.h[p] * ct };
            }
        }
        return pts;
    }

    /**
     * @brief Move the origin, re-expressing the shape so the surface itself
     *        does not move.
     *
     * Setting the centre alone keeps h(theta,phi) as the radius about the *old*
     * origin, which rigidly translates the surface by the displacement -- on a
     * horizon of radius 2 moved by 0.5 that takes max|Theta| from 1e-12 to 0.8,
     * destroying exactly the converged shape a warm start is meant to reuse.
     * AHFinderDirect keeps this correction behind `reshape_while_moving`; here
     * it is not optional, because the origin is tracked by default.
     *
     * For each new direction, solve for the radius at which the ray from the
     * new origin meets the existing surface:
     *
     *     f(r) = |c_new + r n - c_old| - h_old(direction of that offset) = 0
     *
     * f(0) < 0 because the new origin lies inside the surface, and f grows
     * without bound, so the root is bracketed and bisection is safe. A ray
     * meets a star-shaped surface exactly once, so the root is unique.
     *
     * @return false if `new_centre` is not inside the surface, in which case
     *         the surface is left untouched -- it would not be star-shaped
     *         about the new origin and the re-expression is meaningless.
     */
    bool recentre(std::array<double,3> const& new_centre) {
        auto const shape = evaluate();
        double h_max = 0.0, h_min = 1e300;
        for (double v : shape.h) { h_max = std::max(h_max, v);
                                   h_min = std::min(h_min, v); }
        if (!(h_min > 0.0)) return false;

        double const dx = new_centre[0] - _centre[0];
        double const dy = new_centre[1] - _centre[1];
        double const dz = new_centre[2] - _centre[2];
        double const disp = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (disp < 1e-14) { _centre = new_centre; return true; }

        // Cheap necessary condition: a displacement beyond the smallest radius
        // certainly leaves the surface.
        if (disp >= h_min) return false;

        auto const& b = _basis;
        std::vector<double> h_new(b.n_points());

        for (int it = 0; it < b.n_theta(); ++it) {
            double const st = b.sin_theta(it), ct = b.cos_theta(it);
            for (int ip = 0; ip < b.n_phi(); ++ip) {
                double const cp = std::cos(b.phi(ip)), sp = std::sin(b.phi(ip));
                double const n[3] = { st * cp, st * sp, ct };

                auto f = [&](double r) {
                    double const ox = new_centre[0] + r * n[0] - _centre[0];
                    double const oy = new_centre[1] + r * n[1] - _centre[1];
                    double const oz = new_centre[2] + r * n[2] - _centre[2];
                    double const d  = std::sqrt(ox*ox + oy*oy + oz*oz);
                    if (!(d > 1e-30)) return -h_min;
                    double const th = std::acos(std::max(-1.0, std::min(1.0, oz / d)));
                    double const ph = std::atan2(oy, ox);
                    return d - radius_at(th, ph);
                };

                // Every f() is a full spectral evaluation, so stop as soon as
                // the bracket is narrower than the radius can resolve rather
                // than running a fixed count into the round-off floor.
                double lo = 0.0, hi = disp + h_max;
                double const tol = 1e-14 * hi;
                for (int k = 0; k < 60 && (hi - lo) > tol; ++k) {
                    double const mid = 0.5 * (lo + hi);
                    if (f(mid) < 0.0) lo = mid; else hi = mid;
                }
                h_new[b.point_index(it, ip)] = 0.5 * (lo + hi);
            }
        }

        _centre  = new_centre;
        _coeffs  = project(h_new);
        return true;
    }

    /// Project pointwise values onto the basis (the analysis transform).
    ///
    /// This is what lets a Newton solve on this surface be posed as a square
    /// (l_max+1)^2 system rather than an overdetermined one at the collocation
    /// points, avoiding normal equations and the squared condition number they
    /// bring.
    std::vector<double> project(std::vector<double> const& point_values) const {
        require_size(point_values.size(), _basis.n_points(), "point_values");
        std::vector<double> a(_basis.n_coeffs());
        _basis.inverse(point_values.data(), a.data());
        return a;
    }

    //-- surface integrals --------------------------------------------------

    /**
     * @brief Area element sqrt(det q) at each collocation point, where q is the
     *        2-metric induced on the surface by gamma_ij.
     *
     * With x^i = c^i + h n^i, the tangent vectors are
     *     d_theta x = h_theta n + h e_theta
     *     d_phi   x = h_phi   n + h sin(theta) e_phi
     * and q_AB is gamma_ij contracted with them.
     */
    std::vector<double> area_element(std::vector<adm_point_t> const& geom) const;

    /**
     * @brief Integrate a pointwise field over the surface: int f dA.
     *
     * The basis quadrature weights already carry the flat sin(theta) d(theta)
     * factor (they are built for int f dOmega), so the physical area element is
     * divided by sin(theta) before weighting. Gauss-Legendre nodes never reach
     * the poles, so that division is safe.
     */
    double integrate(std::vector<double> const& f,
                     std::vector<adm_point_t> const& geom) const;

    /// Proper area of the surface.
    double area(std::vector<adm_point_t> const& geom) const {
        return integrate(std::vector<double>(_basis.n_points(), 1.0), geom);
    }

    /// Irreducible mass implied by the proper area, sqrt(A / 16 pi).
    double irreducible_mass(std::vector<adm_point_t> const& geom) const {
        return std::sqrt(area(geom) / (16.0 * M_PI));
    }

    /// Areal radius implied by the proper area, sqrt(A / 4 pi).
    double areal_radius(std::vector<adm_point_t> const& geom) const {
        return std::sqrt(area(geom) / (4.0 * M_PI));
    }

    /// Area-weighted centroid, in the same coordinates as centre().
    std::array<double,3> centroid(std::vector<adm_point_t> const& geom) const;

    /// Minimum, maximum and area-weighted mean of h over the surface.
    struct radius_stats_t { double min, max, mean; };
    radius_stats_t radius_stats(std::vector<adm_point_t> const& geom) const;

    /// Every scalar diagnostic of the surface, from one shape evaluation and
    /// one area-element pass.
    ///
    /// Asking for these one at a time costs ten redundant passes -- area(),
    /// irreducible_mass() and areal_radius() each recompute the area, and
    /// centroid() and radius_stats() integrate four more times. The quadrature
    /// here runs in the same order with the same terms, so the results are
    /// bit-identical to the individual calls.
    struct diagnostics_t {
        double               area;
        double               irreducible_mass;
        double               areal_radius;
        std::array<double,3> centroid;
        radius_stats_t       radius;
    };
    diagnostics_t diagnostics(std::vector<adm_point_t> const& geom) const;

    //-- persistence --------------------------------------------------------

    /**
     * @brief Flatten to a portable buffer: [cx, cy, cz, l_max, coeffs...].
     *
     * A surface solve is warm-started from the previous shape, so this state
     * has to survive a restart or every find after one falls back to the
     * initial guess. l_max is stored so that a checkpoint written at a
     * different angular resolution is rejected rather than silently
     * misinterpreted.
     */
    std::vector<double> serialize() const {
        std::vector<double> buf;
        buf.reserve(4 + _coeffs.size());
        buf.push_back(_centre[0]);
        buf.push_back(_centre[1]);
        buf.push_back(_centre[2]);
        buf.push_back(static_cast<double>(_basis.l_max()));
        buf.insert(buf.end(), _coeffs.begin(), _coeffs.end());
        return buf;
    }

    /// Restore from serialize(). Returns false (leaving the surface untouched)
    /// if the buffer is truncated or was written at a different l_max.
    bool deserialize(std::vector<double> const& buf) {
        if (buf.size() < 4) return false;
        int const stored_l_max = static_cast<int>(buf[3]);
        if (stored_l_max != _basis.l_max()) return false;
        if (buf.size() != 4 + _coeffs.size()) return false;

        _centre = { buf[0], buf[1], buf[2] };
        std::copy(buf.begin() + 4, buf.end(), _coeffs.begin());
        return true;
    }

    /// Number of doubles serialize() produces for a given l_max.
    static std::size_t serialized_size(int l_max) {
        return 4 + static_cast<std::size_t>((l_max + 1) * (l_max + 1));
    }

private:
    static void require_size(std::size_t got, int want, char const* what) {
        if (got != static_cast<std::size_t>(want))
            throw std::invalid_argument(
                std::string("strahlkoerper_t: ") + what + " has the wrong length");
    }

    spherical_harmonic_basis_t _basis;
    std::array<double,3>       _centre;
    std::vector<double>        _coeffs;
};

//---------------------------------------------------------------------------

inline std::vector<double>
strahlkoerper_t::area_element(std::vector<adm_point_t> const& geom) const
{
    require_size(geom.size(), _basis.n_points(), "geom");
    auto const s = evaluate();

    std::vector<double> out(_basis.n_points());

    for (int it = 0; it < _basis.n_theta(); ++it) {
        double const st = _basis.sin_theta(it);
        double const ct = _basis.cos_theta(it);
        for (int ip = 0; ip < _basis.n_phi(); ++ip) {
            int const p  = _basis.point_index(it, ip);
            double const cp = std::cos(_basis.phi(ip));
            double const sp = std::sin(_basis.phi(ip));

            // Orthonormal (flat) spherical triad at this angle.
            double const n [3] = { st * cp,  st * sp,  ct };
            double const eth[3] = { ct * cp,  ct * sp, -st };
            double const eph[3] = {     -sp,       cp, 0.0 };

            double const h = s.h[p];

            double dth[3], dph[3];
            for (int i = 0; i < 3; ++i) {
                dth[i] = s.dth[p] * n[i] + h * eth[i];
                dph[i] = s.dph[p] * n[i] + h * st * eph[i];
            }

            auto const& g = geom[p];
            auto dot = [&](double const (&u)[3], double const (&v)[3]) {
                double r = 0;
                for (int i = 0; i < 3; ++i)
                    for (int j = 0; j < 3; ++j)
                        r += g.g[adm_sym(i, j)] * u[i] * v[j];
                return r;
            };

            double const q_tt = dot(dth, dth);
            double const q_tp = dot(dth, dph);
            double const q_pp = dot(dph, dph);

            double const det = q_tt * q_pp - q_tp * q_tp;
            out[p] = (det > 0.0) ? std::sqrt(det) : 0.0;
        }
    }
    return out;
}

inline double
strahlkoerper_t::integrate(std::vector<double> const& f,
                           std::vector<adm_point_t> const& geom) const
{
    require_size(f.size(), _basis.n_points(), "f");
    auto const dA = area_element(geom);

    double total = 0.0;
    for (int it = 0; it < _basis.n_theta(); ++it) {
        double const inv_st = 1.0 / _basis.sin_theta(it);
        for (int ip = 0; ip < _basis.n_phi(); ++ip) {
            int const p = _basis.point_index(it, ip);
            total += _basis.weight(it, ip) * f[p] * dA[p] * inv_st;
        }
    }
    return total;
}

inline std::array<double,3>
strahlkoerper_t::centroid(std::vector<adm_point_t> const& geom) const
{
    auto const pts = collocation_points();
    int const np = _basis.n_points();

    double const total = area(geom);
    if (!(total > 0.0)) return _centre;

    std::array<double,3> c{};
    std::vector<double> comp(np);
    for (int d = 0; d < 3; ++d) {
        for (int p = 0; p < np; ++p) comp[p] = pts[p][d];
        c[d] = integrate(comp, geom) / total;
    }
    return c;
}

inline strahlkoerper_t::diagnostics_t
strahlkoerper_t::diagnostics(std::vector<adm_point_t> const& geom) const
{
    auto const shape = evaluate();
    auto const pts   = collocation_points(shape);
    auto const dA    = area_element(geom);

    double total = 0.0;
    double mom[3]{0.0, 0.0, 0.0};
    double h_moment = 0.0;

    for (int it = 0; it < _basis.n_theta(); ++it) {
        double const inv_st = 1.0 / _basis.sin_theta(it);
        for (int ip = 0; ip < _basis.n_phi(); ++ip) {
            int const p = _basis.point_index(it, ip);
            double const w = _basis.weight(it, ip) * dA[p] * inv_st;
            total    += w;
            mom[0]   += w * pts[p][0];
            mom[1]   += w * pts[p][1];
            mom[2]   += w * pts[p][2];
            h_moment += w * shape.h[p];
        }
    }

    diagnostics_t out{};
    out.area             = total;
    out.irreducible_mass = std::sqrt(total / (16.0 * M_PI));
    out.areal_radius     = std::sqrt(total / (4.0 * M_PI));

    if (total > 0.0)
        for (int d = 0; d < 3; ++d) out.centroid[d] = mom[d] / total;
    else
        out.centroid = _centre;

    out.radius.min  = shape.h[0];
    out.radius.max  = shape.h[0];
    for (double v : shape.h) {
        out.radius.min = std::min(out.radius.min, v);
        out.radius.max = std::max(out.radius.max, v);
    }
    out.radius.mean = (total > 0.0) ? h_moment / total : 0.0;
    return out;
}

inline strahlkoerper_t::radius_stats_t
strahlkoerper_t::radius_stats(std::vector<adm_point_t> const& geom) const
{
    auto const s = evaluate();
    radius_stats_t out{ s.h[0], s.h[0], 0.0 };
    for (double v : s.h) {
        out.min = std::min(out.min, v);
        out.max = std::max(out.max, v);
    }
    double const total = area(geom);
    out.mean = (total > 0.0) ? integrate(s.h, geom) / total : 0.0;
    return out;
}

/**
 * @brief Is a point inside the surface?
 *
 * Uses the spectral shape in the point's own direction rather than the nearest
 * collocation value, so the test is as accurate as the representation.
 *
 * @param margin Fractional enlargement of the surface; 0 tests the surface
 *               itself, positive values test a shell around it.
 */
inline bool point_is_inside(strahlkoerper_t const& s,
                            std::array<double,3> const& x,
                            double margin = 0.0)
{
    auto const& c = s.centre();
    double const dx = x[0] - c[0];
    double const dy = x[1] - c[1];
    double const dz = x[2] - c[2];
    double const r  = std::sqrt(dx*dx + dy*dy + dz*dz);
    if (!(r > 1e-30)) return true;                 // the origin is always inside

    double const ct    = std::max(-1.0, std::min(1.0, dz / r));
    double const theta = std::acos(ct);
    double const phi   = std::atan2(dy, dx);       // radius_at is 2pi-periodic

    return r < s.radius_at(theta, phi) * (1.0 + margin);
}

/**
 * @brief Do two surfaces overlap?
 *
 * Both directions are tested -- points of `a` inside `b` and points of `b`
 * inside `a` -- because one surface may be much larger than the other, and a
 * one-sided test would then miss containment.
 *
 * This is the real predicate, not a spheres-approximation: comparing
 * |c1 - c2| against a multiple of the two radii is wrong exactly when it
 * matters, which is when the horizons are tidally stretched towards each other
 * and their nearest points touch long before their centres are close.
 */
inline bool surfaces_overlap(strahlkoerper_t const& a,
                             strahlkoerper_t const& b,
                             double margin = 0.0)
{
    for (auto const& p : a.collocation_points())
        if (point_is_inside(b, p, margin)) return true;
    for (auto const& p : b.collocation_points())
        if (point_is_inside(a, p, margin)) return true;
    return false;
}

}  // namespace grace

#endif  // GRACE_GEOMETRY_STRAHLKOERPER_HH
