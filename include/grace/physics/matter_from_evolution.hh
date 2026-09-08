/**
 * @file matter_from_evolution.hh
 * @brief Matter and gauge variables at a point, for the horizon flux integrals.
 *
 * The companion to adm_from_evolution.hh, and structured the same way: the
 * index lists and the extraction live in one header so they cannot drift apart.
 * In the finder that was deleted at v1.0 those were two hand-kept lists that had
 * to agree by eye, and that is the failure this shape is designed to prevent.
 *
 * Two arrays are involved, because GRACE keeps them apart: rho, eps, press,
 * z^i and B^i are *auxiliary* (primitive) variables, while the lapse and shift
 * are *evolved*. So a caller samples twice and hands both rows here.
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

#ifndef GRACE_PHYSICS_MATTER_FROM_EVOLUTION_HH
#define GRACE_PHYSICS_MATTER_FROM_EVOLUTION_HH

#include <grace_config.h>

#include <grace/data_structures/variable_indices.hh>
#include <grace/utils/device.h>

#include <vector>

namespace grace {

/// Matter primitives and gauge at one point.
struct matter_point_t {
    double rho   = 0.0;
    double eps   = 0.0;
    double press = 0.0;
    double z[3]  = {0.0, 0.0, 0.0};   ///< Valencia z^i = W v^i (the ZVEC* slots)
    double B[3]  = {0.0, 0.0, 0.0};   ///< cell-centred B^i
    double alp   = 1.0;
    double beta[3] = {0.0, 0.0, 0.0};
    /// Hamiltonian and momentum constraint violations, Z4 only. Already
    /// computed on the grid once per RK step by
    /// compute_constraint_violations_fast(), so sampling them is free.
    double ham    = 0.0;
    double mom[3] = {0.0, 0.0, 0.0};
    /// Whether this point was actually sampled. Consumers must check it rather
    /// than testing for nonzero density: vacuum is a legitimate zero.
    bool   valid = false;
};

/// Slots in the order matter_from_sample() expects, for the *auxiliary* array.
enum matter_aux_slot_t : int {
    MATTER_SLOT_RHO = 0, MATTER_SLOT_EPS, MATTER_SLOT_PRESS,
    MATTER_SLOT_ZX, MATTER_SLOT_ZY, MATTER_SLOT_ZZ,
    MATTER_SLOT_BX, MATTER_SLOT_BY, MATTER_SLOT_BZ,
    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    MATTER_SLOT_HAM, MATTER_SLOT_MOMX, MATTER_SLOT_MOMY, MATTER_SLOT_MOMZ,
    #endif
    MATTER_N_AUX_SLOTS
};

/// Slots in the order matter_from_sample() expects, for the *evolved* array.
enum matter_evol_slot_t : int {
    MATTER_SLOT_ALP = 0, MATTER_SLOT_BETAX, MATTER_SLOT_BETAY, MATTER_SLOT_BETAZ,
    MATTER_N_EVOL_SLOTS
};

/// Variable indices to hand to point_cloud_sampler_t::sample() for the
/// auxiliary array. Do not build this by hand -- it must match the slot order.
inline std::vector<int> matter_aux_var_indices() {
    return { RHO_, EPS_, PRESS_, ZVECX_, ZVECY_, ZVECZ_, BX_, BY_, BZ_
             #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
             , HAM_, MOMX_, MOMY_, MOMZ_
             #endif
           };
}

/// The same, for the evolved array.
inline std::vector<int> matter_evol_var_indices() {
    return { ALP_, BETAX_, BETAY_, BETAZ_ };
}

inline int matter_n_aux_vars()  { return MATTER_N_AUX_SLOTS; }
inline int matter_n_evol_vars() { return MATTER_N_EVOL_SLOTS; }

/// @param evol may be null when only the constraint monitor is wanted: the
///             lapse and shift enter nothing but the flux integrand.
GRACE_HOST_DEVICE inline
matter_point_t matter_from_sample(double const* aux, double const* evol) {
    matter_point_t q;
    q.rho   = aux[MATTER_SLOT_RHO];
    q.eps   = aux[MATTER_SLOT_EPS];
    q.press = aux[MATTER_SLOT_PRESS];
    q.z[0]  = aux[MATTER_SLOT_ZX];
    q.z[1]  = aux[MATTER_SLOT_ZY];
    q.z[2]  = aux[MATTER_SLOT_ZZ];
    q.B[0]  = aux[MATTER_SLOT_BX];
    q.B[1]  = aux[MATTER_SLOT_BY];
    q.B[2]  = aux[MATTER_SLOT_BZ];

    #if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
    q.ham    = aux[MATTER_SLOT_HAM];
    q.mom[0] = aux[MATTER_SLOT_MOMX];
    q.mom[1] = aux[MATTER_SLOT_MOMY];
    q.mom[2] = aux[MATTER_SLOT_MOMZ];
    #endif

    if (evol != nullptr) {
        q.alp     = evol[MATTER_SLOT_ALP];
        q.beta[0] = evol[MATTER_SLOT_BETAX];
        q.beta[1] = evol[MATTER_SLOT_BETAY];
        q.beta[2] = evol[MATTER_SLOT_BETAZ];
    }
    q.valid   = true;
    return q;
}

namespace detail {

/**
 * @brief Reflect a matter sample back across the symmetry planes it was folded
 *        through, mirroring detail::reflect_adm_point().
 *
 * `rho`, `eps`, `press` and `alp` are scalars. `z^i` and `beta^i` are ordinary
 * vectors: one sign per folded axis.
 *
 * **`B^i` is a pseudovector, and this is the trap.** Under a reflection of axis
 * `a` the component `B_a` is *even* and the components `B_{j != a}` are *odd* --
 * exactly the opposite of a true vector. Equivalently, B picks up the extra
 * determinant of the reflection, `(-1)^(number of folded axes)`. This is already
 * the convention the physical boundary conditions use for the staggered field
 * (`phys_bc_kernels.hh:379-388`); getting it backwards would silently corrupt
 * the magnetic flux on every equatorially symmetric run and would leave `mdot`
 * looking perfectly healthy.
 */
GRACE_HOST_DEVICE inline
void reflect_matter_point(matter_point_t& q, unsigned flip) {
    if (flip == 0u) return;

    double sv[3];
    double det = 1.0;
    for (int d = 0; d < 3; ++d) {
        bool const f = (flip & (1u << d)) != 0u;
        sv[d] = f ? -1.0 : 1.0;
        if (f) det = -det;
    }

    for (int d = 0; d < 3; ++d) {
        q.z[d]    *= sv[d];
        q.beta[d] *= sv[d];
        q.mom[d]  *= sv[d];         // ordinary covector
        q.B[d]    *= sv[d] * det;   // pseudovector
    }
    // `ham` is a scalar and is deliberately left alone.
}

}  // namespace detail

}  // namespace grace

#endif  // GRACE_PHYSICS_MATTER_FROM_EVOLUTION_HH
