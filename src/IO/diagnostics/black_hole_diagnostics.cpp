/**
 * @file black_hole_diagnostics.cpp
 * @author Carlo Musolino (carlo.musolino@aei.mpg.de)
 * @brief 
 * @date 2025-11-17
 * 
 * @copyright This file is part of of the General Relativistic Astrophysics
 * Code for Exascale.
 * GRACE is an evolution framework that uses Finite Volume
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
 * 
 */

#include <grace_config.h>

#include <grace/utils/device.h>
#include <grace/utils/inline.h>

#include <grace/utils/metric_utils.hh>
#include <grace/utils/reductions.hh>   // scalar_symmetry_multiplier()

#include <grace/IO/spherical_surfaces.hh>

#include <grace/IO/diagnostics/black_hole_diagnostics.hh>
#include <grace/physics/horizon_flux_integrand.hh>

#include <array>
#include <vector>
#include <string> 

namespace grace {

std::vector<std::string> bh_diagnostics::flux_names = {"Mdot", "Edot", "Ldot", "Phi"} ; 

std::array<double,bh_diagnostics::n_fluxes> 
bh_diagnostics::compute_local_fluxes(
    Kokkos::View<double**> ivals_d, 
    Kokkos::View<double**> ivals_aux_d, 
    spherical_surface_iface const& detector 
) 
{
    GRACE_VERBOSE("Computing BH diagnostics on sphere {}", detector.name ) ; 

    auto npoints = detector.intersecting_points_h.size() ;
    GRACE_VERBOSE("We have {} points", npoints) ; 

    // initialize local flux array
    std::array<double,n_fluxes> flux_loc = {0.,0.,0.,0.} ; 

    // if no local points return 
    if (npoints == 0 ) return flux_loc ; 


    // copy to host 
    auto ivals = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_d);
    auto ivals_aux = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ivals_aux_d);


    // fetch coord system 
    auto const& coord_system = grace::coordinate_system::get() ;

    // local reduction 
    for(int i=0; i<npoints; ++i) {

        auto ip = detector.intersecting_points_h[i] ; 
        #if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4
        double gxx{ivals(i,loc_var_idx_t::GXXL)}
                , gxy{ivals(i,loc_var_idx_t::GXYL)}
                , gxz{ivals(i,loc_var_idx_t::GXZL)}
                , gyy{ivals(i,loc_var_idx_t::GYYL)}
                , gyz{ivals(i,loc_var_idx_t::GYZL)}
                , gzz{ivals(i,loc_var_idx_t::GZZL)}
                , betax{ivals(i,loc_var_idx_t::BETAXL)}
                , betay{ivals(i,loc_var_idx_t::BETAYL)}
                , betaz{ivals(i,loc_var_idx_t::BETAZL)}
                , alp{ivals(i,loc_var_idx_t::ALPL)}
                , rho{ivals_aux(i,loc_aux_idx_t::RHOL)}
                , eps{ivals_aux(i,loc_aux_idx_t::EPSL)}
                , press{ivals_aux(i,loc_aux_idx_t::PRESSL)}
                , vx{ivals_aux(i,loc_aux_idx_t::VELXL)}
                , vy{ivals_aux(i,loc_aux_idx_t::VELYL)}
                , vz{ivals_aux(i,loc_aux_idx_t::VELZL)}
                , bx{ivals_aux(i,loc_aux_idx_t::BXL)}
                , by{ivals_aux(i,loc_aux_idx_t::BYL)}
                , bz{ivals_aux(i,loc_aux_idx_t::BZL)} ;
        #else 
        double gxx{ivals(i,loc_var_idx_t::GTXXL)}
                , gxy{ivals(i,loc_var_idx_t::GTXYL)}
                , gxz{ivals(i,loc_var_idx_t::GTXZL)}
                , gyy{ivals(i,loc_var_idx_t::GTYYL)}
                , gyz{ivals(i,loc_var_idx_t::GTYZL)}
                , gzz{ivals(i,loc_var_idx_t::GTZZL)}
                , chi{ivals(i,loc_var_idx_t::CHIL)}
                , betax{ivals(i,loc_var_idx_t::BETAXL)}
                , betay{ivals(i,loc_var_idx_t::BETAYL)}
                , betaz{ivals(i,loc_var_idx_t::BETAZL)}
                , alp{ivals(i,loc_var_idx_t::ALPL)}
                , rho{ivals_aux(i,loc_aux_idx_t::RHOL)}
                , eps{ivals_aux(i,loc_aux_idx_t::EPSL)}
                , press{ivals_aux(i,loc_aux_idx_t::PRESSL)}
                , vx{ivals_aux(i,loc_aux_idx_t::VELXL)}
                , vy{ivals_aux(i,loc_aux_idx_t::VELYL)}
                , vz{ivals_aux(i,loc_aux_idx_t::VELZL)}
                , bx{ivals_aux(i,loc_aux_idx_t::BXL)}
                , by{ivals_aux(i,loc_aux_idx_t::BYL)}
                , bz{ivals_aux(i,loc_aux_idx_t::BZL)} ;
        #endif 
        auto report_nan = [&](char const* name, double v) {
                if (std::isnan(v)) {
                    GRACE_VERBOSE("NaN detected in {} at i={}", name, i);
                }
            };
        {
            

            report_nan("gxx", gxx);
            report_nan("gxy", gxy);
            report_nan("gxz", gxz);
            report_nan("gyy", gyy);
            report_nan("gyz", gyz);
            report_nan("gzz", gzz);
            report_nan("betax", betax);
            report_nan("betay", betay);
            report_nan("betaz", betaz);
            report_nan("alp", alp);
            report_nan("rho", rho);
            report_nan("eps", eps);
            report_nan("press", press);
            report_nan("vx", vx);
            report_nan("vy", vy);
            report_nan("vz", vz);
            report_nan("bx", bx);
            report_nan("by", by);
            report_nan("bz", bz);
        }
        #if GRACE_METRIC_EVOL != GRACE_METRIC_EVOL_Z4
        metric_array_t metric{
            {gxx,gxy,gxz,gyy,gyz,gzz}, {betax,betay,betaz}, alp
        } ;
        #else 
        metric_array_t metric{
            {gxx,gxy,gxz,gyy,gyz,gzz}, chi, {betax,betay,betaz}, alp
        } ;
        #endif 
        auto r = detector.radius ; 
        auto theta = detector.angles_h[ip][0] ; 
        auto phi   = detector.angles_h[ip][1] ; 
        auto x = detector.points_h[ip].second[0] ; 
        auto y = detector.points_h[ip].second[1] ; 
        auto z = detector.points_h[ip].second[2] ; 

        double sth = sin(theta) ;
        double sph = sin(phi)   ;
        double cph = cos(phi)   ;

        // Direction of the outward normal and of the azimuthal generator, in
        // the coordinates the metric is expressed in. On a coordinate sphere
        // "outward" is dr, so `dir` is the gradient dr/dx^i and the surface
        // measure carries the r^2 sqrt(gamma) (or its CKS equivalent).
        double dir[3], phi_dir[3], sqrtmdet ;
        if ( coord_system.get_is_cks() ) {
            double a = coord_system.get_bh_spin() ;
            double a2 = SQR(a) ;
            double rad2 = SQR(x) + SQR(y) + SQR(z) ;
            double r2 = SQR(r) ;

            dir[0] = r*x/(2.0*r2 - rad2 + a2) ;
            dir[1] = r*y/(2.0*r2 - rad2 + a2) ;
            dir[2] = (r*z + a2 * z / r) / (2.0*r2 - rad2 + a2) ;

            phi_dir[0] = (-r*sph-a*cph)*sth ;
            phi_dir[1] = ( r*cph-a*sph)*sth ;
            phi_dir[2] = 0.0 ;

            sqrtmdet = (r2 + SQR(a*cos(theta))) ;
        } else {
            dir[0] = x/r ;
            dir[1] = y/r ;
            dir[2] = z/r ;

            phi_dir[0] = (-r*sph)*sth ;
            phi_dir[1] = ( r*cph)*sth ;
            phi_dir[2] = 0.0 ;

            sqrtmdet = r * r * metric.sqrtg() ;
        }

        // The MHD algebra itself lives in one place, shared with the
        // apparent-horizon flux integrals, which differ only in the direction
        // and the measure.
        auto const fi = flux_integrand(
            metric, mhd_primitives_t{rho, eps, press, {vx,vy,vz}, {bx,by,bz}},
            dir, phi_dir) ;

        double const domega = detector.weights_h[ip] ;
        report_nan("domega",domega) ;
        report_nan("sqrtmdet",sqrtmdet) ;
        // accretion rate
        flux_loc[0] += domega * sqrtmdet * fi.mdot ;
        // energy flux
        flux_loc[1] += domega * sqrtmdet * fi.edot ;
        // angular momentum flux
        flux_loc[2] += domega * sqrtmdet * fi.ldot ;
        // magnetic flux
        flux_loc[3] += domega * sqrtmdet * fi.phi_B ;

    }

    // The sphere quadrature only spans the active subdomain when reflection
    // symmetries are on (e.g. an octant covers 1/8 of the sphere), so scale the
    // surface integrals up to the full sphere. Mdot/Edot/Phi are parity-even.
    // NB: Ldot (L_z) is parity-ODD under an x- or y-reflection -- there the
    // full-sphere value is zero and this factor over-counts a partial sum, so
    // only trust Ldot for z-only (or no) reflection symmetry.
    int const sym_mult = scalar_symmetry_multiplier() ;
    for ( auto& f : flux_loc ) f *= sym_mult ;

    return flux_loc; 
}


}