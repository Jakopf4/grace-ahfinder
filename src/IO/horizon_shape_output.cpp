/**
 * @file horizon_shape_output.cpp
 * @brief HDF5 writer for a located apparent horizon.
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

#include <grace/IO/horizon_shape_output.hh>

#include <grace/errors/assert.hh>
#include <grace/system/print.hh>

#include <hdf5.h>

#include <cmath>
#include <system_error>

#define HDF5_CALL(result,cmd) \
    do { \
        if ((result = cmd) < 0) { \
            ERROR("HDF5 API call failed with error code " << result) ; \
        } \
    } while (0)

namespace grace::IO {

namespace {

/// Write a rank-1 or rank-2 double dataset, skipping empty ones.
///
/// Fields that were never computed -- the curvature set without the Hessian
/// path, or everything when the horizon was not found -- are simply absent from
/// the file rather than present and zero. A reader can then tell "not computed"
/// from "computed and zero", which is the same distinction `has_hessian` and
/// `fluxes.valid` draw in memory.
void write_field(hid_t grp, char const* name,
                 std::vector<double> const& data,
                 hsize_t d0, hsize_t d1 = 0, hsize_t d2 = 0)
{
    if (data.empty()) return;

    hsize_t dims[3] = { d0, d1, d2 };
    int const rank = (d2 > 0) ? 3 : ((d1 > 0) ? 2 : 1);

    herr_t err;
    hid_t space_id, dset_id;
    HDF5_CALL(space_id, H5Screate_simple(rank, dims, nullptr));
    HDF5_CALL(dset_id, H5Dcreate2(grp, name, H5T_NATIVE_DOUBLE, space_id,
                                  H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                            H5P_DEFAULT, data.data()));
    HDF5_CALL(err, H5Dclose(dset_id));
    HDF5_CALL(err, H5Sclose(space_id));
}

void attr_double(hid_t obj, char const* name, double const* v, hsize_t n) {
    herr_t err;
    hid_t space_id, attr_id;
    if (n == 1) { HDF5_CALL(space_id, H5Screate(H5S_SCALAR)); }
    else        { hsize_t d[1] = { n };
                  HDF5_CALL(space_id, H5Screate_simple(1, d, nullptr)); }
    HDF5_CALL(attr_id, H5Acreate2(obj, name, H5T_NATIVE_DOUBLE, space_id,
                                  H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err, H5Awrite(attr_id, H5T_NATIVE_DOUBLE, v));
    HDF5_CALL(err, H5Aclose(attr_id));
    HDF5_CALL(err, H5Sclose(space_id));
}

void attr_int(hid_t obj, char const* name, int v) {
    herr_t err;
    hid_t space_id, attr_id;
    HDF5_CALL(space_id, H5Screate(H5S_SCALAR));
    HDF5_CALL(attr_id, H5Acreate2(obj, name, H5T_NATIVE_INT, space_id,
                                  H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err, H5Awrite(attr_id, H5T_NATIVE_INT, &v));
    HDF5_CALL(err, H5Aclose(attr_id));
    HDF5_CALL(err, H5Sclose(space_id));
}

void attr_string(hid_t obj, char const* name, std::string const& v) {
    herr_t err;
    hid_t type_id, space_id, attr_id;
    HDF5_CALL(type_id, H5Tcopy(H5T_C_S1));
    HDF5_CALL(err, H5Tset_size(type_id, v.size() + 1));
    HDF5_CALL(space_id, H5Screate(H5S_SCALAR));
    HDF5_CALL(attr_id, H5Acreate2(obj, name, type_id, space_id,
                                  H5P_DEFAULT, H5P_DEFAULT));
    HDF5_CALL(err, H5Awrite(attr_id, type_id, v.c_str()));
    HDF5_CALL(err, H5Aclose(attr_id));
    HDF5_CALL(err, H5Sclose(space_id));
    HDF5_CALL(err, H5Tclose(type_id));
}

}  // namespace

bool write_horizon_shape_hdf5(std::filesystem::path const& path,
                              std::string const& name,
                              strahlkoerper_t const& surface,
                              horizon_result_t const& result,
                              std::vector<std::string> const& columns,
                              std::vector<double> const& row,
                              std::size_t iteration,
                              double time)
{
    // The manager creates this directory at construction, so this is a fallback
    // for a caller that does not go through it. Report a genuine failure here:
    // otherwise it falls through to H5Fcreate on a non-existent parent and
    // surfaces as an opaque HDF5 error instead of the real cause.
    std::error_code ec;
    if (path.has_parent_path()) {
        std::filesystem::create_directories(path.parent_path(), ec);
        if (ec)
            GRACE_WARN("Could not create horizon shape directory {}: {}",
                       path.parent_path().string(), ec.message());
    }

    herr_t err;
    hid_t file_id;
    HDF5_CALL(file_id, H5Fcreate(path.c_str(), H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT));

    // Only the basis comes from the live surface: it is the collocation grid,
    // which recentring does not touch. Everything describing the *shape* comes
    // from the result, so the coefficients, the centre, the points and every
    // per-point field in this file are one consistent snapshot.
    auto const& b  = surface.basis();
    auto const  nt = static_cast<hsize_t>(b.n_theta());
    auto const  npi = static_cast<hsize_t>(b.n_phi());
    auto const  np = static_cast<std::size_t>(surface.n_points());

    //-- structure -----------------------------------------------------------
    attr_string(file_id, "name", name);
    attr_int(file_id, "iteration", static_cast<int>(iteration));
    attr_double(file_id, "time", &time, 1);
    attr_int(file_id, "l_max",   b.l_max());
    attr_int(file_id, "n_theta", b.n_theta());
    attr_int(file_id, "n_phi",   b.n_phi());
    attr_int(file_id, "found",   result.found ? 1 : 0);
    attr_int(file_id, "status",  static_cast<int>(result.status));
    {
        auto const& c = result.fields.centre;
        double const cc[3] = { c[0], c[1], c[2] };
        attr_double(file_id, "centre", cc, 3);
    }

    //-- the surface itself --------------------------------------------------
    // The coefficients are the primary record: everything else in this file is
    // a sampling of them, and a reader that wants a different resolution can
    // rebuild the surface from these alone.
    write_field(file_id, "coefficients", result.fields.coefficients,
                static_cast<hsize_t>(result.fields.coefficients.size()));

    {
        std::vector<double> th(nt), ph(npi);
        for (int i = 0; i < b.n_theta(); ++i) th[static_cast<std::size_t>(i)] = b.theta(i);
        for (int i = 0; i < b.n_phi();   ++i) ph[static_cast<std::size_t>(i)] = b.phi(i);
        write_field(file_id, "theta", th, nt);
        write_field(file_id, "phi",   ph, npi);
    }

    // Built from the retained h and centre rather than from the live surface,
    // so xyz[p] is the point every field[p] was evaluated at.
    if (result.fields.h.size() == np) {
        std::vector<double> xyz(np * 3);
        for (int it = 0; it < b.n_theta(); ++it) {
            double const st = b.sin_theta(it), ct = b.cos_theta(it);
            for (int ip = 0; ip < b.n_phi(); ++ip) {
                auto const p  = static_cast<std::size_t>(b.point_index(it, ip));
                double const cp = std::cos(b.phi(ip)), sp = std::sin(b.phi(ip));
                double const h  = result.fields.h[p];
                xyz[p * 3 + 0] = result.fields.centre[0] + h * st * cp;
                xyz[p * 3 + 1] = result.fields.centre[1] + h * st * sp;
                xyz[p * 3 + 2] = result.fields.centre[2] + h * ct;
            }
        }
        write_field(file_id, "xyz", xyz, nt, npi, 3);
    }

    //-- per-point diagnostic fields ----------------------------------------
    auto const& f = result.fields;
    write_field(file_id, "h",               f.h,               nt, npi);
    write_field(file_id, "expansion",       f.expansion,       nt, npi);
    write_field(file_id, "ricci_scalar",    f.ricci_scalar,    nt, npi);
    write_field(file_id, "gauss_curvature", f.gauss_curvature, nt, npi);
    write_field(file_id, "shear2",          f.shear2,          nt, npi);
    write_field(file_id, "sqrt_q",          f.sqrt_q,          nt, npi);
    write_field(file_id, "zeta",            f.zeta,            nt, npi);

    //-- the full multipole spectrum ----------------------------------------
    // All of it, not the five the time series carries. The series stays narrow
    // enough to scan; the moments belong with the surface they describe.
    {
        std::vector<double> const m(result.multipoles.mass.begin(),
                                    result.multipoles.mass.end());
        std::vector<double> const j(result.multipoles.spin.begin(),
                                    result.multipoles.spin.end());
        write_field(file_id, "multipole_mass", m, static_cast<hsize_t>(m.size()));
        write_field(file_id, "multipole_spin", j, static_cast<hsize_t>(j.size()));
    }

    //-- the scalar row, carrying its own key -------------------------------
    // Written with the column names rather than as forty hand-listed
    // attributes: the two cannot drift apart, and adding a column to the time
    // series adds it here for free.
    write_field(file_id, "scalars", row, static_cast<hsize_t>(row.size()));
    {
        std::string joined;
        for (std::size_t i = 0; i < columns.size(); ++i) {
            if (i) joined += ',';
            joined += columns[i];
        }
        attr_string(file_id, "scalar_columns", joined);
    }

    HDF5_CALL(err, H5Fclose(file_id));
    return true;
}

}  // namespace grace::IO
