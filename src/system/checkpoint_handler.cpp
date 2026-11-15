/**
 * @file checkpoint_handler.cpp
 * @author Carlo Musolino (musolino@itp.uni-frankfurt.de)
 * @brief 
 * @date 2025-01-31
 * 
 * @copyright This file is part of the General Relativistic Astrophysics
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

#include <grace/system/checkpoint_handler.hh>
#include <cstring>

#include <grace_config.h>
#include <grace/utils/grace_utils.hh>
#include <grace/amr/grace_amr.hh>
#include <grace/config/config_parser.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/data_structures/memory_defaults.hh>
#include <grace/system/grace_system.hh>
#include <grace/IO/hdf5_output.hh>
#include <grace/coordinates/coordinate_systems.hh>
#include <grace/coordinates/coordinates.hh>
#include <grace/profiling/profiling_runtime.hh>
#include <grace/data_structures/variables.hh>
#include <grace/system/grace_runtime.hh>
#include <grace/system/runtime_functions.hh>
#include <grace/evolution/auxiliaries.hh>
#include <grace/physics/eos/eos_storage.hh>

#include <grace/amr/p4est_headers.hh>

#include <grace/IO/diagnostics/co_tracker.hh>
#include <grace/geometry/horizon_manager.hh>
#include <grace/physics/b_field_injection.hh>
#ifdef GRACE_ENABLE_PARTICLES
#include <grace/particles/particles_module.hh>
#include <grace/particles/particle_checkpoint.hh>
#endif
#if GRACE_METRIC_EVOL == GRACE_METRIC_EVOL_Z4
#endif

#include <hdf5.h>
#include <omp.h>
#include <filesystem>
#include <regex>
#include <vector> 
#include <algorithm>

#define HDF5_CALL(result,cmd) \
    do {  \
        if((result=cmd)<0) { \
            ERROR("HDF5 API call failed with error code " << result ) ; \
        } \
    } while(false)


namespace grace {

namespace detail {

/**
 * @brief Generate filename given basepath, prefix iteration and extension
 * \cond grace_detail
 * @param dir base path
 * @param base_name name prefix
 * @param iteration iteration
 * @param extension file extension
 * @return std::filesystem::path File path where the iteration number is padded by zeros and inserted
 */
std::filesystem::path inline get_filename(
    std::filesystem::path const& dir,
    std::string const& base_name, 
    int iteration,
    std::string const& extension )
{
    std::stringstream ss ; 
    ss << base_name << "_" << std::setw(6) << std::setfill('0') << iteration << extension ; 
    return dir / ss.str() ; 
}

void read_buffer_hdf5(
    double* data, 
    hid_t file_id,
    hid_t dxpl,
    hid_t dset_id,
    size_t slab_off,
    size_t slab_size
)
{
    herr_t herr ; 


    // retrieve space 
    hid_t filespace ; 
    HDF5_CALL(filespace,H5Dget_space(dset_id));

    /* Create local space for this rank */
    hid_t memspace ; 
    hsize_t dset_dims[1] = {slab_size} ;
    HDF5_CALL(memspace, H5Screate_simple(1, dset_dims, NULL)) ;


    /* Select hyperslab for this rank's output */
    hsize_t slab_start[1]  = {slab_off} ; 
    hsize_t slab_count[1]  = {slab_size} ;
    HDF5_CALL( herr
            , H5Sselect_hyperslab( filespace
                                , H5S_SELECT_SET
                                , slab_start
                                , NULL
                                , slab_count 
                                , NULL )) ;

    HDF5_CALL(
        herr, H5Dread(dset_id, H5T_NATIVE_DOUBLE, memspace, filespace, dxpl, data)
    ) ;

    HDF5_CALL(herr, H5Sclose(memspace)) ; 
    HDF5_CALL(herr, H5Sclose(filespace)) ; 
}

template< typename v_t > 
void read_contiguous_buffer(
    v_t dst, std::vector<double>const & src
)
{
    size_t ipos{0} ; 
    for( int q=0; q<dst.extent(4); ++q) {
        for( int ivar=0; ivar<dst.extent(3); ++ivar) {
            for( int k=0; k<dst.extent(2); ++k){
                for( int j=0; j<dst.extent(1); ++j) {
                    for( int i=0; i<dst.extent(0); ++i) {
                        dst(i,j,k,ivar,q) = src[ipos++]; 
                    }
                }
            }
        }
    } 
}

void read_data_hdf5(
    hid_t file_id,
    hid_t dxpl, 
    std::string const& dset_name,
    grace::var_array_t& data
)
{
    DECLARE_GRID_EXTENTS ; 
    
    herr_t err ; 

    auto rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ;
    /* Number of datapoints/quadrant */ 
    unsigned long const npts_quad = EXPR(data.extent(0),*data.extent(1),*data.extent(2)) ;
    /* Global dataset dimension */
    unsigned long const dim_loc  = npts_quad * data.extent(GRACE_NSPACEDIM) * nq ; 
    unsigned long const dim_glob = npts_quad * data.extent(GRACE_NSPACEDIM) * nq_glob ;
    // offset
    unsigned long const off_loc  = npts_quad * data.extent(GRACE_NSPACEDIM) * local_quad_offset ; 

    // If there are no variables return 
    // FIXME if the code was somehow compiled with different variables than the code that wrote the checkpoint this is a bug! 
    if (dim_glob == 0) return ; 

    // Open the dataset
    hid_t dset_id, space_id; 
    HDF5_CALL(
        dset_id, H5Dopen(file_id, dset_name.c_str(), H5P_DEFAULT)
    ) ; 
    HDF5_CALL(
        space_id, H5Dget_space(dset_id)
    ) ; 

    // Get dataset dimensions
    hsize_t dim[1] ; 
    HDF5_CALL(
        err, H5Sget_simple_extent_dims(space_id, dim, NULL)
    ) ;

    // Check that they match
    ASSERT(dim[0] == dim_glob, "Dataset " << dset_name << " dimensions do not match: " << dim[0] << " != " << dim_glob) ;

    // Select hyperslab for this rank's output
    hid_t memspace_id ; 
    hsize_t count[1] = {dim_loc} ;
    hsize_t offset[1] = {local_quad_offset * npts_quad * data.extent(GRACE_NSPACEDIM)} ;
     HDF5_CALL(memspace_id, H5Screate_simple(1, count, NULL)) ; 
    HDF5_CALL(
        err, H5Sselect_hyperslab(space_id, H5S_SELECT_SET, offset, NULL, count, NULL)
    ) ;

    if constexpr ( Kokkos::SpaceAccessibility<Kokkos::HostSpace, grace::default_space>::accessible ) {
        if ( data.span_is_contiguous() ) {
            read_buffer_hdf5(data.data(), file_id, dxpl, dset_id, off_loc, dim_loc ) ; 
        } else {
            std::vector<double> flatbuf(dim_loc) ; 
            read_buffer_hdf5(flatbuf.data(), file_id, dxpl, dset_id, off_loc, dim_loc ) ; 
            read_contiguous_buffer(data, flatbuf) ; 
        }
    } else {
        auto h_mirror = Kokkos::create_mirror_view(data)  ;
        if ( h_mirror.span_is_contiguous() ) {
            read_buffer_hdf5(h_mirror.data(), file_id, dxpl, dset_id, off_loc, dim_loc ) ; 
        } else {
            std::vector<double> flatbuf(dim_loc) ; 
            read_buffer_hdf5(flatbuf.data(), file_id, dxpl, dset_id, off_loc, dim_loc ) ; 
            read_contiguous_buffer(h_mirror, flatbuf) ; 
        }
         Kokkos::deep_copy(data, h_mirror) ;
    }

    // Close everything
    HDF5_CALL(err, H5Sclose(memspace_id)) ;
    HDF5_CALL(err, H5Dclose(dset_id)) ;
    HDF5_CALL(err, H5Sclose(space_id)) ;
}

void write_buffer_hdf5(
    double* data, 
    hid_t file_id,
    hid_t dxpl,
    hid_t dset_id,
    size_t slab_off,
    size_t slab_size
)
{
    herr_t herr ; 


    // retrieve space 
    hid_t filespace ; 
    HDF5_CALL(filespace,H5Dget_space(dset_id));

    /* Create local space for this rank */
    hid_t memspace ; 
    hsize_t dset_dims[1] = {slab_size} ;
    HDF5_CALL(memspace, H5Screate_simple(1, dset_dims, NULL)) ;


    /* Select hyperslab for this rank's output */
    hsize_t slab_start[1]  = {slab_off} ; 
    hsize_t slab_count[1]  = {slab_size} ;
    HDF5_CALL( herr
            , H5Sselect_hyperslab( filespace
                                , H5S_SELECT_SET
                                , slab_start
                                , NULL
                                , slab_count 
                                , NULL )) ;

    /* write to dataset */
    HDF5_CALL( herr
                , H5Dwrite( dset_id
                        , H5T_NATIVE_DOUBLE
                        , memspace
                        , filespace 
                        , dxpl 
                        , data )) ;

    HDF5_CALL(herr, H5Sclose(memspace)) ; 
    HDF5_CALL(herr, H5Sclose(filespace)) ; 
}

template< typename v_t > 
void fill_contiguous_buffer(
    std::vector<double>& dst, v_t src
)
{
    size_t ipos{0} ; 
    for( int q=0; q<src.extent(4); ++q) {
        for( int ivar=0; ivar<src.extent(3); ++ivar) {
            for( int k=0; k<src.extent(2); ++k){
                for( int j=0; j<src.extent(1); ++j) {
                    for( int i=0; i<src.extent(0); ++i) {
                        dst[ipos++] = src(i,j,k,ivar,q) ; 
                    }
                }
            }
        }
    } 
}


void write_data_hdf5(
    hid_t file_id, 
    hid_t dxpl,
    std::string const& dset_name,
    grace::var_array_t data 
)
{
    DECLARE_GRID_EXTENTS ; 

    herr_t err ; 

    //static constexpr unsigned int compression_level = 6 ;
    auto rank = parallel::mpi_comm_rank() ; 
    /* Get the p4est pointer */
    auto _p4est = amr::forest::get().get() ; 
    /* Get global number of quadrants and quadrant offset for this rank */
    unsigned long const nq_glob = _p4est->global_num_quadrants ; 
    unsigned long const local_quad_offset = _p4est->global_first_quadrant[rank] ;
    /* Number of datapoints/quadrant */ 
    unsigned long const npts_quad = EXPR(data.extent(0),*data.extent(1),*data.extent(2)) ;
    /* Global dataset dimension */
    unsigned long const dim_loc  = npts_quad * data.extent(GRACE_NSPACEDIM) * nq ; 
    unsigned long const dim_glob = npts_quad * data.extent(GRACE_NSPACEDIM) * nq_glob ;
    /* Local offset */
    unsigned long const off_loc = local_quad_offset * npts_quad * data.extent(GRACE_NSPACEDIM) ; 

    // If there are no variables return 
    if (dim_glob == 0) return ; 

    /* Global space for dataset */
    hid_t space_id_glob ; 
    hsize_t dset_dims_glob[1] = {dim_glob} ;
    HDF5_CALL(space_id_glob, H5Screate_simple(1, dset_dims_glob, NULL)) ;
    /* Dataset properties */
    hid_t prop_id ; 
    HDF5_CALL(prop_id, H5Pcreate(H5P_DATASET_CREATE)) ;
    // Contiguous layout, deliberately not chunked.  The state is written
    // uncompressed and never extended, so chunking buys nothing here, and on
    // Hunter (HLRS/APU 2026.1, Cray MPICH) a collective write into a chunked
    // dataset hangs every rank inside H5Dwrite.  grace-m1 has run without
    // chunking on Hunter since 0d6f1535.  The reader does not depend on the
    // layout, so older chunked checkpoints still restore.

    /* Create dataset */
    hid_t dset_id ; 
        HDF5_CALL( dset_id
                , H5Dcreate2( file_id
                            , dset_name.c_str()
                            , H5T_NATIVE_DOUBLE
                            , space_id_glob
                            , H5P_DEFAULT
                            , prop_id
                            , H5P_DEFAULT) ) ;

    double * dptr = nullptr ; 
    std::vector<double> flatbuf ; 
    decltype(Kokkos::create_mirror_view(Kokkos::HostSpace{}, data)) h_mirror;

    if constexpr ( Kokkos::SpaceAccessibility<Kokkos::HostSpace, grace::default_space>::accessible ) {
        if ( data.span_is_contiguous() ) {
            dptr = data.data() ; 
        } else {
            flatbuf.resize(dim_loc) ; 
            fill_contiguous_buffer(flatbuf, data) ; 
            dptr = flatbuf.data() ; 
        }
    } else {
        h_mirror = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, data);
        if ( h_mirror.span_is_contiguous() ) {
            dptr = h_mirror.data() ; 
        } else {
            flatbuf.resize(dim_loc) ; 
            fill_contiguous_buffer(flatbuf, h_mirror) ; 
            dptr = flatbuf.data() ; 
        }
    }
    
    /* Write to file */
    write_buffer_hdf5(
        dptr,
        file_id,
        dxpl,
        dset_id,
        off_loc,
        dim_loc
    ) ; 

    /* Close dataset */
    HDF5_CALL(err, H5Dclose(dset_id)) ; 
    /*****************************************************************************************/
    /*                                  Close data space                                     */
    /*****************************************************************************************/
    HDF5_CALL(err, H5Sclose(space_id_glob)) ;
    HDF5_CALL(err, H5Pclose(prop_id)) ;

}  

}

void checkpoint_handler_impl_t::save_checkpoint()  
{
    GRACE_PROFILING_PUSH_REGION("save_checkpoint") ; 
    DECLARE_GRID_EXTENTS ;
    unsigned int const iter = grace::get_iteration() ;
    double const time = grace::get_simulation_time() ; 
    /* Save the current state to a checkpoint file */
    GRACE_INFO("Saving checkpoint at iteration {} simulation time {}.", iter, time ) ;

    // first write the forest to file 
    auto forest_file = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ; 
    p4est_save_ext(
        forest_file.string().c_str(),
        amr::forest::get().get(),
        true,   // save_data
        0       // save_partition=false: file is mpisize-independent
    ) ;

    // Now we write the state data to an hdf5 file 
    auto state_file = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    // Create property list for parallel file access
    herr_t err ; 
    hid_t plist_id ; 
    HDF5_CALL(plist_id,H5Pcreate(H5P_FILE_ACCESS)) ; 
    HDF5_CALL(err,H5Pset_fapl_mpio(plist_id, mpi_comm_world, MPI_INFO_NULL)) ; 
    // Create a new file 
    hid_t file_id ; 
    HDF5_CALL(file_id,H5Fcreate(state_file.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, plist_id)) ;
    hid_t attr_id, attr_dataspace_id;
    std::string file_attr_name = "Time";
    const double file_attr_data = time ; 
    // Create a dataspace for the attribute
    HDF5_CALL(attr_dataspace_id,H5Screate(H5S_SCALAR));
    // Create the attribute
    HDF5_CALL(attr_id,H5Acreate2(file_id, file_attr_name.c_str(), H5T_NATIVE_DOUBLE, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    // Write the attribute data
    HDF5_CALL(err,H5Awrite(attr_id, H5T_NATIVE_DOUBLE, &file_attr_data));
    // Close the attribute and dataspace
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));
    file_attr_name = "Iteration" ;  
    HDF5_CALL(attr_dataspace_id,H5Screate(H5S_SCALAR));
    // Create the attribute
    HDF5_CALL(attr_id,H5Acreate2(file_id, file_attr_name.c_str(), H5T_NATIVE_UINT, attr_dataspace_id, H5P_DEFAULT, H5P_DEFAULT));
    // Write the attribute data
    HDF5_CALL(err,H5Awrite(attr_id, H5T_NATIVE_UINT, &iter));
    // Close the attribute and dataspace
    HDF5_CALL(err,H5Aclose(attr_id));
    HDF5_CALL(err,H5Sclose(attr_dataspace_id));

    // Write scalar attributes (Time, Iteration) here as before...

    /* ----------------------------------------------------------------------
    Collective write of grid extents (X, Y, Z) on all ranks
    ------------------------------------------------------------------------*/
    hid_t dxpl;
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER));
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl, H5FD_MPIO_COLLECTIVE));

    auto write_extent_dataset = [&](std::string const& name, double const val_min, double const val_max){
        hsize_t dset_dims[1] = {2};
        hid_t space_id;
        HDF5_CALL(space_id, H5Screate_simple(1, dset_dims, NULL));

        hid_t dset_id;
        HDF5_CALL(dset_id, H5Dcreate2(file_id,
                                    name.c_str(),
                                    H5T_NATIVE_DOUBLE,
                                    space_id,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT,
                                    H5P_DEFAULT));

        double data[2] = {val_min, val_max};
        HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, data));

        HDF5_CALL(err, H5Dclose(dset_id));
        HDF5_CALL(err, H5Sclose(space_id));
    };

    // All ranks participate in the write
    write_extent_dataset("grid_extent_x",
                        grace::get_param<double>("amr", "xmin"),
                        grace::get_param<double>("amr", "xmax"));

    write_extent_dataset("grid_extent_y",
                        grace::get_param<double>("amr", "ymin"),
                        grace::get_param<double>("amr", "ymax"));

    write_extent_dataset("grid_extent_z",
                        grace::get_param<double>("amr", "zmin"),
                        grace::get_param<double>("amr", "zmax"));

    // if active, we need to write ns tracker data
    auto write_co_dset = [&] (hid_t gid, int ico, std::unique_ptr<co_tracker_t> const& co ) {

        auto name = co->get_name() ; 

        // Create a group for this CO so h5ls shows it cleanly
        std::string grp_name = "co_" + std::to_string(ico) ; 
        hid_t grp_id;
        HDF5_CALL(grp_id, H5Gcreate2(gid, grp_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));

        // Write location as a 3-vector dataset
        {
            hsize_t dims[1] = {3};
            hid_t space_id, dset_id;
            HDF5_CALL(space_id, H5Screate_simple(1, dims, NULL));
            HDF5_CALL(dset_id, H5Dcreate2(grp_id, "location", H5T_NATIVE_DOUBLE,
                                        space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            auto data = co->get_loc() ; 
            HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, data.data()));
            HDF5_CALL(err, H5Dclose(dset_id));
            HDF5_CALL(err, H5Sclose(space_id));
        }

        // Write radius as a scalar dataset
        {
            double rad = co->get_radius() ; 
            hid_t space_id, dset_id;
            HDF5_CALL(space_id, H5Screate(H5S_SCALAR));
            HDF5_CALL(dset_id, H5Dcreate2(grp_id, "radius", H5T_NATIVE_DOUBLE,
                                        space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, &rad));
            HDF5_CALL(err, H5Dclose(dset_id));
            HDF5_CALL(err, H5Sclose(space_id));
        }

        // Write name as a string attribute on the group
        {
            hid_t aspace_id, atype_id, attr_id;
            HDF5_CALL(aspace_id, H5Screate(H5S_SCALAR));
            HDF5_CALL(atype_id,  H5Tcopy(H5T_C_S1));
            HDF5_CALL(err,       H5Tset_size(atype_id, name.size()+1));
            HDF5_CALL(err,       H5Tset_strpad(atype_id, H5T_STR_NULLTERM));
            HDF5_CALL(attr_id,   H5Acreate2(grp_id, "name", atype_id,
                                            aspace_id, H5P_DEFAULT, H5P_DEFAULT));
            HDF5_CALL(err, H5Awrite(attr_id, atype_id, name.c_str()));
            HDF5_CALL(err, H5Aclose(attr_id));
            HDF5_CALL(err, H5Tclose(atype_id));
            HDF5_CALL(err, H5Sclose(aspace_id));
        }

        // Write kind as a string attribute on the group
        {
            auto kind = co->get_kind() ; 
            hid_t aspace_id, atype_id, attr_id;
            HDF5_CALL(aspace_id, H5Screate(H5S_SCALAR));
            HDF5_CALL(atype_id,  H5Tcopy(H5T_C_S1));
            HDF5_CALL(err,       H5Tset_size(atype_id, kind.size()+1));
            HDF5_CALL(err,       H5Tset_strpad(atype_id, H5T_STR_NULLTERM));
            HDF5_CALL(attr_id,   H5Acreate2(grp_id, "kind", atype_id,
                                            aspace_id, H5P_DEFAULT, H5P_DEFAULT));
            HDF5_CALL(err, H5Awrite(attr_id, atype_id, kind.c_str()));
            HDF5_CALL(err, H5Aclose(attr_id));
            HDF5_CALL(err, H5Tclose(atype_id));
            HDF5_CALL(err, H5Sclose(aspace_id));
        }

        HDF5_CALL(err, H5Gclose(grp_id));
    };
    auto& co_tracker = grace::co_tracker::get() ;
    if (co_tracker.is_active()) {
        int ncos    = co_tracker.get_n_cos() ; 
        int merged  = static_cast<int>(co_tracker.get_merged()) ; 
        hid_t tracker_grp;
        HDF5_CALL(tracker_grp, H5Gcreate2(file_id, "co_tracker", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        {
            hid_t space_id, dset_id;
            HDF5_CALL(space_id, H5Screate(H5S_SCALAR));
            HDF5_CALL(dset_id, H5Dcreate2(tracker_grp, "n_cos", H5T_NATIVE_INT,
                                          space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, dxpl, &ncos));
            HDF5_CALL(err, H5Dclose(dset_id));
            HDF5_CALL(err, H5Sclose(space_id));
        }
        {
            hid_t space_id, dset_id;
            HDF5_CALL(space_id, H5Screate(H5S_SCALAR));
            HDF5_CALL(dset_id, H5Dcreate2(tracker_grp, "has_merged", H5T_NATIVE_INT,
                                          space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, dxpl, &merged));
            HDF5_CALL(err, H5Dclose(dset_id));
            HDF5_CALL(err, H5Sclose(space_id));
        }
        for( int ico=0; ico<ncos; ++ico) write_co_dset(tracker_grp, ico, co_tracker.get(ico)) ;

        HDF5_CALL(err, H5Gclose(tracker_grp));
    }

    // Apparent horizons go to their own small file beside this one, written
    // after this parallel file is closed -- see save_horizon_checkpoint().
    // The state is a few KB one wants to inspect and copy on its own, and this
    // file runs to O(100 GB) in production.

    // persist mid-run B-field injection state
    {
        int has_fired = static_cast<int>(grace::b_field_injection_has_fired()) ;
        hid_t inj_grp ;
        HDF5_CALL(inj_grp, H5Gcreate2(file_id, "b_field_injection", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
        {
            hid_t space_id, dset_id;
            HDF5_CALL(space_id, H5Screate(H5S_SCALAR));
            HDF5_CALL(dset_id, H5Dcreate2(inj_grp, "has_fired", H5T_NATIVE_INT,
                                          space_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT));
            HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, dxpl, &has_fired));
            HDF5_CALL(err, H5Dclose(dset_id));
            HDF5_CALL(err, H5Sclose(space_id));
        }
        HDF5_CALL(err, H5Gclose(inj_grp));
    }

    // particle subsystem state
    #ifdef GRACE_ENABLE_PARTICLES
    grace::particles::save_particles_to_checkpoint(file_id, dxpl);
    #endif


    // write state data
    GRACE_TRACE("Writing state.") ; 
    auto state = grace::variable_list::get().getstate() ; 
    detail::write_data_hdf5(file_id, dxpl, "CellCenteredData", state) ; 
    // write staggered state data
    auto sstate = grace::variable_list::get().getstaggeredstate() ; 
    detail::write_data_hdf5(file_id, dxpl, "CornerCenteredData", sstate.corner_staggered_fields) ;
    detail::write_data_hdf5(file_id, dxpl, "EdgeCenteredDataXY", sstate.edge_staggered_fields_xy) ;
    detail::write_data_hdf5(file_id, dxpl, "EdgeCenteredDataXZ", sstate.edge_staggered_fields_xz) ;
    detail::write_data_hdf5(file_id, dxpl, "EdgeCenteredDataYZ", sstate.edge_staggered_fields_yz) ;
    detail::write_data_hdf5(file_id, dxpl, "FaceCenteredDataX", sstate.face_staggered_fields_x) ;
    detail::write_data_hdf5(file_id, dxpl, "FaceCenteredDataY", sstate.face_staggered_fields_y) ;
    detail::write_data_hdf5(file_id, dxpl, "FaceCenteredDataZ", sstate.face_staggered_fields_z) ;
    // Block all processes until all data is written
    parallel::mpi_barrier() ;
    GRACE_TRACE("Done with write.") ; 
    // Cleanup 
    HDF5_CALL(err, H5Pclose(dxpl)) ;
    GRACE_TRACE("Done with write.") ; 
    /* Close the file */
    HDF5_CALL(err,H5Fclose(file_id)) ; 
    HDF5_CALL(err,H5Pclose(plist_id)) ;
    GRACE_TRACE("Done with write.") ;
    // Apparent-horizon state, in its own file beside this one.  Written after
    // the parallel file is closed so the two never contend, and by rank 0 only.
    save_horizon_checkpoint(iter) ;
    // Append the checkpoint to the list
    checkpoint_list.push_back(iter) ;
    if ( checkpoint_list.size() > max_n_checkpoints ) {
        delete_checkpoint() ;
    }
    GRACE_TRACE("Done with write.") ; 
    next_checkpoint_wtime = grace::get_total_runtime() + checkpoint_wtime_interval * 3600 ; 
    next_checkpoint_iter  = grace::get_iteration() + checkpoint_iter_interval ;
    next_checkpoint_time  = grace::get_simulation_time() + checkpoint_time_interval ;

    GRACE_INFO("Checkpoint saved successfully.") ; 
    GRACE_PROFILING_POP_REGION ; 
}



void checkpoint_handler_impl_t::load_checkpoint(int64_t iter )  
{
    GRACE_PROFILING_PUSH_REGION("load_checkpoint") ; 
    if ( iter < 0 ) {
        iter = checkpoint_list.back() ;
    }
    /**********************************************************************/
    /* We do the following operations here:                               */
    /* 1) Load the forest file and setup the grid                         */
    /* 2) Once the grid is set up we read the data                        */
    /**********************************************************************/
    GRACE_INFO("Loading checkpoint from iteration {}", iter) ; 

    auto grid_fname = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ; 

    p4est_connectivity_t * conn = nullptr;
    p4est_t* p4est  = p4est_load_ext(
        grid_fname.string().c_str(),
        sc_MPI_COMM_WORLD,
        sizeof(amr::grace_quadrant_user_data_t),
        true,       // load_data
        1,          // autopartition: ignore saved partition, allows any core count
        1,          // broadcasthead: rank 0 reads headers and bcasts
        nullptr,    // user_pointer
        &conn
    ) ;
    ASSERT( p4est != nullptr, "Could not load forest file " << grid_fname ) ;
    ASSERT( conn != nullptr, "Could not load connectivity file " << grid_fname ) ;

    amr::connectivity::initialize(conn) ;
    amr::forest::initialize(p4est) ;
    // Repartition with SFC respecting sibling families (autopartition gives uniform)
    p4est_partition(amr::forest::get().get(), 1, nullptr) ;
    /**********************************************************************/
    /* Now we set these static variables from the parameter file          */
    /* Later we will need to check that they haven't changed since        */
    /* when the checkpoint was written.                                   */
    /**********************************************************************/
    grace::amr::detail::_nx = 
        grace::config_parser::get()["amr"]["npoints_block_x"].as<int64_t>() ;
    grace::amr::detail::_ny =
        grace::config_parser::get()["amr"]["npoints_block_y"].as<int64_t>() ;
    grace::amr::detail::_nz = 
        grace::config_parser::get()["amr"]["npoints_block_z"].as<int64_t>() ;
    grace::amr::detail::_ngz =
        grace::config_parser::get()["amr"]["n_ghostzones"].as<int>() ;
    ASSERT(grace::amr::detail::_ngz % 2 == 0,
        "n_ghostzones must be even (required by div-free prolongation), got " << grace::amr::detail::_ngz) ;
    ASSERT(grace::amr::detail::_nx == grace::amr::detail::_ny
        && grace::amr::detail::_ny == grace::amr::detail::_nz,
        "npoints_block_x/y/z must all be equal, got "
        << grace::amr::detail::_nx << ", " << grace::amr::detail::_ny << ", " << grace::amr::detail::_nz) ;
    /**********************************************************************/
    GRACE_INFO("Allocating memory...");
    /**********************************************************************/
    grace::variable_list::initialize() ;
    grace::runtime::initialize() ; 
    grace::coordinate_system::initialize() ;
    grace::eos::initialize() ;
    grace::co_tracker::initialize() ;
    // Horizons too: on a restart the else-branch in grace::initialize() that
    // normally constructs these singletons never runs, so anything this
    // function reads from must be brought up here.
    grace::horizon_manager::initialize() ;
    /**********************************************************************/
    auto data_fname = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    /**********************************************************************/
    herr_t err ;
    // Create property list for parallel file access
    hid_t plist_id ; 
    HDF5_CALL(plist_id,H5Pcreate(H5P_FILE_ACCESS)) ; 
    HDF5_CALL(err,H5Pset_fapl_mpio(plist_id, mpi_comm_world, MPI_INFO_NULL)) ; 
    // Open the file
    hid_t file_id ;
    HDF5_CALL(file_id,H5Fopen(data_fname.string().c_str(), H5F_ACC_RDONLY, plist_id)) ;
    /**********************************************************************/
    /* Read the iteration and time attributes                             */
    /**********************************************************************/
    unsigned int iter_read ;
    double time_read ;
    hid_t attr_id, attr_dataspace_id ;
    HDF5_CALL(attr_id,H5Aopen(file_id, "Iteration", H5P_DEFAULT)) ;
    HDF5_CALL(attr_dataspace_id,H5Aget_space(attr_id)) ;
    HDF5_CALL(err,H5Aread(attr_id, H5T_NATIVE_UINT, &iter_read)) ;
    HDF5_CALL(err,H5Sclose(attr_dataspace_id)) ;    
    HDF5_CALL(err,H5Aclose(attr_id)) ;
    HDF5_CALL(attr_id,H5Aopen(file_id, "Time", H5P_DEFAULT)) ;
    HDF5_CALL(attr_dataspace_id,H5Aget_space(attr_id)) ;
    HDF5_CALL(err,H5Aread(attr_id, H5T_NATIVE_DOUBLE, &time_read)) ;
    HDF5_CALL(err,H5Sclose(attr_dataspace_id)) ;
    HDF5_CALL(err,H5Aclose(attr_id)) ;
    /**********************************************************************/
    ASSERT(iter == iter_read, "Iterations don't match in checkpoint file " << iter << " != " << iter_read) ;
    /**********************************************************************/
    // Set iteration and time in grace runtime 
    grace::set_iteration(iter) ;
    grace::set_simulation_time(time_read) ;
    grace::set_initial_simulation_time(time_read) ;
    /**********************************************************************/
    /* Read the data from the hdf5 file                                   */
    /**********************************************************************/
    hid_t dxpl ;
    HDF5_CALL(dxpl, H5Pcreate(H5P_DATASET_XFER)) ;
    HDF5_CALL(err, H5Pset_dxpl_mpio(dxpl,H5FD_MPIO_COLLECTIVE)) ;
    /**********************************************************************/
    // read coordinate extents 
    auto read_extent_dataset = [&](std::string const& name, double &val_min, double &val_max, std::string const& dim_label){
        hsize_t dset_dims[1];
        hid_t dset_id = H5Dopen(file_id, name.c_str(), H5P_DEFAULT);
        ASSERT(dset_id >= 0, "Failed to open dataset " << name);

        hid_t space_id = H5Dget_space(dset_id);
        ASSERT(space_id >= 0, "Failed to get dataspace for " << name);

        HDF5_CALL(err, H5Sget_simple_extent_dims(space_id, dset_dims, nullptr));
        ASSERT(dset_dims[0] == 2, "Dimension of " << name << " is not 2");

        // Since the dataset is 2 elements, copy properly
        double tmp[2];
        HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, dxpl, tmp));
        val_min = tmp[0];
        val_max = tmp[1];

        // Compare with expected parameters
        double param[2] = {
            grace::get_param<double>("amr", dim_label + "min"),
            grace::get_param<double>("amr", dim_label + "max")
        };
        ASSERT(val_min == param[0] && val_max == param[1],
            name << " extents do not match checkpoint: "
            << val_min << " != " << param[0] << " or "
            << val_max << " != " << param[1]);

        HDF5_CALL(err, H5Dclose(dset_id));
        HDF5_CALL(err, H5Sclose(space_id));
    };

    // All ranks participate collectively
    double xmin, xmax, ymin, ymax, zmin, zmax;
    read_extent_dataset("grid_extent_x", xmin, xmax, "x");
    read_extent_dataset("grid_extent_y", ymin, ymax, "y");
    read_extent_dataset("grid_extent_z", zmin, zmax, "z");

    auto read_co_dataset = [&](hid_t gid, int ico, std::vector<std::unique_ptr<co_tracker_t>>& vec){
        std::string grp_name = "co_" + std::to_string(ico) ; 
        hid_t co_grp;
        HDF5_CALL(co_grp, H5Gopen2(gid, grp_name.c_str(), H5P_DEFAULT));

        // Read location
        std::array<double,3> loc;
        {
            hid_t dset_id;
            HDF5_CALL(dset_id, H5Dopen2(co_grp, "location", H5P_DEFAULT));
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, loc.data()));
            HDF5_CALL(err, H5Dclose(dset_id));
        }

        // Read radius
        double radius;
        {
            hid_t dset_id;
            HDF5_CALL(dset_id, H5Dopen2(co_grp, "radius", H5P_DEFAULT));
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &radius));
            HDF5_CALL(err, H5Dclose(dset_id));
        }

        // Read name attribute
        std::string name;
        {
            hid_t attr_id, atype_id;
            HDF5_CALL(attr_id,  H5Aopen(co_grp, "name", H5P_DEFAULT));
            HDF5_CALL(atype_id, H5Aget_type(attr_id));
            size_t sz = H5Tget_size(atype_id);
            name.resize(sz);
            HDF5_CALL(err, H5Aread(attr_id, atype_id, name.data()));
            HDF5_CALL(err, H5Tclose(atype_id));
            HDF5_CALL(err, H5Aclose(attr_id));
            // H5Tget_size includes the null terminator for H5T_STR_NULLTERM strings
            name.resize(std::strlen(name.c_str()));
        }

        // Read kind attribute -- you need to write this in write_co_dset too
        std::string kind;
        {
            hid_t attr_id, atype_id;
            HDF5_CALL(attr_id,  H5Aopen(co_grp, "kind", H5P_DEFAULT));
            HDF5_CALL(atype_id, H5Aget_type(attr_id));
            size_t sz = H5Tget_size(atype_id);
            kind.resize(sz);
            HDF5_CALL(err, H5Aread(attr_id, atype_id, kind.data()));
            HDF5_CALL(err, H5Tclose(atype_id));
            HDF5_CALL(err, H5Aclose(attr_id));
            // H5Tget_size includes the null terminator for H5T_STR_NULLTERM strings
            kind.resize(std::strlen(kind.c_str()));
        }

        if (kind=="ns") {
            vec.push_back(std::make_unique<ns_tracker_t>(
                loc,radius,name 
            )) ; 
        } else if ( kind == "bh" ) {
            vec.push_back(std::make_unique<puncture_tracker_t>(
                loc,radius,name 
            )) ; 
        } else {
            ERROR("Unknown co kind in checkpoint " << kind ) ;
        }
        HDF5_CALL(err, H5Gclose(co_grp));
    };
    auto& co_tracker = grace::co_tracker::get() ;
    if (co_tracker.is_active()) {
        int ncos_exp = co_tracker.get_n_cos() ; 
        hid_t tracker_grp;
        HDF5_CALL(tracker_grp, H5Gopen2(file_id, "co_tracker", H5P_DEFAULT));

        // Read n_cos
        int ncos;
        {
            hid_t dset_id;
            HDF5_CALL(dset_id, H5Dopen2(tracker_grp, "n_cos", H5P_DEFAULT));
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ncos));
            HDF5_CALL(err, H5Dclose(dset_id));
        }
        bool merged ; 
        // Read has_merged
        {
            int merged_int;
            hid_t dset_id;
            HDF5_CALL(dset_id, H5Dopen2(tracker_grp, "has_merged", H5P_DEFAULT));
            HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &merged_int));
            HDF5_CALL(err, H5Dclose(dset_id));
            merged = (merged_int != 0);
        }

        ASSERT(ncos==ncos_exp, "Mismatch in number of compact objects in the checkpoint.") ; 
        
        co_tracker.set_merged(merged) ;
        // danger zone! make this protected 
        std::vector<std::unique_ptr<co_tracker_t>> co_list ; 
        for( int ico=0; ico<ncos; ++ico ) read_co_dataset(tracker_grp,ico,co_list) ;
        co_tracker.set_co_list(std::move(co_list)) ;
        HDF5_CALL(err, H5Gclose(tracker_grp));
    }

    // Apparent horizons.  A checkpoint written with a different l_max is
    // rejected per horizon by deserialize_horizon(), which re-seeds that
    // horizon rather than misreading the buffer.
    //
    // Horizon state now lives in checkpoint_ah_<iter>.h5 beside this file.  The
    // block below is the legacy path, kept so checkpoints written before that
    // split still restart; it reads the group out of the file already open
    // here, so restart still consults exactly one directory.
    auto& ah_mgr = grace::horizon_manager::get() ;
    if (ah_mgr.is_active() && !load_horizon_checkpoint(iter)) {
        int const nah_exp = ah_mgr.n_horizons() ;
        htri_t const has_ah = H5Lexists(file_id, "apparent_horizon", H5P_DEFAULT) ;
        if (has_ah > 0) {
            GRACE_WARN("No apparent-horizon sidecar for iteration {}; reading "
                       "the horizon state stored inside the checkpoint itself. "
                       "That is a pre-sidecar checkpoint -- the next one this "
                       "run writes will use checkpoint_ah_{:06d}.h5.",
                       iter, iter) ;
        }
        if (has_ah <= 0) {
            // Either the checkpoint predates horizon support, or the earlier
            // run simply had no horizons configured. Both are ordinary, and
            // neither is a reason to refuse to restart: the stored state is
            // only a warm start, worth a few Newton iterations once.
            GRACE_WARN("Checkpoint contains no apparent-horizon state (written "
                       "before horizons were enabled, or by an older version); "
                       "horizons will start from their initial guesses.") ;
        } else {
            hid_t ah_grp;
            HDF5_CALL(ah_grp, H5Gopen2(file_id, "apparent_horizon", H5P_DEFAULT));

            int nah;
            {
                hid_t dset_id;
                HDF5_CALL(dset_id, H5Dopen2(ah_grp, "n_horizons", H5P_DEFAULT));
                HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &nah));
                HDF5_CALL(err, H5Dclose(dset_id));
            }
            // Do not abort on a count mismatch. Adding or removing a horizon
            // between runs is a normal thing to do -- enabling a common-horizon
            // search partway through a binary, say -- and the stored state is
            // only a warm start. Losing it costs a few extra Newton iterations
            // on one step; refusing to restart costs the run.
            if (nah != nah_exp) {
                GRACE_WARN("Checkpoint holds {} apparent horizons but {} are "
                           "configured; horizon shapes will not be restored and "
                           "every horizon starts from its initial guess.",
                           nah, nah_exp) ;
                HDF5_CALL(err, H5Gclose(ah_grp));
                nah = 0 ;
            }

            for (int iah = 0; iah < nah; ++iah) {
                std::string const base = "horizon_" + std::to_string(iah) ;

                std::vector<double> buf ;
                {
                    hid_t dset_id, space_id;
                    HDF5_CALL(dset_id, H5Dopen2(ah_grp, base.c_str(), H5P_DEFAULT));
                    HDF5_CALL(space_id, H5Dget_space(dset_id));
                    hsize_t dims[1] ;
                    H5Sget_simple_extent_dims(space_id, dims, nullptr) ;
                    buf.resize(dims[0]) ;
                    HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                           H5P_DEFAULT, buf.data()));
                    HDF5_CALL(err, H5Sclose(space_id));
                    HDF5_CALL(err, H5Dclose(dset_id));
                }
                int found = 0 ;
                {
                    std::string const fname = base + "_found" ;
                    hid_t dset_id;
                    HDF5_CALL(dset_id, H5Dopen2(ah_grp, fname.c_str(), H5P_DEFAULT));
                    HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                           H5P_DEFAULT, &found));
                    HDF5_CALL(err, H5Dclose(dset_id));
                }
                ah_mgr.deserialize_horizon(iah, buf, found != 0) ;

                // Optional, like the state array below: a checkpoint written
                // before this field existed simply restores without it, and
                // deserialize_last_good() is length-tolerant so a shorter
                // record loads with the missing entries defaulted.
                std::string const lname = base + "_last_good" ;
                if (H5Lexists(ah_grp, lname.c_str(), H5P_DEFAULT) > 0) {
                    std::vector<double> lg ;
                    hid_t dset_id, lspace;
                    HDF5_CALL(dset_id, H5Dopen2(ah_grp, lname.c_str(), H5P_DEFAULT));
                    HDF5_CALL(lspace, H5Dget_space(dset_id));
                    hsize_t ldims[1] = {0} ;
                    H5Sget_simple_extent_dims(lspace, ldims, nullptr) ;
                    lg.resize(ldims[0]) ;
                    HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                           H5P_DEFAULT, lg.data()));
                    HDF5_CALL(err, H5Sclose(lspace));
                    HDF5_CALL(err, H5Dclose(dset_id));
                    ah_mgr.deserialize_last_good(iah, lg) ;
                }

                // State is optional: a checkpoint written before common-horizon
                // support has no such dataset, and those runs had no common
                // horizons to restore.
                std::string const sname = base + "_state" ;
                if (H5Lexists(ah_grp, sname.c_str(), H5P_DEFAULT) > 0) {
                    int sc[3] = {0, 0, 0} ;
                    hid_t dset_id, sspace;
                    HDF5_CALL(dset_id, H5Dopen2(ah_grp, sname.c_str(), H5P_DEFAULT));
                    HDF5_CALL(sspace, H5Dget_space(dset_id));
                    hsize_t sdims[1] = {0} ;
                    H5Sget_simple_extent_dims(sspace, sdims, nullptr) ;
                    HDF5_CALL(err, H5Sclose(sspace));
                    // Older checkpoints stored only (state, failures); read what
                    // is there and let has_existed fall back to the state.
                    HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                           H5P_DEFAULT, sc));
                    HDF5_CALL(err, H5Dclose(dset_id));
                    if (sdims[0] < 3) {
                        sc[2] = (sc[0] == static_cast<int>(horizon_state_t::found)
                              || sc[0] == static_cast<int>(horizon_state_t::abandoned))
                              ? 1 : 0 ;
                    }
                    ah_mgr.set_horizon_state(iah, sc[0], sc[1], sc[2]) ;
                }
            }
            if (nah > 0) { HDF5_CALL(err, H5Gclose(ah_grp)); }
        }
    }

    // if present, load mid-run B-field injection state
    {
        htri_t exists = H5Lexists(file_id, "b_field_injection", H5P_DEFAULT) ;
        if (exists > 0) {
            hid_t inj_grp ;
            HDF5_CALL(inj_grp, H5Gopen2(file_id, "b_field_injection", H5P_DEFAULT));
            int has_fired_int = 0 ;
            {
                hid_t dset_id;
                HDF5_CALL(dset_id, H5Dopen2(inj_grp, "has_fired", H5P_DEFAULT));
                HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &has_fired_int));
                HDF5_CALL(err, H5Dclose(dset_id));
            }
            grace::b_field_injection_set_fired(has_fired_int != 0) ;
            HDF5_CALL(err, H5Gclose(inj_grp));
        }
    }

    // if present, restore particle subsystem state. Routed through the
    // module's initialize() so the config + bcs + min_quad_width are set
    // up before the loader replaces the would-be seed contents. Must run
    // after the grid has been restored — load_particles_from_checkpoint
    // resolves ownership against the current fluid topology shadow.
    #ifdef GRACE_ENABLE_PARTICLES
    grace::particles::particles_module_t::get().initialize(file_id);
    #endif


    /**********************************************************************/
    /* Read the state data                                                */
    /**********************************************************************/
    auto state = grace::variable_list::get().getstate() ; 
    auto sstate = grace::variable_list::get().getstaggeredstate() ; 

    detail::read_data_hdf5(file_id, dxpl, "CellCenteredData", state) ; 
    detail::read_data_hdf5(file_id, dxpl, "CornerCenteredData", sstate.corner_staggered_fields) ;
    detail::read_data_hdf5(file_id, dxpl, "EdgeCenteredDataXY", sstate.edge_staggered_fields_xy) ;
    detail::read_data_hdf5(file_id, dxpl, "EdgeCenteredDataXZ", sstate.edge_staggered_fields_xz) ;
    detail::read_data_hdf5(file_id, dxpl, "EdgeCenteredDataYZ", sstate.edge_staggered_fields_yz) ;
    detail::read_data_hdf5(file_id, dxpl, "FaceCenteredDataX", sstate.face_staggered_fields_x) ;
    detail::read_data_hdf5(file_id, dxpl, "FaceCenteredDataY", sstate.face_staggered_fields_y) ;
    detail::read_data_hdf5(file_id, dxpl, "FaceCenteredDataZ", sstate.face_staggered_fields_z) ;
    /**********************************************************************/
    /* Close the file                                                     */
    /**********************************************************************/
    HDF5_CALL(err,H5Fclose(file_id)) ;
    /**********************************************************************/
    /* Cleanup                                                            */
    /**********************************************************************/
    HDF5_CALL(err,H5Pclose(dxpl)) ;
    HDF5_CALL(err,H5Pclose(plist_id)) ;
    /**********************************************************************/
    /* Compute auxiliary quantities                                       */
    /**********************************************************************/
    /**********************************************************************/
    next_checkpoint_time += grace::get_simulation_time() ; 
    next_checkpoint_iter += grace::get_iteration() ;
    GRACE_INFO("Checkpoint loaded successfully.") ;
    GRACE_PROFILING_POP_REGION ; 
}

//---------------------------------------------------------------------------
// Apparent-horizon sidecar
//
// Kept out of the main checkpoint file because that file is O(100 GB) in
// production while this is a few KB, and horizon state is exactly the thing one
// wants to inspect, copy and diff on its own.  Losing it costs a warm start and
// nothing else, so nothing here is ever required for a restart to succeed.
//---------------------------------------------------------------------------

void checkpoint_handler_impl_t::save_horizon_checkpoint(int64_t iter) const
{
    auto& ah_mgr = grace::horizon_manager::get() ;
    if (!ah_mgr.is_active()) return ;
    // Horizon state is bit-identical on every rank (pinned by
    // horizon_manager_mpi_test), so there is nothing to gather: rank 0 writes
    // serially and every rank reads the result back independently.
    if (parallel::mpi_comm_rank() != 0) return ;

    auto const path = detail::get_filename(checkpoint_dir, "checkpoint_ah", iter, ".h5") ;

    herr_t err ;
    hid_t  file_id ;
    HDF5_CALL(file_id, H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC,
                                 H5P_DEFAULT, H5P_DEFAULT)) ;

    int const nah = ah_mgr.n_horizons() ;

    auto write_scalar_int = [&](hid_t where, char const* name, int value) {
        hid_t space_id, dset_id ;
        HDF5_CALL(space_id, H5Screate(H5S_SCALAR)) ;
        HDF5_CALL(dset_id, H5Dcreate2(where, name, H5T_NATIVE_INT, space_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) ;
        HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, &value)) ;
        HDF5_CALL(err, H5Dclose(dset_id)) ;
        HDF5_CALL(err, H5Sclose(space_id)) ;
    } ;

    // Provenance, so the file says what it is without the run beside it.  The
    // iteration also records how stale the state is: a checkpoint that does not
    // land on a multiple of find_every carries the previous find's result.
    write_scalar_int(file_id, "iteration", static_cast<int>(iter)) ;
    write_scalar_int(file_id, "n_horizons", nah) ;
    {
        double const t = grace::get_simulation_time() ;
        hid_t space_id, dset_id ;
        HDF5_CALL(space_id, H5Screate(H5S_SCALAR)) ;
        HDF5_CALL(dset_id, H5Dcreate2(file_id, "time", H5T_NATIVE_DOUBLE, space_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) ;
        HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, &t)) ;
        HDF5_CALL(err, H5Dclose(dset_id)) ;
        HDF5_CALL(err, H5Sclose(space_id)) ;
    }

    auto write_doubles = [&](char const* name, std::vector<double> const& v) {
        hsize_t dims[1] = { static_cast<hsize_t>(v.size()) } ;
        hid_t space_id, dset_id ;
        HDF5_CALL(space_id, H5Screate_simple(1, dims, nullptr)) ;
        HDF5_CALL(dset_id, H5Dcreate2(file_id, name, H5T_NATIVE_DOUBLE, space_id,
                                      H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) ;
        HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                                H5P_DEFAULT, v.data())) ;
        return dset_id ;   // caller closes, after attaching attributes
    } ;

    for (int iah = 0; iah < nah; ++iah) {
        std::string const base = "horizon_" + std::to_string(iah) ;

        {
            auto const buf = ah_mgr.serialize_horizon(iah) ;
            hid_t dset_id = write_doubles(base.c_str(), buf) ;
            // The name is what makes a restart able to tell one horizon from
            // another.  Positional identity alone cannot: in a binary, AH1 and
            // AH2 have the same l_max and the same initial radius, so a
            // reordered horizons list would otherwise load each into the other
            // with nothing to detect it.
            grace::IO::write_dataset_string_attribute_hdf5(dset_id, "name",
                                                           ah_mgr.get(iah).name) ;
            HDF5_CALL(err, H5Dclose(dset_id)) ;
        }
        write_scalar_int(file_id, (base + "_found").c_str(),
                         ah_mgr.horizon_ever_found(iah) ? 1 : 0) ;
        {
            auto const lg = ah_mgr.serialize_last_good(iah) ;
            hid_t dset_id = write_doubles((base + "_last_good").c_str(), lg) ;
            HDF5_CALL(err, H5Dclose(dset_id)) ;
        }
        {
            int const sc[3] = { ah_mgr.horizon_state_code(iah),
                                ah_mgr.horizon_failure_count(iah),
                                ah_mgr.horizon_has_existed(iah) ? 1 : 0 } ;
            hsize_t dims[1] = { 3 } ;
            hid_t space_id, dset_id ;
            HDF5_CALL(space_id, H5Screate_simple(1, dims, nullptr)) ;
            HDF5_CALL(dset_id, H5Dcreate2(file_id, (base + "_state").c_str(),
                                          H5T_NATIVE_INT, space_id,
                                          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)) ;
            HDF5_CALL(err, H5Dwrite(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                    H5P_DEFAULT, sc)) ;
            HDF5_CALL(err, H5Dclose(dset_id)) ;
            HDF5_CALL(err, H5Sclose(space_id)) ;
        }
    }

    HDF5_CALL(err, H5Fclose(file_id)) ;
    GRACE_VERBOSE("Wrote apparent-horizon checkpoint {}", path.string()) ;
}

bool checkpoint_handler_impl_t::load_horizon_checkpoint(int64_t iter) const
{
    auto& ah_mgr = grace::horizon_manager::get() ;
    if (!ah_mgr.is_active()) return false ;

    auto const path = detail::get_filename(checkpoint_dir, "checkpoint_ah", iter, ".h5") ;
    if (!std::filesystem::exists(path)) return false ;

    // Every rank opens it read-only and serially, as the EOS table reader
    // already does: the payload is small and identical for all of them.
    herr_t err ;
    hid_t  file_id ;
    HDF5_CALL(file_id, H5Fopen(path.string().c_str(), H5F_ACC_RDONLY, H5P_DEFAULT)) ;

    int nah = 0 ;
    {
        hid_t dset_id ;
        HDF5_CALL(dset_id, H5Dopen2(file_id, "n_horizons", H5P_DEFAULT)) ;
        HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, &nah)) ;
        HDF5_CALL(err, H5Dclose(dset_id)) ;
    }

    int const nah_exp = ah_mgr.n_horizons() ;
    if (nah != nah_exp) {
        // Same reasoning as the count check has always used: adding or removing
        // a horizon between runs is a normal thing to do, and the stored state
        // is only a warm start.  Refusing to restart would cost the run.
        GRACE_WARN("Apparent-horizon checkpoint holds {} horizons but {} are "
                   "configured; every horizon starts from its initial guess.",
                   nah, nah_exp) ;
        HDF5_CALL(err, H5Fclose(file_id)) ;
        return true ;
    }

    auto read_doubles = [&](std::string const& name, std::vector<double>& out) {
        hid_t dset_id, space_id ;
        HDF5_CALL(dset_id, H5Dopen2(file_id, name.c_str(), H5P_DEFAULT)) ;
        HDF5_CALL(space_id, H5Dget_space(dset_id)) ;
        hsize_t dims[1] = {0} ;
        H5Sget_simple_extent_dims(space_id, dims, nullptr) ;
        out.resize(dims[0]) ;
        HDF5_CALL(err, H5Dread(dset_id, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL,
                               H5P_DEFAULT, out.data())) ;
        HDF5_CALL(err, H5Sclose(space_id)) ;
        return dset_id ;   // caller closes, after reading attributes
    } ;

    for (int iah = 0; iah < nah; ++iah) {
        std::string const base = "horizon_" + std::to_string(iah) ;

        std::vector<double> buf ;
        hid_t dset_id = read_doubles(base, buf) ;

        // Identity check.  On a mismatch this horizon keeps its configured
        // guess rather than inheriting another horizon's shape and centre --
        // the failure that positional identity alone could not see.
        std::string stored_name ;
        if (H5Aexists(dset_id, "name") > 0) {
            hid_t attr_id, atype ;
            HDF5_CALL(attr_id, H5Aopen(dset_id, "name", H5P_DEFAULT)) ;
            HDF5_CALL(atype, H5Aget_type(attr_id)) ;
            // write_dataset_string_attribute_hdf5() writes a *variable-length*
            // string, and for those H5Tget_size reports the size of a pointer
            // rather than of the text.  Reading one as fixed-length yields the
            // pointer's bytes as characters, which then fails the comparison
            // below and silently skips the restore.  Handle both encodings.
            if (H5Tis_variable_str(atype) > 0) {
                char* s = nullptr ;
                HDF5_CALL(err, H5Aread(attr_id, atype, &s)) ;
                if (s) { stored_name = s ; H5free_memory(s) ; }
            } else {
                std::string raw(H5Tget_size(atype), '\0') ;
                HDF5_CALL(err, H5Aread(attr_id, atype, raw.data())) ;
                raw.erase(raw.find_last_not_of('\0') + 1) ;
                stored_name = raw ;
            }
            HDF5_CALL(err, H5Tclose(atype)) ;
            HDF5_CALL(err, H5Aclose(attr_id)) ;
        }
        HDF5_CALL(err, H5Dclose(dset_id)) ;

        std::string const& want = ah_mgr.get(iah).name ;
        if (!stored_name.empty() && stored_name != want) {
            GRACE_WARN("Apparent-horizon checkpoint slot {} holds '{}' but '{}' "
                       "is configured there; '{}' starts from its initial guess "
                       "rather than inheriting another horizon's shape.",
                       iah, stored_name, want, want) ;
            continue ;
        }

        int found = 0 ;
        {
            hid_t d ;
            HDF5_CALL(d, H5Dopen2(file_id, (base + "_found").c_str(), H5P_DEFAULT)) ;
            HDF5_CALL(err, H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                   H5P_DEFAULT, &found)) ;
            HDF5_CALL(err, H5Dclose(d)) ;
        }
        if (!ah_mgr.deserialize_horizon(iah, buf, found != 0)) {
            GRACE_WARN("Apparent horizon '{}' was checkpointed at a different "
                       "l_max; it starts from its initial guess.", want) ;
            continue ;
        }

        {
            std::vector<double> lg ;
            hid_t d = read_doubles(base + "_last_good", lg) ;
            HDF5_CALL(err, H5Dclose(d)) ;
            ah_mgr.deserialize_last_good(iah, lg) ;
        }
        {
            int sc[3] = {0, 0, 0} ;
            hid_t d, sspace ;
            HDF5_CALL(d, H5Dopen2(file_id, (base + "_state").c_str(), H5P_DEFAULT)) ;
            HDF5_CALL(sspace, H5Dget_space(d)) ;
            HDF5_CALL(err, H5Sclose(sspace)) ;
            HDF5_CALL(err, H5Dread(d, H5T_NATIVE_INT, H5S_ALL, H5S_ALL,
                                   H5P_DEFAULT, sc)) ;
            HDF5_CALL(err, H5Dclose(d)) ;
            ah_mgr.set_horizon_state(iah, sc[0], sc[1], sc[2]) ;
        }
    }

    HDF5_CALL(err, H5Fclose(file_id)) ;
    GRACE_INFO("Restored apparent-horizon state from {}", path.string()) ;
    return true ;
}

void checkpoint_handler_impl_t::delete_checkpoint()
{
    auto iter = checkpoint_list.front() ; 
    GRACE_VERBOSE("Deleting checkpoint at iter {}",iter ) ; 
    // first the forest file 
    auto fname = detail::get_filename(checkpoint_dir, "checkpoint_grid", iter, ".bin") ;
    std::filesystem::remove(fname) ;
    // then the data
    auto data_fname = detail::get_filename(checkpoint_dir, "checkpoint_data", iter, ".h5") ;
    std::filesystem::remove(data_fname) ;
    // and the apparent-horizon sidecar, which need not exist: checkpoints
    // written before it, or by a run with no horizons, have none.  The
    // error_code overload keeps that from being an error.
    {
        auto ah_fname = detail::get_filename(checkpoint_dir, "checkpoint_ah", iter, ".h5") ;
        std::error_code ec ;
        std::filesystem::remove(ah_fname, ec) ;
    }
    // remove from the list
    checkpoint_list.pop_front() ;
}

void checkpoint_handler_impl_t::detect_checkpoints() 
{
    /**********************************************************************/
    /* Clear the list of checkpoints                                      */
    /**********************************************************************/
    checkpoint_list.clear() ; 
    /**********************************************************************/
    /* We search for checkpoints in the directory specified by the        */
    /* parameter <code>checkpoint_dir</code>.                             */
    /**********************************************************************/
    std::filesystem::path dir(checkpoint_dir) ;
    if ( !std::filesystem::exists(dir) ) {
        GRACE_VERBOSE("Checkpoint directory {} does not exist. No checkpoints found.", dir.string()) ;
        return ; 
    }
    /**********************************************************************/
    /* We search for files with the pattern checkpoint_data_*.h5          */
    /**********************************************************************/
    std::regex data_pattern(R"(checkpoint_data_(\d+)\.h5)") ;
    std::regex grid_pattern(R"(checkpoint_grid_(\d+)\.bin)") ;

    std::vector<int64_t> data_iters ; 
    std::vector<int64_t> grid_iters ; 

    for( auto const& entry: std::filesystem::directory_iterator(dir) ) {
        std::string filename = entry.path().filename().string() ; 
        GRACE_INFO("checking {}...", filename) ; 
        std::smatch match ;

        if ( std::regex_match(filename, match, data_pattern)) {
            data_iters.push_back(std::stoll(match[1].str())) ; 
        } else if ( std::regex_match(filename, match, grid_pattern)) {
            grid_iters.push_back(std::stoll(match[1].str())) ; 
        }
    }
    std::sort(data_iters.begin(), data_iters.end()); std::sort(grid_iters.begin(),grid_iters.end()) ; 
    /**********************************************************************/
    /* Check if there are any data files without corresponding grid files */
    /**********************************************************************/
    for( int i=0; i<data_iters.size(); ++i) {
        if ( std::find(grid_iters.begin(), grid_iters.end(), data_iters[i]) == grid_iters.end() ) {
            GRACE_WARN("Checkpoint data file {} does not have a corresponding grid file.", data_iters[i]) ;
        } else {
            checkpoint_list.push_back(data_iters[i]) ;
        }
    }
    /**********************************************************************/
    if (checkpoint_list.size() > 0) {
        GRACE_INFO("Found {} checkpoints in directory {}, most recent at iteration {}", checkpoint_list.size(), dir.string(), checkpoint_list.front()) ;
    } else {
        GRACE_INFO("No checkpoints found") ;
    }
}

bool checkpoint_handler_impl_t::need_checkpoint()  {
    if ( checkpoint_interval_type == "iteration" ) {
        return grace::get_iteration() > next_checkpoint_iter ;
    } else if ( checkpoint_interval_type == "time" ) {
        return grace::get_simulation_time() > next_checkpoint_time ;
    } else {
        return grace::get_total_runtime() > next_checkpoint_wtime ;
    } 
}

checkpoint_handler_impl_t::checkpoint_handler_impl_t() {
    // Read parameters for max number of checkpoints 
    max_n_checkpoints = grace::get_param<unsigned int>("checkpoints", "max_n_checkpoints") ; 
    // And for the directory 
    checkpoint_dir = grace::get_param<std::string>("checkpoints", "checkpoint_dir") ;
    // If the directory does not exist rank 0 should create it 
    if ( parallel::mpi_comm_rank() == 0 ) {
        if ( !std::filesystem::exists(checkpoint_dir) ) {
            std::filesystem::create_directory(checkpoint_dir) ;
        }
    }
    // Find the method to save checkpoints
    checkpoint_interval_type = grace::get_param<std::string>("checkpoints", "interval_kind") ;
    std::vector<std::string> methods{ "iteration", "time", "walltime" } ;
    if (std::find(methods.begin(), methods.end(), checkpoint_interval_type) == methods.end()) {
        ERROR("Invalid checkpointing method " << checkpoint_interval_type << " valid methods are 'iteration', 'time' and 'walltime'.") ;
    }
    if ( checkpoint_interval_type == "iteration" ) {
        checkpoint_iter_interval = grace::get_param<int64_t>("checkpoints", "iteration_interval") ;
        ASSERT(checkpoint_iter_interval > 0, "Iteration interval must be greater than zero!") ;
        next_checkpoint_iter = checkpoint_iter_interval ;
    } else if ( checkpoint_interval_type == "time" ) {
        checkpoint_time_interval = grace::get_param<double>("checkpoints", "time_interval") ;
        ASSERT(checkpoint_time_interval > 0, "Time interval must be greater than zero!") ;
        next_checkpoint_time = checkpoint_time_interval ;
    } else if ( checkpoint_interval_type == "walltime" ) {
        checkpoint_wtime_interval = grace::get_param<double>("checkpoints", "walltime_interval") ;
        ASSERT(checkpoint_wtime_interval > 0, "Walltime interval must be greater than zero!") ;
        // This is in hours! 
        next_checkpoint_wtime = checkpoint_wtime_interval * 3600;
    }
    _checkpoint_at_termination = grace::get_param<bool>("checkpoints", "checkpoint_at_termination") ; 
    // Detect checkpoints
    detect_checkpoints() ;

    have_checkpoint_ = checkpoint_list.size() > 0 ; 
}   


checkpoint_handler_impl_t::~checkpoint_handler_impl_t() {
    if ( _checkpoint_at_termination ) { save_checkpoint(); }
}

} // namespace grace 

#undef HDF5_CALL
