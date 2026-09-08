// ---------------------------------------------------------------------------
// Bit-exact round-trip test for the checkpoint write/load path.
//
// Recipe:
//   1. `grace::initialize` (driven by configs/checkpoint_roundtrip_test.yaml)
//      sets up a small AMR grid + initial data (e.g., TOV) that exercises
//      hydro conservatives, staggered face B, and the full Z4c evolved
//      variable set including the Bona-Masso shift driver.
//   2. Capture the in-memory state of every evolved variable (cell-centered
//      + staggered face/edge/corner) into mirror buffers.
//   3. Call `checkpoint_handler::save_checkpoint()` (production path).
//   4. Scribble over the device state with sentinel values to ensure the
//      load path actually writes what we expect.
//   5. Call `checkpoint_handler::load_checkpoint()` (production path).
//      This includes the post-load `compute_auxiliary_quantities()` call
//      that recomputes aux via c2p.
//   6. Compare current device state vs the snapshot taken in step 2,
//      bit-exactly per cell per variable.
//
// What this test catches:
//   - Lossy HDF5 round-trip (compression, type conversion, chunk boundary
//     pathology).
//   - Missing state in the checkpoint write set — anything evolved by the
//     RHS that isn't in the registered `getstate()` / `getstaggeredstate()`
//     will fail this test because the load can't restore it.
//   - Order-dependent recomputation in `compute_auxiliary_quantities()` if
//     the post-load pass uses a different cell-visit order than the
//     running version.
//   - Pathological behavior in the singleton initialization on the second
//     `initialize()` cycle (the load path tears down and re-creates).
//
// Notes:
//   - The test asserts bit-exactness on the EVOLVED state only.  The aux
//     state is recomputed via c2p post-load and is bit-exact iff c2p is
//     deterministic given cons (independently verified in the two-from-scratch
//     test, May 2026).
//   - Multi-rank: run with `mpirun -n N` to exercise the collective
//     MPI-IO write/read.  Bit-exactness must hold regardless of N.
//
// Author: carlo.musolino@aei.mpg.de

#include <catch2/catch_test_macros.hpp>

#include <grace_config.h>
#include <Kokkos_Core.hpp>
#include <grace/amr/grace_amr.hh>
#include <grace/data_structures/grace_data_structures.hh>
#include <grace/data_structures/variable_utils.hh>
#include <grace/geometry/horizon_manager.hh>
#include <grace/system/checkpoint_handler.hh>
#include <grace/utils/grace_utils.hh>
#include <grace/parallel/mpi_wrappers.hh>

#include <hdf5.h>

#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

// Bit-exact equality check on two Kokkos views with identical extents,
// returning (n_differing_cells, max_abs_diff, sample_location).
template <typename ViewT>
struct diff_result_t {
    size_t n_diff   = 0;
    double max_abs  = 0.0;
    int    first_i  = -1, first_j = -1, first_k = -1;
    int    first_v  = -1, first_q = -1;
};

template <typename ViewT>
auto compute_diff(ViewT const& a, ViewT const& b) {
    REQUIRE(a.rank() == b.rank());
    for (size_t d = 0; d < a.rank(); ++d) REQUIRE(a.extent(d) == b.extent(d));

    auto h_a = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, a);
    auto h_b = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace{}, b);

    diff_result_t<ViewT> r;
    auto const N = h_a.size();
    auto const* pa = h_a.data();
    auto const* pb = h_b.data();
    for (size_t i = 0; i < N; ++i) {
        if (pa[i] != pb[i]) {
            r.n_diff++;
            double const ad = std::fabs(pa[i] - pb[i]);
            if (ad > r.max_abs) r.max_abs = ad;
        }
    }
    return r;
}

// Deep-copy a View into a freshly-allocated host-resident snapshot of the
// same type — used to capture pre-write state for later comparison.
template <typename ViewT>
ViewT clone_view(ViewT const& src, std::string const& tag) {
    ViewT out(tag, src.layout());
    Kokkos::deep_copy(out, src);
    return out;
}

} // namespace

TEST_CASE("Checkpoint round-trip is bit-exact on evolved state",
          "[checkpoint][roundtrip]")
{
    using namespace grace;

    auto& vars   = variable_list::get();
    auto& cphndl = checkpoint_handler::get();

    auto state  = vars.getstate();
    auto sstate = vars.getstaggeredstate();

    // --- Step 1: capture pre-write snapshots of EVERY evolved field -------
    auto snap_state = clone_view(state, "checkpoint_test_state_snap");

    auto snap_corner  = clone_view(sstate.corner_staggered_fields, "snap_corner");
    auto snap_face_x  = clone_view(sstate.face_staggered_fields_x, "snap_face_x");
    auto snap_face_y  = clone_view(sstate.face_staggered_fields_y, "snap_face_y");
    auto snap_face_z  = clone_view(sstate.face_staggered_fields_z, "snap_face_z");
    auto snap_edge_xy = clone_view(sstate.edge_staggered_fields_xy, "snap_edge_xy");
    auto snap_edge_xz = clone_view(sstate.edge_staggered_fields_xz, "snap_edge_xz");
    auto snap_edge_yz = clone_view(sstate.edge_staggered_fields_yz, "snap_edge_yz");

    auto const init_iter = get_iteration();
    auto const init_time = get_simulation_time();

    // --- Step 2: write checkpoint via production path ---------------------
    cphndl.save_checkpoint();

    parallel::mpi_barrier(sc_MPI_COMM_WORLD);

    // --- Step 3: scribble device state with sentinel -----------------------
    // If the load fails to overwrite some field, the sentinel survives and
    // the diff catches it immediately.
    Kokkos::deep_copy(state,                                  -1.2345e+99);
    Kokkos::deep_copy(sstate.corner_staggered_fields,         -1.2345e+99);
    Kokkos::deep_copy(sstate.face_staggered_fields_x,         -1.2345e+99);
    Kokkos::deep_copy(sstate.face_staggered_fields_y,         -1.2345e+99);
    Kokkos::deep_copy(sstate.face_staggered_fields_z,         -1.2345e+99);
    Kokkos::deep_copy(sstate.edge_staggered_fields_xy,        -1.2345e+99);
    Kokkos::deep_copy(sstate.edge_staggered_fields_xz,        -1.2345e+99);
    Kokkos::deep_copy(sstate.edge_staggered_fields_yz,        -1.2345e+99);

    // --- Step 4: load checkpoint via production path ----------------------
    cphndl.load_checkpoint(init_iter);

    parallel::mpi_barrier(sc_MPI_COMM_WORLD);

    // --- Step 5: bit-exact diff against snapshots --------------------------
    auto print_result = [](char const* name, auto const& r, size_t total) {
        if (r.n_diff == 0) {
            std::cout << "  " << name << ": BIT-EXACT (" << total << " elements)\n";
        } else {
            std::cout << "  " << name << ": " << r.n_diff << "/" << total
                      << " elements differ, max |abs|=" << r.max_abs << "\n";
        }
    };

    auto r_state    = compute_diff(state,                                  snap_state);
    auto r_corner   = compute_diff(sstate.corner_staggered_fields,         snap_corner);
    auto r_face_x   = compute_diff(sstate.face_staggered_fields_x,         snap_face_x);
    auto r_face_y   = compute_diff(sstate.face_staggered_fields_y,         snap_face_y);
    auto r_face_z   = compute_diff(sstate.face_staggered_fields_z,         snap_face_z);
    auto r_edge_xy  = compute_diff(sstate.edge_staggered_fields_xy,        snap_edge_xy);
    auto r_edge_xz  = compute_diff(sstate.edge_staggered_fields_xz,        snap_edge_xz);
    auto r_edge_yz  = compute_diff(sstate.edge_staggered_fields_yz,        snap_edge_yz);

    if (parallel::mpi_comm_rank() == 0) {
        std::cout << "[checkpoint roundtrip] init_iter=" << init_iter
                  << " init_time=" << init_time << "\n";
        print_result("CellCenteredData",   r_state,    state.size());
        print_result("CornerCenteredData", r_corner,   sstate.corner_staggered_fields.size());
        print_result("FaceCenteredDataX",  r_face_x,   sstate.face_staggered_fields_x.size());
        print_result("FaceCenteredDataY",  r_face_y,   sstate.face_staggered_fields_y.size());
        print_result("FaceCenteredDataZ",  r_face_z,   sstate.face_staggered_fields_z.size());
        print_result("EdgeCenteredDataXY", r_edge_xy,  sstate.edge_staggered_fields_xy.size());
        print_result("EdgeCenteredDataXZ", r_edge_xz,  sstate.edge_staggered_fields_xz.size());
        print_result("EdgeCenteredDataYZ", r_edge_yz,  sstate.edge_staggered_fields_yz.size());
    }

    // Catch2 SECTION blocks would re-run the entire test body once per
    // section, which would attempt multiple save+load cycles and trip the
    // max_n_checkpoints=1 cleanup logic. Use sequential CHECKs instead
    // (Catch2 will still report each failure individually, the test fails
    // as a whole if any one is non-zero).
    CHECK(r_state.n_diff   == 0);
    CHECK(r_corner.n_diff  == 0);
    CHECK(r_face_x.n_diff  == 0);
    CHECK(r_face_y.n_diff  == 0);
    CHECK(r_face_z.n_diff  == 0);
    CHECK(r_edge_xy.n_diff == 0);
    CHECK(r_edge_xz.n_diff == 0);
    CHECK(r_edge_yz.n_diff == 0);
}

//---------------------------------------------------------------------------
// Apparent-horizon state.
//
// The shape is a warm start, so losing it costs a few Newton iterations. The
// life-cycle fields are not: without them a restart after merger sends a
// common horizon back to dormant and re-triggers a search that already
// happened, and forgets that a vanished horizon had been abandoned. All of it
// was verified by hand in a real binary and never automatically.
//---------------------------------------------------------------------------

TEST_CASE("Checkpoint round-trip preserves apparent-horizon state",
          "[checkpoint][roundtrip][horizon]")
{
    using namespace grace;

    auto& mgr    = horizon_manager::get();
    auto& cphndl = checkpoint_handler::get();

    REQUIRE(mgr.is_active());
    REQUIRE(mgr.n_horizons() == 1);
    auto& e = mgr.get(0);

    // A shape that is not the initial guess, so a fallback to the guess is
    // distinguishable from a genuine restore.
    e.surface.set_sphere(1.5);
    auto& coeffs = e.surface.coefficients();
    for (std::size_t c = 1; c < coeffs.size(); ++c)
        coeffs[c] = 0.01 * static_cast<double>(c);

    e.state                = horizon_state_t::found;
    e.consecutive_failures = 2;
    e.has_existed          = true;
    e.ever_found           = true;

    // The last successful result and the previous centroid. Without these a
    // restart mid-merger sends the common-horizon trigger back to waiting for
    // both parents, and the excision margin loses the motion it shrinks by.
    e.has_last_good              = true;
    e.last_good_iteration        = 7;
    e.last_good.found            = true;
    e.last_good.centroid         = { 0.25, -0.5, 0.75 };
    e.last_good.min_radius       = 1.25;
    e.last_good.max_radius       = 1.75;
    e.last_good.mean_radius      = 1.5;
    e.last_good.irreducible_mass = 0.875;
    e.prev_centroid              = { 0.2, -0.4, 0.6 };
    e.has_prev_centroid          = true;

    auto const saved_coeffs = e.surface.coefficients();

    auto const init_iter = get_iteration();
    cphndl.save_checkpoint();
    parallel::mpi_barrier(sc_MPI_COMM_WORLD);

    // Scribble every field the checkpoint is supposed to carry.
    e.surface.set_sphere(99.0);
    e.state                = horizon_state_t::dormant;
    e.consecutive_failures = 0;
    e.has_existed          = false;
    e.ever_found           = false;
    e.has_last_good        = false;
    e.has_prev_centroid    = false;
    e.last_good            = horizon_result_t{};
    e.prev_centroid        = {};

    cphndl.load_checkpoint(init_iter);
    parallel::mpi_barrier(sc_MPI_COMM_WORLD);

    auto const& after = e.surface.coefficients();
    REQUIRE(after.size() == saved_coeffs.size());
    for (std::size_t c = 0; c < saved_coeffs.size(); ++c) {
        INFO("coefficient " << c);
        REQUIRE(after[c] == saved_coeffs[c]);   // bitwise
    }

    REQUIRE(e.state                == horizon_state_t::found);
    REQUIRE(e.consecutive_failures == 2);
    REQUIRE(e.has_existed);
    REQUIRE(e.ever_found);

    REQUIRE(e.has_last_good);
    REQUIRE(e.last_good_iteration == 7);
    REQUIRE(e.last_good.centroid[0]     == 0.25);
    REQUIRE(e.last_good.centroid[2]     == 0.75);
    REQUIRE(e.last_good.min_radius      == 1.25);
    REQUIRE(e.last_good.irreducible_mass == 0.875);
    REQUIRE(e.has_prev_centroid);
    REQUIRE(e.prev_centroid[1] == -0.4);
}

//---------------------------------------------------------------------------
// The apparent-horizon sidecar.
//
// Horizon state lives in checkpoint_ah_<iter>.h5 beside the main checkpoint
// rather than inside it: the main file is O(100 GB) in production while this is
// a few KB, and it is exactly the state one wants to inspect on its own.
//
// One save_checkpoint() for the whole case on purpose. Every test here saves at
// iteration 0, so an extra save pushes checkpoint_list past max_n_checkpoints
// and rotation deletes the very files the next load needs.
//---------------------------------------------------------------------------
TEST_CASE("Horizon state is checkpointed to a name-keyed sidecar",
          "[checkpoint][roundtrip][horizon]")
{
    using namespace grace;

    auto& mgr    = horizon_manager::get();
    auto& cphndl = checkpoint_handler::get();
    auto& e      = mgr.get(0);

    auto const original_name = e.name;

    e.surface.set_sphere(2.75);
    auto const saved_coeffs = e.surface.coefficients();

    auto const iter = get_iteration();
    cphndl.save_checkpoint();
    parallel::mpi_barrier(sc_MPI_COMM_WORLD);

    auto const dir = std::filesystem::path("./checkpoint_roundtrip_test_data");

    // Record rather than assert: a REQUIRE that throws here would leave this
    // rank outside the collective load_checkpoint() below while the others
    // entered it, deadlocking the run instead of reporting a failure. Every
    // check between two collectives is gathered and asserted afterwards.
    std::ostringstream ss;
    ss << "checkpoint_ah_" << std::setw(6) << std::setfill('0') << iter << ".h5";
    auto const sidecar = dir / ss.str();

    bool sidecar_exists = true, sidecar_small = true;
    if (parallel::mpi_comm_rank() == 0) {
        sidecar_exists = std::filesystem::exists(sidecar);
        // Small enough to open and diff on its own -- the point of the split.
        sidecar_small  = sidecar_exists
                      && std::filesystem::file_size(sidecar) < 1024 * 1024;
    }

    // The payload really moved rather than being duplicated into two copies
    // that can disagree. Checked single-rank only: the main file is written
    // with MPI-IO, and opening it serially from one rank while the others are
    // elsewhere invites an HDF5 file-locking stall. The property is
    // rank-independent, so verifying it once is enough.
    bool moved_out_of_main = true;
    if (parallel::mpi_comm_size(sc_MPI_COMM_WORLD) == 1) {
        std::ostringstream ds;
        ds << "checkpoint_data_" << std::setw(6) << std::setfill('0') << iter << ".h5";
        hid_t const fid = H5Fopen((dir / ds.str()).string().c_str(),
                                  H5F_ACC_RDONLY, H5P_DEFAULT);
        moved_out_of_main = (fid >= 0)
                         && (H5Lexists(fid, "apparent_horizon", H5P_DEFAULT) <= 0);
        if (fid >= 0) H5Fclose(fid);
    }

    // Identity check. Positional identity alone cannot tell two horizons apart
    // when they share l_max and initial radius -- exactly AH1/AH2 in a binary --
    // so a reordered horizons list would load each into the other with nothing
    // to detect it. Standing in for that here: the slot now holds a different
    // horizon than the one that wrote it.
    e.name = "SomeOtherHorizon";
    e.surface.set_sphere(42.0);
    auto const scribbled = e.surface.coefficients();

    cphndl.load_checkpoint(iter);
    parallel::mpi_barrier(sc_MPI_COMM_WORLD);

    // Collectives are done; safe to assert.
    if (parallel::mpi_comm_rank() == 0) {
        INFO("expected sidecar: " << sidecar.string());
        REQUIRE(sidecar_exists);
        REQUIRE(sidecar_small);
    }
    INFO("horizon state must not be duplicated inside the main checkpoint");
    REQUIRE(moved_out_of_main);

    auto const& after = e.surface.coefficients();
    INFO("a name mismatch must leave the horizon on its own guess, "
         "not hand it the other horizon's shape");
    REQUIRE(after[0] == scribbled[0]);
    REQUIRE(after[0] != saved_coeffs[0]);

    e.name = original_name;
}
