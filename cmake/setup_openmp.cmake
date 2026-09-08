# OpenMP runtime resolution.
#
# Must be included BEFORE the first find_package(OpenMP), because CMake caches
# the result and later attempts to steer it are silently ignored.
#
# macOS: pin the OpenMP runtime to the one Kokkos links.
#
# Homebrew's Kokkos links the keg-only libomp
# (/opt/homebrew/opt/libomp/lib/libomp.dylib), while Homebrew LLVM's clang
# resolves a bare `-fopenmp` against its own /opt/homebrew/opt/llvm/lib/
# libomp.dylib.  The two dylibs carry different install_names, so both are
# loaded into the process, and the second runtime to initialise crashes inside
# __kmp_suspend_* as soon as a parallel region joins.  `-Xpreprocessor -fopenmp`
# suppresses clang's implicit -lomp so only the explicitly-named runtime links.
#
# Does nothing when the caller has already pinned OpenMP, when building the
# bundled dependencies (which use one consistent toolchain), or off macOS.
if(APPLE AND NOT GRACE_USE_BUNDLED_DEPS AND NOT DEFINED OpenMP_omp_LIBRARY)
    find_library(_grace_keg_libomp NAMES omp
        HINTS "$ENV{LIBOMP_ROOT}/lib"
              "/opt/homebrew/opt/libomp/lib"
              "/usr/local/opt/libomp/lib"
        NO_DEFAULT_PATH)
    if(_grace_keg_libomp)
        get_filename_component(_grace_libomp_prefix "${_grace_keg_libomp}" DIRECTORY)
        get_filename_component(_grace_libomp_prefix "${_grace_libomp_prefix}" DIRECTORY)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I${_grace_libomp_prefix}/include"
            CACHE STRING "OpenMP compile flags (pinned to match Kokkos)")
        set(OpenMP_C_FLAGS   "-Xpreprocessor -fopenmp -I${_grace_libomp_prefix}/include"
            CACHE STRING "OpenMP compile flags (pinned to match Kokkos)")
        set(OpenMP_CXX_LIB_NAMES omp CACHE STRING "OpenMP runtime library name")
        set(OpenMP_C_LIB_NAMES   omp CACHE STRING "OpenMP runtime library name")
        set(OpenMP_omp_LIBRARY "${_grace_keg_libomp}"
            CACHE FILEPATH "OpenMP runtime pinned to match Kokkos")
        message(STATUS "macOS: pinning OpenMP runtime to ${_grace_keg_libomp}")
    else()
        message(WARNING
            "macOS: no keg-only libomp found to pin against.  If the build "
            "crashes inside libomp, verify that GRACE and Kokkos link the same "
            "libomp.dylib (otool -L).")
    endif()
endif()
