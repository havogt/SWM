cmake_minimum_required(VERSION 3.16.2)

project(swm_gritools LANGUAGES CXX)# CUDA)

include(FetchContent)
FetchContent_Declare(
    gridtools
    GIT_REPOSITORY https://github.com/GridTools/gridtools.git
    GIT_TAG v2.3.2
)
FetchContent_MakeAvailable(gridtools)

add_executable(swm_gridtools swm_gridtools.cpp)
target_link_libraries(swm_gridtools GridTools::stencil_cpu_ifirst)
target_compile_features(swm_gridtools PRIVATE cxx_std_20)

add_executable(swm_gridtools_lr_halo swm_gridtools_lr_halo.cpp)
target_link_libraries(swm_gridtools_lr_halo GridTools::stencil_cpu_ifirst)
target_compile_features(swm_gridtools_lr_halo PRIVATE cxx_std_20)
