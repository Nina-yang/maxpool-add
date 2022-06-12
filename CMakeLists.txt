cmake_minimum_required(VERSION 3.0)
project(max_pool_add)

OPTION(USE_OMP "Use OpenMP" OFF)
OPTION(USE_AVX "Use AVX" OFF)
OPTION(USE_BATCHOMP "Use batch omp" OFF)

IF(USE_BATCHOMP)
    ADD_DEFINITIONS(-DUSE_BATCHOMP)
ENDIF(USE_BATCHOMP)

IF(USE_OMP)
    ADD_DEFINITIONS(-DUSE_OMP)
ENDIF(USE_OMP)

IF(USE_AVX)
    ADD_DEFINITIONS(-DUSE_AVX)
ENDIF(USE_AVX)

IF(USE_OP_FUSION)
    ADD_DEFINITIONS(-DUSE_OP_FUSION)
ENDIF(USE_OP_FUSION)

add_executable(mpa_baseline main.cpp utils.cpp)

target_include_directories(mpa_baseline
    PUBLIC {PROJECT_SOURCE_DIR}
)

set (CMAKE_CXX_FLAGS "-fopenmp -mavx2")
