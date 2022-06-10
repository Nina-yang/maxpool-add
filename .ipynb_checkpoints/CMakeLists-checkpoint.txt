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


set (CMAKE_CXX_FLAGS "-fopenmp -mavx2")


IF(USE_OMP AND USE_AVX)
    add_executable(mpa_omp_avx main.cpp utils.cpp)

    target_include_directories(mpa_omp_avx
        PUBLIC {PROJECT_SOURCE_DIR}
    )
ENDIF(USE_OMP AND USE_AVX)

IF(NOT USE_OMP AND NOT USE_AVX)
    add_executable(mpa main.cpp utils.cpp)

    target_include_directories(mpa
        PUBLIC {PROJECT_SOURCE_DIR}
    )
ENDIF(NOT USE_OMP AND NOT USE_AVX)

IF(USE_OMP AND NOT USE_AVX)
    add_executable(mpa_omp main.cpp utils.cpp)

    target_include_directories(mpa_omp
        PUBLIC {PROJECT_SOURCE_DIR}
    )
ENDIF(USE_OMP AND NOT USE_AVX)

IF(NOT USE_OMP AND USE_AVX)
    add_executable(mpa_avx main.cpp utils.cpp)

    target_include_directories(mpa_avx
        PUBLIC {PROJECT_SOURCE_DIR}
    )
ENDIF(NOT USE_OMP AND USE_AVX)

IF(USE_BATCHOMP)
    add_executable(mpa_batch main.cpp utils.cpp)

    target_include_directories(mpa_batch
        PUBLIC {PROJECT_SOURCE_DIR}
    )
ENDIF(USE_BATCHOMP)