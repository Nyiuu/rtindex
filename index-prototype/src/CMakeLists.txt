cmake_minimum_required(VERSION 3.18)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_C_COMPILER "gcc-12")
#set(CMAKE_CXX_COMPILER "g++-12")

project(index_prototype C CXX CUDA)

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

include(cmake/configure_optix.cmake)

include(${CMAKE_CURRENT_SOURCE_DIR}/../ext/nvtx/nvtxImportedTargets.cmake)

include_directories(${OptiX_INCLUDE})
include_directories(../btree)

cuda_compile_and_embed(embedded_ptx_code device_code.cu)

add_executable(index_prototype
        ${embedded_ptx_code}
        main.cu
        benchmarks.cuh
        cuda_buffer.cuh
        definitions.h
        impl_binsearch.cuh
        impl_hashtable.cuh
        impl_rtx_index.cuh
        impl_tree.cuh
        impl_scan.cuh
        input_generation.h
        launch_parameters.cuh
        optix_helpers.cuh
        optix_pipeline.cu
        optix_pipeline.h
        optix_wrapper.cu
        optix_wrapper.h
        result_collector.h
        utilities.h
        )

set_target_properties(index_prototype PROPERTIES CUDA_ARCHITECTURES "75;80;86;89")
# 75 TURING
# 80 AMPERE
# 86 AMPERE
# 89 ADA

target_compile_options(index_prototype PRIVATE $<$<COMPILE_LANGUAGE:CXX>:
        # profiling
        #-g -shared-libgcc -fno-omit-frame-pointer
        -Wall -Wextra -Wpedantic -Werror
        ${OpenMP_CXX_FLAGS}
        >)

target_compile_options(index_prototype PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        # profiling
        #--debug --profile --generate-line-info
        #-funwind-tables -fno-omit-frame-pointer
        --extended-lambda --use_fast_math --expt-relaxed-constexpr
        -O3
        -Xcompiler=-fopenmp
#        -Xcompiler=-Wall -Xcompiler=-Wextra -Xcompiler=-Wpedantic
        >)

target_link_libraries(index_prototype
        ${CUDA_LIBRARIES}
        ${CUDA_CUDA_LIBRARY}
        OpenMP::OpenMP_CXX
        # profiling
        nvtx3-cpp
        )
