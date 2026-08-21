cmake_minimum_required(VERSION 3.18)
project(bayesfilter_symmetric_sylvester LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

foreach(QUERY IN ITEMS get_include get_lib get_compile_flags get_link_flags)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c
            "import tensorflow as tf; r = tf.sysconfig.${QUERY}(); print(r if isinstance(r, str) else ' '.join(r))"
        OUTPUT_VARIABLE TF_${QUERY}
        OUTPUT_STRIP_TRAILING_WHITESPACE
        COMMAND_ERROR_IS_FATAL ANY)
endforeach()

cmake_path(GET Python3_EXECUTABLE PARENT_PATH _py_bin_dir)
cmake_path(GET _py_bin_dir PARENT_PATH PY_ENV_PREFIX)

set(HAS_CUDA OFF)
set(CUDA_CONDA_INCLUDE "")
set(CUDA_CONDA_LIB "")
set(PYTHON_SITE_SUFFIX
    "python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages")
set(CUDA_PYTHON_INCLUDE
    "${PY_ENV_PREFIX}/lib/${PYTHON_SITE_SUFFIX}/nvidia/cuda_runtime/include")
set(CUDA_PYTHON_LIB
    "${PY_ENV_PREFIX}/lib/${PYTHON_SITE_SUFFIX}/nvidia/cuda_runtime/lib")
if(EXISTS "${PY_ENV_PREFIX}/targets/x86_64-linux/include/cuda_runtime_api.h")
    set(CUDA_CONDA_INCLUDE "${PY_ENV_PREFIX}/targets/x86_64-linux/include")
    set(CUDA_CONDA_LIB "${PY_ENV_PREFIX}/targets/x86_64-linux/lib")
elseif(EXISTS "${PY_ENV_PREFIX}/include/cuda_runtime_api.h")
    set(CUDA_CONDA_INCLUDE "${PY_ENV_PREFIX}/include")
    set(CUDA_CONDA_LIB "${PY_ENV_PREFIX}/lib")
endif()
set(TF_CUDA_INCLUDE "${TF_get_include}/third_party/gpus/cuda/include")
find_path(CUDA_INCLUDE NAMES cuda_runtime_api.h
    PATHS "${CUDA_CONDA_INCLUDE}" "${CUDA_PYTHON_INCLUDE}" "${TF_CUDA_INCLUDE}" "/usr/local/cuda/include"
    NO_DEFAULT_PATH)
find_library(CUDART_LIB NAMES cudart
    PATHS "${CUDA_CONDA_LIB}" "${CUDA_PYTHON_LIB}" "${PY_ENV_PREFIX}/lib" "/usr/local/cuda/lib64"
    NO_DEFAULT_PATH)
if(NOT CUDART_LIB)
    foreach(_dir "${CUDA_CONDA_LIB}" "${CUDA_PYTHON_LIB}" "${PY_ENV_PREFIX}/lib" "/usr/local/cuda/lib64")
        if(EXISTS "${_dir}/libcudart.so.12")
            set(CUDART_LIB "${_dir}/libcudart.so.12" CACHE FILEPATH "cudart versioned" FORCE)
            break()
        endif()
    endforeach()
endif()
if(CUDA_INCLUDE AND CUDART_LIB)
    set(HAS_CUDA ON)
    message(STATUS "CUDA found: include=${CUDA_INCLUDE} lib=${CUDART_LIB}")
else()
    message(STATUS "CUDA not found; SymmetricSylvester GPU XLA target disabled")
endif()

set(EXTRA_INCLUDES "")
set(CONDA_INCLUDE "${PY_ENV_PREFIX}/include")
if(EXISTS "${CONDA_INCLUDE}/absl/container/flat_hash_map.h")
    list(APPEND EXTRA_INCLUDES "${CONDA_INCLUDE}")
endif()
set(TF_LLVM_PROJECT_ROOT "${TF_get_include}/external/llvm-project")
set(TF_MLIR_INCLUDE "${TF_LLVM_PROJECT_ROOT}/mlir/include")
if(EXISTS "${TF_MLIR_INCLUDE}/mlir/IR/MLIRContext.h")
    list(APPEND EXTRA_INCLUDES "${TF_MLIR_INCLUDE}")
endif()
find_path(LLVM_INCLUDE_DIR
    NAMES llvm/ADT/ArrayRef.h
    PATHS
        "${CONDA_INCLUDE}"
        "${TF_LLVM_PROJECT_ROOT}/llvm/include"
        "/usr/include"
        "/usr/lib/llvm-19/include"
        "/usr/lib/llvm-18/include"
        "/usr/lib/llvm-17/include"
        "/usr/lib/llvm-16/include")
if(LLVM_INCLUDE_DIR)
    list(APPEND EXTRA_INCLUDES "${LLVM_INCLUDE_DIR}")
else()
    message(STATUS
        "Standalone LLVM headers not found; using TensorFlow's packaged "
        "include tree and allowing the compiler to enforce actual dependencies")
endif()
if(EXISTS "${TF_get_include}/external/highwayhash/highwayhash/arch_specific.h")
    list(APPEND EXTRA_INCLUDES "${TF_get_include}/external/highwayhash")
endif()
if(EXISTS "${TF_get_include}/external/farmhash_archive/src/farmhash.h")
    list(APPEND EXTRA_INCLUDES "${TF_get_include}/external/farmhash_archive/src")
endif()

add_library(
    symmetric_sylvester
    SHARED
    bayesfilter/ops/symmetric_sylvester_op.cc
    custom_call_status_stub.cc)
set_target_properties(
    symmetric_sylvester
    PROPERTIES
    PREFIX ""
    OUTPUT_NAME "_symmetric_sylvester_ops"
    SUFFIX ".so"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bayesfilter/ops"
    INSTALL_RPATH "${TF_get_lib};${PY_ENV_PREFIX}/lib")

target_include_directories(
    symmetric_sylvester
    PRIVATE
    ${TF_get_include}
    ${Python3_INCLUDE_DIRS}
    ${EXTRA_INCLUDES})
if(HAS_CUDA)
    target_include_directories(symmetric_sylvester PRIVATE ${CUDA_INCLUDE})
    target_compile_definitions(symmetric_sylvester PRIVATE HAS_CUDA)
endif()

set_target_properties(
    symmetric_sylvester
    PROPERTIES
    COMPILE_FLAGS "${TF_get_compile_flags}")

target_link_options(
    symmetric_sylvester
    PRIVATE
    -static-libstdc++
    -static-libgcc
    "LINKER:--export-dynamic")

target_link_directories(symmetric_sylvester PRIVATE ${TF_get_lib})
target_link_libraries(
    symmetric_sylvester
    PRIVATE
    ${Python3_LIBRARIES}
    "${TF_get_lib}/libtensorflow_cc.so.2"
    "${TF_get_lib}/libtensorflow_framework.so.2")
if(HAS_CUDA)
    target_link_libraries(symmetric_sylvester PRIVATE "${CUDART_LIB}")
endif()
