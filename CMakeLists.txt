cmake_minimum_required(VERSION 3.18)
project(TVMInferenceEngine CXX)

# Enforce Modern C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define optimized release configurations
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -Wall -Wextra")

# Locate TVM Runtime library (without requiring python in production)
# First search in TVM_HOME, then system paths, fallback to default directories
find_path(TVM_RUNTIME_INCLUDE_DIR
    NAMES tvm/runtime/object.h
    PATHS
        ${TVM_HOME}/include
        /usr/local/include
        /usr/include
)

find_library(TVM_RUNTIME_LIB
    NAMES tvm_runtime
    PATHS
        ${TVM_HOME}/build
        /usr/local/lib
        /usr/lib
)

if(NOT TVM_RUNTIME_INCLUDE_DIR)
    message(FATAL_ERROR "TVM runtime headers not found. Please set the TVM_HOME environment variable or install TVM runtime.")
endif()

if(NOT TVM_RUNTIME_LIB)
    message(FATAL_ERROR "libtvm_runtime not found. Please compile/install libtvm_runtime.so first.")
endif()

message(STATUS "Found TVM runtime include: ${TVM_RUNTIME_INCLUDE_DIR}")
message(STATUS "Found TVM runtime library: ${TVM_RUNTIME_LIB}")

# Create high-performance C++ executable
add_executable(tvm_inference_app main.cpp)

# Header inclusions
target_include_directories(tvm_inference_app PRIVATE
    ${TVM_RUNTIME_INCLUDE_DIR}
    ${TVM_RUNTIME_INCLUDE_DIR}/../3rdparty/dlpack/include
    ${TVM_RUNTIME_INCLUDE_DIR}/../3rdparty/dmlc-core/include
)

# Link lean libraries
target_link_libraries(tvm_inference_app PRIVATE
    ${TVM_RUNTIME_LIB}
    pthread
    dl
)

# Compile-time optimizations for multi-threading & backend execution
target_compile_definitions(tvm_inference_app PRIVATE DMLC_USE_LOGGING_LIBRARY=<dmlc/logging.h>)
