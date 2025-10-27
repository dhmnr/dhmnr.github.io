cmake_minimum_required(VERSION 3.18)
project(CUDAVectorAdd LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Add CUDA flags for better profiling with NCU
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/kernels)

# Add the executable
add_executable(vector_add
    main.cu
    kernels/vector_add.cu
)

# Link CUDA libraries
target_link_libraries(vector_add CUDA::cudart)

# Set CUDA properties
set_target_properties(vector_add PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

# Print configuration info
message(STATUS "CUDA Toolkit Root: ${CUDAToolkit_ROOT}")
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

