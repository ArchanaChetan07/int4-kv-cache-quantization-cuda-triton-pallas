cmake_minimum_required(VERSION 3.18)
project(flash_decode_int4 LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Optional: only build CUDA if FLASH_DECODE_FORCE_CUDA env var is set
if(NOT DEFINED ENV{FLASH_DECODE_FORCE_CUDA})
  message(STATUS "CUDA build disabled (set FLASH_DECODE_FORCE_CUDA to enable)")
  return()
endif()

# CUDA architecture targets: T1000/Turing (75), A100 (80), A10/3090 (86),
# L40S/4090 (89), H100 (90)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Find PyTorch
find_package(Torch REQUIRED)
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${TORCH_INCLUDE_DIRS})

# CUDA sources
file(GLOB CUDA_SOURCES "csrc/*.cu")
file(GLOB CPP_SOURCES "csrc/*.cpp")

# Compile CUDA kernels
add_library(flash_decode_int4 SHARED ${CUDA_SOURCES} ${CPP_SOURCES})

# Link libraries
target_link_libraries(flash_decode_int4 ${TORCH_LIBRARIES})

# Output directory
set_target_properties(flash_decode_int4 PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

message(STATUS "Flash Decode INT4 CUDA build configured")
message(STATUS "  Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  CUDA sources: ${CUDA_SOURCES}")
message(STATUS "  C++ sources: ${CPP_SOURCES}")
