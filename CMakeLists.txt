cmake_minimum_required(VERSION 3.18)

# Project definition
project(matri_mult LANGUAGES CXX CUDA)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Enable experimental C++ modules (if needed)
set(CMAKE_EXPERIMENTAL_CXX_MODULE_CMAKE_API ON)

# Set CUDA architectures (adjust based on your GPU)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring for DEBUG build")
    # Enable debug symbols and disable optimizations
    add_compile_definitions(DEBUG _DEBUG)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring for RELEASE build")
    # Enable optimizations and disable debug info
    add_compile_definitions(NDEBUG RELEASE)
elseif(CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    message(STATUS "Configuring for RELEASE WITH DEBUG INFO build")
    add_compile_definitions(NDEBUG RELEASE)
endif()

# Print configuration info
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Configure vcpkg (make path more flexible)
if(DEFINED ENV{VCPKG_ROOT})
    set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake")
    message(STATUS "Using vcpkg from: $ENV{VCPKG_ROOT}")
else()
    # Fallback path - adjust as needed
    set(CMAKE_TOOLCHAIN_FILE "C:/vcpkg/scripts/buildsystems/vcpkg.cmake")
    message(WARNING "VCPKG_ROOT not set, using default path")
endif()

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc)

# Print found package info
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "CUDA version: ${CUDAToolkit_VERSION}")

# Define source files
set(SOURCES
    src/matri_mult.cu
    # Add CUDA files here
    # src/matri_mult_kernels.cu
)

set(HEADERS
    # Add CUDA headers here
    # include/matri_mult_kernels.cuh
)

# Create executable
add_executable(matri_mult ${SOURCES} ${HEADERS})

# Set target properties
set_target_properties(matri_mult PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES}
    # Use dynamic runtime for better compatibility with vcpkg
    MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL"
)

# Include directories
target_include_directories(matri_mult PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)



# Compiler-specific options
target_compile_options(matri_mult PRIVATE
    # MSVC C++ options
    $<$<AND:$<COMPILE_LANGUAGE:CXX>,$<CXX_COMPILER_ID:MSVC>>:
        /W4          # Warning level 4
        /permissive- # Strict C++ conformance
        
        # Debug-specific flags
        $<$<CONFIG:Debug>:
            /Od      # Disable optimizations
            /Zi      # Generate debug info
            /RTC1    # Runtime checks
            /MDd     # Debug runtime library
        >
        
        # Release-specific flags  
        $<$<CONFIG:Release>:
            /O2      # Maximize speed
            /Ob2     # Inline expansion
            /DNDEBUG # No debug
            /MD      # Release runtime library
        >
        
        # RelWithDebInfo-specific flags
        $<$<CONFIG:RelWithDebInfo>:
            /O2      # Optimize
            /Zi      # Debug info
            /DNDEBUG # No debug asserts
            /MD      # Release runtime library
        >
        
        # MinSizeRel-specific flags
        $<$<CONFIG:MinSizeRel>:
            /O1      # Minimize size
            /DNDEBUG # No debug
            /MD      # Release runtime library
        >
    >
    
    # CUDA compiler options
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math              # Fast math optimizations
        --extended-lambda            # Extended lambda support
        --expt-relaxed-constexpr     # Relaxed constexpr
        -Xcompiler=/W3               # Pass warning level to host compiler
        
        # Debug-specific CUDA flags
        $<$<CONFIG:Debug>:
            -G       # Generate debug info for device code
            -g       # Generate debug info for host code
            -O0      # No optimization
            --ptxas-options=-v  # Verbose ptxas output
        >
        
        # Release-specific CUDA flags
        $<$<CONFIG:Release>:
            -O3      # Maximum optimization
            --use_fast_math
            -DNDEBUG # No debug
        >
        
        # RelWithDebInfo-specific CUDA flags
        $<$<CONFIG:RelWithDebInfo>:
            -O3      # Optimize
            -g       # Debug info
            --lineinfo # Line info for profiling
        >
    >
)

target_link_options(matri_mult PRIVATE
    # Debug linking options
    $<$<CONFIG:Debug>:
        $<$<CXX_COMPILER_ID:MSVC>:/DEBUG:FULL>
    >
    
    # Release linking options  
    $<$<CONFIG:Release>:
        $<$<CXX_COMPILER_ID:MSVC>:/INCREMENTAL:NO /OPT:REF /OPT:ICF>
    >
)

# Link libraries
target_link_libraries(matri_mult PRIVATE
    # OpenCV libraries
    ${OpenCV_LIBS}
    
    # CUDA libraries (add as needed)
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::cublas
    # CUDA::curand      # If you need random numbers
    # CUDA::cufft       # If you need FFT
)

# Optional: Add custom commands for better development experience
if(WIN32)
    # Copy OpenCV DLLs to output directory (if using dynamic linking)
    add_custom_command(TARGET matri_mult POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:opencv_core>
        $<TARGET_FILE_DIR:matri_mult>
    )
    add_custom_command(TARGET matri_mult POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:opencv_highgui>
        $<TARGET_FILE_DIR:matri_mult>
    )
endif()

add_custom_command(TARGET matri_mult POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
    "${CMAKE_CURRENT_SOURCE_DIR}/data"
    "$<TARGET_FILE_DIR:matri_mult>/data"
    COMMENT "Copying data folder..."
)

# Optional: Installation rules
install(TARGETS matri_mult
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)