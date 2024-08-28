cmake_minimum_required(VERSION 3.15)
project(MLPP LANGUAGES CXX)
project(Tests_MLPP LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Check for CUDA
include(CheckLanguage)
check_language(CUDA)

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CUDA_AVAILABLE TRUE)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    message(STATUS "CUDA is available. Adding CUDA support.")

    # Set CUDA architectures
    set(CMAKE_CUDA_ARCHITECTURES 75) # Adjust this to match your GPU architecture
    # You can specify multiple architectures like this:
    # set(CMAKE_CUDA_ARCHITECTURES 75 86)
else()
    set(CUDA_AVAILABLE FALSE)
    message(STATUS "CUDA is not available. Building without CUDA support.")
endif()

# Add all source files in src directory
file(GLOB CPP_SOURCES "../src/*.cpp")
if(CUDA_AVAILABLE)
    file(GLOB CUDA_SOURCES "../src/*.cu")
endif()

# Create executable
add_executable(mlpp ${CPP_SOURCES})

# Add CUDA sources as a separate library
if(CUDA_AVAILABLE)
    add_library(cuda_lib STATIC ${CUDA_SOURCES})
    set_target_properties(cuda_lib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(mlpp cuda_lib)
endif()

file(GLOB TEST_SOURCES "../tests/*.cpp")
add_executable(tests_mlpp ${TEST_SOURCES})

# Specify required C++ features for your target
target_compile_features(mlpp PRIVATE cxx_std_17)
target_compile_features(tests_mlpp PRIVATE cxx_std_17)

# Link libraries
target_link_libraries(mlpp ${OpenCV_LIBS})
target_link_libraries(tests_mlpp ${OpenCV_LIBS})

if(CUDA_AVAILABLE)
    target_compile_definitions(mlpp PRIVATE CUDA_AVAILABLE)
    target_compile_definitions(tests_mlpp PRIVATE CUDA_AVAILABLE)
endif()