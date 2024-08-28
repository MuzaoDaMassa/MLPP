cmake_minimum_required(VERSION 3.15)
project(MLPP LANGUAGES CXX)
project(Tests_MLPP LANGUAGES CXX)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add all source files in src directory
file(GLOB CPP_SOURCES "../src/*.cpp")

# Create executable
add_executable(mlpp ${CPP_SOURCES})
file(GLOB TEST_SOURCES "../tests/*.cpp")
add_executable(tests_mlpp ${TEST_SOURCES})

# Specify required C++ features for your target
target_compile_features(mlpp PRIVATE cxx_std_17)
target_compile_features(tests_mlpp PRIVATE cxx_std_17)

# Link libraries
target_link_libraries(mlpp ${OpenCV_LIBS})
target_link_libraries(tests_mlpp ${OpenCV_LIBS})
