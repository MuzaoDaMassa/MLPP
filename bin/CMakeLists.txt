cmake_minimum_required(VERSION 3.0)
project(MLPP)
project(Tests_MLPP)

# Find packages
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Add all source files in src directory
file(GLOB SOURCES "../src/*.cpp")
file(GLOB TEST_SOURCES "../tests/*.cpp")

# Create executable
add_executable(mlpp ${SOURCES})
add_executable(tests_mlpp ${TEST_SOURCES})

# Link libraries
target_link_libraries(mlpp ${OpenCV_LIBS})
target_link_libraries(tests_mlpp ${OpenCV_LIBS})