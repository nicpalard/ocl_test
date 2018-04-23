# Minimal version of CMake
cmake_minimum_required (VERSION 3.2)

# Define project name
project (OpenCL_Example)

# Tell CMake to compile with C++14
set(CMAKE_CXX_STANDARD 14)

# Activate optimisation when in release mode
# Currently CMake automatically uses -o2 option in CXX flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -o3 -w")

find_package( OpenCL REQUIRED )
if (NOT OPENCL_FOUND)
	message(FATAL_ERROR, "OpenCL not found.")
endif()

include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})

# Source code of application
set (SOURCES 
		main.cpp
)

# Set up executable
add_executable (${PROJECT_NAME} ${SOURCES})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARY})