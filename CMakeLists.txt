# Minimal version of CMake
cmake_minimum_required (VERSION 3.2)

# Define project name
project (ocl)

# Tell CMake to compile with C++14
set(CMAKE_CXX_STANDARD 14)

# Activate optimisation when in release mode
# Currently CMake automatically uses -o2 option in CXX flags
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -o3 -w")

find_package( OpenCL REQUIRED )
if (NOT OPENCL_FOUND)
	message(FATAL_ERROR, "OpenCL not found.")
endif()

find_package( OpenCV REQUIRED core opencv_imgcodecs)
if (NOT OPENCV_FOUND)
	message(FATAL_ERROR, "OpenCV not found.")
endif()

include_directories(${OpenCL_INCLUDE_DIRS} ${OPENCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include)
link_directories(${OpenCL_LIBRARY} ${OpenCV_LIB_DIR})

# Source code of application
set (VECADD_SRC 
		src/vecadd.cpp
		include/benchmark.hpp
)

set (IMBLUR_SRC
		src/imblur.cpp
		include/benchmark.hpp
)

# Set up executable
add_executable (ocl_vecadd ${VECADD_SRC})
target_link_libraries(ocl_vecadd ${OpenCL_LIBRARY})

add_executable(ocl_imblur ${IMBLUR_SRC})
target_link_libraries(ocl_imblur ${OpenCL_LIBRARY} ${OpenCV_LIBS})