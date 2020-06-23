cmake_minimum_required(VERSION 2.8)
project( display )
find_package( OpenCV REQUIRED )
#set(CMAKE_CXX_STANDARD 11)
SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
add_compile_options(-fopenmp)
set(PROJECT_LINK_LIBS libdfacepro.so libMNN.so)
link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)
include_directories(${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/include/dface ${CMAKE_CURRENT_SOURCE_DIR}/include/spdlog ${CMAKE_CURRENT_SOURCE_DIR}/include/stb ${CMAKE_CURRENT_SOURCE_DIR}/include)
#
add_executable( display display.cpp )
#
target_link_libraries( display ${OpenCV_LIBS} ${PROJECT_LINK_LIBS} dl)
