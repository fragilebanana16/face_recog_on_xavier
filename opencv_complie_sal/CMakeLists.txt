cmake_minimum_required(VERSION 2.8)


project( display )
set( OpenCV_DIR "/home/appltini/opencv_3.3.1/opencv-3.3.1/build")
find_package(OpenCV 3.3 REQUIRED)

include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( display display.cpp )
target_link_libraries( display ${OpenCV_LIBS} )
