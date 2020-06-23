cmake_minimum_required(VERSION 2.8.12)
project(examples)
# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)
#find_package(Freetype REQUIRED)
#add_subdirectory(recogHrt)
add_subdirectory(/home/appltini/Desktop/faceDetectCpp/dlib-19.19_2/dlib-19.19/dlib dlib_build)

#   mkdir build
#   cd build
#   cmake ..
#   cmake --build . --config Release
#SET(CMAKE_BUILD_TYPE "Debug")  
#SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g2 -ggdb")  
#SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

if (DLIB_NO_GUI_SUPPORT)
   message("No GUI support, so we won't build the webcam_face_pose_ex example.")
else()
   find_package(OpenCV QUIET)
   if (OpenCV_FOUND)
      # include opencv for all targets
      include_directories(${OpenCV_INCLUDE_DIRS})
     
      add_executable(webcam_face_recog webcam_face_recog.cpp recogHrt/recog.cpp)

      target_link_libraries(webcam_face_recog dlib::dlib ${OpenCV_LIBS} )


      target_include_directories(webcam_face_recog PUBLIC "${PROJECT_BINARY_DIR}" "${PROJECT_SOURCE_DIR}/recogHrt")
   else()
      message("OpenCV not found, so we won't build the webcam_face_pose_ex example.")
   endif()
endif()





