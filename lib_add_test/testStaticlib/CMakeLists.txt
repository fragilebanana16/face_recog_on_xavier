cmake_minimum_required(VERSION 3.10)

# set the project name and version
project(So VERSION 1.0)

include_directories(include)

file(GLOB SOURCES "src/*.cpp")
# a testSo named so
add_library(testSo STATIC ${SOURCES})

# specify the install dir
install(TARGETS testSo DESTINATION /home/appltini/Desktop/faceDetectCpp/test/testSharelib/install)


