cmake_minimum_required(VERSION 2.8)
project( myApp )
find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )
add_executable( myApp MyApplication.cpp Utilities.cpp)
target_link_libraries( myApp ${OpenCV_LIBS} stdc++fs)


