cmake_minimum_required(VERSION 2.8)
project(aur_odometry)

set(CMAKE_CXX_FLAGS "-O3 -std=c++11")

find_package(OpenCV REQUIRED)
file(GLOB OCVL "opencv/lib/*.so")
message("libs " ${OCVL})
message(${OpenCV_INCLUDE_DIRS})

include_directories(
	${OpenCV_INCLUDE_DIRS}
	#"opencv/include"
)

set(LIBS
	"opencv_aruco"
	"opencv_calib3d"
	"opencv_core"
	"opencv_flann"
	"opencv_highgui"
	"opencv_imgproc"
	"opencv_video"
	"opencv_videoio"
)

add_executable(aur_odometry
	src/main.cpp
	src/util.cpp
	src/AURTracker.cpp
	src/VideoSource.cpp
	src/CubeMesh.cpp
)

#target_link_libraries(aur_odometry ${LIBS})
target_link_libraries(aur_odometry ${OpenCV_LIBS})


install(TARGETS aur_odometry RUNTIME DESTINATION bin)
