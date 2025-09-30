cmake_minimum_required(VERSION 3.10)

project(DeviceManager)

set(CMAKE_CXX_STANDARD 17)
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
include_directories(${CMAKE_CURRENT_LIST_DIR}/embedmediaStorage)
link_directories(${CMAKE_CURRENT_LIST_DIR}/lib)


add_executable(DeviceManager deviceManager.cpp TCPClientManager.cpp)

target_link_libraries(DeviceManager fmod fmodstudio pthread)