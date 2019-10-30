CMAKE_MINIMUM_REQUIRED(VERSION 3.3)
project(hanalearn)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
    "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/rela)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/hanabi-learning-environment)

pybind11_add_module(hanalearn ${CMAKE_CURRENT_SOURCE_DIR}/cpp/pybind.cc)
target_link_libraries(hanalearn PUBLIC hanabi)
target_link_libraries(hanalearn PUBLIC _rela)
target_include_directories(hanalearn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
