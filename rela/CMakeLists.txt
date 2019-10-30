cmake_minimum_required(VERSION 3.0 FATAL_ERROR)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -fPIC -march=native -Wfatal-errors")


# get and append paths for finding dep
execute_process(
  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
  OUTPUT_VARIABLE TorchPath
)
list(APPEND CMAKE_PREFIX_PATH ${TorchPath})


# find packages & third_party
find_package(PythonInterp 3.7 REQUIRED)
find_package(PythonLibs 3.7 REQUIRED)
find_package(Torch REQUIRED)


# message(${CMAKE_CURRENT_SOURCE_DIR}/../)
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11 third_party/pybind11
)


# lib for other c++ programs
add_library(_rela
  types.cc
)
target_include_directories(_rela PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_include_directories(_rela PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(_rela PUBLIC ${PYTHON_INCLUDE_DIRS})
target_link_libraries(_rela PUBLIC ${TORCH_LIBRARIES})


# python lib
pybind11_add_module(rela pybind.cc)
target_link_libraries(rela PUBLIC _rela)
