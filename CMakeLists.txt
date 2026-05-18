cmake_minimum_required(VERSION 3.20)

project(magi_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(magi_engine
  src/main.cpp
  src/layers.cpp
  src/ops.cpp
  src/tensor.cpp
)

target_compile_options(magi_engine PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)
