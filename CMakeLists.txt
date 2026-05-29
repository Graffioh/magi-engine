cmake_minimum_required(VERSION 3.20)

project(magi_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(magi_engine
  src/main.cpp
  src/layers.cpp
  src/mmap.cpp
  src/ops.cpp
  src/tensor.cpp
)

target_compile_options(magi_engine PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

add_executable(magi_tests
  tests/test_main.cpp
  tests/test_tensor.cpp
  tests/test_ops.cpp
  tests/test_mmap.cpp
  tests/test_layers.cpp
  tests/test_golden.cpp
  src/layers.cpp
  src/mmap.cpp
  src/ops.cpp
  src/tensor.cpp
)

target_include_directories(magi_tests PRIVATE src)

# Absolute path to the generated golden fixtures, so tests don't depend on the
# working directory. Blobs are gitignored; regenerate with tools/gen_golden_weights.py.
target_compile_definitions(magi_tests PRIVATE
  MAGI_GOLDEN_DIR="${CMAKE_SOURCE_DIR}/tests/golden"
)

target_compile_options(magi_tests PRIVATE
  -Wall
  -Wextra
  -Wpedantic
)

enable_testing()
add_test(NAME magi_tests COMMAND magi_tests)

# Convenience targets so you can do:
#   cmake --build build --target run    -> build and run the engine
#   cmake --build build --target check  -> build and run the test suite
# (`test` is reserved by CTest, hence `check`. You can also use `ctest --test-dir build`.)
add_custom_target(run
  COMMAND magi_engine
  DEPENDS magi_engine
  USES_TERMINAL
  COMMENT "Running magi_engine"
)

add_custom_target(check
  COMMAND magi_tests
  DEPENDS magi_tests
  USES_TERMINAL
  COMMENT "Running magi_tests"
)
