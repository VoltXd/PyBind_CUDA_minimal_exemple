cmake_minimum_required(VERSION 3.9) 

project(TEST_PYBIND_CUDA LANGUAGES CXX CUDA)

# Source & Header location
file(GLOB_RECURSE SOURCE_FILES src/*.cpp)
file(GLOB_RECURSE HEADER_FILES include/*.hpp)
file(GLOB_RECURSE CUDA_SOURCE_FILES src/*.cu)
file(GLOB_RECURSE CUDA_HEADER_FILES include/*.cuh)

add_subdirectory(libraries/pybind11)

pybind11_add_module(${PROJECT_NAME} ${SOURCE_FILES} ${HEADER_FILES} ${CUDA_SOURCE_FILES} ${CUDA_HEADER_FILES})

# Set the directory where CMakeLists.txt is to be the working directory
set_property(TARGET ${PROJECT_NAME} PROPERTY VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")

set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES 60)

# Allow separate .cu files compilation (allows you to write libraries/toolbox)
set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Target the include folder (used to make headers visible in visual studio)
target_include_directories(${PROJECT_NAME} PRIVATE include)

# Enable C++17 (Used for std::filesystem)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_17)

