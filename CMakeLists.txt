cmake_minimum_required(VERSION 3.20)
project(NZD CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# For Release builds (-D CMAKE_BUILD_TYPE=Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native" CACHE STRING "Release C++ flags")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE "-flto" CACHE STRING "Release linker flags")

# For Debug builds (-D CMAKE_BUILD_TYPE=Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fno-omit-frame-pointer" CACHE STRING "Debug C++ flags")
set(CMAKE_EXE_LINKER_FLAGS_DEBUG "" CACHE STRING "Debug linker flags")

include_directories(include)

file(GLOB_RECURSE SOURCES "src/*.cpp")

list(FILTER SOURCES EXCLUDE REGEX ".*\\.disable$")

find_package(Threads REQUIRED)
find_package(OpenMP)

add_executable(NZD ${SOURCES})

target_compile_options(NZD PRIVATE -Wall)

# 나중에 pwd 동적으로 os맞게 shell에서 받아오게해서 상대경로 지정
target_compile_definitions(NZD PRIVATE
        PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}/"
)

target_link_libraries(NZD PRIVATE Threads::Threads)

if (OpenMP_CXX_FOUND)
    target_link_libraries(NZD PRIVATE OpenMP::OpenMP_CXX)
endif ()