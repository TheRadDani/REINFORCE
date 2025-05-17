cmake_minimum_required(VERSION 3.25)
project(REINFORCE_RL)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
elseif(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W4")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
endif()

set(SOURCES
    src/main.cpp
    src/policy_network.cpp
    src/grid_world.cpp
    src/reinforce.cpp
)

# Executable
add_executable(reinforce ${SOURCES})

# === Include Directories ===
target_include_directories(reinforce PRIVATE
    ${CMAKE_SOURCE_DIR}/include
)

# === Security Hardening ===
target_compile_options(reinforce PRIVATE
    -Wformat -Wformat-security
)

install(TARGETS reinforce DESTINATION bin)

option(BUILD_TESTS "Build tests" OFF)
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

message(STATUS "Building REINFORCE Reinforcement Learning Algorithm")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")