cmake_minimum_required(VERSION 3.10)
project(fluid)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(lib/flgl)

include_directories(${PROJECT_SOURCE_DIR}/lib/flgl/inc)
include_directories(${PROJECT_SOURCE_DIR}/lib/fftw/inc)
include_directories(${PROJECT_SOURCE_DIR}/lib/sw)

file(GLOB_RECURSE PROJECT_SOURCES "lib/sw/*.cpp" "src/*.cpp")

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES})

target_link_libraries(${PROJECT_NAME} FLGL)

if (WIN32)
  set(FFTW_NAME fftw3w)
else()
  set(FFTW_NAME fftw3f)
endif()

find_library(FFTW
             NAMES ${FFTW_NAME}
             HINTS /usr/local/lib /opt/homebrew/lib lib/fftw/bin 
             )

if(FFTW)
    message("FFTW found ${FFTW}")
    target_link_libraries(${PROJECT_NAME} ${FFTW})
else()
    message(FATAL_ERROR "FFTW library not found")
endif()
