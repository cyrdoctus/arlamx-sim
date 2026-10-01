cmake_minimum_required(VERSION 3.16)
project(arlamx_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-O3 -Wall -Wextra -Wno-unknown-pragmas)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# pybind11 from the active interpreter
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
  OUTPUT_VARIABLE pybind11_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
find_package(pybind11 CONFIG REQUIRED)

set(ARLAMX_SRC
  src/aero/sentman.cpp
  src/aero/cll.cpp
  src/srp/panel_srp.cpp
  src/orbit/frames.cpp
  src/orbit/gravity.cpp
  src/orbit/third_body.cpp
  src/attitude/mrp.cpp
  src/attitude/integrate.cpp
  src/control/mrp_feedback.cpp
  src/control/quat_feedback.cpp
  src/control/bdot.cpp
  src/control/magnetorquer.cpp
  src/mag/field.cpp
  src/onboard/propagator_f32.cpp
  src/api.cpp
  src/bindings.cpp
)

# Optional CSPICE
set(CSPICE_INCLUDE "" CACHE PATH "CSPICE include directory (SpiceUsr.h)")
set(CSPICE_LIB "" CACHE FILEPATH "CSPICE static library")
if(NOT CSPICE_INCLUDE)
  if(EXISTS "/home/nekolny/.conan2/p/b/cspic056cf88ad347a/b/src/include/SpiceUsr.h")
    set(CSPICE_INCLUDE "/home/nekolny/.conan2/p/b/cspic056cf88ad347a/b/src/include")
    set(CSPICE_LIB "/home/nekolny/.conan2/p/b/cspic056cf88ad347a/b/src/lib/cspice.a")
  endif()
endif()

pybind11_add_module(arlamx_cpp ${ARLAMX_SRC})
target_include_directories(arlamx_cpp PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

if(CSPICE_INCLUDE AND EXISTS "${CSPICE_INCLUDE}/SpiceUsr.h" AND EXISTS "${CSPICE_LIB}")
  target_compile_definitions(arlamx_cpp PRIVATE ARLAMX_HAVE_CSPICE)
  target_include_directories(arlamx_cpp PRIVATE "${CSPICE_INCLUDE}")
  target_link_libraries(arlamx_cpp PRIVATE "${CSPICE_LIB}")
  message(STATUS "ARLAMX: CSPICE enabled (${CSPICE_LIB})")
else()
  message(STATUS "ARLAMX: CSPICE not found — analytic Sun/Moon only")
endif()

set_target_properties(arlamx_cpp PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../python/arlamx_v2"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../python/arlamx_v2"
)
