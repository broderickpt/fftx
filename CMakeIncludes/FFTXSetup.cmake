##
## Copyright (c) 2018-2021, Carnegie Mellon University
## All rights reserved.
##
## See LICENSE file for full information
##

##  Define variables and items required for building FFTX.  This module is intended to be
##  included by the top level CMake to build the entire project, but also may be included
##  at a lower level to support generating multiple sizes/sample builds within a single
##  example in the population of FFTX/examples

## specify this only here, lower levels use CMAKE_MINIMUM_REQUIRED_VERSION that this sets
cmake_minimum_required (VERSION 3.14)

##  Start by finding things -- the list varies by what we're building for...
##  Get SPIRAL home...

if ( DEFINED ENV{SPIRAL_HOME} )
    message ( STATUS "SPIRAL_HOME = $ENV{SPIRAL_HOME}" )
    set ( SPIRAL_SOURCE_DIR $ENV{SPIRAL_HOME} )
else ()
    if ( "x${SPIRAL_HOME}" STREQUAL "x" )
        message ( FATAL_ERROR "SPIRAL_HOME environment variable undefined and not specified on command line" )
    endif ()
    set ( SPIRAL_SOURCE_DIR ${SPIRAL_HOME} )
endif ()

##  Find python3 -- used to marshall/run examples

find_package (Python3 COMPONENTS Interpreter)
if (${Python3_FOUND})
    ##  It exists, executable is ${Python3_EXECUTABLE}
    message ( STATUS "Found Python3: Version = ${Python3_VERSION}, Executable = ${Python3_EXECUTABLE}")
else ()
    message ( SEND_ERROR "Python3 NOT FOUND: Python is required to build/run examples")
endif ()

##  Define paths and include other CMake functions needed for building

set ( SPIRAL_INCLUDE_PATH ${SPIRAL_SOURCE_DIR}/config/CMakeIncludes )
include ("${SPIRAL_INCLUDE_PATH}/RunSpiral.cmake")

set ( FFTX_CMAKE_INCLUDE_DIR ${FFTX_PROJECT_SOURCE_DIR}/CMakeIncludes )
set ( BACKEND_SOURCE_DIR ${FFTX_PROJECT_SOURCE_DIR}/examples/backend )

include ( "${FFTX_CMAKE_INCLUDE_DIR}/FFTXCmakeFunctions.cmake" )

##  Get hip/rocm stuff if _codegen == HIP

if ( ${_codegen} STREQUAL "HIP" )
    ##  Setup what we need to build for HIP/ROCm
    list ( APPEND CMAKE_PREFIX_PATH /opt/rocm/hip /opt/rocm )
    find_package ( hip REQUIRED )
    if ( ${hip_FOUND} )
	##  HIP/ROCm support found
	message ( STATUS "Found HIP: Version = ${hip_VERSION}" )
    else ()
	message ( SEND_ERROR "HIP NOT FOUND: HIP is required to build")
    endif ()

    # ##  Set the compiler/linker
    ##  Specify these on command line -- if done here cmake re-evaluates (reruns) and loses the _codegen value
    # if ( NOT WIN32 )
    # 	set (  CMAKE_CXX_COMPILER ${HIP_HIPCC_EXECUTABLE} )
    # 	set ( CMAKE_CXX_LINKER   ${HIP_HIPCC_EXECUTABLE} )
    # endif ()

    ##  Adjust include and library directories
    ##  Need to add $ROCM_PATH for includes and libraries
    if ( DEFINED ENV{ROCM_PATH} )
	message ( STATUS "ROCM_PATH is defined: $ENV{ROCM_PATH}" )
	include_directories ( $ENV{ROCM_PATH}/hipfft/include $ENV{ROCM_PATH}/include )
    endif ()
    list ( APPEND LIBS_FOR_HIP hipfft rocfft )
    list ( APPEND ADDL_COMPILE_FLAGS -DFFTX_HIP )
endif ()

##  Set flags and options for use when building code
if ( WIN32 )
    list ( APPEND ADDL_COMPILE_FLAGS -D_USE_MATH_DEFINES )
endif ()

##  relocatable code doesn't work if multiple spiral files are included (name collisions)
##  Default setting is false; only running on 64 bit machines.

if ( ${_codegen} STREQUAL "GPU" )
    if (WIN32)
	##  set ( CUDA_COMPILE_FLAGS -rdc=false )
	set ( GPU_COMPILE_DEFNS )			## -Xptxas -v
	set ( LIBS_FOR_CUDA cufft )
	list ( APPEND ADDL_COMPILE_FLAGS -DWIN64 )
    else ()
	##  set ( CUDA_COMPILE_FLAGS -m64 -rdc=false )
	set ( GPU_COMPILE_DEFNS -dc )		## -Xptxas -v
	set ( LIBS_FOR_CUDA cufft culibos )
    endif ()

    set ( CMAKE_CUDA_ARCHITECTURES 52 )
endif ()

if ( "x${DIM_X}" STREQUAL "x" )
    ##  DIM_X is not defined (on command line).  Assume building with default sizes only
    message ( STATUS "Building for default size example only" )
else ()
    ## DIM_X is defined (presumably DIM_Y & DIM_Z also since they come form a script)
    list ( APPEND ADDL_COMPILE_FLAGS -Dfftx_nx=${DIM_X} -Dfftx_ny=${DIM_Y} -Dfftx_nz=${DIM_Z} )
    message ( STATUS "Building for size [ ${DIM_X}, ${DIM_Y}, ${DIM_Z} ]" )
endif ()

##  Set include paths and require C++ 11 standard

set ( FFTX_INCLUDE ${FFTX_PROJECT_SOURCE_DIR}/include )
set ( CMAKE_C_STANDARD 11)
set ( CMAKE_CXX_STANDARD 11)

include_directories ( ${FFTX_INCLUDE} ${SPIRAL_SOURCE_DIR}/profiler/targets )

if ( (NOT DEFINED CMAKE_BUILD_TYPE) OR (NOT CMAKE_BUILD_TYPE) )
    set ( CMAKE_BUILD_TYPE Release )
endif ()
