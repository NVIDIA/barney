# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------
# global options
# ------------------------------------------------------------------

# ----------- enable/disable different devices we can build
option(BARNEY_USE_HIP        "Use HIP instead of CUDA"     OFF)
option(BARNEY_BACKEND_CPU    "Build Embree/CPU Backend"    OFF)
option(BARNEY_BACKEND_CUDA   "Build (native) CUDA Backend" OFF)
option(BARNEY_BACKEND_OPTIX  "Build OptiX Backend"         ON)
option(BARNEY_BACKEND_HIP    "Build HIP Backend"           OFF)
option(BARNEY_BACKEND_HIPRT  "Build HIPRT Backend"         OFF)
option(BARNEY_MPI            "Enable MPI Support"          OFF)

# ------------------------------------------------------------------
# cmake project/output config
# ------------------------------------------------------------------
set(BARNEY_VERSION_MAJOR 1)
set(BARNEY_VERSION_MINOR 4)
set(BARNEY_VERSION_PATCH 0)
set(BARNEY_VERSION ${BARNEY_VERSION_MAJOR}.${BARNEY_VERSION_MINOR}.${BARNEY_VERSION_PATCH})

# ----------- which versions of owl/cubql we expect for correct build 
set(EXPECTED_OWL_VERSION "1.2.9")
set(EXPECTED_CUBQL_VERSION "1.3.1")

cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME ON)

cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0063 NEW)

set(CMAKE_BUILD_TYPE_INIT "Release")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

project(Barney VERSION ${BARNEY_VERSION} LANGUAGES C CXX)

if (CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" 
    "MinSizeRel" "RelWithDebInfo")
endif()
include(GNUInstallDirs)
include(CheckLanguage)


set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)
set(CMAKE_CXX_VISIBILITY_PRESET  hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

# barney's own cmake modules (e.g. Findhiprt.cmake for the HIPRT backend)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(anari 0.15.0 COMPONENTS code_gen)

find_package(Python3 COMPONENTS Interpreter)
option(BARNEY_AUTO_GENERATE_QUERIES "auto-generate the anari device queries"
  ${Python3_FOUND})
#if (BARNEY_AUTO_GENERATE_QUERIES)
#  find_package(Python3 REQUIRED COMPONENTS Interpreter)
#endif()

# ==================================================================
# check if there's any compile language conflicts in the
# backends. currently we can't build both hip and cuda at the same
# time (mostly due to this meaning we'd have to build two different
# versions of cuBQL within the same cmake build dir), so let's disable
# the respective backends based on whether hip is or isn't enabled
# ==================================================================
mark_as_advanced(BARNEY_USE_HIP)
if (BARNEY_USE_HIP)
  set(CUBQL_USE_HIP ON)
  set(BARNEY_BACKEND_OPTIX OFF)
  set(BARNEY_BACKEND_CUDA  OFF)
else()
  set(CUBQL_USE_HIP OFF)
  set(BARNEY_BACKEND_HIP   OFF)
  set(BARNEY_BACKEND_HIPRT OFF)
endif()

set(BARNEY_PLUGIN_DIRS "" CACHE STRING "List of barney external plugin directories")
mark_as_advanced(BARNEY_PLUGIN_DIRS)

# ==================================================================
# configure language(s): in theory we support either cuda, hip, or
# cxx, BUT a) hip or cuda might not acutally be installed even if the
# user requested it (we need to check this, and disable selected
# backends if that's the case; and b) for the cpu backend we have to
# know whether cuda/hip are available or not (because if it is the
# respective 'device' source files have to be tagged as CUDA, and if
# not, as CXX.
# ==================================================================

if (BARNEY_BACKEND_CUDA OR BARNEY_BACKEND_OPTIX)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(BARNEY_HAVE_CUDA ON)
    set(CMAKE_CUDA_ARCHITECTURES
      "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
      "Which CUDA architecture to build for")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr --extended-lambda")
    set(BARNEY_DEVICE_LANGUAGE CUDA)
  else()
    set(BARNEY_HAVE_CUDA OFF)
    set(BARNEY_DEVICE_LANGUAGE CXX)
    set(BARNEY_BACKEND_OPTIX OFF)
    set(BARNEY_BACKEND_CUDA OFF)
  endif()
endif()

if (BARNEY_BACKEND_HIP OR BARNEY_BACKEND_HIPRT)
  check_language(HIP)
  if (CMAKE_HIP_COMPILER)
    enable_language(HIP)
    set(BARNEY_DEVICE_LANGUAGE HIP)
    set(BARNEY_HAVE_HIP ON)
  else()
    set(BARNEY_HAVE_HIP OFF)
    set(BARNEY_DEVICE_LANGUAGE CXX)
    set(BARNEY_BACKEND_HIP OFF)
    set(BARNEY_BACKEND_HIPRT OFF)
  endif()
endif()

macro(rtc_configure_source)
  foreach(src ${ARGN})
    get_filename_component(ext "${src}" EXT)
    if ((${ext} STREQUAL ".cu") OR (${ext} STREQUAL ".dev.cu"))
      if (BARNEY_HAVE_CUDA)
        set_source_files_properties(${src} PROPERTIES
          LANGUAGE CUDA
          COMPILE_OPTIONS "--expt-relaxed-constexpr;--extended-lambda;-rdc=true"
        )
      elseif (BARNEY_HAVE_HIP)
        set_source_files_properties(${src} PROPERTIES
          LANGUAGE HIP
          COMPILE_OPTIONS "-fgpu-rdc"
        )
      else ()
        set_source_files_properties(${src} PROPERTIES
          LANGUAGE CXX
          #COMPILE_OPTIONS "--extended-lambda;-rdc=true"
        )
      endif()
    endif()
  endforeach()
endmacro()






# ==================================================================
# external projects
# ==================================================================

# ------------------------------------------------------------------
# owl - even if we build only for cpu (where we wouldn't use the owl
# _library_) we'd still need the owl common classes ... so this
# _always_ gets included first
# ------------------------------------------------------------------
# if (BARNEY_HAVE_CUDA)
#   set(OWL_CUDA_STATIC ON)
#   message(" checking ${CMAKE_CURRENT_LIST_DIR}/../optix")
#   if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/submodules/optix/include/optix.h")
#     set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/submodules/optix)
#   elseif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/../optix/include/optix.h")
#     message(" found ${CMAKE_CURRENT_LIST_DIR}/../optix")
#     set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../optix)
#   else()
#   endif()
#   set(OWL_BUILD_VIEWER OFF)
#   add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
#   if ((NOT (DEFINED OWL_VERSION)) OR (${OWL_VERSION} VERSION_LESS ${EXPECTED_OWL_VERSION}))
#     message(FATAL_ERROR " OWL version is too old. make sure to update your owl submodule")
# endif()
# else()
#   add_library(owl-config INTERFACE)
#   target_include_directories(owl-config INTERFACE
#     ${CMAKE_CURRENT_LIST_DIR}/submodules/owl/include
#   )
# endif()
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# cubql
# ------------------------------------------------------------------

if (BARNEY_HAVE_CUDA)
  # nothing to config, cuda is default
elseif (BARNEY_HAVE_HIP)
  set(CUBQL_USE_HIP ON)
else()
  set(CUBQL_DISABLE_CUDA ON)
endif()
  
add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# mpi
# ------------------------------------------------------------------

if (BARNEY_MPI)
  message("#barney: configuring MPI")
  find_package(MPI)
  if (MPI_FOUND)
    message("#barney: MPI found, including in default barney and banari builds")
  else()
    message("#barney: MPI build requested, but MPI not found... disabling")
    set(BARNEY_MPI OFF)
  endif()
endif()


# ==================================================================
# actual build config
# ==================================================================

add_library(barney_config INTERFACE)
target_link_libraries(barney_config INTERFACE anari::helium)
target_include_directories(barney_config INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
#  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdParty/glm/include/>
#  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/barney/include>
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/barney
  # for any owl::common stuff
  ${CMAKE_CURRENT_LIST_DIR}/submodules/owl/include/
  ${CMAKE_CURRENT_LIST_DIR}/3rdParty/nanovdb/include
)
if (WIN32)
  target_compile_definitions(barney_config INTERFACE -DNOMINMAX)
  target_compile_options(barney_config INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/Zc:preprocessor">
  )
endif()
if(MSVC) # That's also clang-cl
  # /bigobj is needed for bigger binding projects due to the limit to 64k
  # addressable sections
  set_property(
    TARGET barney_config
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

include(CMakePackageConfigHelpers)


# ==================================================================
option(BARNEY_DISABLE_DENOISING "DISable denoising" OFF)
#option(BARNEY_HAVE_NANOVDB "Include Support for NanoVDB" ON)
set(BARNEY_HAVE_NANOVDB ON)


# ------------------------------------------------------------------
# External field plugins
# ------------------------------------------------------------------
if(DEFINED BARNEY_EXTERNAL_FIELDS_DIR AND EXISTS "${BARNEY_EXTERNAL_FIELDS_DIR}/CMakeLists.txt")
  message("#barney: including external fields from ${BARNEY_EXTERNAL_FIELDS_DIR}")
  add_subdirectory("${BARNEY_EXTERNAL_FIELDS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/external_fields")
endif()

# ------------------------------------------------------------------
# cmake install stuff
# ------------------------------------------------------------------

set(BARNEY_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/barney-${BARNEY_VERSION}
)






# build_barney(config device) -> tests if config exists as a target,
# and if so build device (and mpi device, if enabled) with this
# config. This builds only one device, specific to the given
# ${backend}, and named barney_{backend}. in addition to the
# respective 'full' anari library for that backend this also creates
# some statically linkable library that contains only the anari
# device, not the anari library or entrypoint,t hus allowing us to
# link those same devices also the multi-device 'barney' library
function(build_barney backend)
  string(TOUPPER ${backend} BACKEND)
  if (NOT (${BARNEY_BACKEND_${BACKEND}}))
    message("#barney: backend '${backend}' disabled by user request")
  else()
    add_subdirectory(barney build_barney_${backend})
    # ==================================================================
    # now finally, link the respective single-backend
    # 'barney_{backend}' device to the multi-device 'barney' library,
    # and use a compile defitnion to tell it which ones are available
    # ==================================================================
    target_link_libraries(anari_library_barney PRIVATE
      anari_library_barney_${backend}_static)
    target_compile_definitions(anari_library_barney PRIVATE
      -DBARNEY_BACKEND_${BACKEND}=1)
  endif()
endfunction()

# ==================================================================
# this is the MULTI-device library; the ANARILibrary created by that
# will link to each individual device built in this repo, and select
# the best at runtime. The 'build_barney()' function will
# automatically link each generated device to this multi-library
# ==================================================================
add_library(anari_library_barney SHARED
  barney/anari/MultiDeviceLibrary.cpp)
target_include_directories(anari_library_barney PRIVATE
  ${CMAKE_CURRENT_BINARY_DIR}/)
target_link_libraries(anari_library_barney PRIVATE
  barney_config)

# ==================================================================
# now, build - and link-add to multi-dev 'barney' - all possible
# devices. those that lack dependencies or are disblaed by used will
# simply exit-and-ignore.
# ==================================================================
build_barney(cpu)
build_barney(cuda)
build_barney(optix)
build_barney(hip)
build_barney(hiprt)

# this installs the MULTI-library (ie, the 'barney' device that
# internally switches to barney_cuda, barney_optix, etc)
set_target_properties(anari_library_barney
  PROPERTIES
  POSITION_INDEPENDENT_CODE    ON
  VISIBILITY_INLINES_HIDDEN    ON
  CUDA_SEPARABLE_COMPILATION   ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
  CUDA_VISIBILITY_PRESET       hidden
  CXX_VISIBILITY_PRESET        hidden
)
install(
  TARGETS
  anari_library_barney
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

