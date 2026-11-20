# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------
# optix backend: instantiates all the barney classes,Data device
# programs, etcpp - except the api itself - in a optix
# configuration.
# ------------------------------------------------------------------

if (WIN32)
  set(BARNEY_CUDA_ARCHITECTURES_INIT "all-major")
else()
  set(BARNEY_CUDA_ARCHITECTURES_INIT "native")
endif()

set(CMAKE_CUDA_ARCHITECTURES
  "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
  "Which CUDA architecture to build for")


# ==================================================================
# barney_config already has include dirs for owl (for owl::common
# stuff), but for optix backend we need the full of owl
# ==================================================================
set(OptiX_INSTALL_DIR ${PROJECT_SOURCE_DIR}/submodules/optix)
set(OWL_BUILD_VIEWER OFF)
set(OWL_CUDA_STATIC  ON)
add_subdirectory(${PROJECT_SOURCE_DIR}/submodules/owl builddir_owl EXCLUDE_FROM_ALL)
if ((NOT (DEFINED OWL_VERSION)) OR (${OWL_VERSION} VERSION_LESS ${EXPECTED_OWL_VERSION}))
  message(FATAL_ERROR " OWL version is too old. make sure to update your owl submodule")
endif()


function(rtc_library_properties lib)
  set_target_properties(${lib}
    PROPERTIES
    POSITION_INDEPENDENT_CODE    ON
    VISIBILITY_INLINES_HIDDEN    ON
    CUDA_SEPARABLE_COMPILATION   ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    CUDA_VISIBILITY_PRESET       hidden
    CXX_VISIBILITY_PRESET        hidden
  )
  if (APPLE)
    set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$loader_path")
  else()
    set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
endfunction()

add_library(barney_config_ptx INTERFACE)
target_link_libraries(barney_config_ptx INTERFACE barney_config)
target_compile_definitions(barney_config_ptx INTERFACE -DBARNEY_DEVICE_PROGRAM=1)

function(rtc_build_device_sources libname)
  add_library(${libname} INTERFACE)

  set(DEVICE_PROGRAM_SOURCES ${ARGN})
  foreach(src  ${DEVICE_PROGRAM_SOURCES})
    get_filename_component(basename "${src}" NAME_WE)
    embed_ptx(
      OUTPUT_TARGET      barney_${basename}_ptx
      PTX_LINK_LIBRARIES barney_config_ptx barney_rtc_optix owl::owl_static
      SOURCES            ${src}
    )
    target_link_libraries(
      ${libname}
      INTERFACE
      #PRIVATE
      barney_${basename}_ptx)
  endforeach()
  
  rtc_library_properties(${libname})
endfunction()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cudaCommon buildDir_cudaCommon_optix)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/optix)

