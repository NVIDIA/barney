# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if (WIN32)
  set(BARNEY_CUDA_ARCHITECTURES_INIT "all-major")
else()
  set(BARNEY_CUDA_ARCHITECTURES_INIT "native")
endif()

set(CMAKE_CUDA_ARCHITECTURES
  "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
  "Which CUDA architecture to build for")

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

function(rtc_build_device_sources libname)
  set(DEVICE_PROGRAM_SOURCES ${ARGN})
  add_library(${libname} OBJECT
    ${DEVICE_PROGRAM_SOURCES}
  )
  target_compile_definitions(${libname} PRIVATE
    -DBARNEY_DEVICE_PROGRAM=1)
  target_link_libraries(${libname}
    barney_rtc_cuda
    barney_rtc_cudaCommon_${backend}
    barney_config_${backend}
    )
  rtc_library_properties(${libname})
endfunction()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cudaCommon)# buildDir_rtc_cudaCommon)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/cuda)# buildDir_rtc_optix)

