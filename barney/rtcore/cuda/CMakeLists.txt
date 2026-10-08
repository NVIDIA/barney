# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(rtc_target barney_rtc_${backend})
set(rtc_sources
  Buffer.cpp
  GeomType.cpp
  Geom.cpp
  TraceKernel.cpp
  Device.cu
  Group.cu
)

rtc_configure_source(${rtc_sources})
add_library(${rtc_target} STATIC ${rtc_sources})
target_link_libraries(${rtc_target} PUBLIC
  barney_config_${backend}
  barney_rtc_cudaCommon_${backend} 
  cuBQL
)
target_include_directories(${rtc_target} PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)

rtc_library_properties(${rtc_target})
set_property(
  TARGET ${rtc_target}
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_property(
  TARGET ${rtc_target} 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>
)
