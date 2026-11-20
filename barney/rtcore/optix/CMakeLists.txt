# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(barney_rtc_optix STATIC
  Buffer.cpp
  Device.cpp
  Group.cpp
  Denoiser.cpp
  Geom.cpp
)
target_link_libraries(barney_rtc_optix PUBLIC
  barney_config
  barney_rtc_cudaCommon_${backend} 
  cuBQL
  owl::owl_static
)
target_include_directories(barney_rtc_optix PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)


rtc_library_properties(barney_rtc_optix)
set_property(
  TARGET barney_rtc_optix 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_property(
  TARGET barney_rtc_optix 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
