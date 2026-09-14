# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 'cudaCommon' is a single set of sources, but it might get compiled
# differently for differnet backends - different compiler, different
# defines, and in particular different namespace - so we actually have
# t have a different cmake target for each 'isntantiation' of this
# code.

if (BARNEY_USE_HIP)
  set(CMAKE_HIP_PLATFORM amd)
  enable_language(HIP)
else()
  find_package(CUDAToolkit REQUIRED)
endif()

add_library(barney_rtc_cudaCommon_${backend} STATIC
  Device.cu
  Texture.cu
  TextureData.cu
#  cuda-helper.cpp
)
target_link_libraries(barney_rtc_cudaCommon_${backend} PUBLIC
  $<BUILD_INTERFACE:barney_config_${backend}>
  cuBQL
)
if (BARNEY_USE_HIP)
  target_compile_definitions(barney_rtc_cudaCommon_${backend} PUBLIC
    -DUSE_HIP=1 -D__HIP_PLATFORM_AMD__=1
  )
else()
  target_link_libraries(barney_rtc_cudaCommon_${backend} PUBLIC
    CUDA::cudart_static
  )
endif()


target_include_directories(barney_rtc_cudaCommon_${backend} PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)
rtc_library_properties(barney_rtc_cudaCommon_${backend})
