# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==================================================================
# create barney_${backend} library; this will contain only one
# anari device, exactly the one as encoded in the name.
# ==================================================================
set(BARNEY_BACKEND_NAME ${backend})
set(BARNEY_LIBRARY_NAME "barney_${backend}")
set(BARNEY_INSTALL_TARGETS)

add_library(barney_config_${backend} INTERFACE)
target_link_libraries(barney_config_${backend} INTERFACE barney_config)
configure_file( 
  "${PROJECT_SOURCE_DIR}/barney/barneyConfig.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/barneyDeviceConfig.h"
)
#configure_file(
#  ${PROJECT_SOURCE_DIR}/barney/anari/Library_entryPoint.h.in
#  ${CMAKE_CURRENT_BINARY_DIR}/Library_entryPoint.h
#)
message(" barney_config_${backend} adding ${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(barney_config_${backend} INTERFACE
  "${CMAKE_CURRENT_BINARY_DIR}"
  )

# ==================================================================
# now build the respective barney device for this backend, which
# consists of three stacked abstraction layers: rtcore,
# barney(native), and (b)anari:
# ==================================================================
add_subdirectory(rtcore)
add_subdirectory(native)
add_subdirectory(anari)






