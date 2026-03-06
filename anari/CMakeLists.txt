# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


option(BARNEY_BUILD_ANARI "Build ANARI device" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(anari 0.15.0 COMPONENTS code_gen)
if (NOT anari_FOUND)
  message("barney: ANARI SDK not found - skipping this")
  return()
endif()

include(GNUInstallDirs)

option(BARNEY_AUTO_GENERATE_QUERIES "auto-generate the anari device queries" OFF)


## Device target ##

if (BARNEY_HAVE_CUDA)
  project(anari_library_barney LANGUAGES CXX CUDA)
  set(DUMMY_CU dummy.cu)
else()
  project(anari_library_barney LANGUAGES CXX)
  set(DUMMY_CU)
endif()

set(BANARI_SOURCES
  Array.cpp
  BarneyGlobalState.cpp
  Camera.cpp
  Device.cpp
  Frame.cpp
  Geometry.cpp
  Group.cpp
  Instance.cpp
  Library.cpp
  Material.cpp
  Object.cpp
  Renderer.cpp
  Sampler.cpp
  SpatialField.cpp
  Surface.cpp
  Volume.cpp
  World.cpp
  # -----------
  light/Light.h
  light/Light.cpp
  light/HDRI.h
  light/HDRI.cpp
  light/Directional.h
  light/Directional.cpp
  light/Point.h
  light/Point.cpp
  # -----------
  ${DUMMY_CU}
  )
if (NOT BARNEY_AUTO_GENERATE_QUERIES)
  LIST(APPEND BANARI_SOURCES generated/anari_library_barney_queries.cpp)
endif()



# helper 'interface-'library that contains common settings for both
# static and shared anari
add_library(banari-config INTERFACE)
target_link_libraries(banari-config INTERFACE barney_config)
set_target_properties(banari-config PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(banari-config
INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
)

# ==================================================================
# default shared-linkage for anari loader library to load when barney
# device is requested
# ==================================================================
add_library(anari_library_barney SHARED ${BANARI_SOURCES})

if(WIN32)
  target_link_libraries(anari_library_barney PRIVATE
    $<BUILD_INTERFACE:banari-config>
    barney_static
    anari::helium
  )
  set_target_properties(anari_library_barney PROPERTIES
    LINK_FLAGS "/WHOLEARCHIVE:barney_static"
  )
else()
  target_link_libraries(anari_library_barney PRIVATE
    $<BUILD_INTERFACE:banari-config>
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney_static>"
    -Wl,--export-dynamic
    anari::helium
  )
endif()

if (BARNEY_MPI)
  add_library(anari_library_barney_mpi SHARED ${BANARI_SOURCES})
  if(WIN32)
    target_link_libraries(anari_library_barney_mpi PRIVATE
      $<BUILD_INTERFACE:banari-config>
      barney_mpi_static
      anari::helium
    )
    set_target_properties(anari_library_barney_mpi PROPERTIES
      LINK_FLAGS "/WHOLEARCHIVE:barney_mpi_static"
    )
  else()
    target_link_libraries(anari_library_barney_mpi PRIVATE
      $<BUILD_INTERFACE:banari-config>
      "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney_mpi_static>"
      -Wl,--export-dynamic
      anari::helium
    )
  endif()
endif()


if (BARNEY_HAVE_CUDA)
  target_compile_definitions(anari_library_barney PRIVATE -DBANARI_HAVE_CUDA=1)
  if (BARNEY_MPI)
    target_compile_definitions(anari_library_barney_mpi PRIVATE -DBANARI_HAVE_CUDA=1)
  endif()
endif()


if (BARNEY_HAVE_CUDA)
  set_source_files_properties(World.cpp
    PROPERTIES
    LANGUAGE CUDA
  )
  set_target_properties(anari_library_barney
    PROPERTIES
    LANGUAGE CUDA
    CUDA_SEPARABLE_COMPILATION ON
  )
endif()

#target_link_libraries(anari_library_barney PRIVATE )
# if(MSVC)
#     target_compile_options(anari_library_barney INTERFACE
#         "$<$<CONFIG:Debug>:/MTd>"
#         "$<$<CONFIG:Release>:/MT>"
#         "$<$<CONFIG:RelWithDebInfo>:/MT>"
#         "$<$<CONFIG:MinSizeRel>:/MT>"
#     )
# endif()
set_target_properties(anari_library_barney
  PROPERTIES
  CXX_VISIBILITY_PRESET  hidden
  CUDA_VISIBILITY_PRESET hidden
  CUDA_SEPARABLE_COMPILATION   ON
  POSITION_INDEPENDENT_CODE    ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
)

include(GenerateExportHeader)
generate_export_header(anari_library_barney
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
)

## Installation ##

if (APPLE)
  set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
if (OFF)
install(TARGETS# barney
  anari_library_barney
  EXPORT barney-config
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif()

file(GLOB DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.json")

foreach(file ${DEVICE_JSONS})
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file} INPUT ${file})
endforeach()

if (BARNEY_AUTO_GENERATE_QUERIES)
anari_generate_queries(
  DEVICE_TARGET anari_library_barney
  CPP_NAMESPACE barney_device
  JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
  JSON_EXTENSION_FILES ${DEVICE_JSONS}
  )
endif()

if (BARNEY_MPI)
  set_target_properties(anari_library_barney_mpi
    PROPERTIES
    CXX_VISIBILITY_PRESET  hidden
    CUDA_VISIBILITY_PRESET hidden
    CUDA_SEPARABLE_COMPILATION   ON
    POSITION_INDEPENDENT_CODE    ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    )
  if (BARNEY_AUTO_GENERATE_QUERIES)
    anari_generate_queries(
      DEVICE_TARGET anari_library_barney_mpi
      CPP_NAMESPACE barney_device
      JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
      JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
      JSON_EXTENSION_FILES ${DEVICE_JSONS}
      )
  endif()
endif()

