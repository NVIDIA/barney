# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ==================================================================
# create barney_${backend} library; this will contain only one
# anari device, exactly the one as encoded in the name.
# ==================================================================
set(BARNEY_BACKEND_NAME ${backend})

#############################################################################
# main step - create the actual 'backends'; ie, the 'instantiations' of
# barney's actuall classes, algorithms, kernels, programs, etc, for a
# given rtcore.
#############################################################################

# define all the host-side sources
set(HOST_SOURCES

  # all the barney "plumbing" stuff on the host
  WorkerTopo.h
  WorkerTopo.cpp
  Context.h
  Context.cpp
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cpp
  Object.h
  Object.cpp
  FromEnv.h
  FromEnv.cpp
  Camera.h
  Camera.cu
  ImageRegion.h
  
  # all frame buffer related stuff (DistFB in barney_mpi)
  fb/FrameBuffer.h
  fb/FrameBuffer.cu
  fb/LocalFB.h
  fb/LocalFB.cpp
  fb/TiledFB.h
  fb/TiledFB.cu
  
  # model/group/data group handling
  GlobalModel.h
  GlobalModel.cpp
  render/Renderer.h
  render/Renderer.cpp
  Group.h
  Group.cpp
  render/GeometryAttributes.h
  render/GeometryAttributes.cpp

  # samplers
  render/Sampler.h
  render/Sampler.cpp
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  kernels/traceRays.cu
  
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  umesh/mc/UMeshCuBQLSampler.h
  umesh/mc/UMeshCuBQLSampler.cu
 
  amr/BlockStructuredCuBQLSampler.cu
  amr/BlockStructuredField.cu
  
  # *structured* volumes
  volume/StructuredData.h
  volume/StructuredData.cu
  # *NanoVB* volumes
  volume/NanoVDB.h
  volume/NanoVDB.cu
  
  # actual geometry types
  geometry/Geometry.h
  geometry/Geometry.cpp
  geometry/Triangles.h
  geometry/Triangles.cpp
  geometry/Cylinders.h
  geometry/Cylinders.cpp
  geometry/Capsules.h
  geometry/Capsules.cpp
  geometry/Spheres.h
  geometry/Spheres.cpp
  geometry/Cones.h
  geometry/Cones.cpp
  geometry/IsoSurface.h
  geometry/IsoSurface.cpp
  # volume data 
  volume/Volume.h
  volume/Volume.cpp
  volume/ScalarField.h
  volume/ScalarField.cpp
  volume/MCAccelerator.h
  volume/TransferFunction.h
  volume/TransferFunction.cpp
  volume/MCGrid.h
  volume/MCGrid.cu
  volume/MCAccelerator.h
  
  common/barney-common.h
  
  render/RayQueue.h
  render/RayQueue.cpp
  render/World.h
  render/World.cpp
  render/MaterialRegistry.h
  render/MaterialRegistry.cpp
  render/SamplerRegistry.h
  render/SamplerRegistry.cpp
  
  ModelSlot.h
  ModelSlot.cpp
  
  # general csommon/data-related stuff
  common/Texture.h
  common/Texture.cpp
  common/Data.h
  common/Data.cpp

  # lights
  light/Light.h
  light/Light.cpp
  light/DirLight.h
  light/DirLight.cu
  light/PointLight.h
  light/PointLight.cu
  light/QuadLight.h
  light/QuadLight.cu
  light/EnvMap.h
  light/EnvMap.cu

  # materials
  material/HostMaterial.h
  material/HostMaterial.cpp
  material/AnariPBR.h
  material/AnariPBR.cpp
  material/AnariMatte.h
  material/AnariMatte.cpp
  material/NVisii.h
  material/NVisii.cpp
  material/Glass.h
  material/Glass.cpp

  # different traceRays() implementations
  globalTrace/RQSBase.h
  globalTrace/RQSBase.cpp
  globalTrace/RQSLocal.h
  globalTrace/RQSLocal.cu

  barney.cu
)

# all the source that eventually go into barney_mpi
if (BARNEY_MPI)
  set(MPI_SOURCES
    fb/DistFB.h
    fb/DistFB.cu
    MPIContext.h
    MPIContext.cpp
    globalTrace/RQSMPI.h
    globalTrace/RQSMPI.cu
    globalTrace/TwoStage.h
    globalTrace/TwoStage.cu
    globalTrace/All2all.h
    globalTrace/All2all.cu
    common/MPIWrappers.h
    common/MPIWrappers.cpp
  )
else()
  set(MPI_SOURCES)
endif()

set(DEVICE_PROGRAM_SOURCES
  geometry/Spheres.dev.cu
  geometry/Triangles.dev.cu
  geometry/Cylinders.dev.cu
  geometry/Capsules.dev.cu
  geometry/Cones.dev.cu
  volume/StructuredData.dev.cu
  volume/NanoVDB.dev.cu
  umesh/mc/UMeshMC.dev.cu
  amr/BlockStructuredMC.dev.cu
  kernels/traceRays.dev.cu
)

# this will build a static library 'barney_device_programs' that
# either contains the compiled device programs, or library containing
# the precomiled ptx sources
rtc_build_device_sources(barney_${backend}_programs
  ${DEVICE_PROGRAM_SOURCES}
)

foreach(src ${HOST_SOURCES} ${MPI_SOURCES} ${DEVICE_PROGRAM_SOURCES})
  rtc_configure_source(${src})
endforeach()

# ------------------------------------------------------------------
# barney_static
# ------------------------------------------------------------------
add_library(barney_${backend}_static STATIC
  ${HOST_SOURCES}
  # this will be empty if MPI isn't enabled
  ${MPI_SOURCES}
)
target_link_libraries(barney_${backend}_static PUBLIC
  barney_rtc_${backend}  
  $<BUILD_INTERFACE:barney_${backend}_programs>
  $<BUILD_INTERFACE:barney_config_${backend}>
)
if (BARNEY_MPI)
  target_link_libraries(barney_${backend}_static PUBLIC MPI::MPI_CXX)
endif()
if (${backend} STREQUAL "cpu")
  target_link_libraries(barney_${backend}_static PUBLIC cuBQL_cpu_float3_static)
else()
  target_link_libraries(barney_${backend}_static PUBLIC cuBQL_cuda_float3_static)
endif()

rtc_library_properties(barney_${backend}_static)

if (BARNEY_BUILD_BSDF_TESTS)
  set(BSDF_TEST_SOURCE ${PROJECT_SOURCE_DIR}/tests/bsdf_quality.cu)
  rtc_configure_source(${BSDF_TEST_SOURCE})
  add_executable(bsdf_quality_${backend} ${BSDF_TEST_SOURCE})
  target_link_libraries(bsdf_quality_${backend} PRIVATE barney_${backend}_static)
  rtc_library_properties(bsdf_quality_${backend})
  add_test(NAME bsdf_quality_${backend} COMMAND bsdf_quality_${backend})
  set_tests_properties(bsdf_quality_${backend} PROPERTIES TIMEOUT 180)
endif()
