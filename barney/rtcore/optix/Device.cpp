// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/optix/Device.h"
#include "rtcore/optix/Denoiser.h"
#include "rtcore/optix/Buffer.h"
#include "rtcore/optix/Geom.h"
#include "rtcore/optix/Group.h"
#include <owl/InstanceGroup.h>
#include <optix.h>
#include <optix_function_table.h>
#include <optix_stubs.h>

namespace BARNEY_NS {
  namespace rtc {

    int physicalDeviceCount() 
    {
      /* iw - this isn't entirely correct - there are some machines
         that do have working CUDA, but still do not have optix
         support in the driver, in which case you cannot actually use
         all cuda devices for optix. We _should_ be checking for this
         here by trying to do optixInit() on each device, but for now
         let's assume that optix is support, and if not, the user
         can/will manually select `barney_cuda` as anari device */
      int count = 0;
      BARNEY_CUDA_CALL(GetDeviceCount(&count));
      return count;
    }
      
    
    rtc::AccelHandle getAccelHandle(Group *ig)
    { return ig->getDD(); }
    
    Device::Device(int physicalGPU)
      : CudaDeviceBase(physicalGPU)
    {
      owl = owlContextCreate(&physicalGPU,1);
    }

    Device::~Device()
    {
      destroy();
    }

    std::string Device::toString() const
    { return "optix::Device(physical="+std::to_string(physicalID)+")"; }

    Denoiser *Device::createDenoiser()
    {
#if OPTIX_VERSION >= 80000
      auto *d = new Optix8Denoiser(this);
      if (!d->available) {
        delete d;
        return nullptr;
      }
      return d;
#else
      return nullptr;
#endif
    }

    void Device::destroy()
    {
      if (owl) owlContextDestroy(owl);
      owl = 0;
    }

    void Device::sync()
    {
      CudaDeviceBase::sync();
      for (auto s : activeTraceStreams) {
        cudaStreamSynchronize(s);
        auto rc = cudaGetLastError();
        if (rc) {
          PING; PRINT(rc); PRINT(cudaGetErrorString(rc));
        }
        assert(rc == 0);
      }
      activeTraceStreams.clear();
    }
    
    // ==================================================================
    // rtcore/pipeline stuff
    // ==================================================================
    void Device::buildPipeline() 
    {
      if (!programsDirty) return;
      owlBuildPipeline(owl);
    }
      
    void Device::buildSBT() 
    {
      owlBuildSBT(owl);
    }
      
    // ==================================================================
    // buffer
    // ==================================================================

    void Device::freeBuffer(Buffer *buffer) 
    {
      delete buffer;
    }
    
    Buffer *Device::createBuffer(size_t numBytes,
                                 const void *initValues) 
    {
      return new Buffer(this,numBytes,initValues);
    }


    // ==================================================================
    // geom
    // ==================================================================

    void Device::freeGeom(Geom *geom)
    { assert(geom); delete geom; }
    
    // ==================================================================
    // geomtype
    // ==================================================================
    
    void Device::freeGeomType(GeomType *gt)
    { assert(gt); delete gt; }
    
    // ==================================================================
    // kernels
    // ==================================================================


    TraceKernel2D::TraceKernel2D(Device *device,
                                 const std::string &ptxCode,
                                 const std::string &kernelName,
                                 size_t sizeOfLP)
      : device(device)
    {
      OWLVarDecl rg_args[]
        = {
        { nullptr }
      };
      mod = owlModuleCreate(device->owl,ptxCode.c_str());
      rg = owlRayGenCreate(device->owl,mod,
                           kernelName.c_str(),
                           0,rg_args,-1);
      owlBuildPrograms(device->owl);
      
      OWLVarDecl lp_args[]
        = {
        { "raw", (OWLDataType)(OWL_USER_TYPE_BEGIN+sizeOfLP), 0 },
        { nullptr }
      };
      lp = owlParamsCreate(device->owl,sizeOfLP,lp_args,-1);
      lpStream = owlParamsGetCudaStream(lp,0);
      device->programsDirty = true;
    }
    
    void TraceKernel2D::launch(vec2i dims,
                               const void *kernelData)
    {
      SetActiveGPU forDuration(device);
      BARNEY_CUDA_CALL(StreamSynchronize(/*inherited!*/device->stream));

      owlParamsSetRaw(lp,"raw",kernelData,0);
      if (dims.x > 0 && dims.y > 0) {
        owlAsyncLaunch2D(rg,dims.x,dims.y,lp);
        device->activeTraceStreams.push_back(lpStream);
      }
    }
    
    // ==================================================================
    // groups
    // ==================================================================
    void Device::freeGroup(Group *group)
    {
      delete group;
    }

    Group *
    Device::createTrianglesGroup(const std::vector<Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->owl);
      OWLGroup g = owlTrianglesGeomGroupCreate(owl,
                                               owlGeoms.size(),
                                               owlGeoms.data());
      return new Group(this,g);
    }

    Group *
    Device::createUserGeomsGroup(const std::vector<Geom *> &geoms)
    {
      std::vector<OWLGeom> owlGeoms;
      for (auto geom : geoms)
        owlGeoms.push_back(((Geom *)geom)->owl);
      OWLGroup g = owlUserGeomGroupCreate(owl,
                                          owlGeoms.size(),
                                          owlGeoms.data());
      return new Group(this,g);
    }

    Group *
    Device::createInstanceGroup(const std::vector<Group *>  &groups,
                                const std::vector<int>      &instIDs,
                                const std::vector<affine3f> &xfms)
    {
      std::vector<OWLGroup> owls;
      for (auto group : groups)
        owls.push_back(((Group *)group)->owl);

      OWLGroup g
        = owlInstanceGroupCreate(owl,
                                 owls.size(),
                                 owls.data(),
                                 (const uint32_t*)instIDs.data(),
                                 (const float *)xfms.data(),
                                 OWL_MATRIX_FORMAT_OWL,
                                 owl::InstanceGroup::defaultBuildFlags
                                 | OPTIX_BUILD_FLAG_ALLOW_UPDATE);
      Group *gg = new Group(this,g);
      return gg;
    }

  }
}

