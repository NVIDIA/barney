// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cpu/embree-common.h"

#define RTC_DEVICE_CODE 1

#ifndef BARNEY_NS
# error "BARNEY_NS not defined"
#endif

namespace BARNEY_NS {
  namespace rtc {

    struct LaunchSystem;
    LaunchSystem *createLaunchSystem();


    /*! get a unique hash for a given physical device. */
    size_t getPhysicalDeviceHash(int gpuID);
    
    struct Device;
    struct Denoiser;
    struct Group;
    struct Buffer;
    struct Geom;
    struct GeomType;
    struct TextureData;
    struct Texture;

    struct TraceInterface;
    
    typedef void (*TraceKernelFct)(BARNEY_NS::rtc::TraceInterface &);
    
    struct TraceKernel2D {
      TraceKernel2D(Device *device,
                    TraceKernelFct kernelFct)
        : device(device),
          kernelFct(kernelFct)
      {}
      
      void launch(vec2i launchDims,
                  const void *kernelData);
      TraceKernelFct const kernelFct;
      Device *const device;
    };
    
    struct Device {
      Device(int physicalGPU);
      virtual ~Device();

      Denoiser *createDenoiser();

      void destroy();
      std::string toString() const;
 
      // ==================================================================
      // basic compute stuff
      // ==================================================================
      void copyAsync(void *dst, const void *src, size_t numBytes)
      { memcpy(dst,src,numBytes); }

      void copy(void *dst, const void *src, size_t numBytes)
      { memcpy(dst,src,numBytes); }
      
      void *allocHost(size_t numBytes)
      { return numBytes?malloc(numBytes):0; }
      
      void freeHost(void *mem)
      { if(mem) free(mem); }
      
      void memsetAsync(void *mem,int value, size_t size)
      { if (size) memset(mem,value,size); }
      
      void *allocMem(size_t numBytes)
      { return numBytes?malloc(numBytes):nullptr; }
      
      void freeMem(void *mem)
      { if (mem) free(mem); }
      
      void sync()
      {/*no-op*/}
      
      /*! sets this gpu as active, and returns physical ID of GPU that
        was active before */
      int setActive() const
      {/*no-op*/ return 0; }
      
      /*! restores the gpu whose ID was previously returend by setActive() */
      void restoreActive(int oldActive) const
      {/*no-op*/}

      TextureData *createTextureData(vec3i dims,
                                     BARNEY_NS::rtc::DataType format,
                                     const void *texels);
      void freeTextureData(TextureData *td);
      void freeTexture(Texture *tex);
      
      // ==================================================================
      // kernels
      // ==================================================================

      // ==================================================================
      // buffer stuff
      // ==================================================================
      Buffer *createBuffer(size_t numBytes,
                                const void *initValues = 0);
     
      void freeBuffer(Buffer *buffer);
      
      // ==================================================================
      // texture stuff
      // ==================================================================

      // in parent

      // ==================================================================
      // ray tracing pipeline related stuff
      // ==================================================================


      // ------------------------------------------------------------------
      // rt pipeline/sbtstuff
      // ------------------------------------------------------------------

      void buildPipeline();
      void buildSBT();

      // ------------------------------------------------------------------
      // geomtype stuff
      // ------------------------------------------------------------------
      
      void freeGeomType(GeomType *);

      // ------------------------------------------------------------------
      // geom stuff
      // ------------------------------------------------------------------
      
      void freeGeom(Geom *);
      
      // ------------------------------------------------------------------
      // group/accel stuff
      // ------------------------------------------------------------------
      Group *
      createTrianglesGroup(const std::vector<Geom *> &geoms);
      
      Group *
      createUserGeomsGroup(const std::vector<Geom *> &geoms);
      
      Group *
      createInstanceGroup(const std::vector<Group *>  &groups,
                          const std::vector<int>      &instIDs,
                          const std::vector<affine3f> &xfms);
      
      void freeGroup(Group *group);

      LaunchSystem *ls = 0;
      RTCDevice embreeDevice = 0;
    };

    int physicalDeviceCount();
  }
}

#define RTC_DECLARE_GLOBALS(ignore) /* ignore */

#define RTC_IMPORT_TRACE2D(name,Class,sizeOfLP)                         \
  BARNEY_NS::rtc::TraceKernel2D *                                       \
  createTrace_##name(BARNEY_NS::rtc::Device *device);

#define RTC_EXPORT_TRACE2D(name,Class)                                  \
  BARNEY_NS::rtc::TraceKernel2D *                                       \
  createTrace_##name(BARNEY_NS::rtc::Device *device)                    \
  { return new BARNEY_NS::rtc::TraceKernel2D(device,Class::run); }
