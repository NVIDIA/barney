// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/Device.h"
#include "rtcore/cpu/Texture.h"
#include "rtcore/cpu/Buffer.h"
#include "rtcore/cpu/Triangles.h"
#include "rtcore/cpu/UserGeom.h"
#include "rtcore/cpu/Group.h"
#include "rtcore/cpu/Denoiser.h"
#include "rtcore/cpu/TraceInterface.h"
#include "rtcore/cpu/ComputeInterface.h"

namespace BARNEY_NS {
  namespace rtc {

    std::string Device::toString() const
    { return "rtc::cpu::Device()"; }

    int physicalDeviceCount() { return 1; }
    
    /*! get a unique hash for a given physical device. */
    size_t getPhysicalDeviceHash(int gpuID)
    { return (size_t)gpuID; }
    
    // ------------------------------------------------------------------
    // rt core interface
    // ------------------------------------------------------------------
    
    vec3f TraceInterface::transformNormalFromObjectToWorldSpace(vec3f v) const
    {
      return xfmVector(worldToObjectXfm->l.transposed(),
                         (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformPointFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmPoint(*objectToWorldXfm,
                        (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformVectorFromObjectToWorldSpace(vec3f v) const
    { 
      return xfmVector(objectToWorldXfm->l,
                         (const owl::common::vec3f &)v);
    }

    vec3f TraceInterface::transformNormalFromWorldToObjectSpace(vec3f v) const
    {
      return xfmVector(objectToWorldXfm->l.transposed(),
                         (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformPointFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmPoint(*worldToObjectXfm,
                        (const owl::common::vec3f &)v);
    }
    vec3f TraceInterface::transformVectorFromWorldToObjectSpace(vec3f v) const
    { 
      return xfmVector(worldToObjectXfm->l,
                         (const owl::common::vec3f &)v);
    }


    /*! intersection filter for TRIANGLES geoms, that allows us to
        hook in any-hit program on top of embree hardcoded
        triangles. User geoms will NOT use this function, and will
        handle ah programs directly in their embree isec callback */
    void intersectionFilter(const RTCFilterFunctionNArguments* args)
    {
      /* avoid crashing when debug visualizations are used */
      if (args->context == nullptr) return;

      assert(args->N == 1);
      int* valid = args->valid;
      if (valid[0] != -1) return;
      
      RTCRay* ray = (RTCRay*)args->ray;
      RTCHit* hit = (RTCHit*)args->hit;
  
      TraceInterface *ti = (TraceInterface *)args->context;

      int primID = hit->primID;
      int geomID = hit->geomID;
      int instIdx = hit->instID[0];

      InstanceGroup *ig = ti->world;
      GeomGroup *group = ig->getGroup(instIdx);
      Geom *geom = (Geom*)group->getGeom(geomID);
      GeomType *gt = geom->type;
      if (gt->ah) {
        ti->geomData = (void*)geom->programData.data();
        ti->ignoreThisHit = false;
        ti->primID = primID;
        ti->geomID = geomID;
        ti->instIdx = instIdx;
        ti->triangleBarycentrics = { hit->u,hit->v };
        ti->objectToWorldXfm = &ig->xfms[instIdx];
        ti->worldToObjectXfm = &ig->inverseXfms[instIdx];
        ti->embreeRay = ray;
        ti->embreeHit = hit;
    
        gt->ah(*ti);

        if (ti->ignoreThisHit) {
          valid[0] = 0;
          return;
        } 
      }
    }

    
    void TraceInterface::traceRay(rtc::AccelHandle world,
                                  vec3f rayOrigin,
                                  vec3f rayDirection,
                                  float tmin,
                                  float tmax,
                                  void *prdPtr) 
    {
      InstanceGroup *ig = (InstanceGroup *)world;
      RTCScene embreeScene = ig->embreeScene;
      assert(embreeScene);

      RTCRayHit rayHit;
  
      TraceInterface *ti = this;
      ti->world = ig;

      ti->worldOrigin = rayOrigin;
      ti->worldDirection = rayDirection;
      ti->prd = prdPtr;
      ti->instIDs = ig->instIDs.data();

      rayHit.ray.org_x = rayOrigin.x;        // x coordinate of ray origin
      rayHit.ray.org_y = rayOrigin.y;        // y coordinate of ray origin
      rayHit.ray.org_z = rayOrigin.z;        // z coordinate of ray origin
      rayHit.ray.tnear = tmin;        // start of ray segment
  
      rayHit.ray.dir_x = rayDirection.x;        // x coordinate of ray direction
      rayHit.ray.dir_y = rayDirection.y;        // y coordinate of ray direction
      rayHit.ray.dir_z = rayDirection.z;        // z coordinate of ray direction
      rayHit.ray.time = 0;         // time of this ray for motion blur
  
      rayHit.ray.tfar = tmax;         // end of ray segment (set to hit distance)
      
      rayHit.ray.mask = -1;
      rayHit.ray.flags = 0;
      rayHit.hit.Ng_x = 0.f;
      rayHit.hit.Ng_y = 0.f;
      rayHit.hit.Ng_z = 0.f;
      rayHit.hit.primID    = RTC_INVALID_GEOMETRY_ID;
      rayHit.hit.geomID    = RTC_INVALID_GEOMETRY_ID;
      rayHit.hit.instID[0] = RTC_INVALID_GEOMETRY_ID;
      
      ti->embreeRay = &rayHit.ray;
      ti->embreeHit = &rayHit.hit;
      
      rtcInitRayQueryContext(&ti->embreeRayQueryContext);
      ti->world = ig;

      
      /* intersect ray with scene */
      RTCIntersectArguments iargs;
      rtcInitIntersectArguments(&iargs);
      iargs.context = &ti->embreeRayQueryContext;
#if 0
#define FEATURE_MASK                                    \
      RTC_FEATURE_FLAG_TRIANGLE |                       \
        RTC_FEATURE_FLAG_FILTER_FUNCTION_IN_ARGUMENTS


      iargs.feature_mask = (RTCFeatureFlags) (FEATURE_MASK);
#endif
      iargs.filter = intersectionFilter;

      rtcIntersect1(embreeScene,&rayHit,&iargs);
      if ((int)rayHit.hit.geomID >= 0) {
    
        int primID = rayHit.hit.primID;
        int geomID = rayHit.hit.geomID;
        int instIdx = rayHit.hit.instID[0];
    
        GeomGroup *group = (GeomGroup *)ig->groups[instIdx];
        Geom *geom = (Geom *)group->geoms[geomID];
        GeomType *gt = geom->type;
        if (gt->ch) {

          ti->geomData = (void*)geom->programData.data();
          ti->primID = primID;
          ti->geomID = geomID;
          ti->instIdx = instIdx;
          ti->triangleBarycentrics = { rayHit.hit.u,rayHit.hit.v };
          ti->objectToWorldXfm = &ig->xfms[instIdx];
          ti->worldToObjectXfm = &ig->inverseXfms[instIdx];
          ti->embreeRay = &rayHit.ray;
          ti->embreeHit = &rayHit.hit;
    
          gt->ch(*ti);
        }
      }
      
    }
      
    

    // ------------------------------------------------------------------
    // device
    // ------------------------------------------------------------------
    
    Device::Device(int physicalGPU)
    {
      const char *fromEnv = std::getenv("BARNEY_MAX_CPU_THREADS");
      if (fromEnv) {
        std::stringstream ss;
        ss << "verbose=0,threads=" << fromEnv;
        // ss << "verbose=1,threads=" << fromEnv;
        embreeDevice = rtcNewDevice(ss.str().c_str());
      } else {
        embreeDevice = rtcNewDevice("verbose=0");
      }
      ls = createLaunchSystem();
    }

    Device::~Device()
    {
      // todo: shut down launch system
      destroy();
    }

    Denoiser *Device::createDenoiser()
    {
#if BARNEY_OIDN_CPU
      return new DenoiserOIDN(this);
#else
      // we have no way of denoising
      return nullptr;
#endif
    }
    
    void Device::destroy()
    {
      rtcReleaseDevice(embreeDevice);
      embreeDevice = 0;
    }


    // ------------------------------------------------------------------
    // group/accel stuff
    // ------------------------------------------------------------------

    Group *
    Device::createTrianglesGroup(const std::vector<Geom *> &geoms)
    { return new TrianglesGroup(this,geoms); }
    
    Group *
    Device::createUserGeomsGroup(const std::vector<Geom *> &geoms) 
    { return new UserGeomGroup(this,geoms); }
      
    Group *
    Device::createInstanceGroup(const std::vector<Group *>  &groups,
                                const std::vector<int>      &instIDs,
                                const std::vector<affine3f> &xfms) 
    { return new InstanceGroup(this,groups,instIDs,xfms); }
    
    void Device::freeGroup(Group *group) 
    { delete group; }

    // ------------------------------------------------------------------
    // geom stuff
    // ------------------------------------------------------------------

    void Device::freeGeom(Geom *geom) 
    { delete geom; }


    void Device::freeGeomType(GeomType *gt) 
    {
      delete gt;
    }
    
    TextureData *Device::createTextureData(vec3i dims,
                                                rtc::DataType format,
                                                const void *texels) 
    { return new TextureData(this,dims,format,texels); }

    Buffer *Device::createBuffer(size_t numBytes,
                                      const void *initValues) 
    {
      return new Buffer(this,numBytes,initValues);
    }

    void Device::freeTextureData(TextureData *td) 
    {
      delete (TextureData*)td;
    }
      
    void Device::freeTexture(Texture *tex) 
    {
      delete (Texture *)tex;
    }


    void Device::freeBuffer(Buffer *buffer) 
    {
      delete (Buffer*)buffer;
    }
    
    void Device::buildPipeline()
    {}
    
    void Device::buildSBT()
    {}
    
  }
}

