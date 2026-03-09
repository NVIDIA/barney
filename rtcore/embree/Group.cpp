// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/embree/Group.h"
#include "rtcore/embree/Device.h"
#include "rtcore/embree/Triangles.h"
#include "rtcore/embree/UserGeom.h"
// 
#include "rtcore/embree/TraceInterface.h"

namespace rtc {
  namespace embree {

    void virtualBoundsFunc(const struct RTCBoundsFunctionArguments* args)
    {
      TraceInterface *ti = /* just need the type*/0;//TraceInterface::get();
      const UserGeom *geom = (const UserGeom *)args->geometryUserPtr;
      const void *geomData = (const void *)geom->programData.data();
      int primID = args->primID;
      box3f bounds;
      UserGeomType *type = (UserGeomType *)geom->type;
      type->bounds(*ti,geomData,bounds,primID);
      
      RTCBounds* bounds_o = args->bounds_o;
      bounds_o->lower_x = bounds.lower.x;
      bounds_o->lower_y = bounds.lower.y;
      bounds_o->lower_z = bounds.lower.z;
      bounds_o->upper_x = bounds.upper.x;
      bounds_o->upper_y = bounds.upper.y;
      bounds_o->upper_z = bounds.upper.z;
    }

    void virtualIntersect(const RTCIntersectFunctionNArguments* args)
    {
      int *valid = args->valid;
      void *ptr  = args->geometryUserPtr;
      UserGeom *user = (UserGeom *)ptr;
      RTCRayHit *rayHit = (RTCRayHit*)args->rayhit;
      unsigned int primID = args->primID;
      unsigned int geomID = args->geomID;
      int instIdx = args->context->instID[0];
      TraceInterface *ti = (TraceInterface *)args->context;
      ti->primID = primID;
      ti->geomID = geomID;
      ti->instIdx = instIdx;
      ti->geomData = user->programData.data();
      ti->embreeRay = &rayHit->ray;
      ti->embreeHit = &rayHit->hit;

      InstanceGroup *ig = ti->world;
      Group *group = (embree::Group*)ig->groups[instIdx];
      ti->objectToWorldXfm = &ig->xfms[instIdx];
      ti->worldToObjectXfm = &ig->inverseXfms[instIdx];
      
      UserGeomType *type = (UserGeomType *)user->type;
      /* set to 'no hit found' - this is NOT the ray.tmax value that
        the isec code will test against */
      ti->isec_t = INFINITY;
      type->intersect(*ti);
      // check if isec did 'save' a hit
      if (ti->isec_t < INFINITY) {
        float save_t = ti->embreeRay->tfar;
        ti->embreeRay->tfar = ti->isec_t;
        ti->ignoreThisHit = false;
        if (type->ah) 
          type->ah(*ti);
        if (!ti->ignoreThisHit) {
          // "accept" this hit
          rayHit->hit.primID    = ti->primID;
          rayHit->hit.geomID    = ti->geomID;
          rayHit->hit.instID[0] = ti->instIdx;
          args->valid[0] = -1;
        } else {
          ti->embreeRay->tfar = save_t;
          args->valid[0] = 0;
        }
      }
    }

    InstanceGroup::InstanceGroup(Device *device,
                                 const std::vector<Group *>  &groups,
                                 const std::vector<int>      &instIDs,
                                 const std::vector<affine3f> &xfms)
      : Group(device),
        groups(groups),
        xfms(xfms),
        instIDs(instIDs)
    {}

    void UserGeomGroup::buildAccel() 
    {
      if (embreeScene) {
        rtcReleaseScene(embreeScene);
        embreeScene = 0;
      }

      embree::Device *device = (embree::Device *)this->device;
      embreeScene = rtcNewScene(device->embreeDevice);
      for (auto geom : geoms) {
        UserGeom *user = (UserGeom *)geom;
        RTCGeometry eg
          = rtcNewGeometry(device->embreeDevice,RTC_GEOMETRY_TYPE_USER);

        rtcSetGeometryUserPrimitiveCount(eg,user->primCount);
        rtcSetGeometryUserData(eg,user);
        rtcSetGeometryBoundsFunction(eg,virtualBoundsFunc,user);
      
        rtcSetGeometryEnableFilterFunctionFromArguments(eg,true);
        rtcSetGeometryIntersectFunction(eg,virtualIntersect);
        rtcCommitGeometry(eg);
        rtcAttachGeometry(embreeScene,eg);
        rtcEnableGeometry(eg);

        rtcReleaseGeometry(eg);
      }
      rtcCommitScene(embreeScene);
    }


    TrianglesGroup::TrianglesGroup(Device *device,
                                   const std::vector<Geom *> &geoms)
      : GeomGroup(device,geoms)
    {}

  
    void TrianglesGroup::buildAccel() 
    {
      if (embreeScene) {
        rtcReleaseScene(embreeScene);
        embreeScene = 0;
      }
    
      embree::Device *device = (embree::Device *)this->device;
      embreeScene = rtcNewScene(device->embreeDevice);
      for (auto geom : geoms) {
        TrianglesGeom *triangles = (TrianglesGeom *)geom;
        assert(triangles);
        RTCGeometry eg
          = rtcNewGeometry(device->embreeDevice,RTC_GEOMETRY_TYPE_TRIANGLE);

        RTCBuffer indexBuffer
          = rtcNewSharedBuffer(device->embreeDevice,
                               triangles->indices,
                               triangles->numIndices*sizeof(vec3i));
        rtcSetGeometryBuffer(eg, RTC_BUFFER_TYPE_INDEX, 0,
                             RTC_FORMAT_UINT3, indexBuffer, 0,
                             sizeof(vec3i),
                             triangles->numIndices);
      
        RTCBuffer vertexBuffer
          = rtcNewSharedBuffer(device->embreeDevice,
                               triangles->vertices,
                               triangles->numVertices*sizeof(vec3f));
        rtcSetGeometryBuffer(eg, RTC_BUFFER_TYPE_VERTEX, 0,
                             RTC_FORMAT_FLOAT3,
                             vertexBuffer, 0,
                             sizeof(vec3f), triangles->numVertices);
        
        rtcSetGeometryEnableFilterFunctionFromArguments(eg,true);
        rtcCommitGeometry(eg);
        rtcAttachGeometry(embreeScene,eg);
        rtcEnableGeometry(eg);

        rtcReleaseBuffer(vertexBuffer);
        rtcReleaseBuffer(indexBuffer);
        rtcReleaseGeometry(eg);
      }
      rtcCommitScene(embreeScene);
    }


    GeomGroup *InstanceGroup::getGroup(int groupID) 
    {
      assert(groupID >= 0 && groupID < groups.size());
      Group *g = groups[groupID];
      assert(g);
      return (GeomGroup*)g;
    }
    
    void InstanceGroup::setTransforms(const std::vector<affine3f> &newXfms)
    {
      xfms = newXfms;
    }

    void InstanceGroup::buildAccel()
    {
      embree::Device *device = (embree::Device *)this->device;
      if (embreeScene) {
        rtcReleaseScene(embreeScene);
        embreeScene = 0;
      }

      if (xfms.empty()) {
        xfms.resize(groups.size());
        for (auto &xfm : xfms) xfm = affine3f();
      }
      inverseXfms = xfms;
      for (auto &xfm : inverseXfms) xfm = rcp(xfm);
    
      embreeScene = rtcNewScene(device->embreeDevice);
      for (int instIdx=0;instIdx<groups.size();instIdx++) {
        embree::Group *group = (embree::Group *)groups[instIdx];
        RTCGeometry geom
          = rtcNewGeometry(device->embreeDevice,RTC_GEOMETRY_TYPE_INSTANCE);
        assert(group);
        assert(group->embreeScene);
        rtcSetGeometryInstancedScene(geom,group->embreeScene);
        rtcSetGeometryTransform(geom,0,RTC_FORMAT_FLOAT3X4_COLUMN_MAJOR,
                                &xfms[instIdx]);
              
        rtcAttachGeometry(embreeScene,geom);
        rtcCommitGeometry(geom);
        rtcEnableGeometry(geom);
      }
      rtcCommitScene(embreeScene);
    }
  
  }
}
