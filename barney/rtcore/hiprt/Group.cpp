// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>
//
// HIPRT acceleration-structure build for the hardware-RT backend. Each geom
// group becomes one HIPRT geometry (BLAS): a hiprtTriangleMeshPrimitive for
// triangle geoms, or a hiprtAABBListPrimitive (per-prim AABBs + a func-table
// intersect thunk) for user geoms. The instance group becomes a hiprtScene
// (TLAS). We keep barney's per-prim (geomID, primID) remap and per-geom SBT so
// the device-side shading dispatch is identical to the software backend; HIPRT
// supplies only the BVH and the traversal that yields the geometry-local primID.

#include "rtcore/hiprt/Group.h"
#include "rtcore/hiprt/Geom.h"
#include "rtcore/hiprt/GeomType.h"
#include "rtcore/hiprt/Buffer.h"

#include <hiprt/hiprt.h>
#include <stdexcept>
#include <cstring>

namespace BARNEY_NS {
  namespace rtc {

    static void hiprtCheck(hiprtError e, const char *where)
    {
      if (e != hiprtSuccess)
        throw std::runtime_error(std::string("HIPRT error in ")+where
                                 +" code "+std::to_string((int)e));
    }
#define HC(call) rtc::hiprtCheck(call,#call)

    // ------------------------------------------------------------------
    Group::Group(Device *device) : device(device) {}
    Group::~Group() {}

    // ------------------------------------------------------------------
    GeomGroup::GeomGroup(Device *device, const std::vector<Geom *> &geoms)
      : Group(device), geoms(geoms)
    {}

    GeomGroup::~GeomGroup()
    {
      SetActiveGPU forDuration(device);
      if (geom)       hiprtDestroyGeometry(device->hiprtCtx, geom);
      if (d_buildTmp) BARNEY_CUDA_CALL_NOTHROW(Free(d_buildTmp));
      if (sbt)        BARNEY_CUDA_CALL_NOTHROW(Free(sbt));
      if (prims)      BARNEY_CUDA_CALL_NOTHROW(Free(prims));
      if (d_vertices) BARNEY_CUDA_CALL_NOTHROW(Free(d_vertices));
      if (d_indices)  BARNEY_CUDA_CALL_NOTHROW(Free(d_indices));
      if (d_aabbs)    BARNEY_CUDA_CALL_NOTHROW(Free(d_aabbs));
    }

    // assemble the per-geom SBT (shared with the software backend's layout) and
    // the (geomID, primID) prim remap, uploaded to `sbt`/`prims`. Returns the
    // total prim count.
    int GeomGroup_buildSBTandPrims(GeomGroup *gg, bool isTriangles)
    {
      Device *device = gg->device;
      gg->sbtEntrySize = 0;
      size_t align = sizeof(float4);
      for (size_t i=0;i<gg->geoms.size();i++) {
        gg->sbtEntrySize = std::max(gg->sbtEntrySize, gg->geoms[i]->gt->sizeOfDD);
        gg->sbtEntrySize = align*((gg->sbtEntrySize+align-1)/align);
      }
      gg->sbtEntrySize += sizeof(Geom::SBTHeader);

      std::vector<uint8_t> hostSBT(gg->geoms.size()*gg->sbtEntrySize);
      uint8_t *p = hostSBT.data();
      for (size_t i=0;i<gg->geoms.size();i++) {
        Geom *geom = gg->geoms[i];
        Geom::SBTHeader *header = (Geom::SBTHeader *)p;
        if (isTriangles) {
          TrianglesGeom *tg = (TrianglesGeom *)geom;
          header->ah = ((TrianglesGeomType*)tg->gt)->ah;
          header->ch = ((TrianglesGeomType*)tg->gt)->ch;
          header->triangles.vertices = (const vec3f*)tg->vertices->getDD();
          header->triangles.indices  = (const vec3i*)tg->indices->getDD();
        } else {
          UserGeom *ug = (UserGeom *)geom;
          header->ah = ((UserGeomType*)ug->gt)->ah;
          header->ch = ((UserGeomType*)ug->gt)->ch;
          header->user.intersect = ((UserGeomType*)ug->gt)->intersect;
        }
        memcpy(p+sizeof(Geom::SBTHeader),geom->data.data(),geom->data.size());
        p += gg->sbtEntrySize;
      }
      if (gg->sbt) BARNEY_CUDA_CALL(Free(gg->sbt));
      BARNEY_CUDA_CALL(Malloc((void**)&gg->sbt,hostSBT.size()));
      BARNEY_CUDA_CALL(Memcpy(gg->sbt,hostSBT.data(),hostSBT.size(),cudaMemcpyDefault));

      // count prims and build the (geomID, primID) remap, in geom order (which
      // is exactly the order we concatenate triangles / AABBs into the BLAS, so
      // HIPRT's geometry-local primID indexes straight into prims[]).
      std::vector<GeomGroup::Prim> hostPrims;
      for (size_t i=0;i<gg->geoms.size();i++) {
        int count = isTriangles
          ? ((TrianglesGeom*)gg->geoms[i])->numIndices
          : ((UserGeom*)gg->geoms[i])->primCount;
        for (int j=0;j<count;j++)
          hostPrims.push_back({ (int)i, j });
      }
      gg->numPrims = (int)hostPrims.size();
      if (gg->prims) BARNEY_CUDA_CALL(Free(gg->prims));
      if (gg->numPrims) {
        BARNEY_CUDA_CALL(Malloc((void**)&gg->prims,
                                gg->numPrims*sizeof(GeomGroup::Prim)));
        BARNEY_CUDA_CALL(Memcpy(gg->prims,hostPrims.data(),
                                gg->numPrims*sizeof(GeomGroup::Prim),
                                cudaMemcpyDefault));
      }
      return gg->numPrims;
    }

    static void buildHiprtGeometry(GeomGroup *gg, hiprtGeometryBuildInput &bi)
    {
      Device *device = gg->device;
      hiprtBuildOptions bo{};
      bo.buildFlags = hiprtBuildFlagBitPreferFastBuild;

      size_t tempSize = 0;
      HC(hiprtGetGeometryBuildTemporaryBufferSize(device->hiprtCtx,bi,bo,tempSize));
      if (tempSize > gg->buildTmpSize) {
        if (gg->d_buildTmp) BARNEY_CUDA_CALL(Free(gg->d_buildTmp));
        gg->d_buildTmp = nullptr;
        if (tempSize) BARNEY_CUDA_CALL(Malloc(&gg->d_buildTmp,tempSize));
        gg->buildTmpSize = tempSize;
      }
      if (gg->geom) { hiprtDestroyGeometry(device->hiprtCtx,gg->geom); gg->geom=nullptr; }
      HC(hiprtCreateGeometry(device->hiprtCtx,bi,bo,gg->geom));
      HC(hiprtBuildGeometry(device->hiprtCtx,hiprtBuildOperationBuild,bi,bo,
                            gg->d_buildTmp,device->stream,gg->geom));
      device->sync();
    }

    // ------------------------------------------------------------------
    TrianglesGeomGroup::TrianglesGeomGroup(Device *device,
                                           const std::vector<Geom *> &geoms)
      : GeomGroup(device,geoms)
    {}

    void TrianglesGeomGroup::buildAccel()
    {
      SetActiveGPU forDuration(device);
      GeomGroup_buildSBTandPrims(this,/*isTriangles*/true);

      // concatenate all geoms' triangles into one vertex+index buffer; HIPRT
      // primID then indexes straight into prims[].
      std::vector<vec3f> hostVerts;
      std::vector<vec3i> hostIdx;
      for (size_t i=0;i<geoms.size();i++) {
        TrianglesGeom *tg = (TrianglesGeom*)geoms[i];
        int base = (int)hostVerts.size();
        std::vector<vec3f> v(tg->numVertices);
        std::vector<vec3i> idx(tg->numIndices);
        BARNEY_CUDA_CALL(Memcpy(v.data(),tg->vertices->getDD(),
                                tg->numVertices*sizeof(vec3f),cudaMemcpyDefault));
        BARNEY_CUDA_CALL(Memcpy(idx.data(),tg->indices->getDD(),
                                tg->numIndices*sizeof(vec3i),cudaMemcpyDefault));
        for (auto &vv : v) hostVerts.push_back(vv);
        for (auto t : idx) hostIdx.push_back(vec3i{t.x+base,t.y+base,t.z+base});
      }

      if (d_vertices) { BARNEY_CUDA_CALL(Free(d_vertices)); d_vertices=nullptr; }
      if (d_indices)  { BARNEY_CUDA_CALL(Free(d_indices));  d_indices=nullptr; }
      BARNEY_CUDA_CALL(Malloc(&d_vertices,hostVerts.size()*sizeof(vec3f)));
      BARNEY_CUDA_CALL(Memcpy(d_vertices,hostVerts.data(),
                              hostVerts.size()*sizeof(vec3f),cudaMemcpyDefault));
      BARNEY_CUDA_CALL(Malloc(&d_indices,hostIdx.size()*sizeof(vec3i)));
      BARNEY_CUDA_CALL(Memcpy(d_indices,hostIdx.data(),
                              hostIdx.size()*sizeof(vec3i),cudaMemcpyDefault));

      hiprtTriangleMeshPrimitive mesh{};
      mesh.vertices       = d_vertices;
      mesh.vertexCount    = (uint32_t)hostVerts.size();
      mesh.vertexStride   = sizeof(vec3f);
      mesh.triangleIndices= d_indices;
      mesh.triangleCount  = (uint32_t)hostIdx.size();
      mesh.triangleStride = sizeof(vec3i);

      hiprtGeometryBuildInput bi{};
      bi.type = hiprtPrimitiveTypeTriangleMesh;
      bi.primitive.triangleMesh = mesh;
      bi.geomType = 0;
      buildHiprtGeometry(this,bi);
    }

    GeomGroup::DeviceRecord TrianglesGeomGroup::getRecord()
    {
      DeviceRecord r;
      r.sbt = sbt;
      r.prims = prims;
      r.sbtEntrySize = (uint32_t)sbtEntrySize;
      r.isTrianglesGroup = 1;
      return r;
    }

    // ------------------------------------------------------------------
    UserGeomGroup::UserGeomGroup(Device *device, const std::vector<Geom *> &geoms)
      : GeomGroup(device,geoms)
    {}

    void UserGeomGroup::buildAccel()
    {
      SetActiveGPU forDuration(device);
      GeomGroup_buildSBTandPrims(this,/*isTriangles*/false);

      // per-prim AABBs via each geom type's bounds kernel, concatenated in geom
      // order; HIPRT builds an AABB-list BLAS over them and calls the func-table
      // intersect thunk (see TraceKernel.cpp) per candidate prim.
      box3f *d_bounds = nullptr;
      if (numPrims) BARNEY_CUDA_CALL(Malloc((void**)&d_bounds,numPrims*sizeof(box3f)));
      size_t ofs = 0;
      for (size_t i=0;i<geoms.size();i++) {
        UserGeom *geom = (UserGeom*)geoms[i];
        int count = geom->primCount;
        if (!count) continue;
        void *mySBT = sbt + i*sbtEntrySize;
        void *geomData = ((Geom::SBTHeader *)mySBT)+1;
        ((UserGeomType*)geom->gt)->bounds(device,geomData,
                                          d_bounds+ofs,count);
        ofs += count;
      }
      device->sync();

      // HIPRT's AABB list wants packed (lower.xyz, upper.xyz) float pairs; owl's
      // box3f is exactly {vec3f lower; vec3f upper;} so the layout already
      // matches a hiprtFloat4-pair-free tight 6-float AABB.
      if (d_aabbs) { BARNEY_CUDA_CALL(Free(d_aabbs)); d_aabbs=nullptr; }
      d_aabbs = d_bounds; // box3f == 6 contiguous floats per prim

      hiprtAABBListPrimitive list{};
      list.aabbCount  = (uint32_t)numPrims;
      list.aabbStride = sizeof(box3f);
      list.aabbs      = d_aabbs;

      hiprtGeometryBuildInput bi{};
      bi.type = hiprtPrimitiveTypeAABBList;
      bi.primitive.aabbList = list;
      bi.geomType = 0;
      buildHiprtGeometry(this,bi);
    }

    GeomGroup::DeviceRecord UserGeomGroup::getRecord()
    {
      DeviceRecord r;
      r.sbt = sbt;
      r.prims = prims;
      r.sbtEntrySize = (uint32_t)sbtEntrySize;
      r.isTrianglesGroup = 0;
      return r;
    }

    // ------------------------------------------------------------------
    InstanceGroup::InstanceGroup(Device *device,
                                 const std::vector<Group *>  &groups,
                                 const std::vector<int>      &instanceIDs,
                                 const std::vector<affine3f> &xfms)
      : Group(device), groups(groups), instanceIDs(instanceIDs), xfms(xfms)
    {}

    InstanceGroup::~InstanceGroup()
    {
      SetActiveGPU forDuration(device);
      if (scene)             hiprtDestroyScene(device->hiprtCtx,scene);
      if (d_sceneTmp)        BARNEY_CUDA_CALL_NOTHROW(Free(d_sceneTmp));
      if (d_instances)       BARNEY_CUDA_CALL_NOTHROW(Free(d_instances));
      if (d_frames)          BARNEY_CUDA_CALL_NOTHROW(Free(d_frames));
      if (d_instanceRecords) BARNEY_CUDA_CALL_NOTHROW(Free(d_instanceRecords));
      if (d_deviceRecord)    BARNEY_CUDA_CALL_NOTHROW(Free(d_deviceRecord));
    }

    void InstanceGroup::setTransforms(const std::vector<affine3f> &newXfms)
    { xfms = newXfms; }

    void InstanceGroup::buildAccel()
    {
      SetActiveGPU forDuration(device);
      int numInstances = (int)groups.size();

      // per-instance shading records (transforms + the group's SBT/prim record).
      std::vector<InstanceRecord> hostRecs(numInstances);
      std::vector<hiprtInstance>    hostInst(numInstances);
      std::vector<hiprtFrameMatrix> hostFrames(numInstances);
      for (int i=0;i<numInstances;i++) {
        GeomGroup *gg = (GeomGroup*)groups[i];
        hostRecs[i].objectToWorldXfm = xfms[i];
        hostRecs[i].worldToObjectXfm = rcp(xfms[i]);
        hostRecs[i].group = gg->getRecord();
        hostRecs[i].ID = instanceIDs.empty()? (uint32_t)i : (uint32_t)instanceIDs[i];

        hostInst[i].type = hiprtInstanceTypeGeometry;
        hostInst[i].geometry = gg->getGeometry();

        // owl affine3f: linear l (vx,vy,vz columns) + translation p. HIPRT's
        // hiprtFrameMatrix is a 3x4 row-major object->world transform.
        const affine3f &m = xfms[i];
        hostFrames[i].matrix[0][0]=m.l.vx.x; hostFrames[i].matrix[0][1]=m.l.vy.x;
        hostFrames[i].matrix[0][2]=m.l.vz.x; hostFrames[i].matrix[0][3]=m.p.x;
        hostFrames[i].matrix[1][0]=m.l.vx.y; hostFrames[i].matrix[1][1]=m.l.vy.y;
        hostFrames[i].matrix[1][2]=m.l.vz.y; hostFrames[i].matrix[1][3]=m.p.y;
        hostFrames[i].matrix[2][0]=m.l.vx.z; hostFrames[i].matrix[2][1]=m.l.vy.z;
        hostFrames[i].matrix[2][2]=m.l.vz.z; hostFrames[i].matrix[2][3]=m.p.z;
      }

      if (d_instanceRecords) { BARNEY_CUDA_CALL(Free(d_instanceRecords)); d_instanceRecords=nullptr; }
      if (numInstances) {
        BARNEY_CUDA_CALL(Malloc((void**)&d_instanceRecords,numInstances*sizeof(InstanceRecord)));
        BARNEY_CUDA_CALL(Memcpy(d_instanceRecords,hostRecs.data(),
                                numInstances*sizeof(InstanceRecord),cudaMemcpyDefault));
      }
      if (d_instances) { BARNEY_CUDA_CALL(Free(d_instances)); d_instances=nullptr; }
      if (d_frames)    { BARNEY_CUDA_CALL(Free(d_frames));    d_frames=nullptr; }
      BARNEY_CUDA_CALL(Malloc(&d_instances,numInstances*sizeof(hiprtInstance)));
      BARNEY_CUDA_CALL(Memcpy(d_instances,hostInst.data(),
                              numInstances*sizeof(hiprtInstance),cudaMemcpyDefault));
      BARNEY_CUDA_CALL(Malloc(&d_frames,numInstances*sizeof(hiprtFrameMatrix)));
      BARNEY_CUDA_CALL(Memcpy(d_frames,hostFrames.data(),
                              numInstances*sizeof(hiprtFrameMatrix),cudaMemcpyDefault));

      hiprtSceneBuildInput si{};
      si.instanceCount             = (uint32_t)numInstances;
      si.instances                 = d_instances;
      si.instanceTransformHeaders  = nullptr; // one frame per instance
      si.instanceFrames            = d_frames;
      si.frameType                 = hiprtFrameTypeMatrix;

      hiprtBuildOptions bo{};
      bo.buildFlags = hiprtBuildFlagBitPreferFastBuild;
      size_t tempSize = 0;
      HC(hiprtGetSceneBuildTemporaryBufferSize(device->hiprtCtx,si,bo,tempSize));
      if (tempSize > sceneTmpSize) {
        if (d_sceneTmp) BARNEY_CUDA_CALL(Free(d_sceneTmp));
        d_sceneTmp = nullptr;
        if (tempSize) BARNEY_CUDA_CALL(Malloc(&d_sceneTmp,tempSize));
        sceneTmpSize = tempSize;
      }
      if (scene) { hiprtDestroyScene(device->hiprtCtx,scene); scene=nullptr; }
      HC(hiprtCreateScene(device->hiprtCtx,si,bo,scene));
      HC(hiprtBuildScene(device->hiprtCtx,hiprtBuildOperationBuild,si,bo,
                         d_sceneTmp,device->stream,scene));
      device->sync();

      DeviceRecord dd;
      dd.scene = scene;
      dd.instanceRecords = d_instanceRecords;
      if (!d_deviceRecord)
        BARNEY_CUDA_CALL(Malloc((void**)&d_deviceRecord,sizeof(DeviceRecord)));
      BARNEY_CUDA_CALL(Memcpy(d_deviceRecord,&dd,sizeof(DeviceRecord),cudaMemcpyDefault));
      device->sync();
      d_accel = d_deviceRecord;
    }

  }
}
