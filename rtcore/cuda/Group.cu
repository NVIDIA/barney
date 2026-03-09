// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Group.h"
#include "rtcore/cuda/Geom.h"

#define BARNEY_DEVICE_PROGRAM 1
#include "rtcore/cuda/TraceInterface.h"

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"

namespace rtc {
  namespace cuda {
    
    Group::Group(Device *device)
      : device(device)
    {}
    
    Group::~Group()
    {}

    InstanceGroup::InstanceGroup(Device *device,
                                 const std::vector<Group *>  &groups,
                                 const std::vector<int>      &instanceIDs,
                                 const std::vector<affine3f> &xfms)
      : Group(device),
        groups(groups),
        instanceIDs(instanceIDs),
        xfms(xfms)
    {}

    GeomGroup::GeomGroup(Device *device,
                         const std::vector<Geom *> &geoms)
      : Group(device),
        geoms(geoms)
    {}

    TrianglesGeomGroup::TrianglesGeomGroup(Device *device,
                                           const std::vector<Geom *> &geoms)
      : GeomGroup(device, geoms)
    {}

    __global__
    void writePrims(GeomGroup::Prim *prims,
                    int meshID,
                    int count)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= count) return;
      GeomGroup::Prim prim;
      prim.geomID = meshID;
      prim.primID = tid;
      prims[tid] = prim;
    }

    __global__
    void createTriangleBounds(box3f *primBounds,
                              const uint8_t *groupSBT,
                              size_t sbtEntrySize,
                              GeomGroup::Prim *prims,
                              int numPrims)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numPrims) return;

      GeomGroup::Prim prim = prims[tid];
      int primID = prim.primID;
      const uint8_t *mySBT = groupSBT + prim.geomID * sbtEntrySize;
      const Geom::SBTHeader *header = (const Geom::SBTHeader *)mySBT;
      vec3i idx = header->triangles.indices[primID];
      vec3f v0 = header->triangles.vertices[idx.x];
      vec3f v1 = header->triangles.vertices[idx.y];
      vec3f v2 = header->triangles.vertices[idx.z];
      box3f bb = box3f().including(v0).including(v1).including(v2);

      primBounds[tid] = bb;
    }
    
    __global__
    void reorderPrims(GeomGroup::Prim *out,
                      GeomGroup::Prim *in,
                      const uint32_t *primIDs,
                      int numPrims)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numPrims) return;
      out[tid] = in[primIDs[tid]];
    }

    __global__
    void computeInstanceBounds(box3f *instBounds,
                               InstanceGroup::InstanceRecord *instances,
                               int numInstances)
    {
      int tid = threadIdx.x+blockIdx.x*blockDim.x;
      if (tid >= numInstances) return;

      box3f bounds;
      InstanceGroup::InstanceRecord inst = instances[tid];
      bounds = (const box3f&)inst.group.bvhNodes[0].bounds;
      instBounds[tid] = xfmBounds(inst.objectToWorldXfm,bounds);
    }

    void InstanceGroup::setTransforms(const std::vector<affine3f> &newXfms)
    {
      xfms = newXfms;
    }

    void InstanceGroup::buildAccel()
    {
      SetActiveGPU forDuration(device);

      // ------------------------------------------------------------------
      // create and upload instance records
      // ------------------------------------------------------------------
      int numInstances = groups.size();
      std::vector<InstanceRecord> h_instances(numInstances);
      for (int instID=0;instID<numInstances;instID++) {
        auto &inst = h_instances[instID];
        inst.group = ((GeomGroup*)groups[instID])->getRecord();
        inst.worldToObjectXfm = rcp(xfms[instID]);
        inst.objectToWorldXfm = xfms[instID];
        inst.ID = instanceIDs.empty()?instID:instanceIDs[instID];
      }
      if (d_instanceRecords) {
        BARNEY_CUDA_CALL(Free(d_instanceRecords));
        d_instanceRecords = 0;
      }
      if (numInstances) {
        BARNEY_CUDA_CALL(Malloc((void **)&d_instanceRecords,
                                numInstances*sizeof(InstanceRecord)));
        BARNEY_CUDA_CALL(Memcpy(d_instanceRecords,h_instances.data(),
                                numInstances*sizeof(InstanceRecord),
                                cudaMemcpyDefault));
      }
      h_instances.clear();
      device->sync();
      
      // ------------------------------------------------------------------
      // compute bounds for bvh constuction
      // ------------------------------------------------------------------
      box3f *instBounds = 0;
      if (numInstances) {
        BARNEY_CUDA_CALL(Malloc((void **)&instBounds,
                                numInstances*sizeof(box3f)));
        computeInstanceBounds<<<divRoundUp(numInstances,1024),1024,0,device->stream>>>
          (instBounds,d_instanceRecords,numInstances);
      }
      device->sync();

      // ------------------------------------------------------------------
      // build the bvh
      // ------------------------------------------------------------------
      cuBQL::DeviceMemoryResource memResource;
      if (bvh.nodes) {
        cuBQL::cuda::free(bvh,device->stream,memResource);
        bvh.nodes = 0;
        bvh.primIDs = 0;
      }
      cuBQL::BuildConfig buildConfig;
      buildConfig.maxAllowedLeafSize = 1;
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)instBounds,
                        numInstances,
                        buildConfig,
                        device->stream,
                        memResource);
      device->sync();
      BARNEY_CUDA_CALL(Free(instBounds));
      
      // ------------------------------------------------------------------
      // allocate device descriptor
      // ------------------------------------------------------------------
      DeviceRecord dd;
      dd.bvh.nodes = bvh.nodes;
      dd.bvh.primIDs = bvh.primIDs;
      bvh.nodes = 0;
      bvh.primIDs = 0;
      dd.instanceRecords = d_instanceRecords;

      if (!d_deviceRecord)
        BARNEY_CUDA_CALL(Malloc((void **)&d_deviceRecord,
                                sizeof(DeviceRecord)));
      BARNEY_CUDA_CALL(Memcpy(d_deviceRecord,&dd,
                              sizeof(DeviceRecord),
                              cudaMemcpyDefault));
      device->sync();
      d_accel = d_deviceRecord;
    }
    
    
    void TrianglesGeomGroup::buildAccel() 
    {
      SetActiveGPU forDuration(device);

      // ------------------------------------------------------------------
      // compute SBT entry size
      // ------------------------------------------------------------------
      sbtEntrySize = 0;
      size_t align = sizeof(float4);
      for (int i=0;i<geoms.size();i++) {
        TrianglesGeom *geom = (TrianglesGeom *)geoms[i];
        sbtEntrySize = std::max(sbtEntrySize,geom->gt->sizeOfDD);
        sbtEntrySize = align*divRoundUp(sbtEntrySize,align);
      }
      sbtEntrySize += sizeof(Geom::SBTHeader);

      // ------------------------------------------------------------------
      // write SBT, and upload
      // ------------------------------------------------------------------
      std::vector<uint8_t> hostSBT(geoms.size()*sbtEntrySize);
      uint8_t *sbtPointer = hostSBT.data();
      for (int i=0;i<geoms.size();i++) {
        TrianglesGeom *geom = (TrianglesGeom *)geoms[i];
        Geom::SBTHeader *header
          = (Geom::SBTHeader *)sbtPointer;
        header->ah = ((TrianglesGeomType*)geom->gt)->ah;
        header->ch = ((TrianglesGeomType*)geom->gt)->ch;
        header->triangles.vertices = (const vec3f*)geom->vertices->getDD();
        header->triangles.indices  = (const vec3i*)geom->indices->getDD();
        assert(header->triangles.vertices);
        assert(header->triangles.indices);
        memcpy(sbtPointer+sizeof(Geom::SBTHeader),
               geom->data.data(),geom->data.size());
        sbtPointer += sbtEntrySize;
      }
      if (sbt) BARNEY_CUDA_CALL(Free(sbt));
      BARNEY_CUDA_CALL(Malloc((void**)&sbt,hostSBT.size()));
      BARNEY_CUDA_CALL(Memcpy(sbt,hostSBT.data(),hostSBT.size(),cudaMemcpyDefault));
      
      // ------------------------------------------------------------------
      // count prims and alloc geom/prim descriptors
      // ------------------------------------------------------------------
      numPrims = 0;
      for (int i=0;i<geoms.size();i++) {
        TrianglesGeom *geom = (TrianglesGeom *)geoms[i];
        assert(geom->numIndices);
        numPrims += geom->numIndices;
      }
      if (prims) BARNEY_CUDA_CALL(Free(prims));
      BARNEY_CUDA_CALL(Malloc((void**)&prims,numPrims*sizeof(Prim)));

      // ------------------------------------------------------------------
      // write geom/prim descriptors
      // ------------------------------------------------------------------
      size_t ofs = 0;
      for (int i=0;i<geoms.size();i++) {
        TrianglesGeom *geom = (TrianglesGeom *)geoms[i];
        int count = geom->numIndices;
        writePrims<<<divRoundUp(count,1024),1024,0,device->stream>>>
          (prims+ofs,i,count);
        ofs += count;
      }

      device->sync();

      // ------------------------------------------------------------------
      // alloc and write triangle bboxes 
      // ------------------------------------------------------------------
      box3f *primBounds = 0;
      BARNEY_CUDA_CALL(Malloc((void**)&primBounds,numPrims*sizeof(box3f)));
      
      createTriangleBounds
        <<<divRoundUp(numPrims,1024),1024,0,device->stream>>>
        (primBounds,sbt,sbtEntrySize,prims,numPrims);
      
      device->sync();
      
      // ------------------------------------------------------------------
      // build the bvh
      // ------------------------------------------------------------------
      if (this->bvhNodes) {
        BARNEY_CUDA_CALL(Free(this->bvhNodes));
        this->bvhNodes = 0;
      }
      cuBQL::bvh3f bvh;
      cuBQL::DeviceMemoryResource memResource;
      cuBQL::BuildConfig buildConfig;
      buildConfig.maxAllowedLeafSize = 4;
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        buildConfig,
                        device->stream,
                        memResource);
      device->sync();
      BARNEY_CUDA_CALL(Free(primBounds));
      
      // ------------------------------------------------------------------
      // reorder prims, store bvh, and release what we no longer need
      // ------------------------------------------------------------------
      this->bvhNodes = bvh.nodes;
      GeomGroup::Prim *reorderedPrims = 0;
      BARNEY_CUDA_CALL(Malloc((void**)&reorderedPrims,numPrims*sizeof(GeomGroup::Prim)));
      reorderPrims
        <<<divRoundUp(numPrims,1024),1024,0,device->stream>>>
        (reorderedPrims,prims,bvh.primIDs,numPrims);
      
      device->sync();
      BARNEY_CUDA_CALL(Free(prims));
      this->prims = reorderedPrims;

      BARNEY_CUDA_CALL(Free(bvh.primIDs));
      bvh.primIDs = 0;
    }
    



    void UserGeomGroup::buildAccel() 
    {
      SetActiveGPU forDuration(device);

      // ------------------------------------------------------------------
      // compute SBT entry size
      // ------------------------------------------------------------------
      sbtEntrySize = 0;
      size_t align = sizeof(float4);
      for (int i=0;i<geoms.size();i++) {
        UserGeom *geom = (UserGeom *)geoms[i];
        sbtEntrySize = std::max(sbtEntrySize,geom->gt->sizeOfDD);
        sbtEntrySize = align*divRoundUp(sbtEntrySize,align);
      }
      sbtEntrySize += sizeof(Geom::SBTHeader);

      // ------------------------------------------------------------------
      // write SBT, and upload
      // ------------------------------------------------------------------
      std::vector<uint8_t> hostSBT(geoms.size()*sbtEntrySize);
      uint8_t *sbtPointer = hostSBT.data();
      for (int i=0;i<geoms.size();i++) {
        UserGeom *geom = (UserGeom *)geoms[i];
        Geom::SBTHeader *header
          = (Geom::SBTHeader *)sbtPointer;
        header->ah = ((UserGeomType*)geom->gt)->ah;
        header->ch = ((UserGeomType*)geom->gt)->ch;
        header->user.intersect = ((UserGeomType*)geom->gt)->intersect;
        // header->user.bounds = ((UserGeomType*)geom->gt)->bounds;
        memcpy(sbtPointer+sizeof(Geom::SBTHeader),
               geom->data.data(),geom->data.size());
        sbtPointer += sbtEntrySize;
      }
      if (sbt) BARNEY_CUDA_CALL(Free(sbt));
      BARNEY_CUDA_CALL(Malloc((void**)&sbt,hostSBT.size()));
      BARNEY_CUDA_CALL(Memcpy(sbt,hostSBT.data(),hostSBT.size(),cudaMemcpyDefault));
      
      // ------------------------------------------------------------------
      // count prims and alloc geom/prim descriptors
      // ------------------------------------------------------------------
      numPrims = 0;
      for (int i=0;i<geoms.size();i++) {
        UserGeom *geom = (UserGeom *)geoms[i];
        numPrims += geom->primCount;
      }
      
      if (prims) BARNEY_CUDA_CALL(Free(prims));
      BARNEY_CUDA_CALL(Malloc((void**)&prims,numPrims*sizeof(Prim)));

      // ------------------------------------------------------------------
      // write geom/prim descriptors and bounding boxes
      // ------------------------------------------------------------------
      box3f *primBounds = 0;
      BARNEY_CUDA_CALL(Malloc((void**)&primBounds,numPrims*sizeof(box3f)));
      
      size_t ofs = 0;
      for (int i=0;i<geoms.size();i++) {
        UserGeom *geom = (UserGeom *)geoms[i];
        int count = geom->primCount;
        if (!count) continue;

        writePrims<<<divRoundUp(count,1024),1024,0,device->stream>>>
          (prims+ofs,i,count);
        
        void *mySBT = sbt+i*sbtEntrySize;
        void *geomData = ((Geom::SBTHeader *)mySBT)+1;
        ((UserGeomType*)geom->gt)->bounds(device,
                                          geomData,
                                          primBounds+ofs,
                                          count);
        ofs += count;
      }

      device->sync();

      // ------------------------------------------------------------------
      // build the bvh
      // ------------------------------------------------------------------
      if (this->bvhNodes) {
        BARNEY_CUDA_CALL(Free(this->bvhNodes));
        this->bvhNodes = 0;
      }
      cuBQL::bvh3f bvh;
      cuBQL::DeviceMemoryResource memResource;
      cuBQL::BuildConfig buildConfig;
      buildConfig.maxAllowedLeafSize = 4;
      buildConfig.enableSAH();
      // buildConfig.makeLeafThreshold = 4;
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        buildConfig,
                        device->stream,
                        memResource);
      device->sync();
      BARNEY_CUDA_CALL(Free(primBounds));
      
      // ------------------------------------------------------------------
      // reorder prims, store bvh, and release what we no longer need
      // ------------------------------------------------------------------
      this->bvhNodes = bvh.nodes;
      GeomGroup::Prim *reorderedPrims = 0;
      BARNEY_CUDA_CALL(Malloc((void**)&reorderedPrims,numPrims*sizeof(GeomGroup::Prim)));
      reorderPrims
        <<<divRoundUp(numPrims,1024),1024,0,device->stream>>>
        (reorderedPrims,prims,bvh.primIDs,numPrims);
      
      device->sync();
      BARNEY_CUDA_CALL(Free(prims));
      this->prims = reorderedPrims;

      BARNEY_CUDA_CALL(Free(bvh.primIDs));
    }
    
    UserGeomGroup::UserGeomGroup(Device *device,
                                 const std::vector<Geom *> &geoms)
      : GeomGroup(device, geoms)
    {}

    GeomGroup::DeviceRecord TrianglesGeomGroup::getRecord() 
    {
      DeviceRecord record;
      record.isTrianglesGroup = true;
      record.sbtEntrySize = sbtEntrySize;
      record.sbt = sbt;
      record.prims = prims;
      record.bvhNodes = bvhNodes;
      return record;
    }
    
    GeomGroup::DeviceRecord UserGeomGroup::getRecord() 
    {
      DeviceRecord record;
      record.isTrianglesGroup = false;
      record.sbtEntrySize = sbtEntrySize;
      record.sbt = sbt;
      record.prims = prims;
      record.bvhNodes = bvhNodes;
      return record;
    }

  }
}
