// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0/

// #define FORCE_HOST_BUILDER 1
// #define CUBQL_CPU_BUILDER_IMPLEMENTATION 1

#define CUBQL_GPU_BUILDER_IMPLEMENTATION 1

#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cuda/Device.h"
#include "rtcore/cuda/Group.h"
#include "rtcore/cuda/Geom.h"

#define BARNEY_DEVICE_PROGRAM 1
#include "rtcore/cuda/TraceInterface.h"

#include "cuBQL/bvh.h"
#include "cuBQL/builder/cuda.h"

namespace BARNEY_NS {
  namespace rtc {
    
    Group::Group(Device *device)
      : device(device)
    {}
    
    Group::~Group()
    {
      SetActiveGPU forDuration(device);
      if (bvhNodes)
        BARNEY_CUDA_CALL_NOTHROW(Free(bvhNodes));
    }

    InstanceGroup::InstanceGroup(Device *device,
                                 const std::vector<Group *>  &groups,
                                 const std::vector<int>      &instanceIDs,
                                 const std::vector<affine3f> &xfms)
      : Group(device),
        groups(groups),
        instanceIDs(instanceIDs),
        xfms(xfms)
    {}

    InstanceGroup::~InstanceGroup()
    {
      SetActiveGPU forDuration(device);
      cuBQL::DeviceMemoryResource memResource;
      if (bvh.nodes)
        cuBQL::cuda::free(bvh,device->stream,memResource);
      if (d_instanceRecords)
        BARNEY_CUDA_CALL_NOTHROW(Free(d_instanceRecords));
      if (d_deviceRecord)
        BARNEY_CUDA_CALL_NOTHROW(Free(d_deviceRecord));
    }

    GeomGroup::GeomGroup(Device *device,
                         const std::vector<Geom *> &geoms)
      : Group(device),
        geoms(geoms)
    {}

    GeomGroup::~GeomGroup()
    {
      SetActiveGPU forDuration(device);
      if (sbt)
        BARNEY_CUDA_CALL_NOTHROW(Free(sbt));
      if (prims)
        BARNEY_CUDA_CALL_NOTHROW(Free(prims));
      // bvhNodes is freed by ~Group()
    }

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
      
      BARNEY_CUDA_SYNC_CHECK();
      
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
#if FORCE_HOST_BUILDER
      BARNEY_CUDA_SYNC_CHECK();
      int numPrims = numInstances;
      std::vector<cuBQL::box3f> h_boxes(numPrims);
      BARNEY_CUDA_CALL(Memcpy(h_boxes.data(),
                              instBounds,
                              numPrims*sizeof(*instBounds),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      cuBQL::cpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)h_boxes.data(),
                        numPrims,
                        buildConfig);
      typedef typename cuBQL::BinaryBVH<float,3>::node_t node3f;
      node3f *d_nodes;
      BARNEY_CUDA_CALL(Malloc((void **)&d_nodes,
                              bvh.numNodes*sizeof(*d_nodes)));
      BARNEY_CUDA_CALL(Memcpy(d_nodes,bvh.nodes,
                              bvh.numNodes*sizeof(*d_nodes),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      uint32_t *d_primIDs;
      BARNEY_CUDA_CALL(Malloc((void **)&d_primIDs,
                              numPrims*sizeof(*d_primIDs)));
      BARNEY_CUDA_CALL(Memcpy(d_primIDs,bvh.primIDs,
                              numPrims*sizeof(*d_primIDs),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();

      delete[] bvh.nodes; bvh.nodes = d_nodes;
      delete[] bvh.primIDs; bvh.primIDs = d_primIDs;
      
#else
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)instBounds,
                        numInstances,
                        buildConfig,
                        device->stream,
                        memResource);
#endif
      device->sync();
      BARNEY_CUDA_CALL(Free(instBounds));
      
      // ------------------------------------------------------------------
      // allocate device descriptor
      // ------------------------------------------------------------------
      DeviceRecord dd;
      dd.bvh.nodes = bvh.nodes;
      dd.bvh.primIDs = bvh.primIDs;
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
      BARNEY_CUDA_SYNC_CHECK();
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

      BARNEY_CUDA_SYNC_CHECK();
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

      BARNEY_CUDA_SYNC_CHECK();
      device->sync();

      // ------------------------------------------------------------------
      // alloc and write triangle bboxes 
      // ------------------------------------------------------------------
      box3f *primBounds = 0;
      BARNEY_CUDA_CALL(Malloc((void**)&primBounds,numPrims*sizeof(box3f)));
      
      BARNEY_CUDA_SYNC_CHECK();
      createTriangleBounds
        <<<divRoundUp(numPrims,128),128,0,device->stream>>>
        (primBounds,sbt,sbtEntrySize,prims,numPrims);
      
      BARNEY_CUDA_SYNC_CHECK();
      device->sync();
      BARNEY_CUDA_SYNC_CHECK();
      
      // ------------------------------------------------------------------
      // build the bvh
      // ------------------------------------------------------------------
      if (this->bvhNodes) {
        BARNEY_CUDA_CALL(Free(this->bvhNodes));
        this->bvhNodes = 0;
      }
      cuBQL::bvh3f bvh;
      cuBQL::BuildConfig buildConfig;
#if FORCE_HOST_BUILDER
      BARNEY_CUDA_SYNC_CHECK();
      std::vector<cuBQL::box3f> h_boxes(numPrims);
      BARNEY_CUDA_CALL(Memcpy(h_boxes.data(),
                              primBounds,
                              numPrims*sizeof(*primBounds),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      cuBQL::cpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)h_boxes.data(),
                        numPrims,
                        buildConfig);
      typedef typename cuBQL::BinaryBVH<float,3>::node_t node3f;
      node3f *d_nodes;
      BARNEY_CUDA_CALL(Malloc((void **)&d_nodes,
                              bvh.numNodes*sizeof(*d_nodes)));
      BARNEY_CUDA_CALL(Memcpy(d_nodes,bvh.nodes,
                              bvh.numNodes*sizeof(*d_nodes),
                              cudaMemcpyDefault));
      uint32_t *d_primIDs;
      BARNEY_CUDA_CALL(Malloc((void **)&d_primIDs,
                              numPrims*sizeof(*d_primIDs)));
      BARNEY_CUDA_CALL(Memcpy(d_primIDs,bvh.primIDs,
                              numPrims*sizeof(*d_primIDs),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      delete[] bvh.nodes; bvh.nodes = d_nodes;
      delete[] bvh.primIDs; bvh.primIDs = d_primIDs;
#else
      buildConfig.maxAllowedLeafSize = 4;
      cuBQL::DeviceMemoryResource memResource;

      BARNEY_CUDA_SYNC_CHECK();

# if 1
      cuBQL::sahBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        buildConfig,
                        device->stream,
                        memResource);
# else
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        buildConfig,
                        device->stream,
                        memResource);
# endif
      device->sync();
#endif
      BARNEY_CUDA_CALL(Free(primBounds));
      
      // ------------------------------------------------------------------
      // reorder prims, store bvh, and release what we no longer need
      // ------------------------------------------------------------------
      this->bvhNodes = bvh.nodes;
      GeomGroup::Prim *reorderedPrims = 0;
      BARNEY_CUDA_CALL(Malloc((void**)&reorderedPrims,numPrims*sizeof(GeomGroup::Prim)));
      reorderPrims
        <<<divRoundUp(numPrims,128),128,0,device->stream>>>
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
      // buildConfig.maxAllowedLeafSize = 4;
      buildConfig.maxAllowedLeafSize = 1;
      // buildConfig.enableSAH();
      // buildConfig.makeLeafThreshold = 4;

#if FORCE_HOST_BUILDER
      BARNEY_CUDA_SYNC_CHECK();
      std::vector<cuBQL::box3f> h_boxes(numPrims);
      BARNEY_CUDA_CALL(Memcpy(h_boxes.data(),
                              primBounds,
                              numPrims*sizeof(*primBounds),
                              cudaMemcpyDefault));
      BARNEY_CUDA_SYNC_CHECK();
      cuBQL::cpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)h_boxes.data(),
                        numPrims,
                        buildConfig);
      typedef typename cuBQL::BinaryBVH<float,3>::node_t node3f;
      node3f *d_nodes;
      BARNEY_CUDA_CALL(Malloc((void **)&d_nodes,
                              bvh.numNodes*sizeof(*d_nodes)));
      BARNEY_CUDA_CALL(Memcpy(d_nodes,bvh.nodes,
                              bvh.numNodes*sizeof(*d_nodes),
                              cudaMemcpyDefault));
      uint32_t *d_primIDs;
      BARNEY_CUDA_CALL(Malloc((void **)&d_primIDs,
                              numPrims*sizeof(*d_primIDs)));
      BARNEY_CUDA_CALL(Memcpy(d_primIDs,bvh.primIDs,
                              numPrims*sizeof(*d_primIDs),
                              cudaMemcpyDefault));

      delete[] bvh.nodes; bvh.nodes = d_nodes;
      delete[] bvh.primIDs; bvh.primIDs = d_primIDs;
      
#else
# if 0
      cuBQL::cuda::sahBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        buildConfig,
                        device->stream,
                        memResource);
# else
      cuBQL::gpuBuilder(bvh,
                        (const cuBQL::box_t<float,3>*)primBounds,
                        numPrims,
                        buildConfig,
                        device->stream,
                        memResource);
# endif
      device->sync();
#endif
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
      bvh.primIDs = 0;

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
