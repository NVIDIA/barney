// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/cudaCommon/Device.h"
#include <cuBQL/bvh.h>

namespace rtc {
  namespace cuda {
    
    using cuBQL::bvh3f;

    struct Device;
    
    struct Group {
      Group(Device *device);
      virtual ~Group();
      virtual void buildAccel() = 0;
      virtual void refitAccel() { buildAccel(); }
      virtual void setTransforms(const std::vector<affine3f> &) {}
      AccelHandle getDD() const { return (AccelHandle)d_accel; }

      Device *const device;
      void *d_accel = 0;
      cuBQL::bvh3f::Node *bvhNodes = 0;
    };

    struct GeomGroup : public Group {
      struct Prim { int geomID; int primID; };

      struct DeviceRecord {
        // 0..8b
        uint8_t *sbt;
        // 8..16b
        cuBQL::bvh3f::Node *bvhNodes;
        // 16..24b
        Prim *prims;
        // 24..28b
        uint32_t sbtEntrySize;
        // 28b..32b
        bool isTrianglesGroup;
      };

      virtual DeviceRecord getRecord() = 0;
      GeomGroup(Device *device,
                const std::vector<Geom *> &geoms);
      const std::vector<Geom *> geoms;

      uint8_t *sbt          = 0;
      size_t   sbtEntrySize = 0;
      
      int    numPrims = 0;
      Prim  *prims    = 0;
    };

    struct InstanceGroup : public Group {
      struct InstanceRecord {
        affine3f worldToObjectXfm;
        affine3f objectToWorldXfm;
        GeomGroup::DeviceRecord group;
        uint32_t ID;
      };
      struct DeviceRecord {
        cuBQL::bvh3f bvh;
        InstanceRecord *instanceRecords;
      };

      InstanceGroup(Device *device,
                    const std::vector<Group *>  &groups,
                    const std::vector<int>      &instanceIDs,
                    const std::vector<affine3f> &xfms);
      void buildAccel() override;
      void setTransforms(const std::vector<affine3f> &newXfms) override;

      DeviceRecord   *d_deviceRecord = 0;
      InstanceRecord *d_instanceRecords = 0;
      bvh3f bvh = { 0,0,0,0 };
      const std::vector<Group *>  groups;
      const std::vector<int>      instanceIDs;
      std::vector<affine3f> xfms;
    };
    
    struct TrianglesGeomGroup : public GeomGroup {
      TrianglesGeomGroup(Device *device, const std::vector<Geom *> &geoms);
      void buildAccel() override;
      DeviceRecord getRecord() override;
    };
    
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(Device *device, const std::vector<Geom *> &geoms);
      void buildAccel() override;
      DeviceRecord getRecord() override;
    };
    
  }
}
