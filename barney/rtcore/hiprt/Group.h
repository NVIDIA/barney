// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>

#pragma once

#include "rtcore/hiprt/Device.h"
#include "rtcore/hiprt/Geom.h"
#include <hiprt/hiprt.h>

namespace BARNEY_NS {
  namespace rtc {

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
    };

    /*! a geometry group (BLAS) -- one HIPRT geometry built over the
        concatenated primitives of all its geoms. We keep barney's per-prim
        (geomID, primID) remap and per-geom SBT exactly as the cuda backend
        does, so the device-side shading dispatch (intersect/anyHit/closestHit
        function pointers) is identical; HIPRT only supplies the BVH + traversal
        that yields the geometry-local primID. */
    struct GeomGroup : public Group {
      struct Prim { int geomID; int primID; };

      /*! the device-visible per-group record stitched into each instance
          record; the trace kernel reads sbt+prims to dispatch shading. Field
          order matches rtc::cuda::GeomGroup::DeviceRecord's leading fields. */
      struct DeviceRecord {
        uint8_t *sbt;
        Prim    *prims;
        uint32_t sbtEntrySize;
        uint32_t isTrianglesGroup;
      };

      GeomGroup(Device *device, const std::vector<Geom *> &geoms);
      ~GeomGroup();

      virtual DeviceRecord getRecord() = 0;
      hiprtGeometry getGeometry() const { return geom; }

      const std::vector<Geom *> geoms;

      uint8_t *sbt          = 0;
      size_t   sbtEntrySize = 0;
      int      numPrims     = 0;
      Prim    *prims        = 0;

      // HIPRT BLAS for this group, plus the reused build-temp scratch.
      hiprtGeometry geom      = nullptr;
      void         *d_buildTmp = nullptr;
      size_t        buildTmpSize = 0;
      // device-side geometry inputs HIPRT references during traversal
      void         *d_vertices = nullptr;
      void         *d_indices  = nullptr;
      void         *d_aabbs    = nullptr;
    };

    /*! the top-level instance group (TLAS) -- a HIPRT scene over the geom
        groups' BLASes. getDD() yields a DeviceRecord with the hiprtScene handle
        and the per-instance records the trace kernel uses to map a hit back to
        a barney shading dispatch. */
    struct InstanceGroup : public Group {
      /*! per-instance record; layout mirrors the leading fields of
          rtc::cuda::InstanceGroup::InstanceRecord so the device shading path is
          identical. */
      struct InstanceRecord {
        affine3f worldToObjectXfm;
        affine3f objectToWorldXfm;
        GeomGroup::DeviceRecord group;
        uint32_t ID;
      };
      struct DeviceRecord {
        hiprtScene      scene;
        InstanceRecord *instanceRecords;
      };

      InstanceGroup(Device *device,
                    const std::vector<Group *>  &groups,
                    const std::vector<int>      &instanceIDs,
                    const std::vector<affine3f> &xfms);
      ~InstanceGroup();
      void buildAccel() override;
      void setTransforms(const std::vector<affine3f> &newXfms) override;

      DeviceRecord   *d_deviceRecord    = 0;
      InstanceRecord *d_instanceRecords = 0;

      hiprtScene scene        = nullptr;
      void      *d_sceneTmp   = nullptr;
      size_t     sceneTmpSize = 0;
      void      *d_instances  = nullptr;
      void      *d_frames     = nullptr;

      const std::vector<Group *>  groups;
      const std::vector<int>      instanceIDs;
      std::vector<affine3f>       xfms;
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
