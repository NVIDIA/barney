// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "rtcore/embree/Triangles.h"

namespace rtc {
  namespace embree {
  
    struct Group {
      Group(Device *device)
        : device(device)
      {}
      virtual ~Group() = default;
      void refitAccel() { buildAccel(); }
      virtual void buildAccel() = 0;
      virtual void setTransforms(const std::vector<affine3f> &) {}
      
      rtc::AccelHandle getDD() const
      {
        rtc::AccelHandle dd;
        (const void *&)dd = (const void *)this;
        return dd;
      }
      RTCScene embreeScene = 0;
      Device *const device;
    };
    
    inline rtc::AccelHandle getAccelHandle(Group *g)
    { return g->getDD(); }

    struct GeomGroup : public Group {
      GeomGroup(Device *device,
                const std::vector<Geom *> &geoms)
        : Group(device),
          geoms(geoms)
      {}
      
      Geom *getGeom(int geomID) 
      { assert(geomID >= 0 && geomID < geoms.size()); return geoms[geomID]; }
      
      std::vector<Geom *> geoms;
    };
      
    struct TrianglesGroup : public GeomGroup {
      TrianglesGroup(Device *device,
                     const std::vector<Geom *> &geoms);
      
      void buildAccel() override;
      
    };
  
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(Device *device,
                    const std::vector<Geom *> &geoms)
        : GeomGroup(device,geoms)
      {}

      void buildAccel() override;
    };
  
    struct InstanceGroup : public Group {
      InstanceGroup(Device *device,
                    const std::vector<Group *>  &groups,
                    const std::vector<int>      &instIDs,
                    const std::vector<affine3f> &xfms);

      GeomGroup *getGroup(int groupID);

      void buildAccel() override;
      void setTransforms(const std::vector<affine3f> &newXfms) override;
    
      std::vector<Group*>   groups;
      std::vector<affine3f> xfms;
      std::vector<affine3f> inverseXfms;
      std::vector<int>      instIDs;
    };
    
  }
    
    
}
