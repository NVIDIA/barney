// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/geometry/Triangles.h"
#include "native/ModelSlot.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_TRIANGLES_GEOM(Triangles,Triangles,Triangles::DD,true,false);

    Triangles::Triangles(Context *context, DevGroup::SP devices)
      : Geometry(context,devices)
    {}

    void Triangles::writeDD(Triangles::DD &dd, Device *device)
    {
      Geometry::writeDD(dd,device);
      dd.vertices = (const vec3f *)vertices->getDD(device);
      dd.indices = (const vec3i *)indices->getDD(device);
    }
  
    /*! handle data arrays for vertices and indices; note that
      'general' geometry attributes of the ANARI material system
      are already handled in parent class */
    bool Triangles::setData(const std::string &member,
                            const Data::SP &value)
    {
      if (Geometry::setData(member,value))
        return true;
    
      if (member == "vertices") {
        vertices = value->as<PODData>();
        return true;
      }
      if (member == "indices") {
        indices = value->as<PODData>();
        return true;
      }
    
      return false;
    }
  
    void Triangles::commit() 
    {
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        if (pld->triangleGeoms.empty()) {
          rtc::GeomType *gt
            = device->geomTypes.get(createGeomType_Triangles);
          rtc::Geom *geom = gt->createGeom();
          pld->triangleGeoms = { geom };
        }
    
        rtc::Geom *geom = pld->triangleGeoms[0];
        rtc::Buffer *verticesBuffer
          = vertices->getPLD(device)->rtcBuffer;
        rtc::Buffer *indicesBuffer
          = indices->getPLD(device)->rtcBuffer;
      
        int numVertices = (int)vertices->count;
        int numIndices  = (int)indices->count;
      
        geom->setVertices(verticesBuffer,numVertices);
        geom->setIndices(indicesBuffer,numIndices);
        Triangles::DD dd;
        writeDD(dd,device);

        // done:
        geom->setDD(&dd);
      }
    }
    
  }
}

