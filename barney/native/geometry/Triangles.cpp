// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/geometry/Triangles.h"
#include "native/geometry/TangentFrames.h"
#include <climits>
#include "native/ModelSlot.h"
#include "native/Context.h"

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_TRIANGLES_GEOM(Triangles,Triangles,Triangles::DD,true,false);

    Triangles::Triangles(Context *context, DevGroup::SP devices)
      : Geometry(context,devices), tangentBuffers(devices->numLogical,nullptr)
    {}
  
    Triangles::~Triangles()
    {
      for (auto device : *devices)
        if (auto buffer = tangentBuffers[device->contextRank()])
          device->rtc->freeBuffer(buffer);
    }

    void Triangles::writeDD(Triangles::DD &dd, Device *device)
    {
      Geometry::writeDD(dd,device);
      dd.vertices = (const vec3f *)vertices->getDD(device);
      dd.indices = (const vec3i *)indices->getDD(device);
      if (auto buffer = tangentBuffers[device->contextRank()]) {
        auto &tangent = dd.attributes.tangentAttribute;
        tangent.scope = GeometryAttribute::FACE_VARYING;
        tangent.fromArray = {buffer->getDD(),BN_FLOAT4,int(3*indices->count)};
      }
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
      if (vertices->count > INT_MAX || indices->count > (INT_MAX-1)/3)
        throw std::runtime_error("triangle mesh exceeds tangent index range");
      const std::array<PODData::SP,8> inputs = {vertices,indices,
        attributes.normalAttribute.perVertex,attributes.normalAttribute.faceVarying,
        attributes.attribute[0].perVertex,attributes.attribute[0].faceVarying,
        attributes.tangentAttribute.perVertex,attributes.tangentAttribute.faceVarying};
      std::array<uint64_t,8> revisions;
      for (size_t i = 0; i < inputs.size(); ++i)
        revisions[i] = inputs[i] ? inputs[i]->revision : 0;
      const bool regenerate = inputs != tangentInputs || revisions != tangentRevisions;
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
        if (regenerate) {
          auto &buffer = tangentBuffers[device->contextRank()];
          if (buffer) device->rtc->freeBuffer(buffer);
          buffer = nullptr;
          if (!attributes.tangentAttribute.perVertex
              && !attributes.tangentAttribute.faceVarying) {
            const auto attrs = attributes.getDD(device);
            TangentMesh mesh{(const vec3f *)vertices->getDD(device),
              (const vec3i *)indices->getDD(device),numVertices,numIndices,
              attrs.normalAttribute,attrs.attribute[0]};
            buffer = generateTangentFrames(device->rtc,mesh);
          }
        }
        writeDD(dd,device);

        // done:
        geom->setDD(&dd);
      }
      tangentInputs = inputs;
      tangentRevisions = revisions;
    }
    
  }
}

