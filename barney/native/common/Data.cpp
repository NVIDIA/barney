// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/common/Data.h"
#include "native/ModelSlot.h"
#include "native/Context.h"
#include "native/DeviceGroup.h"

namespace BARNEY_NS {
  namespace native {

    rtc::DataType toRTC(BNDataType type)
    {
      switch (type) {
      case BN_FLOAT32:      return rtc::FLOAT;
      case BN_FLOAT32_VEC2: return rtc::FLOAT2;
      case BN_FLOAT32_VEC3: return rtc::FLOAT3;
      case BN_FLOAT32_VEC4: return rtc::FLOAT4;
      
      case BN_FLOAT64:      
      case BN_FLOAT64_VEC2: 
      case BN_FLOAT64_VEC3: 
      case BN_FLOAT64_VEC4:
        throw std::runtime_error("rtc does not yet support doubles");
      
      case BN_INT32:
      case BN_UINT32:
        return rtc::INT;
      
      case BN_INT32_VEC2:
      case BN_UINT32_VEC2:
        return rtc::INT2;
      
      case BN_INT32_VEC3:
      case BN_UINT32_VEC3:
        return rtc::INT3;
      
      case BN_INT32_VEC4:
      case BN_UINT32_VEC4:
        return rtc::INT4;
      


      case BN_INT64:
        return rtc::LONG;
      
      case BN_INT64_VEC2:
        return rtc::LONG2;
      
      case BN_INT64_VEC3:
        return rtc::LONG3;
      
      case BN_INT64_VEC4:
        return rtc::LONG4;
      
      case BN_UFIXED8:
        return rtc::UCHAR;
      
      case BN_UFIXED16:
        return rtc::USHORT;
      
      case BN_UFIXED8_VEC2:
        return rtc::UCHAR2;
      
      case BN_UFIXED16_VEC2:
        return rtc::USHORT2;
      
      case BN_FLOAT16:
        return rtc::HALF;
      case BN_FLOAT16_VEC2:
        return rtc::HALF2;
      case BN_FLOAT16_VEC3:
        return rtc::HALF3;
      case BN_FLOAT16_VEC4:
        return rtc::HALF4;
      
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB:
        return rtc::UCHAR4;
      
      default: throw std::runtime_error
          ("un-recognized barney data type #"
           +std::to_string((int)type));
      };
    }
  
    std::string to_string(BNDataType type)
    {
      switch (type) {
      case BN_DATA_UNDEFINED:
        return "BN_DATA_UNDEFINED";
      case BN_DATA:
        return "BN_DATA";
      case BN_OBJECT:
        return "BN_OBJECT";
      case BN_TEXTURE:
        return "BN_TEXTURE";

      case BN_INT8:      return "BN_INT8";
      case BN_INT8_VEC2: return "BN_INT8_VEC2";
      case BN_INT8_VEC3: return "BN_INT8_VEC3";
      case BN_INT8_VEC4: return "BN_INT8_VEC4";

      case BN_UINT8:      return "BN_UINT8";
      case BN_UINT8_VEC2: return "BN_UINT8_VEC2";
      case BN_UINT8_VEC3: return "BN_UINT8_VEC3";
      case BN_UINT8_VEC4: return "BN_UINT8_VEC4";

      case BN_INT16:      return "BN_INT16";
      case BN_INT16_VEC2: return "BN_INT16_VEC2";
      case BN_INT16_VEC3: return "BN_INT16_VEC3";
      case BN_INT16_VEC4: return "BN_INT16_VEC4";

      case BN_UINT16:      return "BN_UINT16";
      case BN_UINT16_VEC2: return "BN_UINT16_VEC2";
      case BN_UINT16_VEC3: return "BN_UINT16_VEC3";
      case BN_UINT16_VEC4: return "BN_UINT16_VEC4";

      case BN_INT32:      return "BN_INT32";
      case BN_INT32_VEC2: return "BN_INT32_VEC2";
      case BN_INT32_VEC3: return "BN_INT32_VEC3";
      case BN_INT32_VEC4: return "BN_INT32_VEC4";

      case BN_UINT32:      return "BN_UINT32";
      case BN_UINT32_VEC2: return "BN_UINT32_VEC2";
      case BN_UINT32_VEC3: return "BN_UINT32_VEC3";
      case BN_UINT32_VEC4: return "BN_UINT32_VEC4";

      case BN_INT64:      return "BN_INT64";
      case BN_INT64_VEC2: return "BN_INT64_VEC2";
      case BN_INT64_VEC3: return "BN_INT64_VEC3";
      case BN_INT64_VEC4: return "BN_INT64_VEC4";

      case BN_UINT64:      return "BN_UINT64";
      case BN_UINT64_VEC2: return "BN_UINT64_VEC2";
      case BN_UINT64_VEC3: return "BN_UINT64_VEC3";
      case BN_UINT64_VEC4: return "BN_UINT64_VEC4";

      case BN_FLOAT32:      return "BN_FLOAT32";
      case BN_FLOAT32_VEC2: return "BN_FLOAT32_VEC2";
      case BN_FLOAT32_VEC3: return "BN_FLOAT32_VEC3";
      case BN_FLOAT32_VEC4: return "BN_FLOAT32_VEC4";

      case BN_FLOAT64:      return "BN_FLOAT64";
      case BN_FLOAT64_VEC2: return "BN_FLOAT64_VEC2";
      case BN_FLOAT64_VEC3: return "BN_FLOAT64_VEC3";
      case BN_FLOAT64_VEC4: return "BN_FLOAT64_VEC4";

      case BN_UFIXED8:      return "BN_UFIXED8";
      case BN_UFIXED8_VEC2: return "BN_UFIXED8_VEC2";
      case BN_UFIXED8_RGBA: return "BN_UFIXED8_RGBA";
      case BN_UFIXED8_RGBA_SRGB: return "BN_UFIXED8_RGBA_SRGB";
      case BN_UFIXED16:      return "BN_UFIXED16";
      case BN_UFIXED16_VEC2: return "BN_UFIXED16_VEC2";
      case BN_FLOAT16:      return "BN_FLOAT16";
      case BN_FLOAT16_VEC2: return "BN_FLOAT16_VEC2";
      case BN_FLOAT16_VEC3: return "BN_FLOAT16_VEC3";
      case BN_FLOAT16_VEC4: return "BN_FLOAT16_VEC4";
      
      default:
        throw std::runtime_error
          ("#bn internal error: to_string not implemented for "
           "numerical BNDataType #"+std::to_string(int(type)));
      };
    }

    std::string to_string(BNFrameBufferChannel channel)
    {
      switch(channel) {
      case BN_FB_COLOR:  return "BN_FB_COLOR";
      case BN_FB_DEPTH:  return "BN_FB_DEPTH";
      case BN_FB_PRIMID: return "BN_FB_PRIMID";
      case BN_FB_INSTID: return "BN_FB_INSTID";
      case BN_FB_OBJID:  return "BN_FB_OBJID";
      case BN_FB_NORMAL: return "BN_FB_NORMAL";
      case BN_FB_MOTION: return "BN_FB_MOTION";
      default:
        throw std::runtime_error
          ("#bn internal error: to_string not implemented for "
           "numerical BNFrameBufferChannel #"+std::to_string(int(channel)));
      };
    }
  
    size_t owlSizeOf(BNDataType type)
    {
      switch (type) {
      case BN_FLOAT32:      return sizeof(float);
      case BN_FLOAT32_VEC2: return sizeof(vec2f);
      case BN_FLOAT32_VEC3: return sizeof(vec3f);
      case BN_FLOAT32_VEC4: return sizeof(vec4f);

      case BN_INT8:      return sizeof(int8_t);
      case BN_INT8_VEC2: return sizeof(vec2c);
      case BN_INT8_VEC3: return sizeof(vec3c);
      case BN_INT8_VEC4: return sizeof(vec4c);

      case BN_UINT8:      return sizeof(uint8_t);
      case BN_UINT8_VEC2: return sizeof(vec2uc);
      case BN_UINT8_VEC3: return sizeof(vec3uc);
      case BN_UINT8_VEC4: return sizeof(vec4uc);

      case BN_INT32:      return sizeof(int32_t);
      case BN_INT32_VEC2: return sizeof(vec2i);
      case BN_INT32_VEC3: return sizeof(vec3i);
      case BN_INT32_VEC4: return sizeof(vec4i);

      case BN_UINT32:      return sizeof(uint32_t);
      case BN_UINT32_VEC2: return sizeof(vec2ui);
      case BN_UINT32_VEC3: return sizeof(vec3ui);
      case BN_UINT32_VEC4: return sizeof(vec4ui);

      case BN_INT64:
        return sizeof(int64_t);
      case BN_INT64_VEC2:
        return sizeof(vec2l);
      case BN_INT64_VEC3:
        return sizeof(vec3l);
      case BN_INT64_VEC4:
        return sizeof(vec4l);

      case BN_UINT64:
        return sizeof(uint64_t);
      case BN_UINT64_VEC2:
        return sizeof(vec2ul);
      case BN_UINT64_VEC3:
        return sizeof(vec3ul);
      case BN_UINT64_VEC4:
        return sizeof(vec4ul);

      case BN_UFIXED8:      return sizeof(uint8_t);
      case BN_UFIXED8_VEC2: return sizeof(vec2uc);
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB:
        return sizeof(vec4uc);
      case BN_UFIXED16:      return sizeof(uint16_t);
      case BN_UFIXED16_VEC2: return sizeof(vec2us);
      case BN_FLOAT16:      return sizeof(uint16_t);
      case BN_FLOAT16_VEC2: return sizeof(vec2us);
      case BN_FLOAT16_VEC3: return 3*sizeof(uint16_t);
      case BN_FLOAT16_VEC4: return 4*sizeof(uint16_t);
      
      default:
        throw std::runtime_error
          ("#bn internal error: owlSizeOf() not implemented for "
           "numerical BNDataType #"+std::to_string(int(type)));
      };
    }

    BaseData::BaseData(Context *context,
                       const DevGroup::SP &devices,
                       BNDataType type)
      : Data(context),
        type(type),
        count(0),
        devices(devices)
    {}

    void PODData::download(Device *device, void *hostPtr)
    {
      void *d_ptr = getPLD(device)->rtcBuffer->getDD();
      auto rtc = device->rtc;
      rtc->copyAsync(hostPtr,d_ptr,numBytes);
      rtc->sync();
    }
  

    void PODData::set(const void *_items, size_t count)
    {
      this->count = count;
      this->numBytes = count*owlSizeOf(type);
      for (auto device : *devices) {
        getPLD(device)->rtcBuffer->resize(count*owlSizeOf(type));
        if (numBytes && !getPLD(device)->rtcBuffer->getDD())
          throw std::bad_alloc();
        getPLD(device)->rtcBuffer->upload(_items,count*owlSizeOf(type));
      }
    }

    PODData::PODData(Context *context,
                     const DevGroup::SP &devices,
                     BNDataType type)
      : BaseData(context,devices,type)
    {
      perLogical.resize(devices->numLogical);
      for (auto device : *devices) {
        getPLD(device)->rtcBuffer 
          = device->rtc->createBuffer(1);//*owlSizeOf(type),_items);
        assert(getPLD(device)->rtcBuffer);
      }
      for (auto device : *devices)
        device->sync();
    }
  
    PODData::~PODData()
    {
      BN_TRACK_LEAKS(std::cout << "#barney: ~PODData is dying" << std::endl);
      for (auto device : *devices)
        device->rtc->freeBuffer(getPLD(device)->rtcBuffer);
    }

    const void *PODData::getDD(Device *device) 
    {
      assert(device);
      PLD *pld = getPLD(device);
      assert(pld);
      assert(pld->rtcBuffer);
      return pld->rtcBuffer->getDD();
    }
  
    PODData::PLD *PODData::getPLD(Device *device) 
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }
  
    BaseData::SP BaseData::create(Context *context,
                                  const DevGroup::SP &devices,
                                  BNDataType type)
    {
      switch(type) {
      case BN_INT8:
      case BN_INT8_VEC2:
      case BN_INT8_VEC3:
      case BN_INT8_VEC4:
      case BN_UINT8:
      case BN_UINT8_VEC2:
      case BN_UINT8_VEC3:
      case BN_UINT8_VEC4:
      case BN_INT32:
      case BN_INT32_VEC2:
      case BN_INT32_VEC3:
      case BN_INT32_VEC4:
      case BN_UINT32:
      case BN_UINT32_VEC2:
      case BN_UINT32_VEC3:
      case BN_UINT32_VEC4:
      case BN_INT64:
      case BN_INT64_VEC2:
      case BN_INT64_VEC3:
      case BN_INT64_VEC4:
      case BN_UINT64:
      case BN_UINT64_VEC2:
      case BN_UINT64_VEC3:
      case BN_UINT64_VEC4:
      case BN_FLOAT32:
      case BN_FLOAT32_VEC2:
      case BN_FLOAT32_VEC3:
      case BN_FLOAT32_VEC4:
      case BN_FLOAT64:
      case BN_FLOAT64_VEC2:
      case BN_FLOAT64_VEC3:
      case BN_FLOAT64_VEC4:
      case BN_UFIXED8:
      case BN_UFIXED8_VEC2:
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB:
      case BN_UFIXED16:
      case BN_UFIXED16_VEC2:
      case BN_FLOAT16:
      case BN_FLOAT16_VEC2:
      case BN_FLOAT16_VEC3:
      case BN_FLOAT16_VEC4:
        return std::make_shared<PODData>
          (context,devices,type);
      case BN_OBJECT:
        return std::make_shared<ObjectRefsData>
          (context,devices,type);
      default:
        throw std::runtime_error("un-implemented data type '"
                                 +to_string(type)
                                 +" in Data::create()");
      }
    }
  
    ObjectRefsData::ObjectRefsData(Context *context,
                                   const DevGroup::SP &devices,
                                   BNDataType type)
      : BaseData(context,devices,type)
    {}

    void ObjectRefsData::set(const void *_items, size_t count)
    {
      this->count = count;
      items.resize(count);
      for (int i=0;i<count;i++)
        items[i] = (((Object **)_items)[i])->shared_from_this();
    }

  }
}
