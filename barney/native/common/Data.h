// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"

namespace BARNEY_NS {
  namespace native {
    
    rtc::DataType toRTC(BNDataType format);
  
    std::string to_string(BNDataType type);
    std::string to_string(BNFrameBufferChannel type);
    size_t owlSizeOf(BNDataType type);
  
    struct BaseData : public Data {
      typedef std::shared_ptr<BaseData> SP;

      BaseData(Context *context,
               const DevGroup::SP &devices,
               BNDataType type);
      virtual ~BaseData() = default;
    
      // virtual void set(const void *data, size_t count) = 0;
      static BaseData::SP create(Context *context,
                                 const DevGroup::SP &devices,
                                 BNDataType type);

      BNDataType type  = BN_DATA_UNDEFINED;
      size_t     count = 0;
      DevGroup::SP const devices;
    };

    /*! a data array for 'plain-old-data' type data (such as int, float,
      vec3f, etc) that does not need reference-counting for object
      lifetime handling; class-type data (any BNWhatEver) needs to be
      in ObecjtsRefData which does the refcounting */
    struct PODData : public BaseData {
      typedef std::shared_ptr<PODData> SP;
    
      /*! constructor for a 'global' data array that lives on the
        context itself, and spans all model slots */
      PODData(Context *context,
              const DevGroup::SP &devices,
              BNDataType type);
    
      virtual ~PODData();

      size_t size() const { return numBytes; }
      const void *getDD(Device *device);
      void set(const void *data, size_t count) override;
      void download(Device *device, void *hostPtr);

      struct PLD {
        rtc::Buffer *rtcBuffer   = 0;
      };
      size_t numBytes = 0;
      uint64_t revision = 0;
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;
    };

    /*! data array over reference-counted barney object handles (e.g.,
      BNTexture's, BNlight's, etc. this has to make sure that objects
      put into this data array will remain properly refcoutned. */
    struct ObjectRefsData : public BaseData {
      typedef std::shared_ptr<ObjectRefsData> SP;
      ObjectRefsData(Context *context,
                     const DevGroup::SP &devices,
                     BNDataType type);
      void set(const void *data, size_t count) override;
      std::vector<Object::SP> items;
    };

  }
}
