// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/fb/FrameBuffer.h"
#include "native/common/MPIWrappers.h"

namespace BARNEY_NS {
  namespace native {

    struct MPIContext;

    /*! for now, do a 24 bit quantization; shoul eventually go
      to 32 or 16, but lets at least try how much this really helps in
      the denoiser */
    struct CompressedNormal {
      inline __rtc_both vec4f get4f() const
      { vec3f v = get(); return vec4f(v.x,v.y,v.z,0.f); }
      inline __rtc_both vec3f get3f() const
      { vec3f v = get(); return vec3f(v.x,v.y,v.z); }
      inline __rtc_both void set(vec3f v) {
        if (v == vec3f(0.f)) { x = y = z = 0; return; }
        v = normalize(v);
        x = encode(v.x);
        y = encode(v.y);
        z = encode(v.z);
      }
      inline __rtc_both vec3f get() const { return vec3f(decode(x),decode(y),decode(z)); }
    private:
      inline __rtc_both int8_t encode(float f) const {
        f = clamp(f*128.f,-127.f,+127.f);
        return int8_t(f);
      }
      inline __rtc_both float decode(int8_t i) const {
        if (i==0) return 0.f;
        return (i<0) ? (i-.5f)*(1.f/128.f) : (i+.5f)*(1.f/128.f);
      }
      int8_t x,y,z;
    };
  
    struct CompressedColorTile {
      /*! rgb are ufixed8 and need to be multiplied by scale, a is
        ufixed8 w/o scale */
      uint32_t rgba[pixelsPerTile];
      half     scale[pixelsPerTile];
    };

    struct CompressedNormalTile {
      CompressedNormal normal[pixelsPerTile];
    };
  
    struct DistFB : public FrameBuffer {
      typedef std::shared_ptr<DistFB> SP;

      DistFB(MPIContext *context,
             const DevGroup::SP &devices);
    
      virtual ~DistFB();

      struct PLD {
        /* can get tile descs and and accumtiles from
           FrameBuffer::PLD->tiledfb */
        struct {
          CompressedColorTile  *compressedColorTiles = 0;
          CompressedNormalTile *compressedNormalTiles = 0;
        } localSend;
        rtc::ComputeKernel1D *compressTiles = 0;
        rtc::ComputeKernel1D *unpackTiles = 0;
      };
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;
    
      /*! resize frame buffer to given number of pixels and the
        indicated types of channels; color will only ever get queries
        in 'colorFormat'. Channels is a bitmask compoosed of
        or'ed-together BN_FB_xyz channel flags; only those bits that
        are set may get queried by the application (ie those that are
        not set do not have to be stored or even computed */
      void resize(BNDataType colorFormat,
                  vec2i size,
                  uint32_t channels,
                  bool isAltFrame) override;

      /*! gather color (and optionally, if not null) linear normal, from
        all GPUs (and ranks). lienarColor and lienarNormal are
        device-writeable 2D linear arrays of numPixel size;
        linearcolor may be null. */
      void gatherColorChannel(/*float4 or rgba8*/void *linearColor,
                              BNDataType gatherType,
                              vec3f *linearNormal) override;
      
      /*! read one of the auxiliary (not color or normal) buffers into
        the given (device-writeable) staging area; this will at the
        least incur some reformatting from tiles to linear (if local
        node), possibly some gpu-gpu transfer (local node w/ more than
        one gpu) and possibly some mpi communication (distFB) */
      void gatherAuxChannel(BNFrameBufferChannel channel) override;
      void writeAuxChannel(void *stagingArea,
                           BNFrameBufferChannel channel) override;

      void gatherMotionChannel() override;
      void writeMotionChannel(void *stagingArea) override;
      
      /*! allocated whatever temporary tile memory we may have allocated */
      void freeChannelData();
    
      /*! @{ _receive_ staging area for gathering tiles from all
        clients; for every tile that any client sends, this has a
        linear array on rank/gpu 0/0 that will be able to store those,
        then there to be decompressed or reformatted as requried for
        final output. Only gpu 0/0 should store those buffers */
      struct {
        CompressedColorTile  *compressedColorTiles  = 0;
        CompressedNormalTile *compressedNormalTiles = 0;
        AuxTiles              auxChannelTiles;
        TileDesc             *tileDescs         = 0;
        int                   numActiveTiles    = 0;
      } gatheredTilesOnOwner;
      /*! @} */

      struct {
        std::vector<int> numTilesOnGPU;
        std::vector<int> firstTileOnGPU;
        int numGPUs;
      } ownerGather;
      // (world)rank that owns this frame buffer
      const int  owningRank = 0;
      const bool isOwner;
      // const bool ownerIsWorker;
      bool needNormals;
      MPIContext *context;
    };

  }
}
