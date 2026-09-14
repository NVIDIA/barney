// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/fb/TiledFB.h"
#include "native/fb/FrameBuffer.h"
#include "native/common/math.h"

namespace BARNEY_NS { 
  namespace native {

    __rtc_global
    void linearizeColorAndNormalKernel(rtc::ComputeInterface ci,
                                       void      *out_rgba,
                                       BNDataType colorFormat,
                                       vec3f     *out_normal,
                                       float      accumScale,
                                       TileDesc  *descs,
                                       AccumTile *tiles,
                                       vec2i      numPixels)
    {
#if RTC_DEVICE_CODE
      int tileIdx = ci.getBlockIdx().x;
      TileDesc desc   = descs[tileIdx];
    
      AccumTile *tile = &tiles[tileIdx];

      int subIdx = ci.getThreadIdx().x;
      int iix = subIdx % tileSize;
      int iiy = subIdx / tileSize;
      int ix = desc.lower.x + iix;
      int iy = desc.lower.y + iiy;
      if (ix >= numPixels.x) return;
      if (iy >= numPixels.y) return;
      int idx = ix + numPixels.x*iy;

      vec4f color = tile->accum[subIdx] * accumScale;

      if (colorFormat == BN_FLOAT4) {
        ((vec4f*)out_rgba)[idx] = color;
      } else if (colorFormat == BN_UFIXED8_RGBA) {
        ((uint32_t*)out_rgba)[idx] = make_rgba(color);
      } else if (colorFormat == BN_UFIXED8_RGBA_SRGB)  {
        ((uint32_t*)out_rgba)[idx] = make_rgba(linear_to_srgb(color));
      } else
        // unsupported type!?
        ;
    
      if (out_normal)
        out_normal[idx] = tile->normal[subIdx];
#endif
    }



    __rtc_global
    void linearizeMotionTilesKernel(const rtc::ComputeInterface &ci,
                                    void              *linearOut,
                                    vec2i              numPixels,
                                    MotionChannelTile *tiles,
                                    TileDesc          *descs)
    {
#if RTC_DEVICE_CODE
      int      tileIdx = ci.getBlockIdx().x;
      TileDesc desc    = descs[tileIdx];

      int subIdx = ci.getThreadIdx().x;
      int iix = subIdx % tileSize;
      int iiy = subIdx / tileSize;
      int ix = desc.lower.x + iix;
      int iy = desc.lower.y + iiy;
      if (ix >= numPixels.x) return;
      if (iy >= numPixels.y) return;
      int idx = ix + numPixels.x*iy;

      ((vec2f*)linearOut)[idx] = tiles[tileIdx].mv[subIdx];
#endif
    }

    void TiledFB::linearizeMotionTiles(Device *device,
                                       void *linearOut,
                                       vec2i numPixels,
                                       MotionChannelTile *tilesIn,
                                       TileDesc          *descsIn,
                                       int numTiles)
    {
      SetActiveGPU forDuration(device);
      __rtc_launch(device->rtc,
                   linearizeMotionTilesKernel,
                   numTiles,pixelsPerTile,
                   linearOut,
                   numPixels,
                   tilesIn,
                   descsIn);
    }
    

    /*! take this GPU's tiles, and write those tiles' color (and
      optionally normal) channels into the linear frame buffers
      provided. The linearColor is guaranteed to be non-null, and to
      be numPixels.x*numPixels.y vec4fs; linearNormal may be
      null. Linear buffers may live on another GPU, but are
      guaranteed to be on the same node. */
    void TiledFB::linearizeColorAndNormal(void *linearColor,
                                          BNDataType colorFormat,
                                          vec3f *linearNormal,
                                          float  accumScale)
    {
      SetActiveGPU forDuration(appDevice?appDevice:device);
      if (appDevice) {
        appDevice->rtc->copyAsync(appAccumTiles,accumTiles,
                                  numActiveTilesThisGPU*sizeof(*appAccumTiles));
        appDevice->rtc->sync();
      }
      __rtc_launch(// device
                   (appDevice?appDevice:device)->rtc,
                   // kernel
                   linearizeColorAndNormalKernel,
                   // launch config
                   numActiveTilesThisGPU,pixelsPerTile,
                   // args
                   linearColor,
                   colorFormat,
                   linearNormal,
                   accumScale,
                   appDevice?appTileDescs:tileDescs,
                   appDevice?appAccumTiles:accumTiles,
                   numPixels);
    }



    __rtc_global void linearizeAuxTilesKernel(rtc::ComputeInterface ci,
                                              void           *linearOut,
                                              vec2i           numPixels,
                                              AuxChannelTile *aux,
                                              TileDesc       *descs)
    {
#if RTC_DEVICE_CODE
      int        tileIdx = ci.getBlockIdx().x;
      TileDesc   desc    = descs[tileIdx];
      
      int subIdx = ci.getThreadIdx().x;
      int iix = subIdx % tileSize;
      int iiy = subIdx / tileSize;
      int ix = desc.lower.x + iix;
      int iy = desc.lower.y + iiy;
      if (ix >= numPixels.x) return;
      if (iy >= numPixels.y) return;
      int idx = ix + numPixels.x*iy;

      ((uint32_t*)linearOut)[idx] = aux[tileIdx].ui[subIdx];
#endif
    }

    /*! linearize given array's aux tiles, on given device. this can be
      used either for local GPUs on a single node, or on the owner
      after it reveived all worker tiles */
    void TiledFB::linearizeAuxTiles(Device *device,
                                    void *linearOut,
                                    vec2i numPixels,
                                    AuxChannelTile *tilesIn,
                                    TileDesc       *descsIn,
                                    int numTiles)
    {
      SetActiveGPU forDuration(device);
      __rtc_launch(// device
                   device->rtc,
                   // kernel
                   linearizeAuxTilesKernel,
                   // launchDims
                   numTiles,pixelsPerTile,
                   // args
                   linearOut,
                   numPixels,
                   tilesIn,
                   descsIn);
    }
  

    /*! linearize _this gpu's_ channels */
    void TiledFB::linearizeAuxChannel(void *linearChannel,
                                      BNFrameBufferChannel channel)
    {
      AuxChannelTile *tgt_aux = 0;
      AuxChannelTile *loc_aux = 0;
      switch(channel) {
      case BN_FB_DEPTH:
        tgt_aux = appDevice?appAuxTiles.depth:auxTiles.depth;
        loc_aux = auxTiles.depth;
        break;
      case BN_FB_PRIMID:
        tgt_aux = appDevice?appAuxTiles.primID:auxTiles.primID;
        loc_aux = auxTiles.primID;
        break;
      case BN_FB_INSTID:
        tgt_aux = appDevice?appAuxTiles.instID:auxTiles.instID;
        loc_aux = auxTiles.instID;
        break;
      case BN_FB_OBJID:
        tgt_aux = appDevice?appAuxTiles.objID:auxTiles.objID;
        loc_aux = auxTiles.objID;
        break;
      case BN_FB_MOTION: {
        MotionChannelTile *tgt_mv = appDevice?appAuxTiles.motion:auxTiles.motion;
        MotionChannelTile *loc_mv = auxTiles.motion;
        if (loc_mv != tgt_mv)
          appDevice->rtc->copyAsync(tgt_mv,loc_mv,
                                    numActiveTilesThisGPU*sizeof(*tgt_mv));
        linearizeMotionTiles(appDevice?appDevice:device,
                             linearChannel,numPixels,
                             tgt_mv,
                             appDevice?appTileDescs:tileDescs,
                             numActiveTilesThisGPU);
        return;
      }        
      default:
        throw std::runtime_error("unsupported aux channel in sending aux!?");
      };
      if (loc_aux != tgt_aux)
        appDevice->rtc->copyAsync(tgt_aux,loc_aux,
                                  numActiveTilesThisGPU*sizeof(*tgt_aux));
      linearizeAuxTiles(appDevice?appDevice:device,
                        linearChannel,numPixels,
                        tgt_aux,
                        appDevice?appTileDescs:tileDescs,
                        numActiveTilesThisGPU);
    }

  
    TiledFB::SP TiledFB::create(Device *device,
                                /*! device for the gpu that the app
                                  lives on, if different from current
                                  GPU, and if no peer access is
                                  available */
                                Device *appDevice,
                                FrameBuffer *owner)
    {
      return std::make_shared<TiledFB>(device, appDevice, owner);
    }

    TiledFB::TiledFB(Device *device,
                     Device *appDevice,
                     FrameBuffer *owner)
      : device(device),
        appDevice(appDevice==device?nullptr:appDevice),
        owner(owner)
    {}

    TiledFB::~TiledFB()
    {
      free();
    }

    template<typename T>
    void freeAndSetNull(Device *device, T *&pMem)
    {
      if (pMem) {
        device->rtc->freeMem(pMem);
        pMem = nullptr;
      }
    }
  
    void TiledFB::free()
    {
      SetActiveGPU forDuration(device);
      freeAndSetNull(device,tileDescs);
      freeAndSetNull(device,accumTiles);
      freeAndSetNull(device,auxTiles.primID);
      freeAndSetNull(device,auxTiles.instID);
      freeAndSetNull(device,auxTiles.objID);
      freeAndSetNull(device,auxTiles.depth);
      freeAndSetNull(device,auxTiles.motion);
      
      if (appDevice) {
        SetActiveGPU forDuration(appDevice);
        freeAndSetNull(appDevice,appTileDescs);
        freeAndSetNull(appDevice,appAccumTiles);
        freeAndSetNull(appDevice,appAuxTiles.primID);
        freeAndSetNull(appDevice,appAuxTiles.instID);
        freeAndSetNull(appDevice,appAuxTiles.objID);
        freeAndSetNull(appDevice,appAuxTiles.depth);
        freeAndSetNull(device,auxTiles.motion);
      }
    }

    __rtc_global
    void setTileCoordsKernel(rtc::ComputeInterface ci,
                             TileDesc *tileDescs,
                             int numActiveTiles,
                             vec2i numTiles,
                             int globalIndex,
                             int globalIndexStep)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= numActiveTiles) return;
    
      int tileID = tid * globalIndexStep + globalIndex;
    
      int tile_x = tileID % numTiles.x;
      int tile_y = tileID / numTiles.x;
      tileDescs[tid].lower = vec2i(tile_x*tileSize,tile_y*tileSize);
#endif
    }
  
    void TiledFB::resize(uint32_t channels,
                         vec2i newSize)
    {
      free();
      SetActiveGPU forDuration(device);

      numPixels = newSize;
      numTiles  = divRoundUp(numPixels,vec2i(tileSize));
      numActiveTilesThisGPU
        = device
        ? divRoundUp(std::max(0,numTiles.x*numTiles.y - device->globalRank()),
                     device->globalSize())
        : 0;
      // ------------------------------------------------------------------
      // accum tiles
      // ------------------------------------------------------------------
      accumTiles
        = (AccumTile *)device->rtc->allocMem(numActiveTilesThisGPU * sizeof(AccumTile));
      if (appDevice) {
        SetActiveGPU forDuration(appDevice);
        appAccumTiles
          = (AccumTile *)appDevice->rtc->allocMem(numActiveTilesThisGPU * sizeof(AccumTile));
      }
      // ------------------------------------------------------------------
      // aux channel tiles
      // ------------------------------------------------------------------
      auto alloc = [&](Device *device, AuxChannelTile *&tiles) 
      {
        assert(device);
        tiles = (AuxChannelTile *)device->rtc->allocMem
          (numActiveTilesThisGPU*sizeof(*tiles));
      };
      auto allocMotion = [&](Device *device, MotionChannelTile *&tiles)
      {
        const size_t bytes = numActiveTilesThisGPU * sizeof(*tiles);
        tiles = (MotionChannelTile *)device->rtc->allocMem(bytes);
        if (tiles) device->rtc->memsetAsync(tiles, 0, bytes);
      };
      

      if (channels & BN_FB_PRIMID) alloc(device,auxTiles.primID);
      if (channels & BN_FB_INSTID) alloc(device,auxTiles.instID);
      if (channels & BN_FB_OBJID)  alloc(device,auxTiles.objID);
      if (channels & BN_FB_DEPTH)  alloc(device,auxTiles.depth);
      if (channels & BN_FB_MOTION) allocMotion(device,auxTiles.motion);      
      if (appDevice) {
        SetActiveGPU forDuration(appDevice);
        if (channels & BN_FB_PRIMID) alloc(appDevice,appAuxTiles.primID);
        if (channels & BN_FB_INSTID) alloc(appDevice,appAuxTiles.instID);
        if (channels & BN_FB_OBJID)  alloc(appDevice,appAuxTiles.objID);
        if (channels & BN_FB_DEPTH)  alloc(appDevice,appAuxTiles.depth);
        if (channels & BN_FB_MOTION) allocMotion(device,auxTiles.motion);      
      }
    
      // ------------------------------------------------------------------
      // tile descs
      // ------------------------------------------------------------------
      tileDescs
        = (TileDesc *)device->rtc->allocMem(numActiveTilesThisGPU * sizeof(TileDesc));
      if (appDevice) {
        SetActiveGPU forDuration(appDevice);
        appTileDescs
          = (TileDesc *)appDevice->rtc->allocMem(numActiveTilesThisGPU * sizeof(TileDesc));
      }
      __rtc_launch(//device
                   device->rtc,
                   // kernel
                   setTileCoordsKernel,
                   // launch config,
                   divRoundUp(numActiveTilesThisGPU,128),128,
                   // args
                   tileDescs,
                   numActiveTilesThisGPU,
                   numTiles, 
                   device->globalRank(),
                   device->globalSize());
      if (appTileDescs)
        device->rtc->copyAsync(appTileDescs,tileDescs,
                               numActiveTilesThisGPU * sizeof(TileDesc));
    }

  }
}




  
