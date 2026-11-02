// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/native/fb/DistFB.h"
#include "barney/native/MPIContext.h"
#include "barney/native/common/math.h"

namespace BARNEY_NS {
  namespace native {

    RTC_IMPORT_COMPUTE1D(compressTiles);
    RTC_IMPORT_COMPUTE1D(unpackTiles);

    inline __rtc_device float from_8bit(uint8_t v) {
      return float(v) * (1.f/255.f);
    }
  
    inline __rtc_device vec4f from_8bit(uint32_t v) {
      return vec4f(from_8bit(uint8_t((v >> 0)&0xff)),
                   from_8bit(uint8_t((v >> 8)&0xff)),
                   from_8bit(uint8_t((v >> 16)&0xff)),
                   from_8bit(uint8_t((v >> 24)&0xff)));
    }
  
    struct UnpackTiles {
      vec2i numPixels;
      void                 *out_rgba;
      BNDataType            colorChannelFormat;
      vec3f                *out_normal;
      CompressedColorTile  *in_color;
      CompressedNormalTile *in_normal;
      TileDesc             *descs;
    
#if RTC_DEVICE_CODE
      __rtc_device void run(const rtc::ComputeInterface &ci);
#endif
    };

#if RTC_DEVICE_CODE
    __rtc_device void UnpackTiles::run(const rtc::ComputeInterface &ci)
    {
      int tileIdx = ci.getBlockIdx().x;
      
      const CompressedColorTile  &colorTile  = in_color[tileIdx];
      const CompressedNormalTile &normalTile = in_normal[tileIdx];
      const TileDesc        desc = descs[tileIdx];
      
      int subIdx = ci.getThreadIdx().x;
      int iix = subIdx % tileSize;
      int iiy = subIdx / tileSize;
      int ix = desc.lower.x + iix;
      int iy = desc.lower.y + iiy;
      if (ix >= numPixels.x) return;
      if (iy >= numPixels.y) return;
      int idx = ix + numPixels.x*iy;
    
      uint32_t rgba8 = colorTile.rgba[subIdx];
      vec4f rgba = from_8bit(rgba8);
      float scale = float(colorTile.scale[subIdx]);
      rgba.x *= scale;
      rgba.y *= scale;
      rgba.z *= scale;

      if (colorChannelFormat == BN_FLOAT4) 
        ((vec4f*)out_rgba)[idx] = rgba;
      else if (colorChannelFormat == BN_UFIXED8_RGBA) 
        ((uint32_t*)out_rgba)[idx] = make_rgba(rgba);
      else if (colorChannelFormat == BN_UFIXED8_RGBA_SRGB) 
        ((uint32_t*)out_rgba)[idx] = make_rgba(linear_to_srgb(rgba));
      else
        // unsupported type!?
        ;

      // and write normal, too, if so required
      if (out_normal && in_normal) {
        vec3f normal = normalTile.normal[subIdx].get3f();
        out_normal[idx] = normal;
      }
    }
#endif
  
    struct CompressTiles {
      CompressedColorTile  *out_color;
      CompressedNormalTile *out_normal;
      AccumTile            *localTiles;
      float                 accumScale;
    
#if RTC_DEVICE_CODE
      __rtc_device void run(const rtc::ComputeInterface &ci);
#endif
    };

#if RTC_DEVICE_CODE
    __rtc_device void CompressTiles::run(const rtc::ComputeInterface &ci)
    {
      int pixelID = ci.getThreadIdx().x;
      int tileID  = ci.getBlockIdx().x;

      vec4f color = vec4f(localTiles[tileID].accum[pixelID])*accumScale;
      vec4f org = color;
      float scale = reduce_max((const vec3f&)color);
      (vec3f&)color *= 1.f/scale;
      out_color[tileID].scale[pixelID] = scale;
      out_color[tileID].rgba[pixelID]  = make_rgba(color);
      if (out_normal) {
        out_normal[tileID].normal[pixelID]
          .set(localTiles[tileID].normal[pixelID]);
      }
    }
#endif


    void DistFB::gatherMotionChannel()
    {
      std::vector<MPI_Request> recv_requests(isOwner ? ownerGather.numGPUs : 0);
      std::vector<MPI_Request> send_requests(context->isActiveWorker ? devices->size() : 0);

      if (isOwner) {
        MotionChannelTile *mv_recv = gatheredTilesOnOwner.auxChannelTiles.motion;
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          auto thisDev = &context->topo->allDevices[ggID];
          context->world.recv(thisDev->worldRank, thisDev->local,
                              mv_recv + ownerGather.firstTileOnGPU[ggID],
                              ownerGather.numTilesOnGPU[ggID],
                              recv_requests[ggID]);
        }
      }
      if (context->isActiveWorker) {
        for (auto device : *devices) {
          SetActiveGPU forDuration(device);
          auto tiledFB = getFor(device);
          MotionChannelTile *mv_send = tiledFB->auxTiles.motion;
          context->world.send(owningRank, device->contextRank(),
                              mv_send,
                              tiledFB->numActiveTilesThisGPU,
                              send_requests[device->contextRank()]);
        }
      }
      if (isOwner)
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++)
          context->world.wait(recv_requests[ggID]);
      if (context->isActiveWorker)
        for (auto device : *devices)
          context->world.wait(send_requests[device->contextRank()]);
    }

    void DistFB::writeMotionChannel(void *stagingArea)
    {
      if (!isOwner) return;
      auto device = getDenoiserDevice();
      TiledFB::linearizeMotionTiles(device,
                                    stagingArea, numPixels,
                                    gatheredTilesOnOwner.auxChannelTiles.motion,
                                    gatheredTilesOnOwner.tileDescs,
                                    gatheredTilesOnOwner.numActiveTiles);
      device->sync();
    }
  
    
    /*! read one of the auxiliary (not color or normal) buffers into
      the given (device-writeable) staging area; this will at the
      least incur some reformatting from tiles to linear (if local
      node), possibly some gpu-gpu transfer (local node w/ more than
      one gpu) and possibly some mpi communication (distFB) */
    void DistFB::gatherAuxChannel(BNFrameBufferChannel channel)
    {
      // ------------------------------------------------------------------
      // gather all (packed) tiles from all clients
      // ------------------------------------------------------------------
      std::vector<MPI_Request> recv_requests(isOwner
                                             ?ownerGather.numGPUs
                                             :0);
      std::vector<MPI_Request> send_requests(context->isActiveWorker
                                             ?devices->size()
                                             :0);
      // ------------------------------------------------------------------
      // trigger all sends and receives - for gpu descs
      // ------------------------------------------------------------------
      if (isOwner) {
        AuxChannelTile *aux_recv = 0;
        switch(channel) {
        case BN_FB_DEPTH:
          aux_recv = gatheredTilesOnOwner.auxChannelTiles.depth;
          break;
        case BN_FB_PRIMID:
          aux_recv = gatheredTilesOnOwner.auxChannelTiles.primID;
          break;
        case BN_FB_INSTID:
          aux_recv = gatheredTilesOnOwner.auxChannelTiles.instID;
          break;
        case BN_FB_OBJID:
          aux_recv = gatheredTilesOnOwner.auxChannelTiles.objID;
          break;
        default:
          throw std::runtime_error("unsupported aux channel in sending aux!?");
        };

        /* OWNER: create receive requests for all compressed tiles from
           all ranks. this also include the ones we send from our own
           GPUs, so do NOT wait here */
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          auto thisDev = &context->topo->allDevices[ggID];
          context->world.recv(thisDev->worldRank,
                              thisDev->local,
                              aux_recv+ownerGather.firstTileOnGPU[ggID],
                              ownerGather.numTilesOnGPU[ggID],
                              recv_requests[ggID]);
          // int rankOfGPU = ggID / context->gpusPerWorker;
          // int localID   = ggID % context->gpusPerWorker;
          // context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
          //                     aux_recv+ownerGather.firstTileOnGPU[ggID],
          //                     ownerGather.numTilesOnGPU[ggID],
          //                     recv_requests[ggID]);
        }
      }
    
      if (context->isActiveWorker) {
        /* worker: compress all local tiles (on all gpus), and trigger
           send requests to transfer those to master */
        for (auto device : *devices) {
          SetActiveGPU forDuration(device);
          auto tiledFB = getFor(device);
          auto pld = getPLD(device);
          AuxChannelTile *aux_send = 0;
          switch(channel) {
          case BN_FB_DEPTH:
            aux_send = tiledFB->auxTiles.depth;
            break;
          case BN_FB_PRIMID:
            aux_send = tiledFB->auxTiles.primID;
            break;
          case BN_FB_INSTID:
            aux_send = tiledFB->auxTiles.instID;
            break;
          case BN_FB_OBJID:
            aux_send = tiledFB->auxTiles.objID;
            break;
          default:
            throw std::runtime_error("unsupported aux channel in sending aux!?");
          };
          context->world.send(owningRank,device->contextRank(),
                              aux_send,
                              tiledFB->numActiveTilesThisGPU,
                              send_requests[device->contextRank()]);
        }
      }
      // ------------------------------------------------------------------
      // wait for all send/recv requests to complete - master should
      // then have all tiles from all gpus, in compressed form
      // ------------------------------------------------------------------
      if (isOwner)
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
          context->world.wait(recv_requests[ggID]);
    
      if (context->isActiveWorker)
        for (auto device : *devices)
          context->world.wait(send_requests[device->contextRank()]);
    }



    /*! read one of the auxiliary (not color or normal) buffers into
      the given (device-writeable) staging area; this will at the
      least incur some reformatting from tiles to linear (if local
      node), possibly some gpu-gpu transfer (local node w/ more than
      one gpu) and possibly some mpi communication (distFB) */
    void DistFB::writeAuxChannel(void *stagingArea,
                                 BNFrameBufferChannel channel)
    {
      if (!isOwner) return;

      AuxChannelTile *inTiles = 0;
      switch(channel) {
      case BN_FB_DEPTH:  inTiles = gatheredTilesOnOwner.auxChannelTiles.depth; break;
      case BN_FB_PRIMID: inTiles = gatheredTilesOnOwner.auxChannelTiles.primID; break;
      case BN_FB_INSTID: inTiles = gatheredTilesOnOwner.auxChannelTiles.instID; break;
      case BN_FB_OBJID:  inTiles = gatheredTilesOnOwner.auxChannelTiles.objID; break;
      default:
        throw std::runtime_error("writeauxchannel - invalid channel "
                                 +std::to_string((int)channel));
      };
    
      auto device = getDenoiserDevice();
      TiledFB::linearizeAuxTiles(device,
                                 stagingArea,numPixels,
                                 inTiles,
                                 gatheredTilesOnOwner.tileDescs,
                                 gatheredTilesOnOwner.numActiveTiles);
      device->sync();
    }
  
    /*! gather color (and optionally, if not null) linear normal, from
      all GPUs (and ranks). lienarColor and lienarNormal are
      device-writeable 2D linear arrays of numPixel size;
      linearcolor may be null. */
    void DistFB::gatherColorChannel(/*float4 or rgba8*/void *linearColor,
                                    BNDataType gatherType,
                                    vec3f *linearNormal)
    {
      // ------------------------------------------------------------------
      // gather all (packed) tiles from all clients
      // ------------------------------------------------------------------
      std::vector<MPI_Request> recv_requests(isOwner
                                             ?((needNormals?2:1)*ownerGather.numGPUs)
                                             :0);
      std::vector<MPI_Request> send_requests((needNormals?2:1)*devices->size());
      // ------------------------------------------------------------------
      // trigger all sends and receives - for gpu descs
      // ------------------------------------------------------------------
      if (isOwner) {
        /* OWNER: create receive requests for all compressed tiles from
           all ranks. this also include the ones we send from our own
           GPUs, so do NOT wait here */
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          auto thisDev = &context->topo->allDevices[ggID];
          context->world.recv(thisDev->worldRank,thisDev->local,
                              gatheredTilesOnOwner.compressedColorTiles
                              +ownerGather.firstTileOnGPU[ggID],
                              ownerGather.numTilesOnGPU[ggID],
                              recv_requests[ggID]);
          if (needNormals)
            context->world.recv(thisDev->worldRank,thisDev->local,
                                gatheredTilesOnOwner.compressedNormalTiles
                                +ownerGather.firstTileOnGPU[ggID],
                                ownerGather.numTilesOnGPU[ggID],
                                recv_requests[ownerGather.numGPUs+ggID]);
          // int rankOfGPU = ggID / context->gpusPerWorker;
          // int localID   = ggID % context->gpusPerWorker;
          // context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
          //                     gatheredTilesOnOwner.compressedColorTiles
          //                     +ownerGather.firstTileOnGPU[ggID],
          //                     ownerGather.numTilesOnGPU[ggID],
          //                     recv_requests[ggID]);
          // if (needNormals)
          //   context->world.recv(context->worldRankOfWorker[rankOfGPU],localID,
          //                       gatheredTilesOnOwner.compressedNormalTiles
          //                       +ownerGather.firstTileOnGPU[ggID],
          //                       ownerGather.numTilesOnGPU[ggID],
          //                       recv_requests[ownerGather.numGPUs+ggID]);
        }
      }
    
      if (context->isActiveWorker) {
        /* worker: compress all local tiles (on all gpus), and trigger
           send requests to transfer those to master */
        for (auto device : *devices) {
          SetActiveGPU forDuration(device);
          auto tiledFB = getFor(device);
          auto pld = getPLD(device);
          float accumScale = 1.f/accumID;
          CompressTiles kernel = {
            pld->localSend.compressedColorTiles,
            pld->localSend.compressedNormalTiles,
            tiledFB->accumTiles,
            accumScale
          };
          pld->compressTiles->launch(tiledFB->numActiveTilesThisGPU,pixelsPerTile,&kernel);
        }
        for (auto device : *devices) {
          SetActiveGPU forDuration(device);
          auto tiledFB = getFor(device);
          auto pld = getPLD(device);
          device->sync();
          context->world.send(owningRank,device->contextRank(),
                              pld->localSend.compressedColorTiles,
                              tiledFB->numActiveTilesThisGPU,
                              send_requests[device->contextRank()]);
          if (needNormals)
            context->world.send(owningRank,device->contextRank(),
                                pld->localSend.compressedNormalTiles,
                                tiledFB->numActiveTilesThisGPU,
                                send_requests[devices->size()+device->contextRank()]);
        }
      }
      // ------------------------------------------------------------------
      // wait for all send/recv requests to complete - master should
      // then have all tiles from all gpus, in compressed form
      // ------------------------------------------------------------------
      if (isOwner)
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          context->world.wait(recv_requests[ggID]);
          if (needNormals)
            context->world.wait(recv_requests[ownerGather.numGPUs+ggID]);
        }
    
      if (context->isActiveWorker)
        for (auto device : *devices) {
          context->world.wait(send_requests[device->contextRank()]);
          if (needNormals)
            context->world.wait(send_requests[devices->size()+device->contextRank()]);
        }
    
      // ------------------------------------------------------------------
      // all (packed) tiles received; unpack (on owner only)
      // ------------------------------------------------------------------
      if (isOwner) {
        UnpackTiles args = {
          numPixels,
          linearColor,
          gatherType,
          linearNormal,
          gatheredTilesOnOwner.compressedColorTiles,
          gatheredTilesOnOwner.compressedNormalTiles,
          gatheredTilesOnOwner.tileDescs
        };
        auto device = getDenoiserDevice();
        SetActiveGPU forDuration(device);
        getPLD(device)->unpackTiles->launch(gatheredTilesOnOwner.numActiveTiles,
                                            pixelsPerTile,
                                            &args);
        device->sync();
      }
    }

  
    DistFB::DistFB(MPIContext *context,
                   const DevGroup::SP &devices)
      : FrameBuffer(context,
                    devices,
                    context->world.rank==0),
        context(context),
        isOwner(context->world.rank == owningRank)
        // ,
        //   ownerIsWorker(context->workerRankOfWorldRank[context->world.rank] != -1)
    {
      perLogical.resize(devices->numLogical);
      if (isOwner) {
        ownerGather.numGPUs = context->topo->allDevices.size();
        // ownerGather.numGPUs = context->numWorkers() * context->gpusPerWorker;
        ownerGather.numTilesOnGPU.resize(ownerGather.numGPUs);
      }
      else {
        ownerGather.numGPUs = context->topo->allDevices.size();
        // ownerGather.numGPUs = context->numWorkers() * context->gpusPerWorker;
        ownerGather.numTilesOnGPU.resize(0);
      }
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        pld->compressTiles = createCompute_compressTiles(device->rtc);
        pld->unpackTiles = createCompute_unpackTiles(device->rtc);
      }
    }

    DistFB::~DistFB()
    {
      freeChannelData();
      for (auto device : *devices) {
        PLD *pld = getPLD(device);
        delete pld->compressTiles;
        delete pld->unpackTiles;
      }
    }

    DistFB::PLD *DistFB::getPLD(Device *device) 
    {
      assert(device);
      assert(device->contextRank() >= 0);
      assert(device->contextRank() < perLogical.size());
      return &perLogical[device->contextRank()];
    }
  
    void DistFB::freeChannelData()
    {
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
      
        TiledFB *tiledFB = getFor(device);
        auto pld = getPLD(device);
        if (pld->localSend.compressedColorTiles) {
          device->rtc->freeMem(pld->localSend.compressedColorTiles);
          pld->localSend.compressedColorTiles = 0;
        }
        if (pld->localSend.compressedNormalTiles) {
          device->rtc->freeMem(pld->localSend.compressedNormalTiles);
          pld->localSend.compressedNormalTiles = 0;
        }
      }
      if (isOwner) {
        Device *device = getDenoiserDevice();
        SetActiveGPU forDuration(device);
        if (!isAltFrame) {
        if (gatheredTilesOnOwner.auxChannelTiles.depth) {
          device->rtc->freeMem(gatheredTilesOnOwner.auxChannelTiles.depth);
          gatheredTilesOnOwner.auxChannelTiles.depth = 0;
        }
        if (gatheredTilesOnOwner.auxChannelTiles.primID) {
          device->rtc->freeMem(gatheredTilesOnOwner.auxChannelTiles.primID);
          gatheredTilesOnOwner.auxChannelTiles.primID = 0;
        }
        if (gatheredTilesOnOwner.auxChannelTiles.instID) {
          device->rtc->freeMem(gatheredTilesOnOwner.auxChannelTiles.instID);
          gatheredTilesOnOwner.auxChannelTiles.instID = 0;
        }
        if (gatheredTilesOnOwner.auxChannelTiles.objID) {
          device->rtc->freeMem(gatheredTilesOnOwner.auxChannelTiles.objID);
          gatheredTilesOnOwner.auxChannelTiles.objID = 0;
        }
        if (gatheredTilesOnOwner.auxChannelTiles.motion) {
          device->rtc->freeMem(gatheredTilesOnOwner.auxChannelTiles.motion);
          gatheredTilesOnOwner.auxChannelTiles.motion = 0;
        }
        if (gatheredTilesOnOwner.tileDescs) {
          device->rtc->freeMem(gatheredTilesOnOwner.tileDescs);
          gatheredTilesOnOwner.tileDescs = 0;
        }
        if (gatheredTilesOnOwner.compressedColorTiles) {
          device->rtc->freeMem(gatheredTilesOnOwner.compressedColorTiles);
          gatheredTilesOnOwner.compressedColorTiles = 0;
        }
        if (gatheredTilesOnOwner.compressedNormalTiles) {
          device->rtc->freeMem(gatheredTilesOnOwner.compressedNormalTiles);
          gatheredTilesOnOwner.compressedNormalTiles = 0;
        }
        }
      }
    }



  
    void DistFB::resize(BNDataType colorFormat,
                        vec2i size,
                        uint32_t channels,
                        bool isAltFrame)
    {
      freeChannelData();
      
      FrameBuffer::resize(colorFormat, size, channels, isAltFrame);

      // ------------------------------------------------------------------
      /* check if we need to have a normal channelf ro deonising (on
         owner), and communicate this to all workers */
      // ------------------------------------------------------------------
      if (isOwner) {
        this->needNormals = (denoiser != 0);
        context->world.bc_send(&this->needNormals,sizeof(this->needNormals));
      } else {
        context->world.bc_recv(&this->needNormals,sizeof(this->needNormals));
      }
    
      // ------------------------------------------------------------------
      /* allocate compressed tiles mem - one for each tiles in the
         corresponding tiledFB */
      // ------------------------------------------------------------------
      std::vector<int> tilesOnGPU(devices->numLogical);
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
      
        TiledFB *tiledFB = getFor(device);
        tilesOnGPU[device->contextRank()]
          = tiledFB->numActiveTilesThisGPU;
      
        auto pld = getPLD(device);
        if (!isAltFrame) {
        pld->localSend.compressedColorTiles
          = (CompressedColorTile*)device->rtc->allocMem
          (tiledFB->numActiveTilesThisGPU*sizeof(CompressedColorTile));
        if (needNormals) 
          pld->localSend.compressedNormalTiles
            = (CompressedNormalTile*)device->rtc->allocMem
            (tiledFB->numActiveTilesThisGPU*sizeof(CompressedNormalTile));
        }
      }
    
      std::vector<MPI_Request> recv_requests(ownerGather.numGPUs);
      std::vector<MPI_Request> send_requests(tilesOnGPU.size());
    
      // ------------------------------------------------------------------
      // trigger all sends and receives - for gpu tile count
      // ------------------------------------------------------------------
      if (isOwner) {
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          auto thisDev = &context->topo->allDevices[ggID];
          context->world.recv(thisDev->worldRank,thisDev->local,
                              &ownerGather.numTilesOnGPU[ggID],1,
                              recv_requests[ggID]);
        }
      }

      if (context->isActiveWorker) {
        for (int localID=0;localID<tilesOnGPU.size();localID++) {
        
          TiledFB *tiledFB = getFor(context->devices->get(localID));
          context->world.send(owningRank,localID,
                              &tiledFB->numActiveTilesThisGPU,1,
                              send_requests[localID]);
        }
      }    

      // ------------------------------------------------------------------
      // and wait for those to complete
      // ------------------------------------------------------------------
      if (isOwner) {
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
          context->world.wait(recv_requests[ggID]);
      }
    
      if (context->isActiveWorker)
        for (int localID=0;localID<tilesOnGPU.size();localID++)
          context->world.wait(send_requests[localID]);    

      // ------------------------------------------------------------------
      // ------------------------------------------------------------------

      if (isOwner) {
        ownerGather.firstTileOnGPU.resize(ownerGather.numGPUs);
        int sumTiles = 0;
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          ownerGather.firstTileOnGPU[ggID] = sumTiles;
          sumTiles += ownerGather.numTilesOnGPU[ggID];
        }
        gatheredTilesOnOwner.numActiveTiles = sumTiles;

        Device *frontDev = getDenoiserDevice();

        if (!isAltFrame) {
        gatheredTilesOnOwner.compressedColorTiles
          = (CompressedColorTile *)frontDev->rtc->allocMem
          (sumTiles*sizeof(*gatheredTilesOnOwner.compressedColorTiles));

        if (denoiser)
          gatheredTilesOnOwner.compressedNormalTiles
            = (CompressedNormalTile *)frontDev->rtc->allocMem
            (sumTiles*sizeof(*gatheredTilesOnOwner.compressedNormalTiles));

        if (channels & BN_FB_DEPTH)
          gatheredTilesOnOwner.auxChannelTiles.depth
            = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
        if (channels & BN_FB_PRIMID)
          gatheredTilesOnOwner.auxChannelTiles.primID
            = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
        if (channels & BN_FB_INSTID)
          gatheredTilesOnOwner.auxChannelTiles.instID
            = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
        if (channels & BN_FB_OBJID)
          gatheredTilesOnOwner.auxChannelTiles.objID
            = (AuxChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(AuxChannelTile));
        if (channels & BN_FB_MOTION)
          gatheredTilesOnOwner.auxChannelTiles.motion
            = (MotionChannelTile*)frontDev->rtc->allocMem(sumTiles*sizeof(MotionChannelTile));
        }
        gatheredTilesOnOwner.tileDescs
          = (TileDesc *)frontDev->rtc->allocMem
          (sumTiles*sizeof(*gatheredTilesOnOwner.tileDescs));
      }
    
      // ------------------------------------------------------------------
      // trigger all sends and receives - for gpu descs
      // ------------------------------------------------------------------
      if (isOwner)  {
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) {
          auto thisDev = &context->topo->allDevices[ggID];
          context->world.recv(thisDev->worldRank,thisDev->local,
                              gatheredTilesOnOwner.tileDescs+ownerGather.firstTileOnGPU[ggID],
                              ownerGather.numTilesOnGPU[ggID],
                              recv_requests[ggID]);
        }
      }

      if (context->isActiveWorker)
        for (auto device : *devices)  {
          context->world.send(owningRank,
                              device->contextRank(),
                              getFor(device)->tileDescs,
                              getFor(device)->numActiveTilesThisGPU,
                              send_requests[device->contextRank()]);
        }

      // ------------------------------------------------------------------
      // and wait for those to complete
      // ------------------------------------------------------------------
      if (isOwner)
        for (int ggID = 0; ggID < ownerGather.numGPUs; ggID++) 
          context->world.wait(recv_requests[ggID]);
    
      if (context->isActiveWorker)
        for (int localID=0;localID<tilesOnGPU.size();localID++)
          context->world.wait(send_requests[localID]);
    }
  
    RTC_EXPORT_COMPUTE1D(unpackTiles,UnpackTiles);
    RTC_EXPORT_COMPUTE1D(compressTiles,CompressTiles);
  }
}
