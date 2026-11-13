// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/Bounce.h"
#include "native/Camera.h"
#include "native/FromEnv.h"
#include "native/GlobalModel.h"
#include "native/fb/FrameBuffer.h"
#include "native/fb/TiledFB.h"
#include "native/render/ShadingInteraction.h"
#include "native/render/RayQueue.h"
#include "native/render/Renderer.h"
#include "native/render/World.h"

namespace BARNEY_NS {
  namespace native {

    __rtc_global void _shadeRays(const rtc::ComputeInterface &rt,
                                 World::DD world,
                                 Renderer::DD renderer,
                                 Camera::DD camera,
                                 vec2i fbSize,
                                 AccumTile *accumTiles,
                                 AuxTiles auxTiles,
                                 int accumID,
                                 SingleQueue readQueue,
                                 int numRays,
                                 SingleQueue writeQueue,
                                 int *d_nextWritePos,
                                 int generation)
    {
#if RTC_DEVICE_CODE
      const uint32_t tid = rt.launchIndex().x;
      if (tid >= numRays)
        return;
      Ray ray = readQueue.rays[tid];
      PathState state = readQueue.states[tid];
      const float alpha =
        (generation == 0)
        // for PRIMARY rays alpha is always true if it hit something,
        // and otherwise it is whatever the background color
        // for that pixel indicated.
        ? (ray.hadHit() ? 1.f : state.miss.alpha)
        // SECONDARY rays alpha does not matter at all.
        : 0.f;
      const float incomingZ = ray.tMax;
      const vec3f incomingN
        = ray.hadHit()
        ? ray.getShadingNormal()
        : vec3f(0.f);
      vec3f fragment(0.f);
      Ray shadowRay;
      PathState shadowState;
      shadowRay.tMax = -1.f;
      bounce(world,
             renderer,
             fragment,
             ray,
             state,
             shadowRay,
             shadowState,
             generation);
#ifndef NDEBUG
      if (ray.crosshair && !ray.dbg())
        fragment = vec3f(1.f, 0.f, 0.f);
#endif
      if (shadowRay.tMax > 0.f) {
        const int pos = rt.atomicAdd(d_nextWritePos, 1);
        writeQueue.rays[pos] = shadowRay;
        writeQueue.states[pos] = shadowState;
      }
      if (ray.tMax > 0.f) {
        const int pos = rt.atomicAdd(d_nextWritePos, 1);
        writeQueue.rays[pos] = ray;
        writeQueue.states[pos] = state;
      }
      const int tileID = int(state.pixelID / pixelsPerTile);
      const int tileOfs = int(state.pixelID % pixelsPerTile);
      vec4f &out = accumTiles[tileID].accum[tileOfs];
      if (accumID == 0 && generation == 0) {
        // aux tiles will only get written for first ray through each
        // pixel (which should be center pixel). colors will get
        // accumulated (ie averaged) over multiple frames, but for
        // alpha, normal, and IDs 'averaging' doesn't really make
        // sense
        out = vec4f(fragment.x, fragment.y, fragment.z, alpha);
        accumTiles[tileID].normal[tileOfs] = incomingN;
        if (auxTiles.depth)
          auxTiles.depth[tileID].f[tileOfs] = incomingZ;
        if (auxTiles.primID)
          auxTiles.primID[tileID].ui[tileOfs] = readQueue.hitIDs[tid].primID;
        if (auxTiles.objID)
          auxTiles.objID[tileID].ui[tileOfs] = readQueue.hitIDs[tid].objID;
        if (auxTiles.instID)
          auxTiles.instID[tileID].ui[tileOfs] = readQueue.hitIDs[tid].instID;
        if (auxTiles.motion && camera.haveMotionMatrices && incomingZ < 1e19f) {
          const int instID = readQueue.hitIDs[tid].instID;
          const vec3f worldHit = ray.P;
          const vec3f prevWorld =
            world.motionDeltas && instID >= 0 && instID < world.numMotionDeltas
            ? xfmPoint(world.motionDeltas[instID], worldHit)
            : worldHit;
          auto reproject = [](const float *m, vec3f p) {
            const float x = m[0] * p.x + m[4] * p.y + m[8] * p.z + m[12];
            const float y = m[1] * p.x + m[5] * p.y + m[9] * p.z + m[13];
            const float w = m[3] * p.x + m[7] * p.y + m[11] * p.z + m[15];
            const float invW = w != 0.f ? 1.f / w : 0.f;
            return vec2f(x * invW, y * invW);
          };
          const vec2f delta = .5f
            * (reproject(camera.currViewProj, worldHit)
               - reproject(camera.prevViewProj, prevWorld));
          auxTiles.motion[tileID].mv[tileOfs] =
            vec2f(delta.x * fbSize.x, delta.y * fbSize.y);
        }
      } else {
        if (generation == 0) {
          if (alpha > 0.f)
            rt.atomicAdd(&out.w, alpha);
          if (auxTiles.depth && incomingZ < auxTiles.depth[tileID].f[tileOfs])
            auxTiles.depth[tileID].f[tileOfs] = incomingZ;
        }
        if (fragment.x > 0.f)
          rt.atomicAdd(&out.x, fragment.x);
        if (fragment.y > 0.f)
          rt.atomicAdd(&out.y, fragment.y);
        if (fragment.z > 0.f)
          rt.atomicAdd(&out.z, fragment.z);
      }
#endif
    }

    void Context::shadeRaysLocally(Renderer *renderer,
                                   GlobalModel *model,
                                   Camera *camera,
                                   FrameBuffer *fb,
                                   int generation,
                                   uint32_t rngSeed)
    {
      const Camera::DD devCamera = camera->getDD();
      const vec2i fbSize = fb->getNumPixels();
      for (auto slotModel : model->modelSlots) {
        World *world = slotModel->world.get();
        for (auto device : *world->devices) {
          SetActiveGPU forDuration(device);
          RayQueue *rayQueue = device->rayQueue;
          rayQueue->resetWriteQueue();
          const int numRays = rayQueue->numActive;
          if (numRays == 0)
            continue;
          const int bs = 128, nb = divRoundUp(numRays, bs);
          TiledFB *devFB = fb->getFor(device);
          const World::DD devWorld = world->getDD(device);
          const Renderer::DD devRenderer = renderer->getDD(device);
          if (FromEnv::logQueues)
            std::cout << "#bn" << myRank() << ": ## ray queue kernel SHADE "
                      << std::endl
                      << "  from " << rayQueue->traceAndShadeReadQueue.rays << " + "
                      << rayQueue->traceAndShadeReadQueue.states << std::endl
                      << "  to " << rayQueue->receiveAndShadeWriteQueue.rays
                      << " + " << rayQueue->receiveAndShadeWriteQueue.states
                      << std::endl;
          __rtc_launch(device->rtc,
                       _shadeRays,
                       nb,
                       bs,
                       devWorld,
                       devRenderer,
                       devCamera,
                       fbSize,
                       devFB->accumTiles,
                       devFB->auxTiles,
                       (int)fb->accumID,
                       rayQueue->traceAndShadeReadQueue,
                       numRays,
                       rayQueue->receiveAndShadeWriteQueue,
                       rayQueue->_d_nextWritePos,
                       generation);
        }
      }
      for (auto device : *devices) {
        SetActiveGPU forDuration(device);
        device->rtc->sync();
        device->rayQueue->swapAfterShade();
        device->rayQueue->numActive = device->rayQueue->readNumActive();
      }
    }
  } // namespace native
} // namespace BARNEY_NS
