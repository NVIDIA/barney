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

    __rtc_global
    void _shadeRays(const rtc::ComputeInterface &rt,
                    World::DD world,
                    Renderer::DD renderer,
                    Camera::DD camera,
                    vec2i fbSize,
                    AccumTile *accumTiles,
                    AuxTiles auxTiles,
                    int accumID,
                    RayQueue::DD rayQueue,
                    // SingleQueue readQueue,
                    // int numRays,
                    // SingleQueue writeQueue,
                    // int *d_nextWritePos,
                    int generation)
    {
#if RTC_DEVICE_CODE
      const uint32_t tid = rt.getLaunchIndex().x;
      if (tid >= rayQueue.currentGen.numActive)
        return;
      HitIDs   *hitIDs = rayQueue.currentGen.hitIDs;
      Ray       ray    = rayQueue.currentGen.rays[tid];
      PathState state  = rayQueue.currentGen.states[tid];
      /*! value to be written into the alpha channel; this is only
          needed for generation==0, as only this generation will ever
          write that channel */
      const float alpha
        =
#if 1
        /* if alpha channel is supposed to be used by the app for
           compositing reasons it should match the hitID channels'
           decision on whether teh ray did or did not hit anything,
           _not_ the output of the possibly stochastic decision on
           whether a semi-transparent hit was or wasn't accepted
           during trace */
        (hitIDs && hitIDs[tid].primID >= 0)
#else
        ray.hadHit()
#endif
        ? /* alpha is always 1 if we hit something - _any_thing */1.f
        : /* if no't hit, alpha comes from background*/state.miss.alpha;
      const float incomingZ = ray.tMax;
      /* this is used for shading, but for generation==0 is ALSO
         written into the normal buffer, so make sure it's initialized
         even if there is no hit */
      const vec3f incomingN
        = ray.hadHit()
        ? ray.getShadingNormal()
        : vec3f(0.f);
      /*! what we write to / add to the color channel */
      vec3f fragment(0.f);
      Ray shadowRay;
      PathState shadowState;
      shadowRay.tMax = -1.f;
      /* for any given input ray ('ray') this can generate: a)
         possibly, a secondary 'bounce' ray (if so, this will modify
         the 'ray' and 'state' of the incoming ray to store the
         outgoing one; if no, ray.t will be set to <= 0); b) possibly,
         a shadow ray (if so, this will set shadowRay and shadowState;
         if not, it will set shadowRay.tMax to <= 0.f); and c)
         possibly, a fragment that needs to be written to the frame
         buffer (if the path hit an emsssive surface, or a shadow ray
         was found to be not occluded).  */
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
      const bool haveShadowRay = shadowRay.valid();
      const bool haveBounceRay = ray.valid();
      int numRaysToWrite = ((int)haveShadowRay+(int)haveBounceRay);
      if (numRaysToWrite > 0) {
        int writePos = rt.atomicAdd(rayQueue.nextGen.pNumActive, numRaysToWrite);
        /* write shadow ray to output queue, but only if valid */
        if (haveShadowRay) {
          /* no need to write hitIDs, shadow rays never write those anyway */
          rayQueue.nextGen.rays[writePos]   = shadowRay;
          rayQueue.nextGen.states[writePos] = shadowState;
          ++writePos;
        }
        /* write bounce ray to output queue, but only if valid */
        if (ray.tMax > 0.f) {
          rayQueue.nextGen.rays[writePos]   = ray;
          rayQueue.nextGen.states[writePos] = state;
        }
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
          auxTiles.depth[tileID].f[tileOfs]   =
            //incomingZ;
            hitIDs[tid].depth;
        if (auxTiles.primID)
          auxTiles.primID[tileID].ui[tileOfs] = hitIDs[tid].primID;
        if (auxTiles.objID)
          auxTiles.objID[tileID].ui[tileOfs]  = hitIDs[tid].objID;
        if (auxTiles.instID)
          auxTiles.instID[tileID].ui[tileOfs] = hitIDs[tid].instID;
        if (auxTiles.motion && camera.haveMotionMatrices && incomingZ < 1e19f) {
          const int instID = hitIDs[tid].instID;
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
#if DO_AUX_CHANNELS_ONLY_ON_FIRST_FRAME
        /* nothing - aux channels only get written in first frame */
#else
        /* iw - this is debatable: currently alpha channel and aux
           tiles will always get updated/accumulated across multiple
           accumulated frames. this means that after N frames the
           depth channel will show the closest of all N frames' depth
           - and the alpha one the average alpha of all N frames -
           rather than just the first frame's. this may be
           undesirable */
        if (generation == 0) {
          if (alpha > 0.f)
            rt.atomicAdd(&out.w, alpha);
          if (auxTiles.depth && incomingZ < auxTiles.depth[tileID].f[tileOfs])
            auxTiles.depth[tileID].f[tileOfs] = incomingZ;
        }
#endif
        /* any non-zero _fragment_ will always get accumulated, for every
           wave and every genertaion */
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
          __rtc_launch(/* device     */device->rtc,
                       /* kernel     */_shadeRays,
                      /* launchdims */ nb, bs,
           /* -----kernel args ----- */
                                       devWorld,
                                       devRenderer,
                                       devCamera,
                                       fbSize,
                                       devFB->accumTiles,
                                       devFB->auxTiles,
                                       (int)fb->accumID,
                                       rayQueue->getDD(),
                                       // rayQueue->traceAndShadeReadQueue,
                                       // numRays,
                                       // rayQueue->receiveAndShadeWriteQueue,
                                       // rayQueue->_d_nextWritePos,
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
