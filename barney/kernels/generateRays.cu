// SPDX-FileCopyrightText:
// Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier:
// Apache-2.0


#ifdef __CUDACC__
# define OWL_DISABLE_TBB
#endif
#include "barney/common/barney-common.h"
#include "barney/DeviceGroup.h"
#include "barney/render/Ray.h"
#include "barney/render/RayQueue.h"
#include "barney/Camera.h"
#include "barney/render/Renderer.h"
#include "barney/fb/FrameBuffer.h"
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  namespace render {
    /*! generates a new wave-front of rays, to be written to
      'rayQueue[]', at (atomically incrementable) positoin
      *d_count. This kernel operates on *tiles* (not complete frames);
      the list of tiles to generate rays for is passed in 'tileDescs';
      there will be one cuda block per tile */
#if RTC_DEVICE_CODE
    inline __rtc_device
    vec3f uvToWorld(const affine3f &toWorld,
                    float f_x,
                    float f_y) 
    {
      const float phi   = TWO_PI * f_x;
      const float theta = ONE_PI * f_y;
      
      vec3f dir;
      dir.z = cosf(theta);
      dir.x = cosf(phi)*sinf(theta);
      dir.y = sinf(phi)*sinf(theta);
      
      return xfmVector(toWorld,dir);
    }


    __rtc_global
    void _generateRays(const rtc::ComputeInterface &rt,
                       Camera::DD camera,
                       Renderer::DD renderer,
                       /*! a unique random number seed value for pixel
                         and lens jitter; probably just accumID */
                       int accumID,
                       /*! full frame buffer size, to check if a given
                         tile's pixel ID is still valid */
                       vec2i fbSize,
                       /*! pointer to a device-side int that tracks the
                         next write position in the 'write' ray
                         queue; can be atomically incremented on the
                         device */
                       int *d_count,
                       /*! pointer to device-side ray queue to write
                         newly generated raysinto */
                       SingleQueue rayQueue,
                       /*! tile descriptors for the tiles that the
                         frame buffer owns on this device; rays
                         should only get generated for these tiles */
                       TileDesc *tileDescs,
                       bool enablePerRayDebug
                       )
    {
      // ------------------------------------------------------------------
      int tileID   = rt.getBlockIdx().x;
      int lPixelID = rt.getThreadIdx().x;

      vec2i tileOffset = tileDescs[tileID].lower;
      int ix = (lPixelID % tileSize) + tileOffset.x;
      int iy = (lPixelID / tileSize) + tileOffset.y;

      Ray ray;
      PathState state;
      state.misWeight = 0.f;
      state.pixelID = tileID * (tileSize*tileSize) + rt.getThreadIdx().x;
      Random rand(unsigned(ix+fbSize.x*accumID),
                  unsigned(iy+fbSize.y*accumID));
// #if NEW_RNG
      // ray.rngSeed.value = (uint32_t)hash(ix,iy,accumID);
      ray.rngSeed.seed(ix+accumID*fbSize.x,iy);
// #else
      // ray.rngSeed.seed(ix+fbSize.x*(accumID),iy+fbSize.y*(accumID));
// #endif


      float pixel_u = ((accumID == 0) ? .5f : rand());
      float pixel_v = ((accumID == 0) ? .5f : rand());
      float image_u = ((ix+pixel_u)/float(fbSize.x));
      float image_v = ((iy+pixel_v)/float(fbSize.y));
      float aspect = fbSize.x / float(fbSize.y);
      if (camera.type == Camera::PERSPECTIVE) {
        auto &perspective = camera.perspective;
        ray.org  = perspective.lens_00;
        vec3f ray_dir
          = perspective.dir_00
          + (1.f*aspect*(image_u - .5f)) * perspective.dir_du
          + (1.f*(image_v - .5f)) * perspective.dir_dv;
      
        if (perspective.apertureRadius > 0.f) {
          vec3f lens_du = normalize(perspective.dir_du);
          vec3f lens_dv = normalize(perspective.dir_dv);
          vec3f lensNormal  = cross(lens_du,lens_dv);

          vec3f D = normalize(ray_dir);
          vec3f pointOnImagePlane
            = D * (perspective.focusDistance / fabsf(dot(D,lensNormal)));
          float lu, lv;
          if (accumID == 0) {
            lu = lv = 0.f;
          } else {
            while (true) {
              lu = 2.f*rand()-1.f;
              lv = 2.f*rand()-1.f;
              float f = lu*lu+lv*lv;
              if (f > 1.f) continue;
              break;
            }
          }
          vec3f lensOffset
            = (perspective.apertureRadius * lu) * lens_du
            + (perspective.apertureRadius * lv) * lens_dv;
          ray.org += lensOffset;
          ray_dir = normalize(pointOnImagePlane - lensOffset);
        } else {
          ray_dir = normalize(ray_dir);
        }
        ray.dir = ray_dir;
      } else if (camera.type == Camera::ORTHOGRAPHIC) {
        auto &orthographic = camera.orthographic;
        ray.dir = normalize(orthographic.dir);
        ray.org
          = orthographic.org_00
          + ((image_u-.5f)*orthographic.aspect*orthographic.height)
          * orthographic.org_du
          + ((image_v-.5f)*orthographic.height)
          * orthographic.org_dv;
      } else if (camera.type == Camera::OMNIDIRECTIONAL) {
        auto &omni = camera.omni;
        ray.org
          = omni.toWorld.p;
        ray.dir =
          uvToWorld(omni.toWorld,image_u,image_v);
       }
      
#ifdef NDEBUG
      ray._dbg        = 0;
      ray.crosshair   = 0;
#else
      int dbg_target_x = fbSize.x/2;
      int dbg_target_y = fbSize.y/2;
      
      // dbg_target_x += 230;
      // dbg_target_y += 80;
      
      bool crossHair_x = (ix == dbg_target_x);
      bool crossHair_y = (iy == dbg_target_y);

      ray._dbg         = enablePerRayDebug && (crossHair_x && crossHair_y);
      ray.crosshair
        = enablePerRayDebug && (crossHair_x || crossHair_y);
#endif

      ray.clearHit();
      ray.isShadowRay = false;
      ray.isInMedium  = false;
      ray.tMax        = 1e30f;

      // Apply cutting plane (disabled if w < -1e28f)
      if (renderer.cutPlane.w > -1e28f) {
        vec3f N = vec3f(renderer.cutPlane.x,
                        renderer.cutPlane.y,
                        renderer.cutPlane.z);
        float d_plane  = renderer.cutPlane.w;
        float dist_org = dot(N, ray.org) + d_plane;
        float denom    = dot(N, ray.dir);
        if (dist_org >= 0.f) {
          // camera on visible side: clip where ray exits the half-space
          if (denom < 0.f) {
            float t_cut  = -dist_org / denom;
            ray.tMax = min(ray.tMax, t_cut);
          }
        } else {
          // camera on invisible side: advance origin to where ray enters
          // the visible half-space, so interior geometry is reachable
          if (denom > 0.f) {
            float t_entry = -dist_org / denom;
            ray.org += t_entry * ray.dir;
            // ray now starts on the plane; tMax stays 1e30f
          } else {
            ray.tMax = 0.f; // ray never enters visible side: show nothing
          }
        }
      }
      // ray.rngSeed     = rand.next;//state;
      state.numDiffuseBounces = 0;
      if (0 && ray.dbg())
        printf("-------------------------------------------------------\n");
             
      if (1 && ray.dbg())
        printf("======================\nspawned %f %f %f dir %f %f %f\n",
               ray.org.x,
               ray.org.y,
               ray.org.z,
               (float)ray.dir.x,
               (float)ray.dir.y,
               (float)ray.dir.z);

      const float t = (iy+.5f)/float(fbSize.y);
      // for *primary* rays we pre-initialize basecolor to a background
      // color; this way the shaderays function doesn't have to reverse
      // engineer pixel pos etc
      
      vec4f bgColor
        = (renderer.bgColor.w >= 0.f)
        ? renderer.bgColor
        : ((1.0f - t)*vec4f(0.9f, 0.9f, 0.9f,1.f)
           + t *      vec4f(0.15f, 0.25f, .8f,1.f));
      if (renderer.bgTexture) {
        float bg_u = ((ix + pixel_u+.5f) / float(fbSize.x-1.f));
        float bg_v = ((iy + pixel_v+.5f) / float(fbSize.y-1.f));
        vec4f v = rtc::tex2D<vec4f>(renderer.bgTexture, bg_u, bg_v);
        bgColor = v;
      }
      (vec4f&)ray.missColor = bgColor;
      if (1 && ray.dbg()) printf("== spawn ray has bg tex %p bg color %f %f %f %f\n",
                               (void*)renderer.bgTexture,
                               bgColor.x,
                               bgColor.y,
                               bgColor.z,
                               bgColor.w);
      state.throughput = 1.f;
      int pos = rt.atomicAdd(d_count,1);

      rayQueue.rays[pos] = ray;
      rayQueue.states[pos] = state;
      rayQueue.hitIDs[pos] = {BARNEY_INF,-1,-1,-1};
    }
#endif
  }
  
  void Context::generateRays(Camera *camera,
                             Renderer *renderer,
                             FrameBuffer *fb)
  {
    auto getPerRayDebug = [&]()
    {
      const char *fromEnv = getenv("BARNEY_DBG_RENDER");
      return fromEnv && std::stoi(fromEnv);
    };
    static bool enablePerRayDebug = getPerRayDebug();
    
    assert(fb);
    int accumID=fb->accumID;
    // ------------------------------------------------------------------
    // launch all GPUs to do their stuff
    // ------------------------------------------------------------------
    Camera::DD cameraDD = camera->getDD();
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      TiledFB *devFB = fb->getFor(device);
      device->rayQueue->resetWriteQueue();

      if (FromEnv::get()->logQueues) {
        std::stringstream ss;
        ss  << "#bn(" << myRank() << "): ## ray queue op GENERATE "
            << device->rayQueue->receiveAndShadeWriteQueue.rays
            << " + " << device->rayQueue->receiveAndShadeWriteQueue.states
            << std::endl;
        std::cout << ss.str();
      }

      __rtc_launch(//device
                   device->rtc,
                   //kernel
                   render::_generateRays,
                   // launch config
                   devFB->numActiveTilesThisGPU,pixelsPerTile,
                   // args
                   cameraDD,
                   renderer->getDD(device),
                   accumID,
                   fb->renderPixels,
                   device->rayQueue->_d_nextWritePos,
                   device->rayQueue->receiveAndShadeWriteQueue,
                   devFB->tileDescs,
                   enablePerRayDebug
                   );
    }
    // ------------------------------------------------------------------
    // wait for all GPUs' completion
    // ------------------------------------------------------------------
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      device->rtc->sync();
      device->rayQueue->swapAfterGeneration();
      device->rayQueue->numActive = device->rayQueue->readNumActive();
    }
  }
  
}


