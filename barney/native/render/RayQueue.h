// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/DeviceGroup.h"
#include "native/render/Ray.h"
#include "native/render/HitIDs.h"

namespace BARNEY_NS {
  namespace native {
    
    struct SingleQueue {
      void alloc(rtc::Device *rtc, int size);
      void free(rtc::Device *rtc);
    
      /*! the actual rays, will all need to be sent every cycle */
      Ray   *rays   = nullptr;
    
      /*! must be sent over the network, but only for generation 0 */
      HitIDs *hitIDs = nullptr;
    
      /*! information we track per ray, but which will NOT be sent over
        the network */
      PathState *states = nullptr;
    };
  
    struct RayQueue {
      
      /*! device data for a given ray queue, tracking both current ray
          queue (the one we're currently shading and/or tracing) as
          well as a secondary queue to write newly generated rays into
          (for raygen or shade output) */
      struct DD {
        struct {
          /*! depth and inst/prim/obj IDs of closest surface or volume
              _irrespective of opacity or stochastic sampling. during
              distributed trace these do need to get tracked/reduced
              across all GPUs */
          HitIDs    *hitIDs;
          
          /*! the actual rays (and hits). during distributed trace
              these do need to get tracked/reduced across all GPUs */
          Ray       *rays;
          
          /*! hit-independent shading state that is _only_ relevant to
              shading, not during trace or intersection - and which
              thus does _not_ have to be tracked/reduced across
              different gpus */
          PathState *states;
          
          /*! number of rays in this queue */
          int numActive;
        } /* !current generation of rays; the one we're currently
             tracing or shading */currentGen;
        
        struct {
          Ray       *rays;
          PathState *states;
          /*! pointer to atomically writable int that tracks number of
              rays written into this queue */
          int       *pNumActive;
        } nextGen;
      };
      
      RayQueue(Device *device);
      ~RayQueue();

      DD getDD() const;
      
      int readNumActive();
      /*! how many rays are active in the *READ* queue */
      int numActiveRays() const;
    
      int *h_numActive = 0;

      /*! the read queue, where local kernels operating on rays (trace
        and shade) can read rays from. this is actually a misnomer
        becasue both shade and trace can actually modify trays (and
        thus, strictly speaking, are 'writing' to those rays), but
        haven't yet found a better name */
      SingleQueue traceAndShadeReadQueue;

      /*! the queue where local kernels that write *new* rays
        (ie, ray gen and shading) will write their rays into */
      SingleQueue receiveAndShadeWriteQueue;

      /*! current write position in the write queue (during shading and
        ray generation) */
      int *_d_nextWritePos  = 0;
    
      /*! how many rays are active in the *READ* queue */
      int  numActive = 0;
      int  size      = 0;

      Device *device = 0;

      void resetWriteQueue();

      /*! swap the two ray queues, where the just received queue becomes
        the new trace queue, etc

        - primID/instID/objID need ONLY be sent along for the first generation

        - shade info will always remain on the rank that generated.
      */
      void swapAfterShade();
      void swapAfterGeneration();
      void swapAfterCycle(int cycleID, int numCycles);

      void resize(int newSize);
    };

  }
}
