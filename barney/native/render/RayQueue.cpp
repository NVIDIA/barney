// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/render/RayQueue.h"
#include "native/Context.h"
#include "native/FromEnv.h"

namespace BARNEY_NS {
  namespace native {
    
    RayQueue::RayQueue(Device *device)
      : device(device)
    {
      auto rtc = device->rtc;
      h_numActive = (int*)rtc->allocHost(sizeof(int));

      // resize(rayQueueSize);
    }
    
    RayQueue::~RayQueue()
    {
      auto rtc = device->rtc;
      traceAndShadeReadQueue.free(rtc);
      receiveAndShadeWriteQueue.free(rtc);
      if (_d_nextWritePos) rtc->freeMem(_d_nextWritePos);
      rtc->freeHost(h_numActive);
    }

    void SingleQueue::alloc(rtc::Device *rtc, int size)
    {
      rays = (Ray *)rtc->allocMem(size*sizeof(Ray));
      states = (PathState *)rtc->allocMem(size*sizeof(PathState));
      hitIDs = (HitIDs *)rtc->allocMem(size*sizeof(HitIDs));
    }
  
    void SingleQueue::free(rtc::Device *rtc)
    {
      if (rays) rtc->freeMem(rays);
      if (hitIDs) rtc->freeMem(hitIDs);
      if (states) rtc->freeMem(states);
      rays   = 0;
      hitIDs = 0;
      states = 0;
    }
  
    RayQueue::DD RayQueue::getDD() const
    {
      RayQueue::DD dd;

      dd.currentGen.hitIDs
        = traceAndShadeReadQueue.hitIDs;
      dd.currentGen.rays
        = traceAndShadeReadQueue.rays;
      dd.currentGen.states
        = traceAndShadeReadQueue.states;
      dd.currentGen.numActive
        = numActive;

      dd.nextGen.rays
        = receiveAndShadeWriteQueue.rays;
      dd.nextGen.states
        = receiveAndShadeWriteQueue.states;
      dd.nextGen.pNumActive
        = _d_nextWritePos;
      
      return dd;
    }

    int RayQueue::readNumActive()
    {
      SetActiveGPU forDuration(device);
      auto rtc = device->rtc;
      rtc->copyAsync(h_numActive,_d_nextWritePos,sizeof(int));
      rtc->sync();
      numActive = *h_numActive;
      if (FromEnv::logQueues)
        printf("#bn: ## ray queue read numactive %i\n",numActive);
      return *h_numActive;
    }
    
    /*! how many rays are active in the *READ* queue */
    int RayQueue::numActiveRays() const
    {
      return numActive;
    }
    
    void RayQueue::resetWriteQueue()
    {
      SetActiveGPU forDuration(device);
      auto rtc = device->rtc;
      rtc->memsetAsync(_d_nextWritePos,0,sizeof(int));
      rtc->sync();
    }
    
    void RayQueue::swapAfterGeneration()
    {
      if (FromEnv::logQueues)
        printf("#bn(%i): ## ray queue swap (after generation)\n",
               device->globalRank());
      std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
      std::swap(receiveAndShadeWriteQueue.states, traceAndShadeReadQueue.states);
      std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
    }

    void RayQueue::swapAfterCycle(int cycleID, int numCycles)
    {
      if (FromEnv::logQueues)
        printf("#bn(%i): ## ray queue swap after cycle (cycle %i/%i)\n",
               device->globalRank(),cycleID,numCycles);
      std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
      std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
    }
    void RayQueue::swapAfterShade()
    {
      if (FromEnv::logQueues)
        printf("#bn(%i): ## ray queue swap after cycle (after shade)\n",
               device->globalRank());
      std::swap(receiveAndShadeWriteQueue.rays, traceAndShadeReadQueue.rays);
      std::swap(receiveAndShadeWriteQueue.states, traceAndShadeReadQueue.states);
      std::swap(receiveAndShadeWriteQueue.hitIDs, traceAndShadeReadQueue.hitIDs);
    }
  
    void RayQueue::resize(int newSize)
    {
      if (newSize <= size) return;
    
      SetActiveGPU forDuration(device);
      auto rtc = device->rtc;
      traceAndShadeReadQueue.free(rtc);
      receiveAndShadeWriteQueue.free(rtc);

      // if (traceAndShadeReadQueue.rays) 
      //   rtc->freeMem(traceAndShadeReadQueue);
      // if (receiveAndShadeWriteQueue)
      //   rtc->freeMem(receiveAndShadeWriteQueue);

      if (!_d_nextWritePos) 
        _d_nextWritePos = (int*)rtc->allocMem(sizeof(int)); 

      // traceAndShadeReadQueue = (Ray*)rtc->allocMem(newSize*sizeof(Ray));
      // receiveAndShadeWriteQueue = (Ray*)rtc->allocMem(newSize*sizeof(Ray));
      traceAndShadeReadQueue.alloc(rtc,newSize);
      receiveAndShadeWriteQueue.alloc(rtc,newSize);
        
      size = newSize;

      resetWriteQueue();
    }
    
  }
}

    
