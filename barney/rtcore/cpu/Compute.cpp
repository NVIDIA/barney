// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/ComputeInterface.h"
#include <owl/common/parallel/parallel_for.h>
#include <mutex>
#include <thread>
#include <condition_variable>
#include <barrier>
#include "rtcore/cpu/TraceInterface.h"

namespace BARNEY_NS {
  namespace rtc {

    int numThreads() {
      int nt = std::thread::hardware_concurrency();
#if 0
      int maxNumThreads = 1;
#else
      int maxNumThreads = 1<<30;
#endif
      return std::min(nt,maxNumThreads);
    }
    
    LaunchSystem::LaunchSystem()
      : barrier(numThreads()+1,[]() noexcept {})
    {
      threads.reserve(numThreads());
      for (int i=0;i<numThreads();i++)
        threads.emplace_back([this](){ this->threadFct(); });

      barrier.arrive_and_wait();
    }
    
    void LaunchSystem::launchAndWait(int numTotal, Task *task)
    {
      // enqueue job
      std::lock_guard<std::mutex> lock(mutex);
      
      // workers are waiting on barrier here
      this->task = task;
      this->numJobs.total = numTotal;
      this->numJobs.taken = 0;
      
      this->barrier.arrive_and_wait();
      // workers do the work here

      this->barrier.arrive_and_wait();
    }

    LaunchSystem *createLaunchSystem() { return new LaunchSystem; }
    
    void LaunchSystem::threadFct()
    {
      barrier.arrive_and_wait();
      while (true) {

        // ------------------------------------------------------------------
        // wait for control thread to submit work
        // ------------------------------------------------------------------
        barrier.arrive_and_wait();
        
        // ------------------------------------------------------------------
        // run the actual task
        // ------------------------------------------------------------------
        while (true) {
          int tid = numJobs.taken++;
          if (tid >= numJobs.total)
            break;
          task->run(tid);
        }
        
        // ------------------------------------------------------------------
        // wait for all to be done
        // ------------------------------------------------------------------
        barrier.arrive_and_wait();
      }
    }
    
    template<typename Kernel>
    void parallel_for_3D(Device *device, vec3ui dims, const Kernel &lambda)
    {
      LaunchSystem *ls = ((BARNEY_NS::rtc::Device *)device)->ls;
      int numTotal = dims.x*dims.y*dims.z;
      TaskWrapper task([&](int tid)
      {
        // printf("in parallel %i / %i %i %i\n",tid,dims.x,dims.y,dims.z);
        int ix = tid % dims.x;
        int iz = tid / (dims.x * dims.y);
        int iy = (tid / dims.x) % dims.y;
        lambda(vec3ui(ix,iy,iz));
      });
      ls->launchAndWait(numTotal,&task);
    }
    
  }
}
