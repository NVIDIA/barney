// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cpu/Device.h"
#include "rtcore/cpu/Texture.h"
#include <atomic>
#include <thread>
#include <barrier>
#include <mutex>

namespace BARNEY_NS {
  namespace rtc {

    struct ComputeInterface;
    struct TraceInterface;


    template<typename T>
    inline __rtc_device T tex1D(rtc::TextureObject to,
                                float x);
    
    template<>
    inline __rtc_device float tex1D<float>(rtc::TextureObject to,
                                           float x)
    {
      return ((TextureSampler *)to)->tex1D(x).x;
    }
    template<>
    inline __rtc_device vec4f tex1D<vec4f>(rtc::TextureObject to,
                                           float x)
    {
      return ((TextureSampler *)to)->tex1D(x);
    }



    
    template<typename T>
    inline __rtc_device T tex2D(rtc::TextureObject to,
                                float x, float y);

    template<>
    inline __rtc_device float tex2D<float>(rtc::TextureObject to,
                                           float x, float y)
    {
      return ((TextureSampler *)to)->tex2D({x,y}).x;
    }

    template<>
    inline __rtc_device vec4f tex2D<vec4f>(rtc::TextureObject to,
                                           float x, float y)
    {
      return ((TextureSampler *)to)->tex2D({x,y});
    }



    
    template<typename T>
    inline __rtc_device T tex3D(rtc::TextureObject to,
                                float x, float y, float z);
    
    template<>
    inline __rtc_device
    float tex3D<float>(rtc::TextureObject to,
                       float x, float y, float z)
    {
      return ((TextureSampler *)to)->tex3D({x,y,z}).x;
    }


    template<>
    inline __rtc_device
    vec4f tex3D<vec4f>(rtc::TextureObject to,
                       float x, float y, float z)
    {
      return ((TextureSampler *)to)->tex3D({x,y,z});
    }



    struct ComputeInterface {
      inline vec3ui getLaunchIndex() const
      {
        return getThreadIdx() + getBlockIdx() * getBlockDim();
      }
      inline vec3ui getThreadIdx() const
      { return this->threadIdx; }
      
      inline vec3ui getBlockDim() const
      { return this->blockDim; }
      
      inline vec3ui getBlockIdx() const
      { return this->blockIdx; }
      
      inline int atomicAdd(int *ptr, int inc) const
      { return ((std::atomic<int> *)ptr)->fetch_add(inc); }
      
      inline float atomicAdd(float *ptr, float inc) const
      { return ((std::atomic<float> *)ptr)->fetch_add(inc); }
      
      vec3ui threadIdx;
      vec3ui blockIdx;
      vec3ui blockDim;
      vec3ui gridDim;
    };
    
    inline
    void fatomicMin(float *addr, float value)
    {
      float current = *(volatile float *)addr;
      while (current > value) {
        bool wasChanged
          = ((std::atomic<int>*)addr)->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }

    inline
    void fatomicMax(float *addr, float value)
    {
      float current = *(volatile float *)addr;
      while (current < value) {
        bool wasChanged
          = ((std::atomic<int>*)addr)->compare_exchange_weak((int&)current,(int&)value);
        if (wasChanged) break;
      }
    }

    struct Task {
      virtual void run(int) = 0;
    };
    template<typename T>
    struct TaskWrapper : public Task {
      TaskWrapper(const T &t) : t(t) {}
      void run(int tid) override { t(tid); }
      const T t;
    };
    
    struct LaunchSystem {
      LaunchSystem();
      void launchAndWait(int numTotal, Task *task);
      void threadFct();

      std::vector<std::thread> threads;

      struct {
        volatile int total = 0;
        std::atomic<int> taken;
      } numJobs;
      
      Task *volatile task = 0;

      std::mutex mutex;
      std::barrier<void(*)() noexcept> barrier;
    };

    

    template<typename TASK_T>
    inline void serial_for(int nTasks, TASK_T&& taskFunction)
    {
      for (int taskIndex = 0; taskIndex < nTasks; ++taskIndex) {
        taskFunction(taskIndex);
      }
    }
    
    struct Dim3Helper : public ::owl::common::vec3ui {
      Dim3Helper(uint32_t ui) : vec3ui(ui,1,1) {}
      Dim3Helper(vec2ui v) : vec3ui(v.x,v.y,1) {}
      Dim3Helper(vec3ui v) : vec3ui(v) {}
    };
  }
}

# define __rtc_global /*static*/
# define __rtc_launch(dev,kernel,nb,bs,...)                             \
  {                                                                     \
    BARNEY_NS::rtc::LaunchSystem *ls                                    \
      = ((BARNEY_NS::rtc::Device *)dev)->ls;                            \
    ::BARNEY_NS::rtc::Dim3Helper _nb(nb);                               \
    ::BARNEY_NS::rtc::Dim3Helper _bs(bs);                               \
    int numTotal = _nb.x*_nb.y*_nb.z;                                   \
    BARNEY_NS::rtc::TaskWrapper task([&](int taskID)                    \
    {                                                                   \
    ::BARNEY_NS::rtc::ComputeInterface ci;                              \
    ci.gridDim  = _nb;                                                  \
    ci.blockDim = _bs;                                                  \
    ci.blockIdx.z = taskID / (_nb.x*_nb.y);                             \
    taskID -= ci.blockIdx.z*(_nb.x*_nb.y);                              \
    ci.blockIdx.y = taskID / (_nb.x);                                   \
    taskID -= ci.blockIdx.y*(_nb.x);                                    \
    ci.blockIdx.x = taskID;                                             \
    ci.blockIdx = {(unsigned)taskID,0u,0u};                             \
    ci.threadIdx = {0u,0u,0u};                                          \
    for (ci.threadIdx.z=0;ci.threadIdx.z<_bs.z;ci.threadIdx.z++)        \
      for (ci.threadIdx.y=0;ci.threadIdx.y<_bs.y;ci.threadIdx.y++)      \
        for (ci.threadIdx.x=0;ci.threadIdx.x<_bs.x;ci.threadIdx.x++){   \
          kernel(ci,__VA_ARGS__);                                       \
        }                                                               \
    });                                                                 \
    ls->launchAndWait(numTotal,&task);                                  \
  }                                               
