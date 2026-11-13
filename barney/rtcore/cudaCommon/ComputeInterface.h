// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/*! file rtcore/cudaCommon/ComputeInterface.h Abstracts device-side
  compute operations (like atomics, texture-sampling, etc), defines
  the interace with which compute kernels can talk to the device (to
  get thread idx etc), and the EXPORT macros to define device
  kernels that host code can later import (using the
  ComputeKernel.h:IMPORT macros) */
#pragma once

#include "rtcore/cudaCommon/cuda-common.h"

namespace BARNEY_NS {
  namespace rtc {

    // ==================================================================
    // INTERFACE
    // ==================================================================

    inline __device__ void fatomicMin(float *addr, float value);
    inline __device__ void fatomicMax(float *addr, float value);

    /* texturing wrappers; can only be instantiated for 'flaot' and
       'vec4f' types */
    template<typename T> inline __device__
    T tex1D(rtc::TextureObject to, float x);
    
    /* texturing wrappers; can only be instantiated for 'flaot' and
       'vec4f' types */
    template<typename T> inline __device__
    T tex2D(rtc::TextureObject to, float x, float y);

    /* texturing wrappers; can only be instantiated for 'flaot' and
       'vec4f' types */
    template<typename T> inline __device__
    T tex3D(rtc::TextureObject to, float x, float y, float z);
    
    struct ComputeInterface
    {
#if RTC_DEVICE_CODE
      inline __device__ vec3ui getLaunchIndex() const
      {
        return getThreadIdx() + getBlockIdx() * getBlockDim();
      }
      inline __device__ vec3ui getThreadIdx() const
      { return vec3ui(threadIdx.x,threadIdx.y,threadIdx.z); }
      inline __device__ vec3ui getBlockDim() const
      { return vec3ui(blockDim.x,blockDim.y,blockDim.z); }
      inline __device__ vec3ui getBlockIdx() const
      { return vec3ui(blockIdx.x,blockIdx.y,blockIdx.z); }
      inline __device__ int atomicAdd(int *ptr, int inc) const
      { return ::atomicAdd(ptr,inc); }
      inline __device__ float atomicAdd(float *ptr, float inc) const
      { return ::atomicAdd(ptr,inc); }
#endif
    };
    
    // ==================================================================
    // INLINE IMPLEMENTATION
    // ==================================================================
#if RTC_DEVICE_CODE
    // ------------------------------------------------------------------
    // cuda texturing
    // ------------------------------------------------------------------

    template<> inline __device__
    float tex2D<float>(rtc::TextureObject to, float x, float y)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      return ::tex2D<float>(texObj,x,y);
    }

    template<> inline __device__
    float tex3D<float>(rtc::TextureObject to, float x, float y, float z)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      float f= ::tex3D<float>(texObj,x,y,z);
      return f;
    }

    template<> inline __device__
    vec4f tex1D<vec4f>(rtc::TextureObject to, float x)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex1D<::float4>(texObj,x);
      return vec4f(v.x,v.y,v.z,v.w);
    }
    
    template<> inline __device__
    vec4f tex2D<vec4f>(rtc::TextureObject to, float x, float y)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex2D<::float4>(texObj,x,y);
      return vec4f(v.x,v.y,v.z,v.w);
    }

    template<> inline __device__
    vec4f tex3D<vec4f>(rtc::TextureObject to, float x, float y, float z)
    {
      cudaTextureObject_t texObj = (const cudaTextureObject_t&)to;
      ::float4 v = ::tex3D<::float4>(texObj,x,y,z);
      return vec4f(v.x,v.y,v.z,v.w);
    }
    
    // ------------------------------------------------------------------
    // atomics
    // ------------------------------------------------------------------
    using ::atomicCAS;

    inline __device__
    void fatomicMin(float *addr, float value)
    {
      float old = *(volatile float *)addr;
      if(old <= value) return;

      int _expected = __float_as_int(old);
      int _desired  = __float_as_int(value);
      while (true) {
        uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
        if (_found == _expected)
          // write went though; we _did_ write the new mininm and
          // are done.
          return;
        // '_expected' changed, so write did not go through, and
        // somebody else wrote something new to that location.
        old = __int_as_float(_found);
        if (old <= value)
          // somebody else wrote something that's already smaller
          // than what we have ... leave it be, and done.
          return;
        else {
          // somebody else wrote something, but ours is _still_ smaller.
          _expected = _found;
          continue;
        }
      } 
    }

    inline __device__
    void fatomicMax(float *addr, float value)
    {
      float old = *(volatile float *)addr;
      if(old >= value) return;

      int _expected = __float_as_int(old);
      int _desired  = __float_as_int(value);
      while (true) {
        uint32_t _found = atomicCAS((int*)addr,_expected,_desired);
        if (_found == _expected)
          // write went though; we _did_ write the new mininm and
          // are done.
          return;
        // '_expected' changed, so write did not go through, and
        // somebody else wrote something new to that location.
        old = __int_as_float(_found);
        if (old >= value)
          // somebody else wrote something that's already smaller
          // than what we have ... leave it be, and done.
          return;
        else {
          // somebody else wrote something, but ours is _still_ smaller.
          _expected = _found;
          continue;
        }
      } 
    }
#endif

    struct Dim3Helper : public ::owl::common::vec3ui {
      Dim3Helper(uint32_t ui) : vec3ui(ui,1,1) {}
      Dim3Helper(vec2ui v) : vec3ui(v.x,v.y,1) {}
      Dim3Helper(vec3ui v) : vec3ui(v) {}
      inline operator dim3() const { return dim3{x,y,z}; }
    };
  }
}

# define __rtc_global __global__
# define __rtc_launch(myRTC,kernel,nb,bs,...)                           \
  {                                                                     \
    ::BARNEY_NS::rtc::SetActiveGPU forDuration(myRTC);                  \
    ::BARNEY_NS::rtc::Dim3Helper _nb(nb);                               \
    ::BARNEY_NS::rtc::Dim3Helper _bs(bs);                               \
    if (_nb.x)                                                          \
      kernel<<<_nb,_bs,0,myRTC->stream>>>                               \
              (::BARNEY_NS::rtc::ComputeInterface(), __VA_ARGS__);      \
  }
