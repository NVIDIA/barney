// SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// \author Jeff Daily <jeff.daily@amd.com>
//
// Single compatibility shim that lets barney's CUDA software ray-tracing
// backend (rtcore/cuda + rtcore/cudaCommon) compile and run on AMD GPUs
// through HIP/ROCm. This is the only header in the backend that knows about
// HIP: it includes the HIP runtime and aliases the small set of CUDA runtime
// spellings the backend actually uses to their hipXxx equivalents, so the
// CUDA sources stay in CUDA spelling and the CUDA/OptiX build is byte
// identical (this whole header is inert unless compiled by hipcc).

#pragma once

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP) || defined(__HIPCC__)

// Pull in the host libc memory routines before the HIP runtime so that
// host-side memcpy/memset resolve to the C library, not a device overload
// the HIP runtime may bring into scope.
#include <cstring>
#include <cstdlib>
#include <hip/hip_runtime.h>

// ------------------------------------------------------------------
// runtime error type + status
// ------------------------------------------------------------------
using cudaError_t = hipError_t;
#define cudaSuccess hipSuccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError   hipGetLastError

// ------------------------------------------------------------------
// device / stream management (used by BARNEY_CUDA_CALL(cuda##X) sites
// via token paste, plus a handful of bare spellings)
// ------------------------------------------------------------------
#define cudaSetDevice              hipSetDevice
#define cudaGetDevice              hipGetDevice
#define cudaGetDeviceCount         hipGetDeviceCount
#define cudaGetDeviceProperties    hipGetDeviceProperties
#define cudaDeviceProp             hipDeviceProp_t
#define cudaDeviceSynchronize      hipDeviceSynchronize
#define cudaDeviceCanAccessPeer    hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess

using cudaStream_t = hipStream_t;
#define cudaStreamNonBlocking      hipStreamNonBlocking
#define cudaStreamCreate           hipStreamCreate
#define cudaStreamCreateWithFlags  hipStreamCreateWithFlags
#define cudaStreamSynchronize      hipStreamSynchronize
#define cudaStreamDestroy          hipStreamDestroy

// ------------------------------------------------------------------
// memory
// ------------------------------------------------------------------
#define cudaMalloc            hipMalloc
#define cudaMallocManaged     hipMallocManaged
#define cudaMallocHost        hipHostMalloc
#define cudaFree              hipFree
#define cudaFreeHost          hipHostFree
#define cudaMemcpy            hipMemcpy
#define cudaMemcpyAsync       hipMemcpyAsync
#define cudaMemsetAsync       hipMemsetAsync
#define cudaMemcpyDefault       hipMemcpyDefault
#define cudaMemcpyHostToDevice  hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost  hipMemcpyDeviceToHost

// ------------------------------------------------------------------
// textures and arrays (cudaArray-backed, no pitched 2D binds)
// ------------------------------------------------------------------
using cudaTextureObject_t  = hipTextureObject_t;
using cudaArray_t          = hipArray_t;
using cudaChannelFormatDesc = hipChannelFormatDesc;
using cudaResourceDesc     = hipResourceDesc;
using cudaTextureDesc      = hipTextureDesc;
using cudaExtent           = hipExtent;
using cudaMemcpy3DParms     = hipMemcpy3DParms;
using cudaTextureFilterMode  = hipTextureFilterMode;
using cudaTextureAddressMode = hipTextureAddressMode;
using cudaTextureReadMode    = hipTextureReadMode;

#define cudaCreateChannelDesc      hipCreateChannelDesc
#define cudaCreateTextureObject    hipCreateTextureObject
#define cudaDestroyTextureObject   hipDestroyTextureObject
#define cudaMallocArray            hipMallocArray
#define cudaMalloc3DArray          hipMalloc3DArray
#define cudaFreeArray              hipFreeArray
#define cudaMemcpy2DToArray        hipMemcpy2DToArray
#define cudaMemcpy3D               hipMemcpy3D
#define make_cudaPitchedPtr        make_hipPitchedPtr

#define cudaResourceTypeArray      hipResourceTypeArray
#define cudaReadModeElementType    hipReadModeElementType
#define cudaReadModeNormalizedFloat hipReadModeNormalizedFloat
#define cudaFilterModePoint        hipFilterModePoint
#define cudaFilterModeLinear       hipFilterModeLinear
#define cudaAddressModeWrap        hipAddressModeWrap
#define cudaAddressModeClamp       hipAddressModeClamp
#define cudaAddressModeBorder      hipAddressModeBorder
#define cudaAddressModeMirror      hipAddressModeMirror

#define cudaCreateChannelDescHalf1 hipCreateChannelDescHalf1
#define cudaCreateChannelDescHalf2 hipCreateChannelDescHalf2
#define cudaCreateChannelDescHalf4 hipCreateChannelDescHalf4

#else
# include <cuda_runtime.h>
#endif
