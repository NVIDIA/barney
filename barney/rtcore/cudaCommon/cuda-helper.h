// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cudaCommon/cuda_to_hip.h"

#if defined(__HIP_PLATFORM_AMD__) || defined(USE_HIP) || defined(__HIPCC__)
# define BARNEY_GPU_RUNTIME "HIP"
#else
# define BARNEY_GPU_RUNTIME "CUDA"
#endif

#ifdef NDEBUG
#define BARNEY_RAISE(MSG) throw std::runtime_error("fatal barney " BARNEY_GPU_RUNTIME " error ... ")
#else
#define BARNEY_RAISE(MSG) { std::cerr << MSG << std::endl; assert(0); }
#endif

#define BARNEY_CUDA_CHECK( call )                                       \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      printf("error code %i\n",rc); fflush(0);                          \
      fprintf(stderr,                                                   \
              BARNEY_GPU_RUNTIME                                   \
              " call (%s) failed with code %d (line %d): %s\n",       \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      BARNEY_RAISE("fatal " BARNEY_GPU_RUNTIME " error");         \
    }                                                                   \
  }

#define BARNEY_CUDA_CALL(call) BARNEY_CUDA_CHECK(cuda##call)

#define BARNEY_CUDA_CHECK2( where, call )                               \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: " BARNEY_GPU_RUNTIME              \
                " call (%s) "                                          \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              BARNEY_GPU_RUNTIME                                   \
              " call (%s) failed with code %d (line %d): %s\n",       \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      BARNEY_RAISE("fatal " BARNEY_GPU_RUNTIME " error");         \
    }                                                                   \
  }

#define BARNEY_CUDA_SYNC_CHECK()                                \
  {                                                             \
    cudaError_t rc0 = cudaDeviceSynchronize();                  \
    assert(rc0 == cudaSuccess);                                 \
    cudaError_t rc = cudaGetLastError();                        \
    if (rc != cudaSuccess) {                                    \
      fprintf(stderr, "error (%s: line %d): %s\n",              \
              __FILE__, __LINE__, cudaGetErrorString(rc));      \
      BARNEY_RAISE("fatal " BARNEY_GPU_RUNTIME " error"); \
    }                                                           \
  }

#define BARNEY_CUDA_CHECK_NOTHROW( call )                               \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if (rc != cudaSuccess) {                                            \
      fprintf(stderr,                                                   \
              BARNEY_GPU_RUNTIME                                   \
              " call (%s) failed with code %d (line %d): %s\n",       \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#define BARNEY_CUDA_CALL_NOTHROW(call) BARNEY_CUDA_CHECK_NOTHROW(cuda##call)

#define BARNEY_CUDA_CHECK2_NOTHROW( where, call )                       \
  {                                                                     \
    cudaError_t rc = call;                                              \
    if(rc != cudaSuccess) {                                             \
      if (where)                                                        \
        fprintf(stderr, "at %s: " BARNEY_GPU_RUNTIME              \
                " call (%s) "                                          \
                "failed with code %d (line %d): %s\n",                  \
                where,#call, rc, __LINE__, cudaGetErrorString(rc));     \
      fprintf(stderr,                                                   \
              BARNEY_GPU_RUNTIME                                   \
              " call (%s) failed with code %d (line %d): %s\n",       \
              #call, rc, __LINE__, cudaGetErrorString(rc));             \
      exit(2);                                                          \
    }                                                                   \
  }

#ifndef CHECK_CUDA_LAUNCH
# define CHECK_CUDA_LAUNCH(kernel,_nb,_bs,_shm,_s,...)  \
  kernel<<<_nb,_bs,_shm,_s>>>(__VA_ARGS__);
#endif
