// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/Buffer.h"
#include <cstdlib>
#include <limits>
#include <new>
#ifdef _WIN32
#include <malloc.h>
#endif

namespace BARNEY_NS {
  namespace rtc {

    static void *allocateBuffer(size_t numBytes)
    {
      if (!numBytes) return nullptr;
      // NanoVDB grids require 32-byte alignment, including on the CPU backend.
      constexpr size_t alignment = 32;
      if (numBytes > std::numeric_limits<size_t>::max() - (alignment - 1))
        throw std::bad_alloc();
      const size_t padded = (numBytes + alignment - 1) & ~(alignment - 1);
#ifdef _WIN32
      void *mem = _aligned_malloc(padded, alignment);
#else
      void *mem = std::aligned_alloc(alignment, padded);
#endif
      if (!mem) throw std::bad_alloc();
      return mem;
    }

    static void freeBuffer(void *mem)
    {
#ifdef _WIN32
      _aligned_free(mem);
#else
      std::free(mem);
#endif
    }

    Buffer::Buffer(Device *device,
                   size_t numBytes,
                   const void *initMem)
    {
      mem = allocateBuffer(numBytes);
      if (initMem)
        memcpy(mem,initMem,numBytes);
    }
    
    Buffer::~Buffer()
    {
      freeBuffer(mem);
    }

    void Buffer::resize(size_t numBytes)
    {
      void *newMem = allocateBuffer(numBytes);
      freeBuffer(mem);
      mem = newMem;
    }
    
    void *Buffer::getDD() const
    {
      return mem;
    }

    void Buffer::upload(const void *hostPtr,
                        size_t numBytes,
                        size_t ofs)
    {
      memcpy(((uint8_t*)mem) + ofs,hostPtr,numBytes);
    }
      
  }
}
