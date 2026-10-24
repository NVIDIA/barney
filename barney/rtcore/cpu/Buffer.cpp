// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/Buffer.h"
#include <cstdlib>
#include <limits>
#include <new>

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
      void *mem = std::aligned_alloc(alignment, padded);
      if (!mem) throw std::bad_alloc();
      return mem;
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
      if (mem) free(mem);
    }

    void Buffer::resize(size_t numBytes)
    {
      void *newMem = allocateBuffer(numBytes);
      if (mem) free(mem);
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
