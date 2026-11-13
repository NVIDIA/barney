// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rtcore/cpu/embree-common.h"
#include "rtcore/cpu/Device.h"
#include "rtcore/cpu/Buffer.h"
#include "rtcore/cpu/Group.h"
#include "rtcore/cpu/Denoiser.h"

namespace BARNEY_NS {
  namespace rtc {
    
    inline bool enablePeerAccess(const std::vector<int> &IDs)
    { /* ignore / no-op on embree backend */; return true; }
    
    /*! get a unique hash for a given physical device. */
    size_t getPhysicalDeviceHash(int gpuID);
    
  }
}

    

