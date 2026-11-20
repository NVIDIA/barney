// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// helium
#include "helium/BaseDevice.h"

// iw - do NOT include barneyDeviceConfig.h here - this is a
// auto-generated header file with backend-specific configs that only
// exists in backend specific build dirs, and that would conflict with
// other backends if included, anyway.
// #include "barneyDeviceConfig.h"

namespace barney {
  
  struct BarneyBaseDevice : public helium::BaseDevice
  {
    BarneyBaseDevice() = default;
    BarneyBaseDevice(ANARIStatusCallback cb, const void *userPtr)
      : helium::BaseDevice(cb,userPtr)
    {}
    BarneyBaseDevice(ANARILibrary library)
      : helium::BaseDevice(library)
    {}
    
    virtual const char **extensions() = 0;
  };
  
}
