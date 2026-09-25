// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

// helium
#include "helium/BaseDevice.h"
#include "anari/BaseDevice.h"
#include "anari/BarneyGlobalState.h"
#include <map>

namespace BARNEY_NS {
  namespace anari {

    // struct BarneyDevice : public helium::BaseDevice
    struct BarneyDevice : public BARNEY_LIBRARY_NAME::BarneyBaseDevice
    {
      BarneyDevice();
      BarneyDevice(ANARILibrary library, const std::string &subType = "default");
      ~BarneyDevice() override;

      // Data Arrays //////////////////////////////////////////////////////////

      ANARIArray1D newArray1D(const void *appMemory,
                              ANARIMemoryDeleter deleter,
                              const void *userdata,
                              ANARIDataType,
                              uint64_t numItems1) override;

      ANARIArray2D newArray2D(const void *appMemory,
                              ANARIMemoryDeleter deleter,
                              const void *userdata,
                              ANARIDataType,
                              uint64_t numItems1,
                              uint64_t numItems2) override;

      ANARIArray3D newArray3D(const void *appMemory,
                              ANARIMemoryDeleter deleter,
                              const void *userdata,
                              ANARIDataType,
                              uint64_t numItems1,
                              uint64_t numItems2,
                              uint64_t numItems3) override;

      // Renderable Objects //////////////////////////////////////////////////

      ANARILight newLight(const char *type) override;

      ANARICamera newCamera(const char *type) override;

      ANARIGeometry newGeometry(const char *type) override;
      ANARISpatialField newSpatialField(const char *type) override;

      ANARISurface newSurface() override;
      ANARIVolume newVolume(const char *type) override;

      // Surface Meta-Data ///////////////////////////////////////////////////

      ANARIMaterial newMaterial(const char *material_type) override;

      ANARISampler newSampler(const char *type) override;

      // Instancing //////////////////////////////////////////////////////////

      ANARIGroup newGroup() override;

      ANARIInstance newInstance(const char *type) override;

      // Top-level Worlds ////////////////////////////////////////////////////

      ANARIWorld newWorld() override;

      // Query functions /////////////////////////////////////////////////////

      const char **getObjectSubtypes(ANARIDataType objectType) override;
      const void *getObjectInfo(ANARIDataType objectType,
                                const char *objectSubtype,
                                const char *infoName,
                                ANARIDataType infoType) override;
      const void *getParameterInfo(ANARIDataType objectType,
                                   const char *objectSubtype,
                                   const char *parameterName,
                                   ANARIDataType parameterType,
                                   const char *infoName,
                                   ANARIDataType infoType) override;

      // Object + Parameter Lifetime Management //////////////////////////////

      int getProperty(ANARIObject object,
                      const char *name,
                      ANARIDataType type,
                      void *mem,
                      uint64_t size,
                      ANARIWaitMask mask) override;

      // FrameBuffer Manipulation ////////////////////////////////////////////

      ANARIFrame newFrame() override;

      // Frame Rendering /////////////////////////////////////////////////////

      ANARIRenderer newRenderer(const char *type) override;

      ////////////////////////////////////////////////////////////////////////
      // Helper/other functions and data members
      ////////////////////////////////////////////////////////////////////////

      virtual BNContext createContext(const std::vector<int> &dataRanks,
                                      const std::vector<int> &gpuIDs);
      virtual void initMPI() { /* nothing by default, mpidev can override */};

      const char **extensions() override;

    protected:
      void initDevice();
      void deviceCommitParameters() override;
      int deviceGetProperty(const char *name,
                            ANARIDataType type,
                            void *mem,
                            uint64_t size,
                            uint32_t flags) override;
      BarneyGlobalState *deviceState(bool commitOnDemand=true);

      bool m_initialized{false};
    
      /*! allows for setting which gpu to use. if set to -1 (the
          default) this means "let backend decide" (so cuda or optix
          might use more than one gpu).  */
      int m_cudaDevice = -1;
      int m_dataGroupID = -1;
      /*! allows the app to say "use as many gpus as you can find", without */
      // #if BARNEY_MPI
      //     int m_enable_multiGPU = 0;
      // #else
      int m_enable_multiGPU = 1;
      // #endif
      const std::string deviceType = "default";
      bool hasBeenCommitted = false;
      BarneyDevice *tetherDevice = 0;
      int tetherIndex = 0;
      int tetherCount = 0;

      struct { int rank=0, size=1; } multiNode;
    };

  }
}
