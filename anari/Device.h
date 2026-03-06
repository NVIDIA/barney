// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

// helium
#include "helium/BaseDevice.h"

#include "BarneyGlobalState.h"
#include <vector>

namespace barney_device {

struct BarneyDevice : public helium::BaseDevice
{
  // Data Arrays //////////////////////////////////////////////////////////////

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

  // Renderable Objects ///////////////////////////////////////////////////////

  ANARILight newLight(const char *type) override;

  ANARICamera newCamera(const char *type) override;

  ANARIGeometry newGeometry(const char *type) override;
  ANARISpatialField newSpatialField(const char *type) override;

  ANARISurface newSurface() override;
  ANARIVolume newVolume(const char *type) override;

  // Surface Meta-Data ////////////////////////////////////////////////////////

  ANARIMaterial newMaterial(const char *material_type) override;

  ANARISampler newSampler(const char *type) override;

  // Instancing ///////////////////////////////////////////////////////////////

  ANARIGroup newGroup() override;

  ANARIInstance newInstance(const char *type) override;

  // Top-level Worlds /////////////////////////////////////////////////////////

  ANARIWorld newWorld() override;

  // Query functions //////////////////////////////////////////////////////////

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

  // Object + Parameter Lifetime Management ///////////////////////////////////

  int getProperty(ANARIObject object,
      const char *name,
      ANARIDataType type,
      void *mem,
      uint64_t size,
      ANARIWaitMask mask) override;

  // FrameBuffer Manipulation /////////////////////////////////////////////////

  ANARIFrame newFrame() override;

  // Frame Rendering //////////////////////////////////////////////////////////

  ANARIRenderer newRenderer(const char *type) override;

  /////////////////////////////////////////////////////////////////////////////
  // Helper/other functions and data members
  /////////////////////////////////////////////////////////////////////////////

  BarneyDevice();
  BarneyDevice(ANARILibrary library, const std::string &subType = "default");
  ~BarneyDevice() override;

  private:
    void initDevice();
    void deviceCommitParameters() override;
    int deviceGetProperty(const char *name,
                          ANARIDataType type,
                          void *mem,
                          uint64_t size,
                          uint32_t flags) override;
    BarneyGlobalState *deviceState(bool commitOnDemand=true);

  bool m_initialized{false};

  /*! allows for setting which gpu to use. must be set before the
      first commit, and should not be changed after that. '-2' means
      'leave it to barney', '-1' means 'use cpu', any value >= 0
      means 'use this specific gpu */
  int m_cudaDevice = -2;
  int m_dataGroupID = -1;

  // Multi-slot: populated from numbered device params (numDataGroups,
  // dataGroupID0..N, cudaDevice0..N). When non-empty these override
  // the scalar m_cudaDevice / m_dataGroupID for context creation.
  std::vector<int> m_multiDataGroupIDs;
  std::vector<int> m_multiCudaDevices;
  const std::string deviceType = "default";
#if BARNEY_MPI
  /*! communicator to use for barney data-parallel rendering, set as
      a uint64_t. If set to 0, we'll use local rendering even if mpi
      support is compiled in, any other value will be interpreted as
      a MPI_Comm type. If device gets created with subtype "mpi" or
      "default", the default value for comm is MPI_COMM_WORLD, if it
      is created with subtype "local" it will default to 0 */
  MPI_Comm comm = MPI_COMM_WORLD;
  bool     commNeedsFree = false;
#endif
  bool hasBeenCommitted = false;
  BarneyDevice *tetherDevice = 0;
  int tetherIndex = 0;
  int tetherCount = 0;
};

} // namespace barney_device
