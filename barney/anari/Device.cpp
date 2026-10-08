// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barneyDeviceConfig.h"
#include "anari/Device.h"
#include "anari/Array.h"
#include "anari/Frame.h"
#include "anari/BaseDevice.h"
// std
#include <cstring>

#include "generated_queries.h"

namespace BARNEY_NS {
  namespace anari {
  
    // Data Arrays ////////////////////////////////////////////////////////////////

    ANARIArray1D BarneyDevice::newArray1D(const void *appMemory,
                                          ANARIMemoryDeleter deleter,
                                          const void *userData,
                                          ANARIDataType type,
                                          uint64_t numItems)
    {
      helium::Array1DMemoryDescriptor md;
      md.appMemory = appMemory;
      md.deleter = deleter;
      md.deleterPtr = userData;
      md.elementType = type;
      md.numItems = numItems;

      if (anari::isObject(type))
        return (ANARIArray1D) new ObjectArray(deviceState(), md);
      else
        return (ANARIArray1D) new Array1D(deviceState(), md);
    }

    ANARIArray2D BarneyDevice::newArray2D(const void *appMemory,
                                          ANARIMemoryDeleter deleter,
                                          const void *userData,
                                          ANARIDataType type,
                                          uint64_t numItems1,
                                          uint64_t numItems2)
    {
      helium::Array2DMemoryDescriptor md;
      md.appMemory = appMemory;
      md.deleter = deleter;
      md.deleterPtr = userData;
      md.elementType = type;
      md.numItems1 = numItems1;
      md.numItems2 = numItems2;

      return (ANARIArray2D) new Array2D(deviceState(), md);
    }

    ANARIArray3D BarneyDevice::newArray3D(const void *appMemory,
                                          ANARIMemoryDeleter deleter,
                                          const void *userData,
                                          ANARIDataType type,
                                          uint64_t numItems1,
                                          uint64_t numItems2,
                                          uint64_t numItems3)
    {
      helium::Array3DMemoryDescriptor md;
      md.appMemory = appMemory;
      md.deleter = deleter;
      md.deleterPtr = userData;
      md.elementType = type;
      md.numItems1 = numItems1;
      md.numItems2 = numItems2;
      md.numItems3 = numItems3;

      return (ANARIArray3D) new Array3D(deviceState(), md);
    }

    // Renderable Objects /////////////////////////////////////////////////////////

    ANARILight BarneyDevice::newLight(const char *subtype)
    {
      return (ANARILight)Light::createInstance(subtype, deviceState());
    }

    ANARICamera BarneyDevice::newCamera(const char *subtype)
    {
      ANARICamera cam = (ANARICamera)Camera::createInstance(subtype, deviceState());
      assert(cam);
      return cam;
    }

    ANARIGeometry BarneyDevice::newGeometry(const char *subtype)
    {
      return (ANARIGeometry)Geometry::createInstance(subtype, deviceState());
    }

    ANARISpatialField BarneyDevice::newSpatialField(const char *subtype)
    {
      return (ANARISpatialField)SpatialField::createInstance(subtype, deviceState());
    }

    ANARISurface BarneyDevice::newSurface()
    {
      return (ANARISurface) new Surface(deviceState());
    }

    ANARIVolume BarneyDevice::newVolume(const char *subtype)
    {
      return (ANARIVolume)Volume::createInstance(subtype, deviceState());
    }

    // Model Meta-Data ////////////////////////////////////////////////////////////

    ANARIMaterial BarneyDevice::newMaterial(const char *subtype)
    {
      return (ANARIMaterial)Material::createInstance(subtype, deviceState());
    }

    ANARISampler BarneyDevice::newSampler(const char *subtype)
    {
      return (ANARISampler)Sampler::createInstance(subtype, deviceState());
    }

    // Instancing /////////////////////////////////////////////////////////////////

    ANARIGroup BarneyDevice::newGroup()
    {
      return (ANARIGroup) new Group(deviceState());
    }

    ANARIInstance BarneyDevice::newInstance(const char * /*subtype*/)
    {
      return (ANARIInstance) new Instance(deviceState());
    }

    // Top-level Worlds ///////////////////////////////////////////////////////////

    ANARIWorld BarneyDevice::newWorld()
    {
      return (ANARIWorld) new World(deviceState());
    }

    // Query functions ////////////////////////////////////////////////////////////

    const char **BarneyDevice::getObjectSubtypes(ANARIDataType objectType)
    {
      return BARNEY_NS::query_object_types(objectType);
    }

    const void *BarneyDevice::getObjectInfo(ANARIDataType objectType,
                                            const char *objectSubtype,
                                            const char *infoName,
                                            ANARIDataType infoType)
    {
      return BARNEY_NS::query_object_info(
                                              objectType, objectSubtype, infoName, infoType);
    }

    const void *BarneyDevice::getParameterInfo(ANARIDataType objectType,
                                               const char *objectSubtype,
                                               const char *parameterName,
                                               ANARIDataType parameterType,
                                               const char *infoName,
                                               ANARIDataType infoType)
    {
      return BARNEY_NS::query_param_info(objectType,
                                             objectSubtype,
                                             parameterName,
                                             parameterType,
                                             infoName,
                                             infoType);
    }

    // Object + Parameter Lifetime Management /////////////////////////////////////

    int BarneyDevice::getProperty(ANARIObject object,
                                  const char *name,
                                  ANARIDataType type,
                                  void *mem,
                                  uint64_t size,
                                  uint32_t mask)
    {
      if (mask == ANARI_WAIT) {
        auto lock = scopeLockObject();
      }

      return helium::BaseDevice::getProperty(object, name, type, mem, size, mask);
    }

    // Frame Manipulation /////////////////////////////////////////////////////////

    ANARIFrame BarneyDevice::newFrame()
    {
      return (ANARIFrame) new Frame(deviceState());
    }

    // Frame Rendering ////////////////////////////////////////////////////////////

    ANARIRenderer BarneyDevice::newRenderer(const char *)
    {
      return (ANARIRenderer) new Renderer(deviceState());
    }

    // Helper/other functions and data members ////////////////////////////////////

    std::vector<std::string> splitString(std::string s, const char delim)
    {
      std::vector<std::string> res;
      while (true) {
        size_t pos = s.find(delim);
        if (pos == s.npos)
          break;
        res.push_back(s.substr(0, pos));
        s = s.substr(pos + 1);
      }
      res.push_back(s);
      return res;
    }

    static void default_statusFunc(const void * /*userData*/,
                                   ANARIDevice /*device*/,
                                   ANARIObject source,
                                   ANARIDataType /*sourceType*/,
                                   ANARIStatusSeverity severity,
                                   ANARIStatusCode /*code*/,
                                   const char *message)
    {
      if (severity == ANARI_SEVERITY_FATAL_ERROR) {
        fprintf(stderr, "[FATAL][%p] %s\n", source, message);
        std::exit(1);
      } else if (severity == ANARI_SEVERITY_ERROR) {
        fprintf(stderr, "[ERROR][%p] %s\n", source, message);
#ifndef NDEBUG
      } else if (severity == ANARI_SEVERITY_WARNING) {
        fprintf(stderr, "[WARN ][%p] %s\n", source, message);
      } else if (severity == ANARI_SEVERITY_PERFORMANCE_WARNING) {
        fprintf(stderr, "[PERF ][%p] %s\n", source, message);
#endif
      }
      // Ignore INFO/DEBUG messages
    }

    BarneyDevice::BarneyDevice(ANARILibrary l, const std::string &subType)
      : barney::BarneyBaseDevice(l), deviceType(subType)
    {
      m_state = std::make_unique<BarneyGlobalState>(this_device());
    }

    BarneyDevice::BarneyDevice()
      : barney::BarneyBaseDevice(default_statusFunc, nullptr)
    {
      m_state = std::make_unique<BarneyGlobalState>(this_device());
    }


    BarneyDevice::~BarneyDevice()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: ~BarneyDevice is deconstructing"
                         << std::endl);
      auto &state = *deviceState(false);
      state.commitBuffer.clear();
      reportMessage(ANARI_SEVERITY_DEBUG, "destroying barney device (%p)", this);
    }

    void BarneyDevice::initDevice()
    {
      if (m_initialized)
        return;

      reportMessage(ANARI_SEVERITY_DEBUG, "initializing barney device (%p)", this);

      auto state = deviceState();
    
      // non-mpi will just route this to a no-op, overloaded mpiDevice
      // will do mpi_init()
      initMPI();
    
      std::vector<int> gpuIDs;
      int numGPUs = rtc::physicalDeviceCount();
      if (state->tether->devices[0]->m_cudaDevice >= 0) {
        // first device DID have a cudadevice explicitly set; let's
        // assume that app then explciitly sets cudaDevice for every
        // other device, too, and let's use that list of GPUs.
        for (auto dev : state->tether->devices) {
          reportMessage(ANARI_SEVERITY_DEBUG,
                        "adding requested GPU ID %i",
                        dev->m_cudaDevice);
          assert(dev->m_cudaDevice >= 0);
          gpuIDs.push_back(dev->m_cudaDevice % numGPUs);
        }
      } else {
        // first device did NOT have a cudadevice set - this means the
        // app is NOT using explicit tethering (where it creates a
        // (tethered) anari device for each gpu), which in turn means
        // we're an independent device that didn't have any cuda gpu
        // explicitly attached to it.... which means we could in theory
        // use multiple GPU" << 
        if (m_enable_multiGPU) {
          std::cout << "#banari: no GPUs selected, but multi-gpu enabled"
                    << " -> picking all " << numGPUs << " rtc devices"
                    << std::endl;
          for (int i=0;i<numGPUs;i++)
            gpuIDs.push_back(i);
        } else {
          std::cout << "#banari: no GPUs selected, and multi-gpu disabled"
                    << " -> picking gpu 0"
                    << std::endl;
          gpuIDs = {0};
        }
      }

      std::vector<int> dgIDs;
      for (auto dev : state->tether->devices) {
        int dgID = dev->m_dataGroupID;
        if (dgID == -1)
          // not set by user, use default of different data group per
          // MPI rank. note we do not consider the number of tehtered
          // devices here - if the app _did_ know how to tether it
          // should also have been able to set the data rank, so the
          // only tiem we have to worry about this is if somebody
          // starts an anari MPI device (which means he _does_ want to
          // render data parallel) withotu even knowing how to specify
          // data ranks or tethering... which only makes sense if what
          // that user is trying to do is data parallel rendering with
          // one data group per rank.
          dgID = multiNode.rank;
        dgIDs.push_back(dgID);
      }

      if (1 || multiNode.size > 1) {
        std::stringstream ss;
        ss << "#banari rank " << multiNode.rank << " (of " << multiNode.size
           << ") creating context GPUs=(";
        for (auto gpu : gpuIDs)
          ss << " " << gpu;
        ss << " ) and data groups=(";
        for (auto dg : dgIDs)
          ss << " " << dg;
        ss << " )";
        
        std::cout << ss.str() << std::endl;
        reportMessage(ANARI_SEVERITY_DEBUG, ss.str().c_str());
      }

      state->tether->context = createContext(dgIDs,gpuIDs);
      
      m_initialized = true;
    }

    void BarneyDevice::deviceCommitParameters()
    {
      helium::BaseDevice::deviceCommitParameters();

      auto state = deviceState(false);
      if (state->hasBeenCommitted) {
        reportMessage(ANARI_SEVERITY_DEBUG, "device committed more than once!");
        return;
      } else {
        state->hasBeenCommitted = true;
      }
      m_cudaDevice = getParam<int>("cudaDevice", m_cudaDevice);
      m_dataGroupID = getParam<int>("dataGroupID", m_dataGroupID);

      m_enable_multiGPU
        = getParam<int>("enable_multiGPU",m_enable_multiGPU);
      if (m_cudaDevice != -2)
        std::cout << "#banari: found 'cudaDevice' = " << m_cudaDevice << std::endl;

      if (!state->tether) {
        tetherIndex = getParam<int>("tetherIndex", tetherIndex);
        tetherCount = getParam<int>("tetherCount", 1);
        state->slot = tetherIndex;
        auto tetherDev = getParam<::anari::Device>("tetherDevice", (::anari::Device)0);
        tetherDevice = (BarneyDevice *)tetherDev;

        assert(tetherCount > 0);
        assert(tetherIndex >= 0);
        assert(tetherIndex < tetherCount);

#ifndef NDEBUG
        std::cout << "#banari: FIRST-TIME device initialization slot "
                  << tetherIndex << "/" << tetherCount << " in tethered dev "
                  << tetherDevice << std::endl;
#endif
        if (tetherDevice) {
          state->tether = tetherDevice->deviceState()->tether;
          assert(state->tether);
          assert(tetherCount == state->tether->devices.size());
          assert(state->tether->devices[tetherIndex] == nullptr);
          state->tether->devices[tetherIndex] = this;
        } else {
          assert(tetherIndex == 0
                 && "first device has to be first to be created");
          state->tether = std::make_shared<Tether>();
          state->tether->devices.resize(tetherCount);
          state->tether->devices[0] = this;
        }
        if (state->tether->allDevicesPresent()) {
          assert(state->tether->context == 0
                 && "only last device to tether should create the barney context");
          initDevice();
        }
      }

    }

    int BarneyDevice::deviceGetProperty(const char *name,
                                        ANARIDataType type,
                                        void *mem,
                                        uint64_t size,
                                        uint32_t mask)
    {
      std::string_view prop = name;
      if (prop == "extension" && type == ANARI_STRING_LIST) {
        helium::writeToVoidP(mem, query_extensions());
        return 1;
      } else if (prop == "barney" && type == ANARI_BOOL) {
        helium::writeToVoidP(mem, true);
        return 1;
      }
      return 0;
    }

    BarneyGlobalState *BarneyDevice::deviceState(bool commitOnDemand)
    {
      BarneyGlobalState *state =
        (BarneyGlobalState *)helium::BaseDevice::m_state.get();
      if (commitOnDemand && !state->hasBeenCommitted)
        deviceCommitParameters();
      return state;
    }

    BNContext BarneyDevice::createContext(const std::vector<int> &dataRanks,
                                          const std::vector<int> &gpuIDs)
    {
      BNContext ctx =
        bnContextCreate(dataRanks.size(),
                        dataRanks.data(),
                        gpuIDs.size(),
                        gpuIDs.data());
      return ctx;
    }

    const char **BarneyDevice::extensions() 
    {
      return query_extensions();
    }
    
    extern "C" barney::BarneyBaseDevice *                               
    CREATE_DEVICE_FUNCTION_NAME                           
    (ANARILibrary library, const char *subType)         
    { return new BarneyDevice(library,subType); }       
  }
}

