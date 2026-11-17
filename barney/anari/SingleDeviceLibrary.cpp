// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Device.h"
// anari
#include "anari/backend/LibraryImpl.h"
#if BARNEY_MPI
# include "MPIDevice.h"
#endif
//#include "generated/anari_library_barney_export.h"

#ifdef ANARI_LIBRARY_BARNEY_STATIC_DEFINE
#  define BARNEY_LIBRARY_INTERFACE
#  define ANARI_LIBRARY_BARNEY_NO_EXPORT
#else
#  ifndef BARNEY_LIBRARY_INTERFACE
#    if defined(_MSC_VER)
#      if defined(anari_library_barney_EXPORTS) || defined(anari_library_barney_mpi_EXPORTS)
/* We are building this library */
#        define BARNEY_LIBRARY_INTERFACE __declspec(dllexport)
#      else
/* We are using this library */
#        define BARNEY_LIBRARY_INTERFACE /* __declspec(dllimport) */
#      endif
#    else
#      if defined(anari_library_barney_EXPORTS) || defined(anari_library_barney_mpi_EXPORTS)
/* We are building this library */
#        define BARNEY_LIBRARY_INTERFACE __attribute__((visibility("default")))
#      else
/* We are using this library */
#        define BARNEY_LIBRARY_INTERFACE __attribute__((visibility("default")))
#      endif
#    endif
#  endif
#endif

// #include "generated/anari_library_barney_queries.h"

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace BARNEY_NS {
  namespace anari {

    // using barney_device::query_extensions;
  
    struct BarneyLibrary : public ::anari::LibraryImpl
    {
      BarneyLibrary(void *lib,
                    ANARIStatusCallback defaultStatusCB,
                    const void *statusCBPtr);

      
      ANARIDevice newDevice(const char *subtype) override;
      const char **getDeviceExtensions(const char *deviceType) override;
    };

    // Definitions ////////////////////////////////////////////////////////////////

    BarneyLibrary::BarneyLibrary(void *lib,
                                 ANARIStatusCallback defaultStatusCB,
                                 const void *statusCBPtr)
      : ::anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
    {}

    ANARIDevice BarneyLibrary::newDevice(const char *_subType)
    {
      std::string subType = _subType ? _subType : "default";
      if (subType == "mpi") {
#if BARNEY_MPI
      try {
        return (ANARIDevice) new BarneyMPIDevice(this_library(), subType);
      } catch (std::exception &e) {
        std::cout << "could not create barney MPI device '" << TOSTRING(BARNEY_NS)
                  << ": " << e.what() << std::endl;
        return (ANARIDevice)0;
      }
#else
        throw std::runtime_error("asking for MPI device, but this version of barney was built without MPI support");
#endif
      }
      
      try {
        return (ANARIDevice) new BarneyDevice(this_library(), subType);
      } catch (std::exception &e) {
        std::cout << "could not create barney device '" << TOSTRING(BARNEY_NS)
                  << ": " << e.what() << std::endl;
        return (ANARIDevice)0;
      }
    }

    const char **BarneyLibrary::getDeviceExtensions(const char *subType)
    {
      try {
        BarneyDevice *device = (BarneyDevice*)newDevice(subType);
        const char **extensions = device->extensions();
        delete device;
        return extensions;
      } catch (std::exception &e) {
        std::cout << "could not create barney device '" << TOSTRING(BARNEY_NS)
                  << ": " << e.what() << std::endl;
        return 0;
      }
    }

    /*! 'entrypoint' for this anari library that creates and
        ANARILibrary object. this function is specific to one fixed
        backend, and will either return a working library object, or a
        null handle if - for some reason - that library could not be
        created. The actual library _should_ test in its constructor
        if it is actually able to create devices on the current system
        (eg, if the system actually has optix support in the driver),
        and bail out with an exception if it cannot. */
    BARNEY_LIBRARY_INTERFACE
    ::anari::LibraryImpl *createAnariLibrary(void *_libShouldBeNull,
                                             ANARIStatusCallback statusCallback,
                                             const void *scbPtr)
    {
      if (bnPhysicalDeviceCount() < 1) 
        return nullptr;
      try {
        ::anari::LibraryImpl *lib
          = new BarneyLibrary(lib, statusCallback, scbPtr);
        return lib;
      } catch (std::exception e) {
        std::cout << "could not create " << TOSTRING(BARNEY_NS)
                  << " library: " << e.what() << std::endl;
        return 0;
      }
    }

    /*! the 'official' anari/helium entry point for this library. we
        simply route that to our C++ namespace'd creator function
        (which we need, too, because multi-device library relies on
        that) */
    extern "C" BARNEY_LIBRARY_INTERFACE
    ANARILibrary BARNEY_LIBRARY_ENTRYPOINT_NAME(void *lib,
                                                ANARIStatusCallback statusCallback,
                                                const void *scbPtr)
    {
      return (ANARILibrary)createAnariLibrary(lib,statusCallback,scbPtr);
    }
    
  }
} // namespace BANARI_NS

// have to have this in a auto-generated file because we have to get
// cmake to insert BARNEY_DEVIEC_NAME - can't do that through #defines
// because the entrypoint macro is itself a macro so it wont'
// substitute recursively
// #include "Library_entryPoint.h"

