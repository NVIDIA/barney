// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// anari
#include "anari/backend/LibraryImpl.h"
#include "anari/BaseDevice.h"
#include <map>
#include <string>
#include "owl/common/owl-common.h"

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

#if BARNEY_BACKEND_OPTIX
// ------------------------------------------------------------------
// defined in anari_library_barney_optix_static
// ------------------------------------------------------------------
extern "C"
barney::BarneyBaseDevice *createDevice_barney_optix(ANARILibrary,const char *);
#endif

#if BARNEY_BACKEND_CUDA
// ------------------------------------------------------------------
// defined in anari_library_barney_cuda_static
// ------------------------------------------------------------------
extern "C"
barney::BarneyBaseDevice *createDevice_barney_cuda(ANARILibrary,const char *);
#endif

#if BARNEY_BACKEND_HIP
// ------------------------------------------------------------------
// defined in anari_library_barney_hip_static
// ------------------------------------------------------------------
extern "C"
barney::BarneyBaseDevice *createDevice_barney_hip(ANARILibrary,const char *);
#endif

#if BARNEY_BACKEND_CPU
// ------------------------------------------------------------------
// defined in anari_library_barney_cpu_static
// ------------------------------------------------------------------
extern "C"
barney::BarneyBaseDevice *createDevice_barney_cpu(ANARILibrary,const char *);
#endif

namespace barney {

//   struct BarneyMultiLibrary : public ::anari::LibraryImpl
//   {
//     BarneyMultiLibrary(void *lib,
//                        ANARIStatusCallback defaultStatusCB,
//                        const void *statusCBPtr);

//     ANARIDevice newDevice(const char *subtype) override;
//     const char **getDeviceExtensions(const char *deviceType) override;
//   };

//   // Definitions ////////////////////////////////////////////////////////////////

//   BarneyMultiLibrary::BarneyMultiLibrary(void *lib,
//                                ANARIStatusCallback defaultStatusCB,
//                                const void *statusCBPtr)
//     : ::anari::LibraryImpl(lib, defaultStatusCB, statusCBPtr)
//   {}

  
//   ANARIDevice BarneyMultiLibrary::newDevice(const char *subType)
//   {
//     PING; PRINT(subType);
    
//     ANARILibrary lib = this_library();
// #if BARNEY_BACKEND_OPTIX
//     try { return (ANARIDevice)createDevice_barney_optix(lib,subType); }
//     catch (...) {};
// #endif
// #if BARNEY_BACKEND_CUDA
//     try { return (ANARIDevice)createDevice_barney_cuda(lib,subType); }
//     catch (...) {};
// #endif
// #if BARNEY_BACKEND_CPU
//     try { return (ANARIDevice)createDevice_barney_cpu(lib,subType); }
//     catch (...) {};
// #endif
//     std::cout << "Warning- couldn't create _any_ barney backend device!?"
//               << std::endl;
//     return (ANARIDevice)0;
//   }

//   const char **BarneyMultiLibrary::getDeviceExtensions(const char *_deviceType)
//   {
//     const std::string deviceType
//       = _deviceType
//       ? _deviceType
//       : "default";
//     static std::map<std::string,const char **> alreadyFound;
//     if (alreadyFound.find(deviceType) != alreadyFound.end())
//       return alreadyFound[deviceType];
//     PING;
//     ANARIDevice dev = newDevice(deviceType.c_str());
//     if (!dev) {
//       alreadyFound[deviceType] = nullptr;
//       return nullptr;
//     }
//     BarneyBaseDevice *base = (BarneyBaseDevice*)dev;
//     alreadyFound[deviceType] = base->extensions();

//     return alreadyFound[deviceType];
//   }

    // /*! helper entry-point for _directly_ creating a banari device
    //   without having to go through the dynamic-library
    //   'anariLoadLibrary' mechanism. This is used in pynari, to allow
    //   static linking of anari sdk */
    // extern "C" ANARIDevice createAnariDeviceBarney()
    // {
    //   ANARIDevice dev = 0;
    //   try {
    //     dev = (ANARIDevice) new BarneyDevice();
    //     return dev;
    //   } catch (std::exception &err) {
    //     std::cerr << "#banari: exception creating anari 'barney' GPU device: "
    //               << err.what() << std::endl;
    //     return 0;
    //   }
    // }


  
} // namespace barney

#if BARNEY_BACKEND_OPTIX
namespace BARNEY_NS_OPTIX { namespace anari {
    ::anari::LibraryImpl *createAnariLibrary(void *lib,
                                             ANARIStatusCallback defaultStatusCB,
                                             const void *statusCBPtr);
  }}
#endif
#if BARNEY_BACKEND_CUDA
namespace BARNEY_NS_CUDA { namespace anari {
    ::anari::LibraryImpl *createAnariLibrary(void *lib,
                                             ANARIStatusCallback defaultStatusCB,
                                             const void *statusCBPtr);
  }}
#endif
#if BARNEY_BACKEND_HIP
namespace BARNEY_NS_HIP { namespace anari {
    ::anari::LibraryImpl *createAnariLibrary(void *lib,
                                             ANARIStatusCallback defaultStatusCB,
                                             const void *statusCBPtr);
  }}
#endif
#if BARNEY_BACKEND_CPU
namespace BARNEY_NS_CPU { namespace anari {
    ::anari::LibraryImpl *createAnariLibrary(void *lib,
                                             ANARIStatusCallback defaultStatusCB,
                                             const void *statusCBPtr);
  }}
#endif

extern "C" BARNEY_LIBRARY_INTERFACE
ANARI_DEFINE_LIBRARY_ENTRYPOINT(barney, handle, scb, scbPtr)
{
// #if 1
# if BARNEY_BACKEND_OPTIX
  if (auto lib = BARNEY_NS_OPTIX::anari::createAnariLibrary(handle,scb,scbPtr))
    return (ANARILibrary)lib;
# endif
# if BARNEY_BACKEND_CUDA
  if (auto lib = BARNEY_NS_CUDA::anari::createAnariLibrary(handle,scb,scbPtr))
    return (ANARILibrary)lib;
# endif
# if BARNEY_BACKEND_HIP
  if (auto lib = BARNEY_NS_HIP::anari::createAnariLibrary(handle,scb,scbPtr))
    return (ANARILibrary)lib;
# endif
# if BARNEY_BACKEND_CPU
  if (auto lib = BARNEY_NS_CPU::anari::createAnariLibrary(handle,scb,scbPtr))
    return (ANARILibrary)lib;
# endif
  std::cout << "#barney - could not create _any_ anari backend library!?"
            << std::endl;
  return 0;
// #else
//   return (ANARILibrary) new barney::BarneyMultiLibrary(handle, scb, scbPtr);
// #endif
}

