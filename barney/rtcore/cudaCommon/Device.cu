// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cudaCommon/Device.h"
#include "rtcore/cudaCommon/Texture.h"
#include "rtcore/cudaCommon/TextureData.h"

namespace BARNEY_NS {
  namespace rtc {

    /*! enable peer access between these gpus, and return truea if
      successful, else if at least one pair does not work */
    bool enablePeerAccess(const std::vector<int> &gpuIDs)
    {
      if (gpuIDs.size() == 1) return true;
#define LOG(a) ss << "#bn." << a << std::endl;

      std::stringstream ss;
      ss << "enabling peer access ('.'=self, '+'=can access other device, '!'=can NOT peer access)" << std::endl;
 
     
      int deviceCount = (int)gpuIDs.size();
      LOG("found " << deviceCount << " CUDA capable devices");
      for (auto gpuID : gpuIDs) {
        cudaDeviceProp prop;
        BARNEY_CUDA_CALL(GetDeviceProperties(&prop, gpuID));
        LOG(" - device #" << gpuID << " : " << prop.name);
      }
      LOG("enabling peer access:");
      
      bool successful = true;

      for (auto gpuID : gpuIDs) {
        SetActiveGPU forLifeTime(gpuID);
        ss << " - device #" << gpuID << " : ";
        int cuda_i = gpuID; 
        for (int j=0;j<deviceCount;j++) {
          int cuda_j = gpuIDs[j];
          if (cuda_i == cuda_j) {
            ss << " ."; 
          } else {
            int canAccessPeer = 0;
            cudaError_t rc = cudaDeviceCanAccessPeer(&canAccessPeer, cuda_i,cuda_j);
            if (rc != cudaSuccess) {
              PRINT(rc);
              PRINT(std::to_string(rc));
              throw std::runtime_error("cuda error in cudaDeviceCanAccessPeer: "
                                       +std::to_string(rc));
            }
            if (!canAccessPeer) {
              // huh. this can happen if you have differnt device
              // types (in my case, a 2070 and a rtx 8000).
              // nvm - yup, this isn't an error. Expect certain configs to not allow peer access.
              // disabling this, as it's concerning end users.
              // std::cerr << "cannot not enable peer access!? ... skipping..." << std::endl;
              successful = false;
              ss << " !"; 
              continue;
            }
            
            rc = cudaDeviceEnablePeerAccess(cuda_j,/* flags - must be 0 */0);
            if (rc == cudaErrorPeerAccessAlreadyEnabled) {
              auto ignore = cudaGetLastError();
            } else if (rc != cudaSuccess)
              throw std::runtime_error("cuda error in cudaDeviceEnablePeerAccess: "
                                       +std::to_string(rc));
            ss << " +";
          }
        }
        ss << "\n";
        // LOG(ss.str());
      }
      std::cout << ss.str();
      return successful;
    }

    /*! get a unique hash for a given physical device. for cuda
      devices we do this by computing a hash from pci bus ID etc */
    size_t getPhysicalDeviceHash(int gpuID)
    {
      cudaDeviceProp props;
      cudaError_t rc = cudaGetDeviceProperties(&props, gpuID);
      if (rc != cudaSuccess)
        throw std::runtime_error("could not query cuda Device properties");
      return ((props.pciDomainID * 256 + props.pciBusID) * 256) + props.pciDeviceID;
    }

    SetActiveGPU::SetActiveGPU(const CudaDeviceBase *device)
    {
      if (device)  {
        savedActiveDeviceID = device->setActive();
      } else {
        BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      }
    }

    SetActiveGPU::SetActiveGPU(int gpuID)
    {
      BARNEY_CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      BARNEY_CUDA_CHECK(cudaSetDevice(gpuID));
    }

    SetActiveGPU::~SetActiveGPU()
    {
      BARNEY_CUDA_CALL_NOTHROW(SetDevice(savedActiveDeviceID));
    }
    
    CudaDeviceBase::CudaDeviceBase(int physicalGPU)
      : physicalID(physicalGPU)
    {
      auto initial_rc = cudaGetLastError();
      if (initial_rc != cudaSuccess)
        printf("#barney WARNING: there seems to be a CUDA error state set _before_ the barney device is being created. This typically indicates an error in the application; I'm going to silently 'eat' this first error, but if the app will later set other error states this is almost certainly going to cause some issues\n");
      
      BARNEY_CUDA_SYNC_CHECK();
      
      int saved = setActive();
#ifdef USE_HIP
      /* HIP: Raise the device stack size,
      which is too low by default, and can overflow.*/
      hipDeviceSetLimit(hipLimitStackSize, 16*1024);
#endif
      BARNEY_CUDA_CALL(StreamCreateWithFlags(&stream,cudaStreamNonBlocking));
      // BARNEY_CUDA_CALL(StreamCreate(&stream));
      restoreActive(saved);
    }

    CudaDeviceBase::~CudaDeviceBase()
    {
      cudaStreamDestroy(stream);
    }
    
    int CudaDeviceBase::setActive() const
    {
      int oldActive = 0;
      BARNEY_CUDA_CHECK(cudaGetDevice(&oldActive));
      if (physicalID != oldActive) {
        BARNEY_CUDA_CHECK(cudaSetDevice(physicalID));
        // BARNEY_CUDA_SYNC_CHECK();
      }
      return oldActive;
    }
    
    void CudaDeviceBase::restoreActive(int oldActive) const
    {
      BARNEY_CUDA_CHECK(cudaSetDevice(oldActive)); 
    }
    
    void *CudaDeviceBase::allocMem(size_t numBytes)
    {
      if (!numBytes) return nullptr;
      SetActiveGPU forDuration(this);
      void *ptr = 0;
      // BARNEY_CUDA_CALL(MallocManaged((void **)&ptr,numBytes));
      BARNEY_CUDA_CALL(Malloc((void **)&ptr,numBytes));
      assert(ptr);
      // BARNEY_CUDA_SYNC_CHECK();
      return ptr;
    }
    
    void *CudaDeviceBase::allocHost(size_t numBytes) 
    {
      if (!numBytes) return nullptr;
      SetActiveGPU forDuration(this);
      void *ptr = 0;
      BARNEY_CUDA_CALL(MallocHost(&ptr,numBytes));
      return ptr;
    }
      
    void CudaDeviceBase::freeHost(void *mem) 
    {
      if (!mem) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(FreeHost(mem));
    }
      
    void CudaDeviceBase::freeMem(void *mem) 
    {
      if (!mem) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(Free(mem));
      // BARNEY_CUDA_SYNC_CHECK();
    }
      
    void CudaDeviceBase::memsetAsync(void *mem,int value, size_t numBytes) 
    {
      if (numBytes == 0) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(MemsetAsync(mem,value,numBytes,stream));
    }
      

    void CudaDeviceBase::copyAsync(void *dst, const void *src, size_t numBytes) 
    {
      if (numBytes == 0) return;
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(MemcpyAsync(dst,src,numBytes,cudaMemcpyDefault,stream));
    }
      
    void CudaDeviceBase::sync() 
    {
      SetActiveGPU forDuration(this);
      BARNEY_CUDA_CALL(StreamSynchronize(stream));
      // BARNEY_CUDA_SYNC_CHECK();
    }

    void CudaDeviceBase::freeTextureData(TextureData *td)
    {
      if (td) delete td;
    }
    
    void CudaDeviceBase::freeTexture(Texture *tex)
    {
      if (tex) delete tex;
    }
    
    TextureData *
    CudaDeviceBase::createTextureData(vec3i dims,
                                      rtc::DataType format,
                                      const void *texels) 
    {
      SetActiveGPU forDuration(this);
      return new TextureData(this,dims,format,texels);
    }

    Texture *TextureData::createTexture(const TextureDesc &desc) 
    {
      SetActiveGPU forDuration(device);
      return new Texture(this,desc);
    }    

  }
}

 
