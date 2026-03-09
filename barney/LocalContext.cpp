// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/LocalContext.h"
#include "barney/fb/LocalFB.h"
#include "barney/globalTrace/RQSLocal.h"
#include "barney/render/RayQueue.h"

#if defined(BARNEY_RTC_EMBREE) && defined(BARNEY_RTC_OPTIX)
# error "should not have both backends on at the same time!?"
#endif

namespace barney_api {
#if BARNEY_RTC_EMBREE
  extern "C" {
    Context *createContext_embree(const std::vector<int> &dgIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *embree (cpu)* context" << std::endl;
      assert(dgIDs.size() == 1);
      std::vector<LocalSlot> localSlots(dgIDs.size());
      for (int lsIdx=0;lsIdx<dgIDs.size();lsIdx++) {
        LocalSlot &slot = localSlots[lsIdx];
        slot.dataRank = dgIDs[lsIdx];
        slot.gpuIDs = { 0 };
      }
      return new BARNEY_NS::LocalContext(localSlots);
    }
  }
#endif
#if BARNEY_RTC_OPTIX
  extern "C" {
    Context *createContext_optix(const std::vector<int> &dgIDs,
                                 int numGPUs, const int *gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *optix* context" << std::endl;
      // std::vector<int> gpuIDs;
      int numDGs = (int)dgIDs.size();
      if (numGPUs == -1) {
        BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
      }

#define ALLOW_OVERSUBSCRIBE 1
#if ALLOW_OVERSUBSCRIBE
      std::vector<int> fakeIDs;
      if (numGPUs < numDGs) {
        for (int i=0;i<numDGs;i++)  {
          int ID
            = gpuIDs
            ? gpuIDs[i%numGPUs]
            : (i%numGPUs)
            ;
          fakeIDs.push_back(ID);
        }
        gpuIDs = (const int *)fakeIDs.data();
        numGPUs = numDGs;
      }
#endif

      if (numGPUs < numDGs)
        throw std::runtime_error
          ("not enough CUDA GPUs for requested number of data groups!");
      int gpusPerDG = numGPUs / numDGs;
      std::vector<LocalSlot> localSlots(dgIDs.size());
      for (int lsIdx=0;lsIdx<dgIDs.size();lsIdx++) {
        LocalSlot &slot = localSlots[lsIdx];
        slot.dataRank = dgIDs[lsIdx];
        for (int j=0;j<gpusPerDG;j++) {
          int idx = lsIdx*gpusPerDG+j;
          slot.gpuIDs.push_back(gpuIDs?gpuIDs[idx]:idx);
        }
      }
      Context *ctx = new BARNEY_NS::LocalContext(localSlots);
      return ctx;
    }
  } 
#endif
#if BARNEY_RTC_CUDA
  extern "C" {
    Context *createContext_cuda(const std::vector<int> &dgIDs,
                                 int numGPUs, const int *gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *(native-)cuda* context" << std::endl;
      int numDGs = dgIDs.size();
      if (numGPUs == -1) {
        BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
      }

#if ALLOW_OVERSUBSCRIBE
      std::vector<int> fakeIDs;
      if (numGPUs < numDGs) {
        for (int i=0;i<numDGs;i++)  {
          int ID
            = gpuIDs
            ? gpuIDs[i%numGPUs]
            : (i%numGPUs)
            ;
          fakeIDs.push_back(ID);
        }
        gpuIDs = (const int *)fakeIDs.data();
        numGPUs = numDGs;
      }
#endif

      if (numGPUs < numDGs)
        throw std::runtime_error
          ("not enough CUDA GPUs for requested number of data groups!");
      int gpusPerDG = numGPUs / numDGs;
      std::vector<LocalSlot> localSlots(dgIDs.size());
      for (int lsIdx=0;lsIdx<dgIDs.size();lsIdx++) {
        LocalSlot &slot = localSlots[lsIdx];
        slot.dataRank = dgIDs[lsIdx];
        for (int j=0;j<gpusPerDG;j++) {
          int idx = lsIdx*gpusPerDG+j;
          slot.gpuIDs.push_back(gpuIDs?gpuIDs[idx]:idx);
        }
      }
      Context *ctx = new BARNEY_NS::LocalContext(localSlots);
      return ctx;
    }
    //   if (numGPUs == -1)
    //     BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
    //   std::vector<int> gpuIDs;
    //   for (int i=0;i<numGPUs;i++)
    //     gpuIDs.push_back(_gpuIDs?_gpuIDs[i]:i);
    //   Context *ctx = new BARNEY_NS::LocalContext(dgIDs,gpuIDs);
    //   return ctx;
    // }
  } 
#endif
}

namespace BARNEY_NS {
  size_t getHostNameHash()
  {
#if defined(_WIN32) || defined(__APPLE__)
    // gethostname() is linux only, but this is only really required
    // for MPI, anyway, so for mac and windows - which won't be used
    // when using MPI - we can just as well return anything we like...
    return 0;
#else
    char hostName[256];
    gethostname(hostName,256);
    size_t hash = 0;
    size_t FNV_PRIME = 0x00000100000001b3ull;
    for (int i=0;hostName[i];i++)
      hash = hash * FNV_PRIME ^ hostName[i];
    return hash;
#endif
  }
  
  WorkerTopo::SP
  LocalContext::makeTopo(const std::vector<LocalSlot> &localSlots)
  {
    std::vector<WorkerTopo::Device> devices;
    for (auto ls : localSlots) {
      for (auto gpuID : ls.gpuIDs) {
        WorkerTopo::Device dev;
        dev.local = (int)devices.size();
        dev.worker = 0;
        dev.worldRank = 0;
        dev.dataRank = ls.dataRank;
        dev.hostNameHash = getHostNameHash();
        dev.physicalDeviceHash = rtc::getPhysicalDeviceHash(gpuID);
        devices.push_back(dev);
      }
    }
    return std::make_shared<WorkerTopo>(devices,0,(int)devices.size());
  }

  LocalContext::LocalContext(const std::vector<LocalSlot> &localSlots)
    : Context(localSlots,
              makeTopo(localSlots))
  {
#if 0
    std::map<int,int> numGPUsInIsland;
    std::map<int,int> numUsesOfDG;

    for (int i=0;i<(int)devices->size();i++) {
      auto dev = (*devices)[i]; assert(dev);
      dev->allGPUsGlobally.rank = i;
      dev->allGPUsGlobally.size = devices->size();
      dev->gpuInNode.rank = i;
      dev->gpuInNode.size = devices->size();
      const int myDG = dataGroupIDs[i % dataGroupIDs.size()];
      const int myIsland = numUsesOfDG[myDG]++;
      dev->islandInWorld.rank = myIsland;
      const int rankInMyIsland = numGPUsInIsland[dev->islandInWorld.rank]++;
      dev->gpuInIsland.rank   = rankInMyIsland;
    }
    // now we know how often every DG and island got used, so now we
    // know num islands, and thus the size of each island.
    for (int i=0;i<(int)devices->size();i++) {
      auto dev = (*devices)[i]; assert(dev);
      int myDG           = dataGroupIDs[i % dataGroupIDs.size()];
      int numIslands     = numUsesOfDG[myDG];
      int myIsland       = dev->islandInWorld.rank;
      int sizeOfMyIsland = numGPUsInIsland[myIsland];
      dev->gpuInIsland.size   = sizeOfMyIsland;
      dev->islandInWorld.size = numIslands;
    }

    // now assign, for each device, it's recv and send devices
    for (int myID=0;myID<(int)devices->size();myID++) {
      auto myDev = (*devices)[myID]; assert(myDev);
      auto &rqs = myDev->rqs;
      const int myIsland = myDev->islandInWorld.rank;
      const int sizeOfIsland = myDev->gpuInIsland.size;
      const int myRankInIsland = myDev->gpuInIsland.rank;
      const int intendedSendRank = (myRankInIsland+sizeOfIsland-1)%sizeOfIsland;
      const int intendedRecvRank = (myRankInIsland+1)%sizeOfIsland;

      // we're all the same 'mpi'-rank 0
      rqs.sendWorkerRank = 0;
      rqs.recvWorkerRank = 0;
          
      for (int otherID=0;otherID<devices->size();otherID++) {
        auto otherDev = (*devices)[otherID]; assert(myDev);
        const int otherIsland = otherDev->islandInWorld.rank;
        const int otherRankInIsland = otherDev->gpuInIsland.rank;
        if (myIsland != otherIsland)
          // different islands, cannot match
          continue;
        if (otherRankInIsland == intendedSendRank) rqs.sendWorkerLocal = otherID;
        if (otherRankInIsland == intendedRecvRank) rqs.recvWorkerLocal = otherID;
      }
    }
    
    // do some sanity checking here:
    auto dev0 = (*devices)[0];
    for (auto dev : *devices) {
      const int sizeOfIsland = dev->gpuInIsland.size;
      // - all islands have to have the same num GPUs
      assert(dev->gpuInIsland.size == dev0->gpuInIsland.size);
      // - every gpu has to have a send peer in rqs.sendlocal
      assert(dev->rqs.sendWorkerLocal >= 0);
      // - every gpu has to have a recv peer in rqs.recvlocal
      assert(dev->rqs.recvWorkerLocal >= 0);
    }
#endif
    globalTraceImpl = new RQSLocal(this);
  }

  LocalContext::~LocalContext()
  {
    /* not doing anything, but leave this in to ensure that derived
       classes' destrcutors get called !*/
  }

  std::shared_ptr<barney_api::FrameBuffer> LocalContext::createFrameBuffer()
  {
    return std::make_shared<LocalFB>(this,devices);
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int LocalContext::numRaysActiveGlobally()
  {
    return numRaysActiveLocally();
  }

  void LocalContext::render(Renderer    *renderer,
                            GlobalModel *model,
                            Camera      *camera,
                            FrameBuffer *fb)
  {
    assert(model);
    assert(fb);

    renderTiles(renderer,model,camera,fb);
    finalizeTiles(fb);
    fb->finalizeFrame();
  }

}
