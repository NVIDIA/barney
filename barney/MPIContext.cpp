// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/MPIContext.h"
#include "barney/fb/DistFB.h"
#include "barney/render/RayQueue.h"
#include "barney/globalTrace/RQSMPI.h"
#include "barney/globalTrace/All2all.h"
#include "barney/globalTrace/TwoStage.h"

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif


#if defined(BARNEY_RTC_EMBREE) && defined(BARNEY_RTC_OPTIX)
# error "should not have both backends on at the same time!?"
#endif

namespace BARNEY_NS {

  size_t getHostNameHash();

  int findLocalRank(barney_api::mpi::Comm &comm)
  {
    size_t myHash = getHostNameHash();
    std::vector<size_t> allHashes(comm.size);
    comm.allGather(allHashes.data(),
                   &myHash,1,
                   sizeof(size_t));
    int count = 0;
    for (int i=0;i<comm.rank;i++)
      if (allHashes[i] == myHash)
        ++count;
    return count;
  }

  
  MPIContext::~MPIContext()
  {}
  
  WorkerTopo::SP
  MPIContext::makeTopo(const barney_api::mpi::Comm &worldComm,
                       const barney_api::mpi::Comm &workerComm,
                       const std::vector<LocalSlot> &localSlots)
  {
    std::vector<WorkerTopo::Device> devices;
    for (auto ls : localSlots) {
      for (auto gpuID : ls.gpuIDs) {
        WorkerTopo::Device dev;
        dev.local = devices.size();
        dev.worker = workerComm.rank;
        dev.worldRank = worldComm.rank;
        dev.dataRank = ls.dataRank;
        dev.hostNameHash = getHostNameHash();
        dev.physicalDeviceHash = rtc::getPhysicalDeviceHash(gpuID);
        devices.push_back(dev);
      }
    }
    int myCount = devices.size();
    int allCount = worldComm.allReduceAdd(myCount);

    std::vector<WorkerTopo::Device> allDevices(allCount);
    worldComm.allGather(allDevices.data(),
                        devices.data(),
                        myCount,
                        sizeof(WorkerTopo::Device));
    int myOfs = 0;
    for (myOfs=0;myOfs<allDevices.size();myOfs++)
      if (allDevices[myOfs].worldRank == worldComm.rank)
        break;
      
    return std::make_shared<WorkerTopo>(allDevices,myOfs,myCount);
  }

  
  inline bool isPassiveNode(const std::vector<LocalSlot> &localSlots)
  { return localSlots.size() == 1 && localSlots[0].dataRank == -1; }
  
  MPIContext::MPIContext(const barney_api::mpi::Comm &worldComm,
                         const barney_api::mpi::Comm &workerComm,
                         const std::vector<LocalSlot> &localSlots,
                         bool userSuppliedGpuListWasEmpty)
    : Context(localSlots,makeTopo(worldComm,workerComm,localSlots)),
      world(worldComm),
      workers(workerComm)
  {
    bool dbg = FromEnv::get()->logConfig;
    if (topo->isDataParallel() && userSuppliedGpuListWasEmpty && world.rank == 0) {
      std::cerr << "#bn.mpi: WARNING - barney is run in 'true' data parallel mode" << std::endl;
      std::cerr << "#bn.mpi: but user did NOT provide an explicit list of GPU ID(s)." << std::endl;
    }
    if (FromEnv::enabled("two-stage") || FromEnv::enabled("two_stage")) {
      std::cout << "ENABLING TwoStage!" << std::endl;
      globalTraceImpl = new TwoStage(this);
    } else if (FromEnv::enabled("all2all")) {
      std::cout << "ENABLING ALL2ALL!" << std::endl;
      globalTraceImpl = new MPIAll2all(this);
    } else {
      globalTraceImpl = new RQSMPI(this);
    }
  }
  
  /*! create a frame buffer object suitable to this context */
  std::shared_ptr<barney_api::FrameBuffer>
  MPIContext::createFrameBuffer()
  {
    return std::make_shared<DistFB>(this,devices);
  }

  /*! returns how many rays are active in all ray queues, across all
    devices and, where applicable, across all ranks */
  int MPIContext::numRaysActiveGlobally()
  {
    assert(isActiveWorker);
    return workers.allReduceAdd(numRaysActiveLocally());
  }

  
  void MPIContext::render(Renderer    *renderer,
                          GlobalModel *model,
                          Camera      *camera,
                          FrameBuffer *_fb)
  {
    auto _context = this;
    DistFB *fb = (DistFB *)_fb;
    if (isActiveWorker) {
      renderTiles(renderer,model,camera,fb);
      finalizeTiles(fb);
    }
    fb->finalizeFrame();
  }

  extern "C" {
# if BARNEY_RTC_EMBREE
    barney_api::Context *
    createMPIContext_embree(barney_api::mpi::Comm world,
                            barney_api::mpi::Comm workers,
                            bool isActiveWorker,
                            const std::vector<int> &dgIDs)
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
      return new BARNEY_NS::MPIContext(world,workers,localSlots,false);
      // std::vector<int> gpuIDs = { 0 }; 
      // return new BARNEY_NS::MPIContext(world,workers,isActiveWorker,
      //                                  dgIDs,gpuIDs);
    }
# endif
# if BARNEY_RTC_OPTIX
    barney_api::Context *
    createMPIContext_optix(barney_api::mpi::Comm world,
                           // barney_api::mpi::Comm workers,
                           // bool isActiveWorker,
                           const std::vector<int> &dgIDs,
                           int numGPUs, const int *gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *optix* context" << std::endl;
      // std::vector<int> gpuIDs;
      bool userSuppliedGpuListWasEmpty = (gpuIDs == nullptr);
      int numDGs = dgIDs.size();
      int localRankGPU = 0;
      if (userSuppliedGpuListWasEmpty) {
        std::cout << "#banari: starting up in data parallel one-gpu-per-rank mode" << std::endl;
        numGPUs = 1;
        localRankGPU = findLocalRank(world) % numGPUs;
        gpuIDs = &localRankGPU;
      }

      if (numGPUs < numDGs)
        throw std::runtime_error
          ("not enough CUDA GPUs for requested number of data groups!");
      if (numGPUs % numDGs != 0)
        throw std::runtime_error
          ("num GPUs not a multiple of num data groups on this rank!");
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

      barney_api::mpi::Comm workers
        = world.split(!isPassiveNode(localSlots));
      return new BARNEY_NS::MPIContext(world,workers,localSlots,
                                       userSuppliedGpuListWasEmpty);
    }
# endif


# if BARNEY_RTC_CUDA
    barney_api::Context *
    createMPIContext_cuda(barney_api::mpi::Comm world,
                           // barney_api::mpi::Comm workers,
                           // bool isActiveWorker,
                           const std::vector<int> &dgIDs,
                           int numGPUs, const int *gpuIDs)
    {
      if (FromEnv::get()->logBackend)
        std::cout << "#bn: creating *cuda* context" << std::endl;
      // std::vector<int> gpuIDs;
      bool userSuppliedGpuListWasEmpty = (gpuIDs == nullptr);
      int numDGs = dgIDs.size();
      int localRankGPU = 0;
      if (numGPUs == -1) {
        BARNEY_CUDA_CALL(GetDeviceCount(&numGPUs));
      }
      if (numGPUs < numDGs)
        throw std::runtime_error
          ("not enough CUDA GPUs for requested number of data groups!");
      if (numGPUs % numDGs != 0)
        throw std::runtime_error
          ("num GPUs not a multiple of num data groups on this rank!");
      int gpusPerDG = numGPUs / numDGs;
      std::vector<LocalSlot> localSlots(dgIDs.size());
      for (int lsIdx=0;lsIdx<dgIDs.size();lsIdx++) {
        LocalSlot &slot = localSlots[lsIdx];
        slot.dataRank = dgIDs[lsIdx];
        for (int j=0;j<gpusPerDG;j++) {
          int idx = lsIdx*gpusPerDG+j;
          int gpuID = gpuIDs?gpuIDs[idx]:idx;
          slot.gpuIDs.push_back(gpuID);
        }
      }

      barney_api::mpi::Comm workers
        = world.split(!isPassiveNode(localSlots));
      return new BARNEY_NS::MPIContext(world,workers,localSlots,
                                       userSuppliedGpuListWasEmpty);
    }
# endif
  }
}
