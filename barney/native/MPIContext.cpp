// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/native/MPIContext.h"
#include "barney/native/FromEnv.h"
#include "barney/native/fb/DistFB.h"
#include "barney/native/render/RayQueue.h"
#include "barney/native/globalTrace/RQSMPI.h"
#include "barney/native/globalTrace/All2all.h"
#include "barney/native/globalTrace/TwoStage.h"

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/
#endif

namespace BARNEY_NS {
  namespace native {
    
    size_t getHostNameHash();

    MPIContext::~MPIContext()
    {}
  
    WorkerTopo::SP
    MPIContext::makeTopo(const Comm &worldComm,
                         const Comm &workerComm,
                         const std::vector<DataGroupDescriptor> &dataGroups)
    {
      std::vector<WorkerTopo::Device> devices;
      for (auto dg : dataGroups) {
        for (auto gpuID : dg.gpuIDs) {
          WorkerTopo::Device dev;
          dev.local = devices.size();
          dev.worker = workerComm.rank;
          dev.worldRank = worldComm.rank;
          dev.dataRank = dg.dataRank;
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

  
    inline bool isPassiveNode(const std::vector<DataGroupDescriptor> &dataGroups)
    { return dataGroups.size() == 1 && dataGroups[0].dataRank == -1; }
  
    MPIContext::MPIContext(const Comm &worldComm,
                           const Comm &workerComm,
                           const std::vector<DataGroupDescriptor> &dataGroups)
      : Context(dataGroups,makeTopo(worldComm,workerComm,dataGroups)),
        world(worldComm),
        workers(workerComm)
    {
      bool dbg = FromEnv::logConfig;
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
    std::shared_ptr<FrameBuffer>
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

      // iw - todo check perf impact of this; check if a allgather and
      // local reduce would be faster
      int lowest_accumID = workers.allReduceMin((int)fb->accumID);
      if (lowest_accumID == 0 && fb->accumID != 0)
        fb->resetAccumulation();
  
      if (isActiveWorker) {
        renderTiles(renderer,model,camera,fb);
        finalizeTiles(fb);
      }
      fb->finalizeFrame();
    }

    void MPIContext::barrier(bool warn) 
    {
      if (warn) PING;
      workers.barrier();
      if (warn) usleep(100);
    }
    
  }

  using namespace native;
  
  // BARNEY_API
  BNContext bnMPIContextCreate(MPI_Comm _comm,
                               int        numDataGroupsOnThisContext,
                               /*! tells which data rank / 'color' of data
                                 will be stored in each of the
                                 `numDataGroupsOnThisContext` data groups
                                 of this context. This array *must* have
                                 `numDataGroupsOnThisContext` entries */
                               const int *dataRanksOnThisContext,
                               /*! which gpu(s) to use for this
                                 process. GPUs will be assigned to data
                                 groups on a round-robin basis, so the i'th
                                 GPU listed here will get assigned to data
                                 group 'i%numDataGroupsOnThisContext' */
                               int numGPUs,
                               const int *gpuIDs)
  {
    assert(numGPUs >= numDataGroupsOnThisContext);
    assert(gpuIDs);
    assert((numGPUs % numDataGroupsOnThisContext) == 0);
    FromEnv::init();
    for (int i=0;i<numGPUs;i++)
      assert(gpuIDs[i] >= 0);

    LOG_API_ENTRY;
    // iw - no need to check if MPI is already initialized, this only
    // gets called through anari MPI device, which already initilaized
    // MPI
    
    Comm world(_comm);
    if (world.size == 1) {
      std::cout << "#bn: MPIContextInit, but only one rank - using local context" << std::endl;
      // if (_gpuIDs == nullptr && numGPUs == 1) {
      //   static const int const_zero = 0;
      //   _gpuIDs = &const_zero;
      // }
      return bnContextCreate(numDataGroupsOnThisContext,
                             dataRanksOnThisContext,
                             numGPUs,
                             gpuIDs);
    }

    // ------------------------------------------------------------------
    // create vector of data groups; if actually specified by user we
    // use those; otherwise we use IDs
    // [0,1,...numModelSlotsOnThisHost)
    // ------------------------------------------------------------------
    int numPhysicalGPUs = rtc::physicalDeviceCount();
    if (numPhysicalGPUs == 0)
      throw std::runtime_error
        ("no phyical devices for this type of rtc backend");
    assert(numPhysicalGPUs > 0);
    
    std::vector<DataGroupDescriptor>
      dataGroupsOnThisContext(numDataGroupsOnThisContext);
    for (int ldgIdx=0;ldgIdx<numDataGroupsOnThisContext;ldgIdx++) {
      auto &ldg = dataGroupsOnThisContext[ldgIdx];
      ldg.dataRank = dataRanksOnThisContext[ldgIdx];
    }

    for (int i=0;i<numGPUs;i++) {
      auto &ldg = dataGroupsOnThisContext[i%numDataGroupsOnThisContext];
      int gpuID = gpuIDs[i];
      assert(gpuID >= 0);
      ldg.gpuIDs.push_back(gpuID);
    }
    Comm workers
      = world.split(!isPassiveNode(dataGroupsOnThisContext));
    return (BNContext)new MPIContext(world,workers,dataGroupsOnThisContext);
  }

}
