// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "barney/native/MPIContext.h"
#include "barney/native/globalTrace/TwoStage.h"
#include "barney/native/DeviceGroup.h"
#include "barney/native/render/RayQueue.h"
#include "barney/native/FromEnv.h"

namespace BARNEY_NS {
  namespace native {

    extern void (*profHook)();
  
    std::vector<std::tuple<double,double,int,const char *>> kernelTimes;

#define TWO_STAGE_PROFILE 0
  
#if TWO_STAGE_PROFILE
# define ENTER() const double prof_t0 = getCurrentTime();
# define LEAVE(count,name)                                              \
    const double prof_t1 = getCurrentTime();                            \
    kernelTimes.push_back(std::make_tuple<double,double,int,const char *>((double)prof_t0,(double)prof_t1,(int)count,(const char *)name));
#else
# define ENTER() /* nothing */
# define LEAVE(count,name) /* nothing */
#endif
  
    int prof_rank;

    void twoStageProfHook()
    {
      std::stringstream ss;
      static double t00 = std::get<0>(kernelTimes[0]);
      for (auto kernel : kernelTimes) {
        double t0 = std::get<0>(kernel)-t00;
        double t1 = std::get<1>(kernel)-t00;
        int numItems = std::get<2>(kernel);
        const char *name = std::get<3>(kernel);

        ss << "r" << prof_rank << " [" << prettyDouble(t0) << "s.."
           << prettyDouble(t1) << "s = "
           << prettyDouble(t1-t0) << "s]: "
           << prettyNumber(numItems) << " items in "
           << name
           << " -> " << prettyDouble(1000000.f*(t1-t0)/numItems) << "s per mio items"
           << std::endl;
      }
      kernelTimes.clear();
      std::cout << ss.str();
    }
  
    __rtc_global
    void buildHitsOnly(const rtc::ComputeInterface &ci,
                       HitOnly *hitOnly,
                       Ray *rayQueue,
                       int N);
  
    __rtc_global
    void reduceReceivedHitsKernel_intraNode(const rtc::ComputeInterface &ci,
                                            HitOnly *hitOnly,
                                            int nRays,
                                            int reduceFactor)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= nRays) return;

      HitOnly reduced = hitOnly[tid];
      for (int peer=1;peer<reduceFactor;peer++) {
        HitOnly *hit = hitOnly+peer*nRays+tid;
      
        if (hit->tHit >= reduced.tHit) continue;
      
        reduced = *hit;
      }
      hitOnly[tid] = reduced;
#endif
    }
  
    __rtc_global
    void reduceReceivedHitsKernel_crossNodes(const rtc::ComputeInterface &ci,
                                             Ray *rayQueueThisRank,
                                             HitOnly *hitOnlyAllRanks,
                                             int nRays,
                                             int reduceFactor)
    {
#if RTC_DEVICE_CODE
      int tid = ci.launchIndex().x;
      if (tid >= nRays) return;
    
      Ray ray = rayQueueThisRank[tid];
      for (int peer=0;peer<reduceFactor;peer++) {
        HitOnly *hit = hitOnlyAllRanks+peer*nRays+tid;

        if (hit->tHit >= ray.tMax) continue;

        ray.tMax     = hit->tHit;
        ray.bsdfType = hit->bsdfType;
        ray.hitBSDF  =  hit->hitBSDF;
        ray.P        =  hit->P;
        ray.shadingNormal = hit->shadingNormal;
        ray.geometricNormal = hit->geometricNormal;
      }
      rayQueueThisRank[tid] = ray;
#endif
    }
  
  
    __rtc_global
    void createRayOnly(const rtc::ComputeInterface &ci,
                       RayOnly *rayOnly,
                       Ray *rayQueue,
                       int N);
    __rtc_global
    void buildStagedRayQueue(const rtc::ComputeInterface &ci,
                             Ray *rayQueue,
                             RayOnly *rayOnly,
                             int N);
  
    TwoStage::TwoStage(MPIContext *context)
      : GlobalTraceImpl(context),
        context(context),
        world(context->world),
        topo(context->topo.get()),
        logTopo(FromEnv::logTopo),
        logQueues(FromEnv::logQueues),
        opt_mpi(FromEnv::enabled("opt_mpi"))
    {
      prof_rank = world.rank;
#if TWO_STAGE_PROFILE
      profHook = twoStageProfHook;
#endif

      if (context->devices->size() != 1)
        throw std::runtime_error
          ("twostage all2all currently only works for one device per rank");
      this->device = context->devices->get(0);
    
      if (topo->islands.size() != 1)
        throw std::runtime_error
          ("twostage all2all currently only works for a single island");

      myGID = device->globalRank();
      numGlobal = topo->allDevices.size();
      global.rayCounts.resize(numGlobal);
      // sanity check that all physical nodes have same number of GPUs
      std::map<size_t,int> gpuCountInHost;
      int numHosts = 0;
      for (int gid=0; gid<context->topo->allDevices.size(); gid++) {
        auto &dev = context->topo->allDevices[gid];
        gpuCountInHost[dev.hostNameHash]++;
        numHosts = std::max(numHosts,topo->physicalHostIndexOf[gid]+1);
      }
      gpusPerHost = gpuCountInHost.begin()->second;
      for (auto count : gpuCountInHost)
        if (count.second != gpusPerHost)
          throw std::runtime_error
            ("twostage all2all currently requires same number of GPUs on all ranks");
      assert(numHosts * gpusPerHost == context->topo->allDevices.size());

      this->hostIdx = topo->physicalHostIndexOf[myGID];
      // allows oversubscription - we enumerate based on (host:process)
      // instead of (host.physialGPU)
      this->gpuIdx = topo->rankOnHost[myGID];
      _rankOf.resize(numGlobal);

      std::vector<int> logicalGidOfRank(numGlobal);
      int myLogicalGID = this->hostIdx * gpusPerHost + this->gpuIdx;
      world.allGather(logicalGidOfRank.data(),&myLogicalGID,1);
      for (int r=0;r<numGlobal;r++)
        _rankOf[logicalGidOfRank[r]] = r;
      this->numHosts = numGlobal / gpusPerHost;

      if (opt_mpi) {
        crossNodes.comm = world.split(this->gpuIdx);
        crossNodes.rayCounts.resize(crossNodes.comm.size);
        intraNode.comm = world.split(this->hostIdx);
        intraNode.rayCounts.resize(intraNode.comm.size);
      }
    
      if (logTopo) {
        world.barrier();
        if (context->myRank() == 0) {
          std::cout << "=========== TwoStage All2all ===========" << std::endl;
          std::cout << "- num MPI ranks (w/ one gpu each) " << numGlobal << std::endl;
          std::cout << "- detected num physical hosts " << numHosts << std::endl;
          std::cout << "- detected num (active) GPUs per host " << gpusPerHost << std::endl;
          PRINT(numHosts);
          PRINT(gpusPerHost);
          for (int h=0;h<numHosts;h++)
            for (int g=0;g<gpusPerHost;g++) {
              std::cout << "- gpu on rank " << (rankOf(h,g))
                        << " is logical h" << h << "g" << g << " {"
                        << topo->toString(rankOf(h,g)) << "}" << std::endl;
            }
        }
        world.barrier();
      }
    }


    void TwoStage::ensureAllOurQueuesAreLargeEnough()
    {
      auto rtc = device->rtc;
      size_t ourRequiredQueueSize
        = device->rayQueue->size * numGlobal;
      if (ourRequiredQueueSize > currentReservedSize) {
        if (logQueues) {
          std::cout << "resizing ray queues from " << currentReservedSize
                    << " to " << ourRequiredQueueSize << std::endl;
        }
        for (int i=0;i<2;i++)
          if (raysOnly[i]) rtc->freeMem(raysOnly[i]);
        for (int i=0;i<2;i++)
          if (hitsOnly[i]) rtc->freeMem(hitsOnly[i]);
      
        if (stagedRayQueue) rtc->freeMem(stagedRayQueue);
      
        size_t N = ourRequiredQueueSize+1024;
        for (int i=0;i<2;i++)
          raysOnly[i] = (RayOnly*)rtc->allocMem(N*sizeof(RayOnly));
        for (int i=0;i<2;i++)
          hitsOnly[i] = (HitOnly*)rtc->allocMem(N*sizeof(HitOnly));
        stagedRayQueue = (Ray *)rtc->allocMem(N*sizeof(Ray));
      
        currentReservedSize = N;
      }
    }

    // step 1: have all ranks exchange which (global) device has how
    // many rays (needed to set up the send/receives)
    void TwoStage::exchangeHowManyRaysEachDeviceHas()
    {
      ENTER();

      if (opt_mpi) {
      } else {
        int myRayCount = device->rayQueue->numActive;
        world.allGather(global.rayCounts.data(),&myRayCount,1);
    
        if (logQueues)  {
          if (myGID == 0) {
            std::cout << "ray counts (" << global.rayCounts.size() << "):";
            for (auto rc : global.rayCounts) std::cout << " " << rc;
            std::cout << std::endl;
          }
        }
      }
      LEAVE(1,"exchangeHowManyRaysEachDeviceHas");
    }
  
  
    /*! in this stage we have all each GPU exchange its rays with
      all GPUs that have same phsycail ID in all OTHER ranks, but NOT
      with other GPUs in same rank
    */
    void TwoStage::sendAndReceiveRays_crossNodes()
    {
      ENTER();
      // -----------------------------------------------------------------------------
      // first, create 'raysOnly[]' array, for each local device
      // -----------------------------------------------------------------------------
      int myRayCount = device->rayQueue->numActive;
      {
        SetActiveGPU forDuration(device);
        int bs = 128;
        int nb = divRoundUp(myRayCount,bs);
        __rtc_launch(device->rtc,
                     createRayOnly,
                     nb,bs,
                     // args
                     raysOnly[0],
                     device->rayQueue->traceAndShadeReadQueue.rays,
                     myRayCount);
      }


      if (opt_mpi) {
        crossNodes.comm.allGather(crossNodes.rayCounts.data(),&myRayCount,1);

        void *sendBuf = raysOnly[0];
        int sendCount = myRayCount*sizeof(RayOnly);
        void *recvBuf = raysOnly[1];
        std::vector<int> recvCounts(crossNodes.comm.size);
        std::vector<int> recvOffsets(crossNodes.comm.size);
        int sumCounts = 0;
        for (int i=0;i<crossNodes.comm.size;i++) {
          recvOffsets[i] = sumCounts;
          recvCounts[i] = crossNodes.rayCounts[i]*sizeof(RayOnly);
          sumCounts += recvCounts[i];
        }
        crossNodes.sumRaysReceived = sumCounts / sizeof(RayOnly);
        if (logQueues) 
          printf("xchg-rays-cross r%i we have %i sumrecv %i first two counts %i %i\n",
                 myGID,myRayCount,crossNodes.sumRaysReceived,
                 crossNodes.rayCounts[0],
                 crossNodes.rayCounts[1]);
      
        device->rtc->sync();
        BN_MPI_CALL(Allgatherv(sendBuf,sendCount,
                               MPI_BYTE,
                               recvBuf,(int*)recvCounts.data(),(int*)recvOffsets.data(),
                               MPI_BYTE,
                               crossNodes.comm));
      } else {
        device->rtc->sync();
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;
          int recvCount = global.rayCounts[rankOf(h,gpuIdx)];
          if (logQueues) 
            printf("splat-cross r%i receiving %i from %i (q 0->1)\n",
                   myGID,recvCount,rankOf(h,gpuIdx));
          world.recv(rankOf(h,gpuIdx),0,raysOnly[1]+recvOfs,
                     recvCount,req);
          recvOfs += recvCount;
          requests.push_back(req);
        }
        crossNodes.sumRaysReceived = recvOfs;
        if (logQueues) 
          printf("splat-cross r%i total received %i\n",
                 myGID,crossNodes.sumRaysReceived);
    
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;
          if (logQueues) 
            printf("splat-cross r%i sending %i to %i (q 0->1)\n",
                   myGID,myRayCount,rankOf(h,gpuIdx));
          world.send(rankOf(h,gpuIdx),0,raysOnly[0],myRayCount,req);
          requests.push_back(req);
        }
    
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
      LEAVE(recvOfs,"sendAndReceiveRays_crossNodes");
    }
  

    /*! in this stage we have all each GPU exchange its rays with
      all GPUs that have same phsycail ID in all OTHER ranks, but NOT
      with other GPUs in same rank
    */
    void TwoStage::sendAndReceiveRays_intraNode()
    {
      ENTER();

      if (opt_mpi) {
        int numRaysWeHave = crossNodes.sumRaysReceived;
        intraNode.comm.allGather(intraNode.rayCounts.data(),&numRaysWeHave,1);

        void *sendBuf = raysOnly[1];
        void *recvBuf = raysOnly[0];

        int sendCount = numRaysWeHave*sizeof(RayOnly);
        std::vector<int> recvCounts(intraNode.comm.size);
        std::vector<int> recvOffsets(intraNode.comm.size);
        int sumCounts = 0;
        for (int i=0;i<intraNode.comm.size;i++) {
          recvOffsets[i] = sumCounts;
          recvCounts[i] = intraNode.rayCounts[i]*sizeof(RayOnly);
          sumCounts += recvCounts[i];
        }
        intraNode.sumRaysReceived = sumCounts / sizeof(RayOnly);

        if (logQueues) 
          printf("xchg-rays-intra r%i we have %i sumrecv %i\n",
                 myGID,numRaysWeHave,intraNode.sumRaysReceived);
        device->rtc->sync();
        BN_MPI_CALL(Allgatherv(sendBuf,sendCount,
                               MPI_BYTE,
                               recvBuf,(int*)recvCounts.data(),(int*)recvOffsets.data(),
                               MPI_BYTE,
                               intraNode.comm));
      } else {
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        for (int g=0;g<gpusPerHost;g++) {
          MPI_Request req;
          int raysOnPeer = 0;
          for (int h=0;h<numHosts;h++)
            raysOnPeer += global.rayCounts[rankOf(h,g)];
          if (logQueues) 
            printf("splat-intra r%i receiving %i from %i (q 1->0)\n",
                   myGID,raysOnPeer,rankOf(hostIdx,g));
          world.recv(rankOf(hostIdx,g),0,
                     raysOnly[0]+recvOfs,raysOnPeer,
                     req);
          recvOfs += raysOnPeer;
          requests.push_back(req);
        }
        intraNode.sumRaysReceived = recvOfs;
        if (logQueues) 
          printf("splat-intra r%i total received %i\n",
                 myGID,intraNode.sumRaysReceived);

        int numRaysWeHave = 0;
        for (int h=0;h<numHosts;h++)
          numRaysWeHave += global.rayCounts[rankOf(h,gpuIdx)];
        for (int g=0;g<gpusPerHost;g++) {
          MPI_Request req;
          if (logQueues) 
            printf("splat-intra r%i sending %i to %i (q 1->0)\n",
                   myGID,numRaysWeHave,rankOf(hostIdx,g));
          world.send(rankOf(hostIdx,g),0,
                     raysOnly[1],numRaysWeHave,
                     req);
          requests.push_back(req);
        }
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
    
      LEAVE(recvOfs,"sendAndReceiveRays_intraNode");
    }

  

    void TwoStage::traceRays(GlobalModel *model,
                             uint32_t rngSeed,
                             bool needHitIDs) 
    {
      assert(needHitIDs == false); // not implemented right now
      ensureAllOurQueuesAreLargeEnough();
      exchangeHowManyRaysEachDeviceHas();
      sendAndReceiveRays_crossNodes();
      sendAndReceiveRays_intraNode();

      traceAllReceivedRays(model,rngSeed,needHitIDs);

      exchangeHits_intraNode();
      reduceHits_intraNode();
      exchangeHits_crossNodes();
      reduceHits_crossNodes();
    }


    void TwoStage::traceAllReceivedRays(GlobalModel *model,
                                        uint32_t rngSeed,
                                        bool needHitIDs)
    {
    
      SetActiveGPU forDuration(device);
      int numRaysWeHaveTotal = intraNode.sumRaysReceived;
      {
        ENTER();
        if (logQueues) 
          printf("buildlocalrays r%i total rays %i (q0)\n",
                 myGID,numRaysWeHaveTotal);
        __rtc_launch(device->rtc,
                     buildStagedRayQueue,
                     divRoundUp(numRaysWeHaveTotal,1024),1024,
                     // args
                     stagedRayQueue,
                     raysOnly[0],
                     numRaysWeHaveTotal);
      
        device->rtc->sync();
        LEAVE(numRaysWeHaveTotal,"buildStagedRayQueue");
      }
    
      auto savedOriginalRayCount = device->rayQueue->numActive;
      auto savedOriginalRayQueue = device->rayQueue->traceAndShadeReadQueue.rays;
      device->rayQueue->traceAndShadeReadQueue.rays = stagedRayQueue;
      device->rayQueue->numActive = numRaysWeHaveTotal;

      {
        ENTER()
          if (logQueues) 
            printf("localtrace r%i total rays %i\n",
                   myGID,numRaysWeHaveTotal);
        context->traceRaysLocally(model,rngSeed,needHitIDs);
        device->rtc->sync();
        LEAVE(numRaysWeHaveTotal,"localTrace");
      }
    
      if (logQueues) 
        printf("buildhits r%i total rays %i (q0)\n",
               myGID,numRaysWeHaveTotal);
      {
        ENTER();
        __rtc_launch(device->rtc,
                     buildHitsOnly,
                     divRoundUp(numRaysWeHaveTotal,1024),1024,
                     // args
                     hitsOnly[0],
                     stagedRayQueue,
                     numRaysWeHaveTotal);
        device->rtc->sync();
        LEAVE(numRaysWeHaveTotal,"buildHitsOnly");
      }
      device->rayQueue->numActive = savedOriginalRayCount;
      device->rayQueue->traceAndShadeReadQueue.rays = savedOriginalRayQueue;
    }
  
    void TwoStage::exchangeHits_intraNode()
    {
      ENTER();
      if (opt_mpi) {
        void *sendBuf = hitsOnly[0];
        void *recvBuf = hitsOnly[1];
        std::vector<int> sendOffsets(intraNode.comm.size);
        std::vector<int> sendCounts(intraNode.comm.size);
        std::vector<int> recvOffsets(intraNode.comm.size);
        std::vector<int> recvCounts(intraNode.comm.size);
        int recvSum = 0;
        int sendSum = 0;
        for (int i=0;i<intraNode.comm.size;i++) {
          int recvCount = crossNodes.sumRaysReceived;
          int sendCount = intraNode.rayCounts[i];
          recvOffsets[i] = recvSum*sizeof(HitOnly);
          sendOffsets[i] = sendSum*sizeof(HitOnly);
          recvCounts[i] = recvCount*sizeof(HitOnly);
          sendCounts[i] = sendCount*sizeof(HitOnly);
          sendSum += sendCount;
          recvSum += recvCount;
        }
        BN_MPI_CALL(Alltoallv(sendBuf,
                              (const int*)sendCounts.data(),
                              (const int*)sendOffsets.data(),
                              MPI_BYTE,
                              recvBuf,
                              (const int*)recvCounts.data(),
                              (const int*)recvOffsets.data(),
                              MPI_BYTE,
                              intraNode.comm));
      } else {
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        for (int g=0;g<gpusPerHost;g++) {
          MPI_Request req;
          int recvCount = 0;
          for (int h=0;h<numHosts;h++)
            recvCount += global.rayCounts[rankOf(h,gpuIdx)];
          world.recv(rankOf(hostIdx,g),0,
                     hitsOnly[1]+recvOfs,recvCount,req);
          if (logQueues) 
            printf("xchg-intra r%i receiving %i from %i (q0->1)\n",
                   myGID,recvCount,rankOf(hostIdx,g));
          requests.push_back(req);
          recvOfs += recvCount;
        }

        // and matching sends
        int sendOfs = 0;
        for (int g=0;g<gpusPerHost;g++) {
          int sendCount = 0;
          for (int h=0;h<numHosts;h++)
            sendCount += global.rayCounts[rankOf(h,g)];
          MPI_Request req;
          world.send(rankOf(hostIdx,g),0,
                     hitsOnly[0]+sendOfs,sendCount,req);
          if (logQueues) 
            printf("xchg-intra r%i sending %i to %i (q0->1)\n",
                   myGID,sendCount,rankOf(hostIdx,g));
          requests.push_back(req);
          sendOfs += sendCount;
        }
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
      LEAVE(recvOfs,"exchangeHits_intraNode");
    }


    void TwoStage::reduceHits_intraNode()
    {
      ENTER();
      SetActiveGPU forDuration(device);
      int numUniqueRaysThisGPU = crossNodes.sumRaysReceived;
      // int g = gpuIdx;
      // for (int h=0;h<numHosts;h++)
      //   numUniqueRaysThisGPU += rayCounts[rankOf(h,g)];

      if (logQueues) 
        printf("r%i intra-reducing %i sets of %i hits (q1)\n",
               myGID,
               gpusPerHost,
               numUniqueRaysThisGPU);
      __rtc_launch(device->rtc,
                   reduceReceivedHitsKernel_intraNode,
                   divRoundUp(numUniqueRaysThisGPU,128),128,
                   // args
                   hitsOnly[1],
                   numUniqueRaysThisGPU,
                   gpusPerHost);
      device->rtc->sync();
      LEAVE(numUniqueRaysThisGPU*gpusPerHost,"reduceHits_intraNode");
    }
  
    void TwoStage::exchangeHits_crossNodes()
    {
      ENTER();
      if (opt_mpi) {
        int myRayCount = device->rayQueue->numActive;
        void *sendBuf = hitsOnly[1];
        void *recvBuf = hitsOnly[0];
        std::vector<int> sendOffsets(crossNodes.comm.size);
        std::vector<int> sendCounts(crossNodes.comm.size);
        std::vector<int> recvOffsets(crossNodes.comm.size);
        std::vector<int> recvCounts(crossNodes.comm.size);
        int recvSum = 0;
        int sendSum = 0;
        for (int i=0;i<crossNodes.comm.size;i++) {
          int recvCount = myRayCount;
          int sendCount = crossNodes.rayCounts[i];
          recvOffsets[i] = recvSum*sizeof(HitOnly);
          sendOffsets[i] = sendSum*sizeof(HitOnly);
          recvCounts[i] = recvCount*sizeof(HitOnly);
          sendCounts[i] = sendCount*sizeof(HitOnly);
          sendSum += sendCount;
          recvSum += recvCount;
        }
        if (logQueues) 
          printf("xchg-hits-cross r%i myRayCount %i sendSum %i recvSum %i\n",
                 myGID,myRayCount,sendSum,recvSum);
        BN_MPI_CALL(Alltoallv(sendBuf,
                              (const int*)sendCounts.data(),
                              (const int*)sendOffsets.data(),
                              MPI_BYTE,
                              recvBuf,
                              (const int*)recvCounts.data(),
                              (const int*)recvOffsets.data(),
                              MPI_BYTE,
                              crossNodes.comm));
      } else {
        std::vector<MPI_Request> requests;
        int recvOfs = 0;
        int recvCount = global.rayCounts[rankOf(hostIdx,gpuIdx)];
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;

          if (logQueues) 
            printf("xchg-intra r%i receiving %i from %i (q1->0)\n",
                   myGID,recvCount,rankOf(h,gpuIdx));
          world.recv(rankOf(h,gpuIdx),0,
                     hitsOnly[0]+recvOfs,recvCount,req);
          requests.push_back(req);
          recvOfs += recvCount;
        }

        // and matching sends
        int sendOfs = 0;
        for (int h=0;h<numHosts;h++) {
          MPI_Request req;
          int sendCount = global.rayCounts[rankOf(h,gpuIdx)];
          if (logQueues) 
            printf("xchg-intra r%i sending %i to %i (q1->0)\n",
                   myGID,sendCount,rankOf(h,gpuIdx));
          world.send(rankOf(h,gpuIdx),0,
                     hitsOnly[1]+sendOfs,sendCount,req);
          requests.push_back(req);
          sendOfs += sendCount;
        }
    
        BN_MPI_CALL(Waitall(requests.size(),requests.data(),MPI_STATUSES_IGNORE));
      }
      LEAVE(recvOfs,"exchangeHits_crossNodes");
    }
  
    void TwoStage::reduceHits_crossNodes()
    {
      ENTER();
      SetActiveGPU forDuration(device);
      if (opt_mpi) {
        int numUniqueRaysThisGPU = device->rayQueue->numActive;
        if (logQueues) 
          printf("r%i cross-reducing %i sets of %i hits (q0)\n",
                 myGID,numHosts,numUniqueRaysThisGPU);
        __rtc_launch(device->rtc,
                     reduceReceivedHitsKernel_crossNodes,
                     divRoundUp(numUniqueRaysThisGPU,128),128,
                     // args
                     device->rayQueue->traceAndShadeReadQueue.rays,
                     hitsOnly[0],
                     numUniqueRaysThisGPU,
                     numHosts);
      } else {
        int numUniqueRaysThisGPU = global.rayCounts[rankOf(hostIdx,gpuIdx)];
        if (logQueues) 
          printf("r%i cross-reducing %i sets of %i hits (q0)\n",
                 myGID,numHosts,numUniqueRaysThisGPU);
        __rtc_launch(device->rtc,
                     reduceReceivedHitsKernel_crossNodes,
                     divRoundUp(numUniqueRaysThisGPU,128),128,
                     // args
                     device->rayQueue->traceAndShadeReadQueue.rays,
                     hitsOnly[0],
                     numUniqueRaysThisGPU,
                     numHosts);
      }
      device->rtc->sync();
      LEAVE(numUniqueRaysThisGPU*numHosts,"reduceHits_crossNodes");
    }
  
  }
}
