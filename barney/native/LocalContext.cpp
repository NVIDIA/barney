// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/LocalContext.h"
#include "native/fb/LocalFB.h"
#include "native/globalTrace/RQSLocal.h"
#include "native/render/RayQueue.h"
#include "native/render/Renderer.h"
#include "native/WorkerTopo.h"

namespace BARNEY_NS {
  namespace native {

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
    LocalContext::makeTopo(const std::vector<DataGroupDescriptor> &localDataGroups)
    {
      std::vector<WorkerTopo::Device> devices;
      for (auto &ldg : localDataGroups) {
        for (auto gpuID : ldg.gpuIDs) {
          WorkerTopo::Device dev;
          dev.local = (int)devices.size();
          dev.worker = 0;
          dev.worldRank = 0;
          dev.dataRank = ldg.dataRank;
          dev.hostNameHash = getHostNameHash();
          dev.physicalDeviceHash = rtc::getPhysicalDeviceHash(gpuID);
          devices.push_back(dev);
        }
      }
      return std::make_shared<WorkerTopo>(devices,0,(int)devices.size());
    }

    LocalContext::LocalContext
    (const std::vector<DataGroupDescriptor> &dataGroupsOnThisContext)
      : Context(dataGroupsOnThisContext,
                makeTopo(dataGroupsOnThisContext))
    {
      globalTraceImpl = new RQSLocal(this);
    }

    LocalContext::~LocalContext()
    {
      /* not doing anything, but leave this in to ensure that derived
         classes' destrcutors get called !*/
    }

    std::shared_ptr<FrameBuffer> LocalContext::createFrameBuffer()
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

      if (fb->accumID >= renderer->sampleLimit)
        return;

      renderTiles(renderer,model,camera,fb);
      finalizeTiles(fb);
      fb->finalizeFrame();
    }

    Context *LocalContext::create(/*! this is the NUMBER of different data
                                    groups/slots inthat context. a data
                                    *group* is one or more GPUs that share a
                                    common set of geometric/volumetric
                                    objects. */
                                  int        numDataGroupsOnThisContext,
                                  /*! tells which data rank / 'color' of data
                                    will be stored in each of the
                                    `numDataGroupsOnThisContext` data groups
                                    of this context. This array *must* have
                                    `numDataGroupsOnThisContext` entries */
                                  const int *dataRanksInDataGroup,
                                  /*! which gpu(s) to use for this
                                    process. GPUs will be assigned to
                                    data groups on a round-robin
                                    basis, so the i'th GPU listed here
                                    will get assigned to data group
                                    'i%numDataGroupsOnThisContext'. if
                                    passed as null, this means 'pick
                                    what you find' */
                                  int numGPUs,
                                  const int *gpuIDs)
    {
      assert(dataRanksInDataGroup);
      assert(gpuIDs);
      assert(numGPUs >= numDataGroupsOnThisContext);
      assert((numGPUs % numDataGroupsOnThisContext) == 0);
      for (int i=0;i<numGPUs;i++) assert(gpuIDs[i] >= 0);

      // build local data group descriptors:

      // 1) create descriptors and fill in data ranks
      assert(numDataGroupsOnThisContext > 0);
      std::vector<DataGroupDescriptor>
        dataGroupsOnThisContext(numDataGroupsOnThisContext);
      for (int ldgIdx=0;ldgIdx<numDataGroupsOnThisContext;ldgIdx++) {
        auto &ldg = dataGroupsOnThisContext[ldgIdx];
        ldg.dataRank = dataRanksInDataGroup[ldgIdx];
      }

      for (int i=0;i<numGPUs;i++) {
        auto &ldg = dataGroupsOnThisContext[i%numDataGroupsOnThisContext];
        int gpuID = gpuIDs[i];
        assert(gpuID >= 0);
        ldg.gpuIDs.push_back(gpuID);
      }

      Context *ctx = new LocalContext(dataGroupsOnThisContext);
      assert(ctx);
      return ctx;
    }
  
  }
}
