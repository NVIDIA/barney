// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "barney/api/Context.h"
#include "barney/Object.h"
#include <set>
#include "barney/WorkerTopo.h"

#define BN_TRACK_LEAKS(a) /* nothing */

namespace BARNEY_NS {
  using namespace owl::common;
  using barney_api::FromEnv;
  using barney_api::LocalSlot;
  
  enum { tileSize = 32 };
  enum { pixelsPerTile = tileSize*tileSize };
  enum { rayQueueSize = 4*1024*1024 };

  struct FrameBuffer;
  struct GlobalModel;
  struct Camera;
  struct Renderer;
  struct Geometry;
  
  namespace render {
    struct HostMaterial;
    struct SamplerRegistry;
    struct MaterialRegistry;
    struct DeviceMaterial;    
  };

  struct SlotContext {
    Context *context;
    int modelRankInThisSlot;//dataGroupID;
    /*! device(s) inside this data group; will be a subset of
      Context::devices */
    std::vector<int>     gpuIDs;
    DevGroup::SP         devices;
    std::shared_ptr<render::HostMaterial>     defaultMaterial = 0;
    std::shared_ptr<render::SamplerRegistry>  samplerRegistry = 0;
    std::shared_ptr<render::MaterialRegistry> materialRegistry = 0;
  };

  struct GlobalTraceImpl;

  // struct LogicalDevice {
  //   /* the worker rank that this device lives on - '0' if local
  //      rendering, and mpi rank in 'workers' mpi goup if mpi */
  //   int worker;
  //   /*! the local device index for the worker that this device is
  //     on */
  //   int local;
  //   /*! the data rank that this gpu holds */
  //   int dataRank;
  // };

  struct Context : public barney_api::Context
  {
    Context(const std::vector<LocalSlot> &localSlots,
            WorkerTopo::SP topo);
    // Context(const std::vector<int> &dataGroupIDs,
    //         const std::vector<int> &gpuIDs,
    //         int globalIndex,
    //         int globalIndexStep);
    virtual ~Context();

    //    std::vector<LogicalDevice> logicalDevices;
    WorkerTopo::SP const topo;
    // int numDifferentModelSlots = -1;
    
    /*! create a frame buffer object suitable to this context */
    // virtual FrameBuffer *createFB(int owningRank) = 0;
    std::shared_ptr<barney_api::Model>
    createModel() override;
    
    std::shared_ptr<barney_api::Renderer>
    createRenderer() override;

    std::shared_ptr<barney_api::Volume>
    createVolume(const std::shared_ptr<barney_api::ScalarField> &sf) override;

    std::shared_ptr<barney_api::TextureData> 
    createTextureData(int slot,
                      BNDataType texelFormat,
                      vec3i dims,
                      const void *texels) override;
    
    std::shared_ptr<barney_api::ScalarField>
    createScalarField(int slot, const std::string &type) override;
    
    std::shared_ptr<barney_api::Geometry>
    createGeometry(int slot, const std::string &type) override;
    
    std::shared_ptr<barney_api::Material>
    createMaterial(int slot, const std::string &type) override;

    std::shared_ptr<barney_api::Sampler>
    createSampler(int slot, const std::string &type) override;

    std::shared_ptr<barney_api::Light>
    createLight(int slot, const std::string &type) override;

    std::shared_ptr<barney_api::Group>
    createGroup(int slot,
                barney_api::Geometry **geoms, int numGeoms,
                barney_api::Volume **volumes, int numVolumes) override;

    std::shared_ptr<barney_api::Data>
    createData(int slot,
               BNDataType dataType) override;
    
    std::shared_ptr<barney_api::Camera>
    createCamera(const std::string &type) override;

    std::shared_ptr<barney_api::Texture>
    createTexture(const std::shared_ptr<barney_api::TextureData> &td,
                  BNTextureFilterMode  filterMode,
                  BNTextureAddressMode addressModes[],
                  BNTextureColorSpace  colorSpace) override;
 
    
    // std::shared_ptr<render::HostMaterial> getDefaultMaterial(int slot);

    static bool logging();
    
    /*! pretty-printer for printf-debugging */
    virtual std::string toString() const 
    { return "<Context(abstract)>"; }

    render::World *getWorld(int slot);
    
    /* goes across all devices, syncs that device, and checks for
       errors - careful, this will be very slow, shoudl only be used
       for debugging multi-gpu race conditions and such */
    void syncCheckAll(const char *where="");
    
    // for debugging ...
    virtual void barrier(bool warn=true) {}
    
    /*! generate a new wave-front of rays */
    void generateRays(Camera *camera,
                      Renderer *renderer,
                      FrameBuffer *fb);
    
    /*! have each *local* GPU trace its current wave-front of rays */
    void traceRaysLocally(GlobalModel *model, uint32_t rngSeed, bool needHitIDs);
    
    /*! trace all rays currently in a ray queue, including forwarding
      if and where applicable, untile every ray in the ray queue as
      found its intersection */
    void traceRaysGlobally(GlobalModel *model, uint32_t rngSeed, bool needHitIDs);

    void shadeRaysLocally(Renderer *renderer,
                          GlobalModel *model,
                          FrameBuffer *fb,
                          int generation,
                          uint32_t rngSeed);
    void finalizeTiles(FrameBuffer *fb);
    
    void renderTiles(Renderer *renderer,
                     GlobalModel *model,
                     Camera      *camera,
                     FrameBuffer *fb);
    
    virtual void render(Renderer    *renderer,
                        GlobalModel *model,
                        Camera      *camera,
                        FrameBuffer *fb) = 0;

    void ensureRayQueuesLargeEnoughFor(FrameBuffer *fb);

    /*! helper function to print a warning when app tries to create an
        object of certain kind and type that barney does not
        support */
    void warn_unsupported_object(const std::string &kind,
                                 const std::string &type);
    DevGroup::SP getDevices(int slot)
    {
      if (slot < 0)
        return this->devices;
      else 
        return getSlot(slot)->devices;
    }

    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks. We use this to
      compute how many rays are active globally, which in turn we need
      to determine if at least one gpu on any rank still has some
      active rays that need to bounced (even if we locally do not) */
    int numRaysActiveLocally();
    
    /*! returns how many rays are active in all ray queues, across all
      devices and, where applicable, across all ranks. We use this to
      decide whether we can terminate a frame, or need more bounces - we
      may actually need to enter another bounce even if *we* do not have
      any rays */
    virtual int numRaysActiveGlobally() = 0;
    
    
    int contextSize() const;

    const bool isActiveWorker;

    /*! whether we have successfully enabled peer access across all
        GPUs (eg, to allow tiledFB to write to gpu 0 linear fb */
    bool havePeerAccess = false;
    
    SlotContext *getSlot(int slot);
    std::vector<SlotContext> perSlot;
    DevGroup::SP devices;
    /*! 'usually' we can rely on all GPUs having peer-(write-)access
        to the memory location that the app wants to have the frame
        buffer read into; but for some hardware configs there is no
        peer access, and non-primary GPUs have to first copy to that
        primary gpu. If this variabel is null, we assume that every
        gpu can just write; if not, we'll have to first create staging
        copies on that device */
    Device *deviceWeNeedToCopyToForFBMap = nullptr;
    // int const globalIndex;
    GlobalTraceImpl *globalTraceImpl = 0;

    /*! cut plane active during the current renderTiles() call;
        populated from Renderer::cutPlane and read by traceRaysLocally */
    vec4f activeCutPlane{0.f, 0.f, 0.f, -1e30f};
  };

  struct GlobalTraceImpl {
    GlobalTraceImpl(Context *context)
      : context(context)
    {}
    
    virtual void traceRays(GlobalModel *model,
                           uint32_t rngSeed,
                           bool needHitIDs) = 0;
    // virtual int maxRaysWeCanHandle() = 0;
    
    Context *const context;
  };

  
  
  // ==================================================================
  // INLINE IMPLEMENTATION SECTION
  // ==================================================================


  /* goes across all devices, syncs that device, and checks for
     errors - careful, this will be very slow, shoudl only be used
     for debugging multi-gpu race conditions and such */
  inline void Context::syncCheckAll(const char *where)
  {
    for (auto device : *devices)
      device->sync();
  }


}

