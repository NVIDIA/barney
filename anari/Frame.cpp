// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "Frame.h"
#include "Device.h"
// std
#include <algorithm>
#include <chrono>
#include <iostream>
// cuda
#if BANARI_HAVE_CUDA
#include <cuda_runtime.h>
#endif

namespace barney_device {

  // Helper functions ///////////////////////////////////////////////////////////

  static BNDataType toBarney(anari::DataType type)
  {
    switch (type) {
    case ANARI_UFIXED8_VEC4:
      return BN_UFIXED8_RGBA;
    case ANARI_UFIXED8_RGBA_SRGB:
      return BN_UFIXED8_RGBA_SRGB;
    case ANARI_FLOAT32:
      return BN_FLOAT;
    case ANARI_FLOAT32_VEC3:
      return BN_FLOAT3;
    case ANARI_FLOAT32_VEC4:
      return BN_FLOAT4;
    }
    return BN_DATA_UNDEFINED;
  }

  // Frame definitions //////////////////////////////////////////////////////////

  Frame::Frame(BarneyGlobalState *s) : helium::BaseFrame(s), m_renderer(this)
  {
    if (s->slot == 0)
      m_bnFrameBuffer = bnFrameBufferCreate(s->tether->context);
  }

  Frame::~Frame()
  {
    wait();
    cleanup();
    if (m_bnFrameBuffer)
      bnRelease(m_bnFrameBuffer);
  }

  bool Frame::isValid() const
  {
    return m_renderer && m_renderer->isValid() && m_camera && m_camera->isValid()
      && m_world && m_world->isValid();
  }

  BarneyGlobalState *Frame::deviceState() const
  {
    return (BarneyGlobalState *)helium::BaseObject::m_state;
  }

  void Frame::commitParameters()
  {
    m_renderer = getParamObject<Renderer>("renderer");
    m_camera = getParamObject<Camera>("camera");
    m_world = getParamObject<World>("world");
    m_channelTypes.color =
      getParam<anari::DataType>("channel.color", ANARI_UNKNOWN);
    m_channelTypes.depth =
      getParam<anari::DataType>("channel.depth", ANARI_UNKNOWN);
    m_channelTypes.primID =
      getParam<anari::DataType>("channel.primitiveId", ANARI_UNKNOWN);
    m_channelTypes.instID =
      getParam<anari::DataType>("channel.instanceId", ANARI_UNKNOWN);
    m_channelTypes.objID =
      getParam<anari::DataType>("channel.objectId", ANARI_UNKNOWN);
    m_channelTypes.normal =
      getParam<anari::DataType>("channel.normal", ANARI_UNKNOWN);
    m_size = getParam<math::uint2>("size", math::uint2(10, 10));
    m_displaySize = m_size;  /* updated in finalize() from actual FB when slot==0 */
    m_enableDenoising = getParam<bool>("denoise", false);
    m_enableUpscaling = getParam<bool>("upscale", false);
  }

  void Frame::finalize()
  {
    cleanup();

    bool denoise = m_enableDenoising;
    if (!m_renderer) {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "missing required parameter 'renderer' on frame");
    }

    if (!m_camera) {
      reportMessage(
                    ANARI_SEVERITY_WARNING, "missing required parameter 'camera' on frame");
    }

    if (!m_world) {
      reportMessage(
                    ANARI_SEVERITY_WARNING, "missing required parameter 'world' on frame");
    }

    if (deviceState()->slot == 0) {
      const auto &size = m_size;
      const auto numPixels = size.x * size.y;

      uint32_t requiredChannels = BN_FB_COLOR;
      if (m_channelTypes.depth == ANARI_FLOAT32) 
        requiredChannels |= BN_FB_DEPTH;
      if (m_channelTypes.primID == ANARI_UINT32)
        requiredChannels |= BN_FB_PRIMID;
      if (m_channelTypes.objID == ANARI_UINT32)
        requiredChannels |= BN_FB_OBJID;
      if (m_channelTypes.instID == ANARI_UINT32)
        requiredChannels |= BN_FB_INSTID;
      if (m_channelTypes.normal == ANARI_FLOAT32_VEC3)
        requiredChannels |= BN_FB_NORMAL;

      if (m_bnFrameBuffer) {
        bnSet1i(m_bnFrameBuffer, "denoise", denoise ? 1 : 0);
        bnSet1i(m_bnFrameBuffer, "upscale", m_enableUpscaling ? 1 : 0);
        bnCommit(m_bnFrameBuffer);

        bnFrameBufferResize(m_bnFrameBuffer,
                            toBarney(m_channelTypes.color),
                            size.x,
                            size.y,
                            requiredChannels);
        int actualW = 0, actualH = 0;
        bnFrameBufferGetSize(m_bnFrameBuffer, &actualW, &actualH);
        m_displaySize.x = static_cast<uint32_t>(actualW);
        m_displaySize.y = static_cast<uint32_t>(actualH);
      } else {
        m_displaySize = m_size;
      }
    } else {
      m_displaySize = m_size;
    }
  }

  bool Frame::getProperty(const std::string_view &name,
                          ANARIDataType type,
                          void *ptr,
                          uint64_t size,
                          uint32_t flags)
  {
    if (type == ANARI_FLOAT32 && name == "duration") {
      if (flags & ANARI_WAIT)
        wait();
      helium::writeToVoidP(ptr, m_duration);
      return true;
    }

    return 0;
  }

  void Frame::renderFrame()
  {
    auto start = std::chrono::steady_clock::now();

    auto *state = deviceState();
    state->commitBuffer.flush();

    bool firstFrame = 0;
    if (m_lastCommitFlush < state->commitBuffer.lastObjectFinalization()) {
      m_lastCommitFlush = helium::newTimeStamp();
      if (m_bnFrameBuffer && state->slot == 0)
        bnAccumReset(m_bnFrameBuffer);
      firstFrame = true;
    }
    if (!isValid()) {
      reportMessage(
                    ANARI_SEVERITY_ERROR, "skipping render of incomplete frame object");
      reportMessage(ANARI_SEVERITY_ERROR,//_DEBUG,
                    "    renderer(%p) - isValid:(%i)",
                    m_renderer.get(),
                    m_renderer ? m_renderer->isValid() : 0);
      reportMessage(ANARI_SEVERITY_ERROR,//_DEBUG,
                    "    world(%p) - isValid:(%i)",
                    m_world.ptr,
                    m_world ? m_world->isValid() : 0);
      reportMessage(ANARI_SEVERITY_ERROR,//_DEBUG,
                    "    camera(%p) - isValid:(%i)",
                    m_camera.ptr,
                    m_camera ? m_camera->isValid() : 0);
      return;
    }

    auto model = m_world->makeCurrent();
    
    if (m_lastFrameWasFirstFrame && m_channelTypes.depth != ANARI_UNKNOWN
        && !m_didMapChannel.depth)
      reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
                    "last frame had a depth buffer request, but never mapped it");
    if (m_lastFrameWasFirstFrame && m_channelTypes.primID != ANARI_UNKNOWN
        && !m_didMapChannel.primID)
      reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
                    "last frame had a primID buffer request, but never mapped it");
    if (m_lastFrameWasFirstFrame && m_channelTypes.objID != ANARI_UNKNOWN
        && !m_didMapChannel.objID)
      reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
                    "last frame had a objID buffer request, but never mapped it");
    if (m_lastFrameWasFirstFrame && m_channelTypes.instID != ANARI_UNKNOWN
        && !m_didMapChannel.instID)
      reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
                    "last frame had a instID buffer request, but never mapped it");

    if (state->slot == 0) {
      auto &peers = state->tether->devices;
      state->tether->numRenderCallsOutstanding = (int)peers.size();
      state->tether->deferredRenderCall.model = model;
      state->tether->deferredRenderCall.renderer = m_renderer->barneyRenderer;
      state->tether->deferredRenderCall.fb = m_bnFrameBuffer;
      state->tether->deferredRenderCall.camera = m_camera->barneyCamera();
    }
    --state->tether->numRenderCallsOutstanding;
    if (state->tether->numRenderCallsOutstanding == 0) {
      bnRender(state->tether->deferredRenderCall.renderer,
               state->tether->deferredRenderCall.model,
               state->tether->deferredRenderCall.camera,
               state->tether->deferredRenderCall.fb
               );
      m_lastFrameWasFirstFrame = firstFrame;
    }
    m_didMapChannel.depth = false;
    m_didMapChannel.primID = false;
    m_didMapChannel.instID = false;
    m_didMapChannel.objID = false;

    auto end = std::chrono::steady_clock::now();
    m_duration = std::chrono::duration<float>(end - start).count();
  }

  void *Frame::map(std::string_view channel,
                   uint32_t *width,
                   uint32_t *height,
                   ANARIDataType *pixelType)
  {
    wait();

    if (deviceState()->slot != 0) {
      return nullptr;
    }
    *width = m_size.x;
    *height = m_size.y;
    int numPixels = *width * *height;
    if (channel == "channel.color") {
      if (m_channelBuffers.color)
        throw std::runtime_error(
                                 "trying to map color buffer, but color buffer already mapped");
      m_channelBuffers.color = new uint32_t[numPixels
                                            * (m_channelTypes.color == ANARI_FLOAT32_VEC4 ? 4 : 1)];
      bnFrameBufferRead(m_bnFrameBuffer,
                        BN_FB_COLOR,
                        m_channelBuffers.color,
                        toBarney(m_channelTypes.color));
      *pixelType = m_channelTypes.color;
      return m_channelBuffers.color;
    } else if (channel == "channel.depth") {
      if (m_channelBuffers.depth)
        throw std::runtime_error
          ("trying to map depth buffer, but depth buffer already mapped");
      m_channelBuffers.depth = new float[numPixels];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_DEPTH, m_channelBuffers.depth, BN_FLOAT);
      m_didMapChannel.depth = true;
      *pixelType = ANARI_FLOAT32;
      return m_channelBuffers.depth;
    } else if (channel == "channel.primitiveId") {
      if (m_channelBuffers.primID)
        throw std::runtime_error
          ("trying to map channel.primitiveId, but seems already mapped");
      m_channelBuffers.primID = new int[numPixels];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_PRIMID, m_channelBuffers.primID, BN_INT);
      m_didMapChannel.primID = true;
      *pixelType = ANARI_UINT32;
      return m_channelBuffers.primID;
    } else if (channel == "channel.objectId") {
      if (m_channelBuffers.objID)
        throw std::runtime_error
          ("trying to map channel.objectId, but seems already mapped");
      m_channelBuffers.objID = new int[numPixels];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_OBJID, m_channelBuffers.objID, BN_INT);
      m_didMapChannel.objID = true;
      *pixelType = ANARI_UINT32;
      return m_channelBuffers.objID;
    } else if (channel == "channel.instanceId") {
      if (m_channelBuffers.instID)
        throw std::runtime_error
          ("trying to map channel.instanceId, but seems already mapped");
      m_channelBuffers.instID = new int[numPixels];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_INSTID, m_channelBuffers.instID, BN_INT);
      m_didMapChannel.instID = true;
      *pixelType = ANARI_UINT32;
      return m_channelBuffers.instID;
    } else if (channel == "channel.normal") {
      if (m_channelBuffers.normal)
        throw std::runtime_error
          ("trying to map channel.normal, but seems already mapped");
      m_channelBuffers.normal = new float[numPixels * 3];
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_NORMAL, m_channelBuffers.normal, BN_FLOAT3);
      m_didMapChannel.normal = true;
      *pixelType = ANARI_FLOAT32_VEC3;
      return m_channelBuffers.normal;
#if BANARI_HAVE_CUDA
    } else if (channel == "channel.colorCUDA") {
      if (m_channelBuffers.color)
        throw std::runtime_error
          ("trying to map color buffer, but color buffer already mapped");
      cudaMalloc((void **)&m_channelBuffers.color,
                 numPixels
                 * (m_channelTypes.color == ANARI_FLOAT32_VEC4 ? sizeof(math::float4)
                    : sizeof(uint32_t)));
      bnFrameBufferRead(m_bnFrameBuffer,
                        BN_FB_COLOR,
                        m_channelBuffers.color,
                        toBarney(m_channelTypes.color));
      *pixelType = m_channelTypes.color;
      return m_channelBuffers.color;
    } else if (channel == "channel.depthCUDA"
               && m_channelTypes.depth == ANARI_FLOAT32) {
      if (m_channelBuffers.depth)
        throw std::runtime_error(
                                 "trying to map depth buffer, but depth buffer already mapped");
      cudaMalloc((void **)&m_channelBuffers.depth, numPixels * sizeof(float));
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_DEPTH, m_channelBuffers.depth, BN_FLOAT);
      *pixelType = ANARI_FLOAT32;
      return m_channelBuffers.depth;
    } else if (channel == "channel.primitiveIdCUDA"
               && m_channelTypes.primID == ANARI_UINT32) {
      if (m_channelBuffers.primID)
        throw std::runtime_error
          ("trying to map primitiveId buffer, but buffer already mapped");
      cudaMalloc((void **)&m_channelBuffers.primID, numPixels * sizeof(uint32_t));
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_PRIMID, m_channelBuffers.primID, BN_INT);
      *pixelType = ANARI_UINT32;
      return m_channelBuffers.primID;
    } else if (channel == "channel.objectIdCUDA"
               && m_channelTypes.objID == ANARI_UINT32) {
      if (m_channelBuffers.objID)
        throw std::runtime_error
          ("trying to map objectId buffer, but buffer already mapped");
      cudaMalloc((void **)&m_channelBuffers.objID, numPixels * sizeof(uint32_t));
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_OBJID, m_channelBuffers.objID, BN_INT);
      *pixelType = ANARI_UINT32;
      return m_channelBuffers.objID;
    } else if (channel == "channel.instanceIdCUDA"
               && m_channelTypes.instID == ANARI_UINT32) {
      if (m_channelBuffers.instID)
        throw std::runtime_error
          ("trying to map instanceId buffer, but buffer already mapped");
      cudaMalloc((void **)&m_channelBuffers.instID, numPixels * sizeof(uint32_t));
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_INSTID, m_channelBuffers.instID, BN_INT);
      *pixelType = ANARI_UINT32;
      return m_channelBuffers.instID;
    } else if (channel == "channel.normalCUDA"
               && m_channelTypes.normal == ANARI_FLOAT32_VEC3) {
      if (m_channelBuffers.normal)
        throw std::runtime_error
          ("trying to map normal buffer, but buffer already mapped");
      cudaMalloc((void **)&m_channelBuffers.normal, numPixels * 3 * sizeof(float));
      bnFrameBufferRead(m_bnFrameBuffer, BN_FB_NORMAL, m_channelBuffers.normal, BN_FLOAT3);
      *pixelType = ANARI_FLOAT32_VEC3;
      return m_channelBuffers.normal;
#endif
    } else {
      reportMessage(ANARI_SEVERITY_WARNING,
                    "trying to map unsupported/unrecognized channel type '%s'",
                    std::string(channel).c_str());
    }

    *width = 0;
    *height = 0;
    *pixelType = ANARI_UNKNOWN;
    return nullptr;
  }

  void Frame::unmap(std::string_view channel)
  {
    if (channel == "channel.color") {
      if (m_channelBuffers.color)
        delete[] m_channelBuffers.color;
      m_channelBuffers.color = 0;
    } else if (channel == "channel.depth" && m_channelBuffers.depth) {
      if (m_channelBuffers.depth)
        delete[] m_channelBuffers.depth;
      m_channelBuffers.depth = 0;
    } else if (channel == "channel.primitiveId" && m_channelBuffers.primID) {
      if (m_channelBuffers.primID)
        delete[] m_channelBuffers.primID;
      m_channelBuffers.primID = 0;
    } else if (channel == "channel.objectId" && m_channelBuffers.objID) {
      if (m_channelBuffers.objID)
        delete[] m_channelBuffers.objID;
      m_channelBuffers.objID = 0;
    } else if (channel == "channel.instanceId" && m_channelBuffers.instID) {
      if (m_channelBuffers.instID)
        delete[] m_channelBuffers.instID;
      m_channelBuffers.instID = 0;
    } else if (channel == "channel.normal" && m_channelBuffers.normal) {
      if (m_channelBuffers.normal)
        delete[] m_channelBuffers.normal;
      m_channelBuffers.normal = 0;
    } else if (channel == "channel.colorCUDA") {
#if BANARI_HAVE_CUDA
      if (m_channelBuffers.color)
        cudaFree(m_channelBuffers.color);
      m_channelBuffers.color = 0;
#endif
    } else if (channel == "channel.depthCUDA") {
#if BANARI_HAVE_CUDA
      if (m_channelBuffers.depth)
        cudaFree(m_channelBuffers.depth);
      m_channelBuffers.depth = 0;
#endif
    } else if (channel == "channel.primitiveIdCUDA") {
#if BANARI_HAVE_CUDA
      if (m_channelBuffers.primID)
        cudaFree(m_channelBuffers.primID);
      m_channelBuffers.primID = 0;
#endif
    } else if (channel == "channel.objectIdCUDA") {
#if BANARI_HAVE_CUDA
      if (m_channelBuffers.objID)
        cudaFree(m_channelBuffers.objID);
      m_channelBuffers.objID = 0;
#endif
    } else if (channel == "channel.instanceIdCUDA") {
#if BANARI_HAVE_CUDA
      if (m_channelBuffers.instID)
        cudaFree(m_channelBuffers.instID);
      m_channelBuffers.instID = 0;
#endif
    } else if (channel == "channel.normalCUDA") {
#if BANARI_HAVE_CUDA
      if (m_channelBuffers.normal)
        cudaFree(m_channelBuffers.normal);
      m_channelBuffers.normal = 0;
#endif
    }
  }

  int Frame::frameReady(ANARIWaitMask m)
  {
    if (m == ANARI_NO_WAIT)
      return ready();
    else {
      wait();
      return 1;
    }
  }

  void Frame::discard()
  {
    // no-op (not yet async)
  }

  bool Frame::ready() const
  {
    return true; // not yet async
  }

  void Frame::wait() const {}

  void Frame::cleanup()
  {
    delete[] m_channelBuffers.color;
    m_channelBuffers.color = nullptr;

    delete[] m_channelBuffers.depth;
    m_channelBuffers.depth = nullptr;

    delete[] m_channelBuffers.primID;
    m_channelBuffers.primID = nullptr;

    delete[] m_channelBuffers.normal;
    m_channelBuffers.normal = nullptr;
  }

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_DEFINITION(barney_device::Frame *);
