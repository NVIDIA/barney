// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Camera.h"
#include "Renderer.h"
#include "World.h"
// helium
#include "helium/BaseFrame.h"
// std
#include <vector>

namespace BARNEY_NS {
  namespace anari {
    
    struct Frame : public helium::BaseFrame
    {
      Frame(BarneyGlobalState *s);
      ~Frame() override;

      bool isValid() const override;

      BarneyGlobalState *deviceState() const;

      bool getProperty(const std::string_view &name,
                       ANARIDataType type,
                       void *ptr,
                       uint64_t size,
                       uint32_t flags) override;

      void commitParameters() override;
      void finalize() override;

      void renderFrame() override;

      void *map(std::string_view channel,
                uint32_t *width,
                uint32_t *height,
                ANARIDataType *pixelType) override;
      void unmap(std::string_view channel) override;
      int frameReady(ANARIWaitMask m) override;
      void discard() override;

      bool ready() const;
      void wait() const;

    private:
      void cleanup();

      bool        m_valid           { false };
      bool        m_isAltFrame      { false };
      math::uint2 m_size            { 0,0 };
      /*! Actual framebuffer dimensions (from bnFrameBufferGetSize after
        resize). Matches buffer layout so map() reports correct stride. */
      math::uint2 m_displaySize     { 0,0 };

      struct {
        // color cold be uint or float4; if float4 we just allocate
        // 4uints and store in those
        uint32_t   *color{nullptr};
        float      *depth{nullptr};
        int        *primID{nullptr};
        int        *instID{nullptr};
        int        *objID{nullptr};
        float      *normal{nullptr};
        float      *motion{nullptr};
      } m_channelBuffers;
      struct {
        /* for performance warnings; initialize all to 'true' so they
           won't throw a perf warning on first time renderframe */
        bool color = true;
        bool depth = true;
        bool primID = true;
        bool instID = true;
        bool objID = true;
        bool normal = true;
        bool motion = true;
      } m_didMapChannel;
      bool m_lastFrameWasFirstFrame = true;

      struct ChannelTypes {
        anari::DataType color{ANARI_UNKNOWN};
        anari::DataType depth{ANARI_UNKNOWN};
        anari::DataType primID{ANARI_UNKNOWN};
        anari::DataType instID{ANARI_UNKNOWN};
        anari::DataType objID{ANARI_UNKNOWN};
        anari::DataType normal{ANARI_UNKNOWN};
        anari::DataType motion{ANARI_UNKNOWN};
      };
      ChannelTypes m_channelTypes;
      /*! channels the last rendered frame actually used; the perf
        warnings must blame that frame, not the freshly flushed params */
      ChannelTypes m_lastFrameChannelTypes;

      helium::ChangeObserverPtr<Renderer> m_renderer;
      helium::IntrusivePtr<Camera>        m_camera;
      helium::IntrusivePtr<World>         m_world;

      helium::TimeStamp m_lastCommitFlush{0};

      mutable float m_duration{0.f};

      BNFrameBuffer m_bnFrameBuffer{nullptr};
      // for device tethering, we need this to know whether all devices
      // have 'checked in'
      int m_numTimesRenderFrameHasBeenCalled = 0;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Frame *, ANARI_FRAME);
