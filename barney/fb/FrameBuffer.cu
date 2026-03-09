// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/common/barney-common.h"
#include "barney/common/math.h"
#include "barney/common/Data.h"
#include "barney/fb/FrameBuffer.h"
#if BARNEY_HAVE_OIDN
# include <OpenImageDenoise/oidn.h>
#endif
#include "rtcore/ComputeInterface.h"

namespace BARNEY_NS {
  RTC_IMPORT_COMPUTE2D(linearToFixed8);

  FrameBuffer::FrameBuffer(Context *context,
                           const DevGroup::SP &devices,
                           const bool isOwner)
    : barney_api::FrameBuffer(context),
      isOwner(isOwner),
      devices(devices)
  {
    if (FromEnv::get()->explicitlyDisabled("denoise")) {
      if (context->myRank() == 0)
        std::cout << "#bn: denoising explicitly disabled in env-config." << std::endl;
      enableDenoising = false;
    }
    perLogical.resize(devices->numLogical);
    for (auto device : *devices) {
      getPLD(device)->tiledFB
        = TiledFB::create(device,context->deviceWeNeedToCopyToForFBMap,this);
    }

    Device *device = getDenoiserDevice();
    assert(device);
    denoiser = device->rtc->createDenoiser();
    linear_toFixed8 = createCompute_linearToFixed8(device->rtc);
  }

  FrameBuffer::~FrameBuffer()
  {
    freeResources();
    denoiser = 0;
  }

  bool FrameBuffer::needHitIDs() const
  {
    return channels & (BN_FB_PRIMID|BN_FB_INSTID|BN_FB_OBJID);
  }

  bool FrameBuffer::set1i(const std::string &member, const int &value)
  {
    if (member == "showCrosshairs") {
      showCrosshairs = value;
      return true;
    }
    if (member == "denoise") {
      enableDenoising = value;
      return true;
    }
    if (member == "upscale") {
      enableUpscaling = value;
      return true;
    }
    return false;
  }

  void FrameBuffer::freeResources()
  {
    Device *device = getDenoiserDevice();
    if (linearColorChannel) {
      device->rtc->freeMem(linearColorChannel);
      linearColorChannel = 0;
    }
    if (linearAuxChannel) {
      device->rtc->freeMem(linearAuxChannel);
      linearAuxChannel = 0;
    }
    if (linearNormalChannel) {
      device->rtc->freeMem(linearNormalChannel);
      linearNormalChannel = 0;
    }
    if (renderAuxChannel) {
      device->rtc->freeMem(renderAuxChannel);
      renderAuxChannel = 0;
    }
    if (renderNormalChannel) {
      device->rtc->freeMem(renderNormalChannel);
      renderNormalChannel = 0;
    }
    if (upscaledColorChannel) {
      device->rtc->freeMem(upscaledColorChannel);
      upscaledColorChannel = 0;
    }
  }

  struct LinearToFixed8 {
    uint32_t *out;
    vec4f    *in;
    vec2i numPixels;
    bool SRGB;
#if RTC_DEVICE_CODE
    __rtc_device void run(const rtc::ComputeInterface &ci);
#endif
  };

#if RTC_DEVICE_CODE
  __rtc_device void LinearToFixed8::run(const rtc::ComputeInterface &ci)
  {
    int ix = ci.getThreadIdx().x+ci.getBlockIdx().x*ci.getBlockDim().x;
    if (ix >= numPixels.x) return;
    int iy = ci.getThreadIdx().y+ci.getBlockIdx().y*ci.getBlockDim().y;
    if (iy >= numPixels.y) return;
    int idx = ix+numPixels.x*iy;
    vec4f v = in[idx];
    v = saturate(v);
    if (SRGB)
      (vec3f&)v = linear_to_srgb((vec3f&)v);
    out[idx] = ::BARNEY_NS::make_rgba(v);
  }
#endif

  // ------------------------------------------------------------------
  // Nearest-neighbor 2x upscale kernels (for AI upscaling mode)
  // Uses 1D flattened indexing to match __rtc_launch convention.
  // ------------------------------------------------------------------

  /*! 2x nearest-neighbor upscale for uint32 data (depth, primID, etc.) */
  __rtc_global
  void upscale2xUint32Kernel(const rtc::ComputeInterface &ci,
                             uint32_t *out, vec2i outSize,
                             const uint32_t *in, vec2i inSize)
  {
    int tid = ci.getThreadIdx().x + ci.getBlockIdx().x * ci.getBlockDim().x;
    int ox = tid % outSize.x;
    int oy = tid / outSize.x;
    if (oy >= outSize.y) return;

    int ix = min(ox / 2, inSize.x - 1);
    int iy = min(oy / 2, inSize.y - 1);
    out[tid] = in[ix + inSize.x * iy];
  }

  /*! 2x nearest-neighbor upscale for vec3f data (normals) */
  __rtc_global
  void upscale2xVec3fKernel(const rtc::ComputeInterface &ci,
                            vec3f *out, vec2i outSize,
                            const vec3f *in, vec2i inSize)
  {
    int tid = ci.getThreadIdx().x + ci.getBlockIdx().x * ci.getBlockDim().x;
    int ox = tid % outSize.x;
    int oy = tid / outSize.x;
    if (oy >= outSize.y) return;

    int ix = min(ox / 2, inSize.x - 1);
    int iy = min(oy / 2, inSize.y - 1);
    out[tid] = in[ix + inSize.x * iy];
  }

  /*! 2x nearest-neighbor upscale for vec4f data (color); used when AI
      upscaling is enabled (we use HDR denoiser at half res then upscale ourselves). */
  __rtc_global
  void upscale2xVec4fKernel(const rtc::ComputeInterface &ci,
                            vec4f *out, vec2i outSize,
                            const vec4f *in, vec2i inSize)
  {
    int tid = ci.getThreadIdx().x + ci.getBlockIdx().x * ci.getBlockDim().x;
    int ox = tid % outSize.x;
    int oy = tid / outSize.x;
    if (oy >= outSize.y) return;

    int ix = min(ox / 2, inSize.x - 1);
    int iy = min(oy / 2, inSize.y - 1);
    out[tid] = in[ix + inSize.x * iy];
  }

  void FrameBuffer::finalizeTiles()
  {}

  void FrameBuffer::finalizeFrame()
  {
    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);

    bool doDenoising = (denoiser != 0) && (enableDenoising || enableUpscaling);

    bool needNormalChannel = (channels & BN_FB_NORMAL) && linearNormalChannel;
    void *colorCopyTarget
      = doDenoising
      ? denoiser->in_rgba
      : linearColorChannel;
    vec3f *normalCopyTarget = nullptr;
    if (doDenoising)
      normalCopyTarget = denoiser->in_normal;
    else if (needNormalChannel)
      normalCopyTarget = (vec3f*)linearNormalChannel;
    BNDataType gatherType
      = doDenoising
      ? BN_FLOAT4
      : colorChannelFormat;

    gatherColorChannel(colorCopyTarget,gatherType,normalCopyTarget);

    if (doDenoising && needNormalChannel) {
      if (enableUpscaling && renderPixels != numPixels) {
        int totalOut = numPixels.x * numPixels.y;
        __rtc_launch(device->rtc,
                     upscale2xVec3fKernel,
                     divRoundUp(totalOut, 256), 256,
                     (vec3f*)linearNormalChannel, numPixels,
                     denoiser->in_normal, renderPixels);
      } else {
        device->rtc->copy(linearNormalChannel, denoiser->in_normal,
                          renderPixels.x*renderPixels.y*sizeof(vec3f));
      }
    }

    if (channels & BN_FB_DEPTH)
      gatherAuxChannel(BN_FB_DEPTH);
    if (channels & BN_FB_PRIMID)
      gatherAuxChannel(BN_FB_PRIMID);
    if (channels & BN_FB_OBJID)
      gatherAuxChannel(BN_FB_OBJID);
    if (channels & BN_FB_INSTID)
      gatherAuxChannel(BN_FB_INSTID);
  }

  /*! gather color (and normal, if required for denoising),
    (re-)format into a linear buffer, perform denoising (if
    required), convert to requested format, and copy to the
    application pointed provided */
  void FrameBuffer::readColorChannel(void *appMemory,
                                     BNDataType requestedFormat)
  {
    assert(requestedFormat == colorChannelFormat);
    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);

    bool doDenoising = denoiser != 0 && (enableDenoising || enableUpscaling);
    if (doDenoising) {
      float blendFactor = (accumID-1) / (accumID+100.f);
      device->rtc->sync();
      denoiser->run(blendFactor);

      // We always use HDR denoiser (no OptiX UPSCALE2X). When enableUpscaling,
      // denoiser output is at renderPixels; we 2x upscale to linearColorChannel
      // at numPixels. When not upscaling, denoiser output is at numPixels.
      vec2i outDims = denoiser->outputDims;
      vec4f *colorSrc = denoiser->out_rgba;
      if (enableUpscaling && renderPixels != numPixels && upscaledColorChannel) {
        int totalOut = numPixels.x * numPixels.y;
        __rtc_launch(device->rtc,
                     upscale2xVec4fKernel,
                     divRoundUp(totalOut, 256), 256,
                     (vec4f*)upscaledColorChannel, numPixels,
                     denoiser->out_rgba, renderPixels);
        device->rtc->sync();
        colorSrc = (vec4f*)upscaledColorChannel;
        outDims = numPixels;
      }

      switch(requestedFormat) {
      case BN_FLOAT4: {
        device->rtc->copy(appMemory, colorSrc,
                          outDims.x*outDims.y*sizeof(vec4f));
      } break;
      case BN_UFIXED8_RGBA:
      case BN_UFIXED8_RGBA_SRGB: {
        bool srgb = (requestedFormat == BN_UFIXED8_RGBA_SRGB);
        vec2ui bs(8,8);
        LinearToFixed8 args = {
          (uint32_t*)linearColorChannel, colorSrc, outDims, srgb
        };
        linear_toFixed8->launch(divRoundUp(vec2ui(outDims),bs),bs,&args);
        device->rtc->copy(appMemory,linearColorChannel,
                          outDims.x*outDims.y*sizeof(uint32_t));
      } break;
      default:
        throw std::runtime_error
          ("requested to read color channel in un-supported format #"
           +std::to_string((int)requestedFormat));
      };
    } else {
      size_t sizeOfPixel
        = (requestedFormat == BN_FLOAT4)
        ? sizeof(vec4f)
        : sizeof(uint32_t);
      device->rtc->copy(appMemory,linearColorChannel,
                        numPixels.x*numPixels.y*sizeOfPixel);
    }
    device->rtc->sync();
  }

  
  /*! "finalize" and read the frame buffer. If this function gets
      called with a null hostPtr we will still finalize the frame
      buffer and run the denoiser, just not copy it to host; the
      result can then be read by framebuffergetpointer() */
  void FrameBuffer::read(BNFrameBufferChannel channel,
                         void *appMemory,
                         BNDataType requestedFormat)
  {
    if (!appMemory) return;

    if (!isOwner) {
      // iw 'in theory' apps shoudln't even call map on any rank other
      // thank rank 0, but if they do, let's report them valid black,
      // zero-opacity, depth-infinity frame in case the app wants to
      // do compositing with that.
      int numPixels = this->numPixels.x * this->numPixels.y;
      if (channel == BN_FB_DEPTH) {
        for (int i=0;i<numPixels;i++)
          ((float*)appMemory)[i] = BARNEY_INF;
        return;
      }
      if (channel == BN_FB_DEPTH ||
          channel == BN_FB_PRIMID ||
          channel == BN_FB_INSTID ||
          channel == BN_FB_OBJID) {
        for (int i=0;i<numPixels;i++)
          ((uint32_t*)appMemory)[i] = 0;
        return;
      }
      if (channel == BN_FB_COLOR && requestedFormat == BN_FLOAT4) {
        for (int i=0;i<numPixels;i++)
          ((vec4f*)appMemory)[i] = vec4f(0.f,0.f,0.f,0.f);
        return;
      }
      if (channel == BN_FB_COLOR && requestedFormat == BN_UFIXED8_RGBA) {
        for (int i=0;i<numPixels;i++)
          ((uint32_t*)appMemory)[i] = 0x00000000;
        return;
      }
      return;
    }
    
    Device *device = getDenoiserDevice();
    SetActiveGPU forDuration(device);

    if (channel == BN_FB_COLOR) {
      readColorChannel(appMemory,requestedFormat);
      return;
    }

    if (channel == BN_FB_DEPTH ||
        channel == BN_FB_PRIMID ||
        channel == BN_FB_INSTID ||
        channel == BN_FB_OBJID) {
      if (enableUpscaling && renderPixels != numPixels && renderAuxChannel) {
        // linearize at render resolution, then upscale to display resolution
        writeAuxChannel(renderAuxChannel,channel);
        int totalOut = numPixels.x * numPixels.y;
        __rtc_launch(device->rtc,
                     upscale2xUint32Kernel,
                     divRoundUp(totalOut, 256), 256,
                     (uint32_t*)linearAuxChannel, numPixels,
                     (const uint32_t*)renderAuxChannel, renderPixels);
        device->rtc->sync();
      } else {
        writeAuxChannel(linearAuxChannel,channel);
      }
      // NOTE: depth + id buffers happen to be the same bytes-per-pixel
      device->rtc->copy(appMemory,linearAuxChannel,
                        numPixels.x*numPixels.y*sizeof(uint32_t));
      return;
    }

    if (channel == BN_FB_NORMAL && linearNormalChannel) {
      // normals were already linearized (and upscaled if needed) during finalizeFrame()
      device->rtc->copy(appMemory,linearNormalChannel,
                        numPixels.x*numPixels.y*sizeof(vec3f));
      return;
    }

    throw std::runtime_error("un-handled frame buffer channel/format combination "
                             +to_string(channel)
                             +" "+to_string(requestedFormat));
  }

  void FrameBuffer::resize(BNDataType colorFormat,
                           vec2i size,
                           uint32_t channels)
  {
    this->channels = channels;
    this->colorChannelFormat = colorFormat;

    freeResources();

    // display resolution - keep exactly as the app requested so the
    // ANARI frame reports the same size back and the pipeline's
    // block-copy fast path is used (no stride mismatch).
    numPixels = size;

    // when upscaling, render at half resolution (ceiling division so
    // that renderPixels covers at least numPixels/2 in each dim; the
    // upscale kernel clamps edge pixels with min()).
    if (enableUpscaling && denoiser) {
      renderPixels = vec2i((numPixels.x + 1) / 2, (numPixels.y + 1) / 2);
    } else {
      renderPixels = numPixels;
    }

    // tiles render at renderPixels
    for (auto device : *devices)
      getFor(device)->resize(channels, renderPixels);

    size_t sizeOfPixel
      = (colorFormat == BN_FLOAT4)
      ? sizeof(vec4f)
      : sizeof(uint32_t);

    if (isOwner) {
      auto rtc = getDenoiserDevice()->rtc;
      int dpNP = numPixels.x * numPixels.y;   // display pixels
      int rpNP = renderPixels.x * renderPixels.y; // render pixels

      // output buffers at display resolution
      linearColorChannel = rtc->allocMem(dpNP * sizeOfPixel);
      linearAuxChannel   = rtc->allocMem(dpNP * sizeof(uint32_t));
      if (channels & BN_FB_NORMAL)
        linearNormalChannel = rtc->allocMem(dpNP * sizeof(vec3f));

      // when upscaling, we need render-resolution staging buffers
      // for aux/normal (tile linearization writes at render res,
      // then we upscale to display res)
      if (enableUpscaling && renderPixels != numPixels) {
        renderAuxChannel    = rtc->allocMem(rpNP * sizeof(uint32_t));
        if (channels & BN_FB_NORMAL)
          renderNormalChannel = rtc->allocMem(rpNP * sizeof(vec3f));
        upscaledColorChannel = rtc->allocMem(dpNP * sizeof(vec4f));
      }

      if (denoiser) {
        // Never use OptiX UPSCALE2X (unreliable); we run HDR denoiser at half res
        // and do 2x upscale ourselves in readColorChannel.
        denoiser->upscaleMode = false;
        denoiser->resize(renderPixels);
      }
    }
  }

  FrameBuffer::PLD *FrameBuffer::getPLD(Device *device)
  {
    assert(device);
    assert(device->contextRank() >= 0);
    assert(device->contextRank() < perLogical.size());
    return &perLogical[device->contextRank()];
  }

  TiledFB *FrameBuffer::getFor(Device *device)
  {
    auto pld = getPLD(device);
    assert(pld);
    return pld->tiledFB.get();
  }

  Device *FrameBuffer::getDenoiserDevice() const
  {
    return (*devices)[0];
  }

  RTC_EXPORT_COMPUTE2D(linearToFixed8,LinearToFixed8);
}

