// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "rtcore/optix/Denoiser.h"
#include <optix.h>
// #include <optix_function_table.h>
#include <optix_stubs.h>
#include <iostream>

namespace rtc {
  namespace optix {
    
#if OPTIX_VERSION >= 80000
    
    Optix8Denoiser::Optix8Denoiser(Device *device)
      : Denoiser(device)
    {
      SetActiveGPU forDuration(device);
      denoiserOptions.guideAlbedo = 0;
      denoiserOptions.guideNormal = 1;
      // denoising alpha can get really funky results when using
      // compositing (eg in paraview/ice-t), so let's not do that.
      denoiserOptions.denoiseAlpha
        = OPTIX_DENOISER_ALPHA_MODE_COPY;

      currentUpscaleMode = upscaleMode;
      OptixDenoiserModelKind modelKind
        = upscaleMode
        ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X
        : OPTIX_DENOISER_MODEL_KIND_HDR;

      OptixDeviceContext optixContext
        = owlContextGetOptixContext(device->owl,0);
      OptixResult res = optixDenoiserCreate(optixContext,
                          modelKind,
                          &denoiserOptions,
                          &denoiser);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "[barney] OptiX denoiser creation failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        denoiser = {};
        available = false;
      }
    }

    void Optix8Denoiser::recreateIfNeeded()
    {
      if (!available) return;
      if (currentUpscaleMode == upscaleMode) return;

      SetActiveGPU forDuration(device);

      // destroy the existing OptixDenoiser handle
      if (denoiser) {
        optixDenoiserDestroy(denoiser);
        denoiser = {};
      }

      currentUpscaleMode = upscaleMode;
      OptixDenoiserModelKind modelKind
        = upscaleMode
        ? OPTIX_DENOISER_MODEL_KIND_UPSCALE2X
        : OPTIX_DENOISER_MODEL_KIND_HDR;

      OptixDeviceContext optixContext
        = owlContextGetOptixContext(device->owl,0);
      OptixResult res = optixDenoiserCreate(optixContext,
                          modelKind,
                          &denoiserOptions,
                          &denoiser);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "[barney] OptiX denoiser re-creation failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        denoiser = {};
        available = false;
      }
    }
    
    Optix8Denoiser::~Optix8Denoiser()
    {
      SetActiveGPU forDuration(device);
      if (denoiserScratch) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      if (denoiserState) {
        BARNEY_CUDA_CALL_NOTHROW(Free(denoiserState));
        denoiserState = 0;
      }
      if (in_rgba) {
        BARNEY_CUDA_CALL_NOTHROW(Free(in_rgba));
        in_rgba = 0;
      }
      if (out_rgba) {
        BARNEY_CUDA_CALL_NOTHROW(Free(out_rgba));
        out_rgba = 0;
      }
      if (in_normal) {
        BARNEY_CUDA_CALL_NOTHROW(Free(in_normal));
        in_normal = 0;
      }
    }
    
    void Optix8Denoiser::resize(vec2i numPixels)
    {
      // If upscale mode changed, destroy and recreate the denoiser
      recreateIfNeeded();

      this->numPixels = numPixels;
      // output is 2x input when upscaling, same as input otherwise
      outputDims = upscaleMode
        ? vec2i(numPixels.x*2, numPixels.y*2)
        : numPixels;

      if (!available) return;

      SetActiveGPU forDuration(device);

      denoiserSizes.overlapWindowSizeInPixels = 0;
      // "Image to be denoised" = input for UPSCALE2X (low-res); output for standard denoise.
      unsigned int memW = upscaleMode ? (unsigned int)numPixels.x : (unsigned int)outputDims.x;
      unsigned int memH = upscaleMode ? (unsigned int)numPixels.y : (unsigned int)outputDims.y;
      OptixResult res = optixDenoiserComputeMemoryResources(denoiser, memW, memH, &denoiserSizes);
      if (res != OPTIX_SUCCESS) {
        std::cerr << "[barney] optixDenoiserComputeMemoryResources failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
      // --------------------------------------------
      if (denoiserScratch) {
        BARNEY_CUDA_CALL(Free(denoiserScratch));
        denoiserScratch = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserScratch,
                              denoiserSizes.withoutOverlapScratchSizeInBytes));
      
      // --------------------------------------------
      if (denoiserState) {
        BARNEY_CUDA_CALL(Free(denoiserState));
        denoiserState = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&denoiserState,
                              denoiserSizes.stateSizeInBytes));
      // --- input buffers at render resolution ---
      if (in_rgba) {
        BARNEY_CUDA_CALL(Free(in_rgba));
        in_rgba = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&in_rgba,
                              numPixels.x*numPixels.y*sizeof(*in_rgba)));
      // --- output buffer at output (possibly 2x) resolution ---
      if (out_rgba) {
        BARNEY_CUDA_CALL(Free(out_rgba));
        out_rgba = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&out_rgba,
                              outputDims.x*outputDims.y*sizeof(*out_rgba)));
      // --- normal guide at render resolution ---
      if (in_normal) {
        BARNEY_CUDA_CALL(Free(in_normal));
        in_normal = 0;
      }
      BARNEY_CUDA_CALL(Malloc(&in_normal,
                              numPixels.x*numPixels.y*sizeof(*in_normal)));
      // --------------------------------------------
      
      // Setup takes INPUT dimensions (max input layer size). For UPSCALE2X
      // the denoiser then produces 2x output; if we passed output dims here
      // it would treat input as that size and produce 4x, and we'd only read
      // the top-left quarter.
      res = optixDenoiserSetup(denoiser,
                         0,
                         numPixels.x,
                         numPixels.y,
                         (CUdeviceptr)denoiserState,
                         denoiserSizes.stateSizeInBytes,
                         (CUdeviceptr)denoiserScratch,
                         denoiserSizes.withoutOverlapScratchSizeInBytes
                         );
      if (res != OPTIX_SUCCESS) {
        std::cerr << "[barney] optixDenoiserSetup failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
    }
    
    void Optix8Denoiser::run(float blendFactor)
    {
      if (!available) return;
      SetActiveGPU forDuration(device);
      OptixDenoiserLayer layer = {};
      
      // --- input at render resolution ---
      layer.input.format = OPTIX_PIXEL_FORMAT_FLOAT4;
      layer.input.rowStrideInBytes = numPixels.x*sizeof(vec4f);
      layer.input.pixelStrideInBytes = sizeof(vec4f);
      layer.input.width  = numPixels.x;
      layer.input.height = numPixels.y;
      layer.input.data   = (CUdeviceptr)in_rgba;
      
      // --- normal guide at render resolution ---
      OptixDenoiserGuideLayer guideLayer = {};
      guideLayer.normal.format = OPTIX_PIXEL_FORMAT_FLOAT3;
      guideLayer.normal.rowStrideInBytes = numPixels.x*sizeof(vec3f);
      guideLayer.normal.pixelStrideInBytes = sizeof(vec3f);
      guideLayer.normal.width  = numPixels.x;
      guideLayer.normal.height = numPixels.y;
      guideLayer.normal.data = (CUdeviceptr)in_normal;
      
      // --- output at outputDims (render res or 2x when upscaling) ---
      layer.output.format = OPTIX_PIXEL_FORMAT_FLOAT4;
      layer.output.rowStrideInBytes = outputDims.x*sizeof(vec4f);
      layer.output.pixelStrideInBytes = sizeof(vec4f);
      layer.output.width  = outputDims.x;
      layer.output.height = outputDims.y;
      layer.output.data = (CUdeviceptr)out_rgba;

      OptixDenoiserParams denoiserParams = {};

      /// blend factor.
      /// If set to 0 the output is 100% of the denoised input. If set to 1, the output is 100% of
      /// the unmodified input. Values between 0 and 1 will linearly interpolate between the denoised
      /// and unmodified input.
      denoiserParams.blendFactor      = blendFactor;
      cudaStream_t denoiserStream = 0;
      OptixResult res = optixDenoiserInvoke
        (
         denoiser,
         denoiserStream,
         &denoiserParams,
         (CUdeviceptr)denoiserState,
         denoiserSizes.stateSizeInBytes,
         &guideLayer,
         &layer,
         1,
         0,
         0,
         (CUdeviceptr)denoiserScratch,
         denoiserSizes.withoutOverlapScratchSizeInBytes
         );
      if (res != OPTIX_SUCCESS) {
        std::cerr << "[barney] optixDenoiserInvoke failed (code "
                  << (int)res << "); denoiser disabled." << std::endl;
        available = false;
        return;
      }
      cudaStreamSynchronize(denoiserStream);
    }
    
#endif
    
  }
}

