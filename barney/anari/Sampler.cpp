// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "Sampler.h"
// std
#include <cassert>
#include <iostream>
#include "Device.h"
#include "Array.h"
#include "Frame.h"
#include "TexelPolicy.h"
// std
#include <cstring>
#include <set>

namespace BARNEY_NS {
  namespace anari {

    // Helper functions /////////////////////////////////////////////////////////

    struct TextureDataRoute {
      BNTextureData        data{0};
      BNTextureColorSpace  colorSpace{BN_COLOR_SPACE_LINEAR};
    };

    /*! upload an ANARI array as barney texel data. the routing policy
      itself lives in TexelPolicy.h; this only executes the route the
      table picked */
    static TextureDataRoute makeBarneyTextureData(Object *objectToReportErrorWith,
                                                  BarneyGlobalState *state,
                                                  helium::Array *input,
                                                  int width,
                                                  int height,
                                                  int depth=-1)
    {
      int slot = state->slot;
      auto context = state->tether->context;
      const auto type = input->elementType();

      const Routed routed = routeTexel(type);
      if (routed.route == Route::UNSUPPORTED) {
        objectToReportErrorWith->reportMessage
          (ANARI_SEVERITY_ERROR,
           "unsupported texel type (%s)",
           anari::toString(type));
        return {};
      }
      if (routed.forcedByCapability && firstTimeSeen(type))
        objectToReportErrorWith->reportMessage
          (ANARI_SEVERITY_PERFORMANCE_WARNING,
           "texel type %s has no native barney format yet and was "
           "converted to float32 on upload; this costs memory but not "
           "accuracy",
           anari::toString(type));

      // host work buffers - only filled for the routes that convert
      std::vector<math::float4> f32vec4;
      std::vector<float>        f32scalar;
      std::vector<uint8_t>      padded;
      std::vector<uint8_t>      rgba8;

      const void *texels = input->data();
      switch (routed.route) {
      case Route::PRISTINE:
        break;
      case Route::PAD: {
        size_t bytesPerTexel = 0;
        padToFourComponents(input, padded, bytesPerTexel);
        texels = padded.data();
      } break;
      case Route::SPLAT_SRGB: {
        // the sRGB decode flag covers the leading colour lanes and
        // leaves alpha linear, so RA has to land as (r,0,0,a) and RGB
        // as (r,g,b,255) - see ANARITypeProperties<>::toFloat4
        const size_t N = input->totalSize();
        const uint8_t *in = (const uint8_t *)input->data();
        const int srcComponents = (int)::anari::componentsOf(type);
        rgba8.resize(N*4);
        for (size_t i = 0; i < N; ++i) {
          if (srcComponents == 2) {
            rgba8[i*4+0] = in[i*2+0];
            rgba8[i*4+1] = 0;
            rgba8[i*4+2] = 0;
            rgba8[i*4+3] = in[i*2+1];
          } else {
            rgba8[i*4+0] = in[i*3+0];
            rgba8[i*4+1] = in[i*3+1];
            rgba8[i*4+2] = in[i*3+2];
            rgba8[i*4+3] = 255;
          }
        }
        texels = rgba8.data();
      } break;
      case Route::DEMOTE_F32: {
        if (routed.target == BN_FLOAT32) {
          if (demoteToF32N(input, f32scalar) != 1) {
            objectToReportErrorWith->reportMessage
              (ANARI_SEVERITY_ERROR,
               "unsupported texel type (%s)",
               anari::toString(type));
            return {};
          }
          texels = f32scalar.data();
        } else {
          if (!demoteToF32Vec4(input, f32vec4)) {
            objectToReportErrorWith->reportMessage
              (ANARI_SEVERITY_ERROR,
               "unsupported texel type (%s)",
               anari::toString(type));
            return {};
          }
          texels = f32vec4.data();
        }
      } break;
      default:
        return {};
      }
      const BNDataType targetType = routed.target;
      const BNTextureColorSpace colorSpace = routed.colorSpace;

      TextureDataRoute res;
      res.colorSpace = colorSpace;
      if (depth == -1)
        res.data = bnTextureData2DCreate(context, slot, targetType,
                                         width, height, texels);
      else
        res.data = bnTextureData3DCreate(context, slot, targetType,
                                         width, height, depth, texels);
      return res;
    }

    // Sampler definitions ////////////////////////////////////////////////////////

    Sampler::Sampler(BarneyGlobalState *s, const char *barneySubtype)
      : Object(ANARI_SAMPLER, s)
    {
      int slot = deviceState()->slot;
      auto context = deviceState()->tether->context;
      m_bnSampler = bnSamplerCreate(context, slot, barneySubtype);
    }

    Sampler::~Sampler()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: ~Sampler deconstructing" << std::endl);
      bnRelease(m_bnSampler);
    }

    Sampler *Sampler::createInstance(std::string_view subtype, BarneyGlobalState *s)
    {
      if (subtype == "image1D")
        return new Image1D(s);
      else if (subtype == "image2D")
        return new Image2D(s);
      else if (subtype == "image3D")
        return new Image3D(s);
      else if (subtype == "transform")
        return new TransformSampler(s);
      else if (subtype == "primitive")
        return new PrimitiveSampler(s);
      else
        return (Sampler *)new UnknownObject(ANARI_SAMPLER, subtype, s);
    }

    BNSampler Sampler::getBarneySampler()
    {
      if (!isValid())
        return {};
      return m_bnSampler;
    }

    void Sampler::commitParameters()
    {
      Object::commitParameters();

      m_inAttribute = getParamString("inAttribute", "attribute0");
      m_outOffset
        = getParam<math::float4>("outOffset",   math::float4(0.f, 0.f, 0.f, 0.f));
      m_outTransform = math::identity;
      getParam("outTransform", ANARI_FLOAT32_MAT4, &m_outTransform);
    }
  
    void Sampler::setBarneyParameters()
    {
      bnSet4x4fv(m_bnSampler, "outTransform", (const bn_float4 *)&m_outTransform);
      bnSet4f(m_bnSampler,
              "outOffset",
              m_outOffset.x,
              m_outOffset.y,
              m_outOffset.z,
              m_outOffset.w);
      bnSetString(m_bnSampler, "inAttribute", m_inAttribute.c_str());
    }

    // Subtypes ///////////////////////////////////////////////////////////////////

    TextureDataSampler::TextureDataSampler(BarneyGlobalState *s,
                                           const char *barneySubtype)
      : Sampler(s, barneySubtype)
    {}
  
    TextureDataSampler::~TextureDataSampler()
    {
      if (m_bnTextureData) {
        bnRelease(m_bnTextureData);
        m_bnTextureData = nullptr;
      }
    }
  
    void TextureDataSampler::setBarneyParameters()
    {
      Sampler::setBarneyParameters();
      BNTextureFilterMode filterMode =
        m_linearFilter ? BN_TEXTURE_LINEAR : BN_TEXTURE_NEAREST;
      bnSet1i(m_bnSampler, "filterMode", (int)filterMode);
      bnSet1i(m_bnSampler, "colorSpace", (int)m_colorSpace);
      bnSet4f(m_bnSampler, "borderColor",
              m_borderColor.x,
              m_borderColor.y,
              m_borderColor.z,
              m_borderColor.w);
      bnSet4x4fv(m_bnSampler, "inTransform", (const bn_float4 *)&m_inTransform);
      bnSet4f(m_bnSampler,
              "inOffset",
              m_inOffset.x,
              m_inOffset.y,
              m_inOffset.z,
              m_inOffset.w);

      bnSetObject(m_bnSampler, "textureData", m_bnTextureData);
    }

    void TextureDataSampler::commitParameters()
    {
      Sampler::commitParameters();
      m_linearFilter = getParamString("filter", "linear") != "nearest";
      m_borderColor
        = getParam<math::float4>("borderColor", math::float4(0.f, 0.f, 0.f, 0.f));
      m_inOffset
        = getParam<math::float4>("inOffset",    math::float4(0.f, 0.f, 0.f, 0.f));
      m_inTransform = math::identity;
      getParam("inTransform", ANARI_FLOAT32_MAT4, &m_inTransform);
    }

    // Image1D //

    Image1D::Image1D(BarneyGlobalState *s)
      : TextureDataSampler(s, "texture2D")
    {}

    Image1D::~Image1D() = default;

    void Image1D::commitParameters()
    {
      TextureDataSampler::commitParameters();
      m_image = getParamObject<helium::Array1D>("image");
      m_wrapMode = toBarneyAddressMode(getParamString("wrapMode", "clampToEdge"));
    }

    bool Image1D::isValid() const
    {
      return m_image;
    }

    void Image1D::finalize()
    {
      if (m_bnTextureData) {
        bnRelease(m_bnTextureData);
        m_bnTextureData = 0;
      }
      if (!m_image) {
        reportMessage(ANARI_SEVERITY_DEBUG,
                      "Image1D::finalize() without a valid 'image' parameter");
        TextureDataSampler::setBarneyParameters();
        bnCommit(m_bnSampler);
        return;
      }
      auto route = makeBarneyTextureData(this,
                                         deviceState(), m_image.ptr,
                                         (int)m_image->size(), 1);
      m_bnTextureData = route.data;
      m_colorSpace   = route.colorSpace;
      // ------------------------------------------------------------------
      // now, create sampler over those texels
      // ------------------------------------------------------------------

      TextureDataSampler::setBarneyParameters();
      bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode);
      bnSet1i(m_bnSampler, "wrapMode1", (int)BN_TEXTURE_CLAMP);
      bnCommit(m_bnSampler);
    }

    // Image2D //

    Image2D::Image2D(BarneyGlobalState *s)
      : TextureDataSampler(s, "texture2D")
    {}

    Image2D::~Image2D() = default;

    void Image2D::commitParameters()
    {
      TextureDataSampler::commitParameters();
      m_image = getParamObject<helium::Array2D>("image");
      m_wrapMode1 = toBarneyAddressMode(getParamString("wrapMode1", "clampToEdge"));
      m_wrapMode2 = toBarneyAddressMode(getParamString("wrapMode2", "clampToEdge"));
    }

    void Image2D::finalize()
    {
      if (m_bnTextureData) {
        bnRelease(m_bnTextureData);
        m_bnTextureData = 0;
      }
      if (!m_image) {
        reportMessage(ANARI_SEVERITY_DEBUG,
                      "Image2D::finalize() without a valid 'image' parameter");
        TextureDataSampler::setBarneyParameters();
        bnCommit(m_bnSampler);
        return;
      }

      auto route = makeBarneyTextureData(this,
                                         deviceState(), m_image.ptr,
                                         m_image->size().x, m_image->size().y);
      m_bnTextureData = route.data;
      m_colorSpace   = route.colorSpace;

      TextureDataSampler::setBarneyParameters();
    
      bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode1);
      bnSet1i(m_bnSampler, "wrapMode1", (int)m_wrapMode2);
    
      bnCommit(m_bnSampler);
    }

    bool Image2D::isValid() const
    {
      return m_image;
    }

    // Image3D //

    Image3D::Image3D(BarneyGlobalState *s)
      : TextureDataSampler(s, "texture3D")
    {}

    Image3D::~Image3D() = default;

    void Image3D::commitParameters()
    {
      TextureDataSampler::commitParameters();
      m_image = getParamObject<helium::Array3D>("image");
      m_wrapMode1 = toBarneyAddressMode(getParamString("wrapMode1", "clampToEdge"));
      m_wrapMode2 = toBarneyAddressMode(getParamString("wrapMode2", "clampToEdge"));
      m_wrapMode3 = toBarneyAddressMode(getParamString("wrapMode3", "clampToEdge"));
    }

    void Image3D::finalize()
    {
      if (m_bnTextureData) {
        bnRelease(m_bnTextureData);
        m_bnTextureData = 0;
      }
      if (!m_image) {
        reportMessage(ANARI_SEVERITY_DEBUG,
                      "Image3D::finalize() without a valid 'image' parameter");
        TextureDataSampler::setBarneyParameters();
        bnCommit(m_bnSampler);
        return;
      }
      auto route = makeBarneyTextureData(this,
                                         deviceState(), m_image.ptr,
                                         m_image->size().x,
                                         m_image->size().y,
                                         m_image->size().z);
      m_bnTextureData = route.data;
      m_colorSpace   = route.colorSpace;

      TextureDataSampler::setBarneyParameters();
    
      bnSet1i(m_bnSampler, "wrapMode0", (int)m_wrapMode1);
      bnSet1i(m_bnSampler, "wrapMode1", (int)m_wrapMode2);
      bnSet1i(m_bnSampler, "wrapMode2", (int)m_wrapMode3);
    
      bnCommit(m_bnSampler);
    }

    bool Image3D::isValid() const
    {
      return m_image;
    }

    /// Transform ///

    TransformSampler::TransformSampler(BarneyGlobalState *s)
      : Sampler(s, "transform")
    {}

    TransformSampler::~TransformSampler() = default;

    void TransformSampler::commitParameters()
    {
      Sampler::commitParameters();
    }

    void TransformSampler::finalize()
    {
      Sampler::setBarneyParameters();
    
      bnCommit(m_bnSampler);
    }

    // Image1D //

    PrimitiveSampler::PrimitiveSampler(BarneyGlobalState *s)
      : Sampler(s, "primitive")
    {}

    PrimitiveSampler::~PrimitiveSampler()
    {
      if (m_bnArrayData)
        bnRelease(m_bnArrayData);
    }

    void PrimitiveSampler::commitParameters()
    {
      Sampler::commitParameters();
      m_array = getParamObject<helium::Array1D>("array");
      // 'offset' predates the KHR_SAMPLER_PRIMITIVE name
      m_offset = getParam<int>("inOffset", getParam<int>("offset", 0));
    }

    bool PrimitiveSampler::isValid() const
    {
      return m_array;
    }

    void PrimitiveSampler::finalize()
    {
      if (!m_array) {
        reportMessage(ANARI_SEVERITY_DEBUG,
                      "PrimitiveSampler::finalize() without a valid 'array' parameter");
        return;
      }

      auto state = deviceState();
      int slot = state->slot;
      auto context = state->tether->context;

      if (m_bnArrayData) {
        bnRelease(m_bnArrayData);
        m_bnArrayData = 0;
      }

      auto type = m_array->elementType();
      const Routed routed = routeArray(type);
      BNDataType barneyType = routed.target;

      if (routed.route == Route::PRISTINE) {
        // pristine: original type, expanded on read by typedRead
        m_bnArrayData
          = bnDataCreate(context, slot, barneyType,
                         m_array->totalSize(), m_array->data());
      } else {
        if (routed.forcedByCapability && firstTimeSeen(type))
          reportMessage(ANARI_SEVERITY_PERFORMANCE_WARNING,
                        "array element type %s has no native barney format "
                        "yet and was converted to float32 on upload",
                        anari::toString(type));
        // upload-time demotion, same as the attribute path
        std::vector<float> demoted;
        int components = demoteToF32N(m_array, demoted);
        if (components == 0) {
          reportMessage
            (ANARI_SEVERITY_ERROR,
             "unsupported array element type (%s) on primitive sampler",
             anari::toString(type));
          bnSetData(m_bnSampler, "arrayData", (BNData)0);
          bnCommit(m_bnSampler);
          return;
        }
        m_bnArrayData
          = bnDataCreate(context, slot, barneyType,
                         demoted.size()/components, demoted.data());
      }

      bnSetData(m_bnSampler, "arrayData", m_bnArrayData);
      bnSet1i(m_bnSampler, "arrayOffset", (int)m_offset);
      bnSet1i(m_bnSampler, "arrayType", (int)barneyType);

      bnCommit(m_bnSampler);
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Sampler *);
