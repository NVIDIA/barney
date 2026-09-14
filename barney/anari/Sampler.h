// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "Object.h"
#include "helium/array/Array1D.h"
#include "helium/array/Array2D.h"
#include "helium/array/Array3D.h"

namespace BARNEY_NS {
  namespace anari {

    struct Sampler : public Object
    {
      Sampler(BarneyGlobalState *s, const char *barneySubtype);
      ~Sampler() override;

      static Sampler *createInstance(std::string_view subtype,
                                     BarneyGlobalState *s);

      BNSampler getBarneySampler();

    protected:
      void commitParameters() override;
      void setBarneyParameters();
    
      BNSampler    m_bnSampler{nullptr};
    
      std::string  m_inAttribute;
    
      math::mat4   m_inTransform{math::identity};
      math::float4 m_inOffset{0.f, 0.f, 0.f, 0.f};
      math::mat4   m_outTransform{math::identity};
      math::float4 m_outOffset{0.f, 0.f, 0.f, 0.f};
    };

    /*! base class for samplers that utilize a barney texturedata object
      - i.e, image1/2/3D; does not have a direct ANARI equivalent */
    struct TextureDataSampler : public Sampler {
    
      TextureDataSampler(BarneyGlobalState *s, const char *barneySubtype);
      ~TextureDataSampler() override;
    
      void commitParameters() override;
      void setBarneyParameters();
    
      bool m_linearFilter{true};
      BNTextureData m_bnTextureData{nullptr};
      BNTextureColorSpace m_colorSpace{BN_COLOR_SPACE_LINEAR};
    
      math::float4  m_borderColor{0.f, 0.f, 0.f, 0.f};
    };
  
    // Subtypes ///////////////////////////////////////////////////////////////////

    struct Image1D : public TextureDataSampler
    {
      Image1D(BarneyGlobalState *s);
      ~Image1D() override;

      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

    private:
      helium::IntrusivePtr<helium::Array1D> m_image;
      BNTextureAddressMode m_wrapMode{BN_TEXTURE_CLAMP};
    };

    struct Image2D : public TextureDataSampler
    {
      Image2D(BarneyGlobalState *s);
      ~Image2D() override;
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

    private:
      helium::IntrusivePtr<helium::Array2D> m_image;
      BNTextureAddressMode m_wrapMode1{BN_TEXTURE_CLAMP};
      BNTextureAddressMode m_wrapMode2{BN_TEXTURE_CLAMP};
    };

    struct Image3D : public TextureDataSampler
    {
      Image3D(BarneyGlobalState *s);
      ~Image3D() override;
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;

    private:
      helium::IntrusivePtr<helium::Array3D> m_image;
      BNTextureAddressMode m_wrapMode1{BN_TEXTURE_CLAMP};
      BNTextureAddressMode m_wrapMode2{BN_TEXTURE_CLAMP};
      BNTextureAddressMode m_wrapMode3{BN_TEXTURE_CLAMP};
    };

    struct TransformSampler : public Sampler
    {
      TransformSampler(BarneyGlobalState *s);
      ~TransformSampler() override;
      void commitParameters() override;
      void finalize() override;

    private:
    };

    struct PrimitiveSampler : public Sampler
    {
      PrimitiveSampler(BarneyGlobalState *s);
      ~PrimitiveSampler() override;
      void commitParameters() override;
      void finalize() override;
      bool isValid() const override;
    
    private:
      BNData m_bnArrayData = 0;
      helium::IntrusivePtr<helium::Array1D> m_array;
      int m_offset = 0;
    };

  }
}

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(BARNEY_NS::anari::Sampler *, ANARI_SAMPLER);
