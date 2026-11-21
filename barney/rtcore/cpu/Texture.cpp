// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "rtcore/cpu/Texture.h"

namespace BARNEY_NS {
  namespace rtc {

    struct LerpAddresses {
      int idx0, idx1;
      float f;
    };
    
    inline float clamp(float f, float lo, float hi)
    {
      return max(min(f,hi),lo);
    }
    
    inline vec4f lerp_l(float f, vec4f a, vec4f b)
    {
      return (1.f-f)*a + f*b;
    }
    
    void computeAddress(LerpAddresses &out,
                        rtc::AddressMode mode,
                        float tc, int N)
    {
      switch (mode) {
      case rtc::WRAP: {
#if 1
        out = {};
        tc = tc * (float)N - .5f;
        tc = fmodf(tc,(float)N);
        if (tc < 0.f) tc += (float)N;
        float fc = floorf(tc);
        int ic = int(fc);
        out.f = tc - ic;
        out.idx0 = ic;
        out.idx1 = int(fmodf(tc+1.f,(float)N));
#else
        out = {};
        tc = tc * N + .5f;
        float fc = floorf(tc);
        int ic = int(fc);
        out.f = tc - ic;

        int i0 = ic-1;
        int i1 = ic+0;
        if (i0 < 0) i0 = -i0;
        if (i1 < 0) i1 = -i1;
        i0 = i0 % (N);
        i1 = i1 % (N);
        out.idx0 = i0;
        out.idx1 = i1;
#endif
      } return;
      case rtc::MIRROR: {
        out = {};
        tc = tc * N + .5f;
        float fc = floorf(tc);
        int ic = int(fc);
        out.f = tc - ic;

        int i0 = ic-1;
        int i1 = ic+0;
        if (i0 < 0) i0 = -i0;
        if (i1 < 0) i1 = -i1;
        i0 = i0 % (2*N);
        i1 = i1 % (2*N);
        if (i0 >= N) i0 = 2*N-1-i0;
        if (i1 >= N) i1 = 2*N-1-i1;
        out.idx0 = i0;
        out.idx1 = i1;
    
      } return;
      case rtc::BORDER:
        out = {};
        printf("border\n"); return;
      case rtc::CLAMP: {
        // NOT NORMALIZED:
        tc = tc * N;

        if (tc-.5f <= 0.f) { out.idx0 = out.idx1 = 0; out.f = 0.f; }
        else if (tc-.5f >= N-1) { out.idx0 = out.idx1 = N-1; out.f = 0.f; }
        else {
          out.f = (tc-.5f) - int(tc-.5f);
          out.idx0 = int(tc-.5f);
          out.idx1 = out.idx0+1;
        }    
      } return;
      };
    }

    





    
    TextureData::TextureData(Device *device,
                             vec3i dims,
                             rtc::DataType format,
                             const void *texels)
      : device(device),
        dims(dims),
        format(format)
    {
      // cudaChannelFormatDesc desc;
      size_t sizeOfScalar;
      size_t numScalarsPerTexel;
      switch (format) {
      case rtc::FLOAT:
        // desc         = cudaCreateChannelDesc<float>();
        sizeOfScalar = 4;
        // readMode     = cudaReadModeElementType;
        numScalarsPerTexel = 1;
        break;
      case rtc::FLOAT4:
        // desc         = cudaCreateChannelDesc<float4>();
        sizeOfScalar = 4;
        // readMode     = cudaReadModeElementType;
        numScalarsPerTexel = 4;
        break;
      case rtc::UCHAR:
        // desc         = cudaCreateChannelDesc<uint8_t>();
        sizeOfScalar = 1;
        // readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 1;
        break;
      case rtc::UCHAR4:
        // desc         = cudaCreateChannelDesc<uchar4>();
        sizeOfScalar = 1;
        // readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 4;
        break;
      case rtc::USHORT:
        // desc         = cudaCreateChannelDesc<uint16_t>();
        sizeOfScalar = 2;
        // readMode     = cudaReadModeNormalizedFloat;
        numScalarsPerTexel = 1;
        break;
      case rtc::UCHAR2:
        sizeOfScalar = 1;
        numScalarsPerTexel = 2;
        break;
      case rtc::USHORT2:
        sizeOfScalar = 2;
        numScalarsPerTexel = 2;
        break;
      case rtc::HALF:
        sizeOfScalar = 2;
        numScalarsPerTexel = 1;
        break;
      case rtc::HALF2:
        sizeOfScalar = 2;
        numScalarsPerTexel = 2;
        break;
      case rtc::HALF3:
        sizeOfScalar = 2;
        numScalarsPerTexel = 3;
        break;
      case rtc::HALF4:
        sizeOfScalar = 2;
        numScalarsPerTexel = 4;
        break;
      default:
        assert(0);
      };

      size_t padded_x = (unsigned)dims.x;
      size_t padded_y = std::max(1u,(unsigned)dims.y);
      size_t padded_z = std::max(1u,(unsigned)dims.z);
      data.resize(padded_x*padded_y*padded_z*numScalarsPerTexel*sizeOfScalar);
      memcpy(data.data(),texels,data.size());
    }

    Texture *TextureData::createTexture(const rtc::TextureDesc &desc) 
    {
      return new Texture(this,desc);
    }



    rtc::TextureObject Texture::getDD() const
    {
      return (const rtc::TextureObject &)sampler;
    }
    
    template<typename T, int FILTER_MODE>
    struct TextureSamplerT
    {};


    // distinct texel types for half - they share storage with
    // uint16 variants, so getTexel<T> needs dedicated keys
    struct half1 { uint16_t v; };
    struct half2 { uint16_t v[2]; };
    struct half3 { uint16_t v[3]; };
    struct half4 { uint16_t v[4]; };

    // sRGB decode matching the GPU's sampling-time texture decode
    // (color channels only - alpha stays linear)
    inline float srgbToLinear(float f)
    {
      return (f <= 0.04045f)
        ? f / 12.92f
        : powf((f + 0.055f) / 1.055f, 2.4f);
    }

    inline vec4f applyColorSpace(vec4f v, const rtc::TextureDesc &desc)
    {
      if (desc.colorSpace == rtc::COLOR_SPACE_SRGB) {
        v.x = srgbToLinear(v.x);
        v.y = srgbToLinear(v.y);
        v.z = srgbToLinear(v.z);
      }
      return v;
    }

    template<typename T>
    vec4f getTexel(TextureData *data,
                    const rtc::TextureDesc &desc,
                    int64_t idx)
    {
      printf("gettexel not implemented for type... %i == %i\n",(int)data->format,
             (int)rtc::UCHAR4);
      return vec4f(0.f);
    }
    
    template<>
    vec4f getTexel<vec4f>(TextureData *data,
                           const rtc::TextureDesc &desc,
                           int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      vec4f v = ((const vec4f*)data->data.data())[idx];
      return v;
    }
    
    template<>
    vec4f getTexel<float>(TextureData *data,
                           const rtc::TextureDesc &desc,
                           int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      float v = ((const float*)data->data.data())[idx];
      return vec4f(v);
    }
    
    
    template<>
    vec4f getTexel<vec4uc>(TextureData *data,
                            const rtc::TextureDesc &desc,
                            int64_t idx)
    {
      // printf("gettexel uchar4 data %p idx %li, eltsize %li\n",data,idx,
      //        sizeof(vec4uc),data);
      if (idx < 0) return desc.borderColor;
      vec4uc v = ((const vec4uc*)data->data.data())[idx];
      vec4f  vf = vec4f(v);
      vf = vf * 1.f/255.f;
      return applyColorSpace(vf,desc);
    }

    template<>
    vec4f getTexel<vec2uc>(TextureData *data,
                            const rtc::TextureDesc &desc,
                            int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      vec2uc v = ((const vec2uc*)data->data.data())[idx];
      vec4f vf = vec4f(v.x * (1.f/255.f), v.y * (1.f/255.f), 0.f, 0.f);
      return applyColorSpace(vf,desc);
    }

    template<>
    vec4f getTexel<unsigned short>(TextureData *data,
                                   const rtc::TextureDesc &desc,
                                   int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      unsigned short v = ((const unsigned short*)data->data.data())[idx];
      vec4f vf = vec4f(v * (1.f/65535.f));
      return applyColorSpace(vf,desc);
    }

    template<>
    vec4f getTexel<vec2us>(TextureData *data,
                            const rtc::TextureDesc &desc,
                            int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      vec2us v = ((const vec2us*)data->data.data())[idx];
      vec4f vf = vec4f(v.x * (1.f/65535.f), v.y * (1.f/65535.f), 0.f, 0.f);
      return applyColorSpace(vf,desc);
    }

    template<>
    vec4f getTexel<half1>(TextureData *data,
                          const rtc::TextureDesc &desc,
                          int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      half1 v = ((const half1*)data->data.data())[idx];
      return vec4f(rtc::halfToFloat(v.v), 0.f, 0.f, 0.f);
    }

    template<>
    vec4f getTexel<half2>(TextureData *data,
                          const rtc::TextureDesc &desc,
                          int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      half2 v = ((const half2*)data->data.data())[idx];
      return vec4f(rtc::halfToFloat(v.v[0]), rtc::halfToFloat(v.v[1]),
                   0.f, 0.f);
    }

    template<>
    vec4f getTexel<half3>(TextureData *data,
                          const rtc::TextureDesc &desc,
                          int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      half3 v = ((const half3*)data->data.data())[idx];
      return vec4f(rtc::halfToFloat(v.v[0]), rtc::halfToFloat(v.v[1]),
                   rtc::halfToFloat(v.v[2]), 0.f);
    }

    template<>
    vec4f getTexel<half4>(TextureData *data,
                          const rtc::TextureDesc &desc,
                          int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      half4 v = ((const half4*)data->data.data())[idx];
      return vec4f(rtc::halfToFloat(v.v[0]), rtc::halfToFloat(v.v[1]),
                   rtc::halfToFloat(v.v[2]), rtc::halfToFloat(v.v[3]));
    }

    template<>
    vec4f getTexel<unsigned char>(TextureData *data,
                                  const rtc::TextureDesc &desc,
                                  int64_t idx)
    {
      if (idx < 0) return desc.borderColor;
      unsigned char v = ((const unsigned char*)data->data.data())[idx];
      vec4f  vf = vec4f(v);
      vf = vf * 1.f/255.f;
      return applyColorSpace(vf,desc);
    }

    
    template<typename T>
    struct TextureSamplerT<T,rtc::FILTER_MODE_POINT>
      : public TextureSampler
    {
      TextureSamplerT(TextureData *data,
                      const rtc::TextureDesc &desc)
        : TextureSampler(data, desc)
      {}
      vec4f tex1D(float tc) override
      {
        int size = data->dims.x;
        int ix = uint32_t(tc * size) % (uint32_t)size;
        return getTexel<T>(data,desc,ix);
      }
      vec4f tex2D(vec2f tc) override
      {
        vec2i size = {data->dims.x,data->dims.y};
        int ix = uint32_t(fabsf(tc.x) * size.x) % (uint32_t)size.x;
        int iy = uint32_t(fabsf(tc.y) * size.y) % (uint32_t)size.y;
        return getTexel<T>(data,desc,ix+iy*size.x);
      }
      vec4f tex3D(vec3f tc) override
      {
        if (desc.normalizedCoords) {
          // not implemented...
          PING;
          return vec4f(0.f);
        } else {
          int Nx = data->dims.x;
          int Ny = data->dims.y;
          int Nz = data->dims.z;
          uint32_t lx = (uint32_t)clamp(tc.x,0.f,Nx-1.f);
          uint32_t ly = (uint32_t)clamp(tc.y,0.f,Ny-1.f);
          uint32_t lz = (uint32_t)clamp(tc.z,0.f,Nz-1.f);
          auto pixelAddress = [](int ix, int iy, int iz, int Nx, int Ny) {
            return (std::min(ix,std::min(iy,iz)) == -1)
              ? size_t(-1)
              : (ix+size_t(Nx)*(iy+size_t(Ny)*(iz)));
          };

          int64_t i = pixelAddress(lx,ly,lz,
                                   data->dims.x,data->dims.y);
          return getTexel<T>(data,desc,i);
        }
        vec2i size = {data->dims.x,data->dims.y};
        int ix = uint32_t(fabsf(tc.x) * size.x) % (uint32_t)size.x;
        int iy = uint32_t(fabsf(tc.y) * size.y) % (uint32_t)size.y;
        return getTexel<T>(data,desc,ix+iy*size.x);
      }
    };

    template<typename T>
    struct TextureSamplerT<T,rtc::FILTER_MODE_LINEAR>
      : public TextureSampler
    {
      TextureSamplerT(TextureData *data,
                      const rtc::TextureDesc &desc)
        : TextureSampler(data, desc)
      {}
      virtual ~TextureSamplerT() {
      }
      vec4f tex1D(float tc) override
      {
        int size = data->dims.x;
        int ix = uint32_t(tc * size) % (uint32_t)size;
        return getTexel<T>(data,desc,ix);
      }
      vec4f tex2D(vec2f tc) override
      {
        if (desc.normalizedCoords == false) {
          return vec4f(0.f,0.f,0.f,0.f);
        } else {
          LerpAddresses lx,ly;
          computeAddress(lx,desc.addressMode[0],tc.x,data->dims.x);
          computeAddress(ly,desc.addressMode[1],tc.y,data->dims.y);

          auto pixelAddress = [](int ix, int iy, int size) {
            return (std::min(ix,iy) == -1) ? -1 : (ix+iy*size);
          };
          int i00 = pixelAddress(lx.idx0,ly.idx0,data->dims.x);
          int i01 = pixelAddress(lx.idx1,ly.idx0,data->dims.x);
          int i10 = pixelAddress(lx.idx0,ly.idx1,data->dims.x);
          int i11 = pixelAddress(lx.idx1,ly.idx1,data->dims.x);

          vec4f v00 = getTexel<T>(data,desc,i00);
          vec4f v01 = getTexel<T>(data,desc,i01);
          vec4f v10 = getTexel<T>(data,desc,i10);
          vec4f v11 = getTexel<T>(data,desc,i11);

          vec4f v0 = lerp_l(lx.f,v00,v01);
          vec4f v1 = lerp_l(lx.f,v10,v11);
          return lerp_l(ly.f,v0,v1);
        }
      }


      inline void lerpCoords_notNormalized(vec2i &out_coords,
                                           float &out_f,
                                           int N,
                                           float tc)
      {
        if (tc-.5f <= 0.f) { out_coords = vec2i{ 0, 0 }; out_f = 0.f; return; }
        if (tc-.5f >= N-1) { out_coords = vec2i{ N-1, N-1 }; out_f = 0.f; return; }

        //    out_f = fracf(tc-.5f);
        out_f = (tc-.5f) - int(tc-.5f);
        out_coords.x = int(tc-.5f);
        out_coords.y = out_coords.x+1;
      }
  
      // LINEAR filter
      vec4f tex3D(vec3f tc) override
      {
        if (desc.normalizedCoords == false) {
          int Nx = data->dims.x;
          int Ny = data->dims.y;
          int Nz = data->dims.z;
          vec2i lx,ly,lz;
          float fx,fy,fz;
          lerpCoords_notNormalized(lx,fx,Nx,tc.x);
          lerpCoords_notNormalized(ly,fy,Ny,tc.y);
          lerpCoords_notNormalized(lz,fz,Nz,tc.z);
          // tc.x = tc.x - .5f;
          // tc.y = tc.y - .5f;
          // tc.z = tc.z - .5f;
          // int Nx = data->dims.x;
          // int Ny = data->dims.y;
          // int Nz = data->dims.z;
          // tc.x = fabsf(tc.x);
          // tc.y = fabsf(tc.y);
          // tc.z = fabsf(tc.z);
          // int ix0 = int(tc.x);
          // int iy0 = int(tc.y);
          // int iz0 = int(tc.z);
          // float fx = tc.x - ix0;
          // float fy = tc.y - iy0;
          // float fz = tc.z - iz0;
          // ix0 = min(ix0,Nx-1);
          // iy0 = min(iy0,Ny-1);
          // iz0 = min(iz0,Nz-1);
          // int ix1 = min(ix0+1,Nx-1);
          // int iy1 = min(iy0+1,Ny-1);
          // int iz1 = min(iz0+1,Nz-1);
          int ix0 = lx.x;
          int ix1 = lx.y;
          int iy0 = ly.x;
          int iy1 = ly.y;
          int iz0 = lz.x;
          int iz1 = lz.y;
        
          auto pixelAddress = [](int ix, int iy, int iz, int Nx, int Ny) {
            return (std::min(ix,std::min(iy,iz)) == -1)
              ? size_t(-1)
              : (ix+size_t(Nx)*(iy+size_t(Ny)*(iz)));
          };

          int64_t i000 = pixelAddress(ix0,iy0,iz0,Nx,Ny);
          int64_t i001 = pixelAddress(ix1,iy0,iz0,Nx,Ny);
          int64_t i010 = pixelAddress(ix0,iy1,iz0,Nx,Ny);
          int64_t i011 = pixelAddress(ix1,iy1,iz0,Nx,Ny);
          int64_t i100 = pixelAddress(ix0,iy0,iz1,Nx,Ny);
          int64_t i101 = pixelAddress(ix1,iy0,iz1,Nx,Ny);
          int64_t i110 = pixelAddress(ix0,iy1,iz1,Nx,Ny);
          int64_t i111 = pixelAddress(ix1,iy1,iz1,Nx,Ny);
      
          vec4f v000 = getTexel<T>(data,desc,i000);
          vec4f v001 = getTexel<T>(data,desc,i001);
          vec4f v010 = getTexel<T>(data,desc,i010);
          vec4f v011 = getTexel<T>(data,desc,i011);
          vec4f v100 = getTexel<T>(data,desc,i100);
          vec4f v101 = getTexel<T>(data,desc,i101);
          vec4f v110 = getTexel<T>(data,desc,i110);
          vec4f v111 = getTexel<T>(data,desc,i111);
      
          vec4f v00 = lerp_l(fx,v000,v001);
          vec4f v01 = lerp_l(fx,v010,v011);
          vec4f v10 = lerp_l(fx,v100,v101);
          vec4f v11 = lerp_l(fx,v110,v111);
      
          vec4f v0 = lerp_l(fy,v00,v01);
          vec4f v1 = lerp_l(fy,v10,v11);
      
          vec4f v = lerp_l(fz,v0,v1);
          return v;
        } else {
          printf("tex3d, IS normalized... not implemented\n");
          return vec4f(0.f,0.f,0.f,0.f);
        }
      }
    };

    template<typename texel_t>
    TextureSampler *createSampler(TextureData *data,
                                  const rtc::TextureDesc &desc)
    {
      if (desc.filterMode == rtc::FILTER_MODE_POINT)
        return new TextureSamplerT
          <texel_t,rtc::FILTER_MODE_POINT>(data,desc);
      else
        return new TextureSamplerT
          <texel_t,rtc::FILTER_MODE_LINEAR>(data,desc);
    }
    

    TextureSampler *createSampler(TextureData *data,
                                  rtc::TextureDesc desc)
    {
      switch (data->format) {
      case rtc::UCHAR:
        return createSampler<unsigned char>(data,desc);
        break;
      case rtc::UCHAR2:
        return createSampler<vec2uc>(data,desc);
        break;
      case rtc::UCHAR4:
        return createSampler<vec4uc>(data,desc);
        break;
      case rtc::USHORT:
        return createSampler<unsigned short>(data,desc);
        break;
      case rtc::USHORT2:
        return createSampler<vec2us>(data,desc);
        break;
      case rtc::HALF:
        return createSampler<half1>(data,desc);
        break;
      case rtc::HALF2:
        return createSampler<half2>(data,desc);
        break;
      case rtc::HALF3:
        return createSampler<half3>(data,desc);
        break;
      case rtc::HALF4:
        return createSampler<half4>(data,desc);
        break;
      case rtc::FLOAT4:
        return createSampler<vec4f>(data,desc);
        break;
      case rtc::FLOAT:
        return createSampler<float>(data,desc);
        break;
      default:
        throw std::runtime_error("sampler channel desc not implemented");
      } 
    }
    
    
    Texture::Texture(TextureData *const data,
                     const rtc::TextureDesc &desc)
    {
      sampler = createSampler(data,desc);
    }

    float tex2D1f(rtc::TextureObject to,
                           float x, float y)
    { return ((TextureSampler *)to)->tex2D({x,y}).x; }
    
    float tex3D1f(rtc::TextureObject to,
                           float x, float y, float z)
    {
      return ((TextureSampler *)to)->tex3D({x,y,z}).x;
    }
    
    vec4f tex2D4f(rtc::TextureObject to,
                            float x, float y)
    {
      // PING;
      // PRINT(to);
      vec4f v = ((TextureSampler *)to)->tex2D({x,y});
      return v;
    }

    vec4f tex3D4f(rtc::TextureObject to,
                           float x, float y, float z)
    { return ((TextureSampler *)to)->tex3D({x,y,z}); }
    

  }
}
