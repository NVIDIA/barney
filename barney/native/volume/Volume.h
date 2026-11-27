// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/Object.h"
#include "native/render/WholeWaveIntersectKernel.h"
#include "native/volume/TransferFunction.h"
#include "native/volume/ScalarField.h"
#include "native/volume/PrincipledVolume.h"
#include <array>

namespace BARNEY_NS {
  namespace native {
    
    using Random = Random2;
    struct Volume;
    struct ScalarField;
    struct MCGrid;
  
    typedef std::array<int,4> TetIndices;
    typedef std::array<int,5> PyrIndices;
    typedef std::array<int,6> WedIndices;
    typedef std::array<int,8> HexIndices;

    struct VolumeAccel {
    
      typedef std::shared_ptr<VolumeAccel> SP;

      VolumeAccel(Volume *volume);
      virtual ~VolumeAccel() = default;

      virtual void build(bool full_rebuild) = 0;

      virtual void rebuildMajorantsOnly();
      virtual void refreshDeviceData();

      const TransferFunction *getXF() const;
    
      Volume      *const volume = 0;
      const DevGroup::SP devices;
    };





    struct VolumeAccel;
  
    /*! a *volume* is a scalar field with a transfer function applied to
      it; it's main job is to create something that can intersect a
      ray with that scalars-plus-transferfct thingy, for which it will
      use some kind of volume accelerator that implements the
      scalar-field type specific stuff (eg, traverse a bvh over
      elements, or look up a 3d texture, etc) */
    struct Volume : public Object
    {
      template<typename SFSampler>
      struct DD {
        inline __rtc_device
        vec4f sampleAndMap(vec3f point, bool dbg=false) const
        {
          float f = sfSampler.sample(point,dbg);
          if (isnan(f)) return vec4f(0.f);
          return xf.map(f,dbg);
        }

        inline __rtc_device
        float sampleScalar(vec3f point, bool dbg=false) const
        {
          return sfSampler.sample(point, dbg);
        }

        ScalarField::DD               sfCommon;
        typename SFSampler::DD        sfSampler;
        TransferFunction::DD          xf;
        TransferFunction::DD          emissionXf;
        int                           emissionXfActive;
        PrincipledVolumeParams::DD    principled;
        float                         anisotropy;
        float                         scatteringAlbedo;
        int                           userID;
      };
    
      template<typename SFSampler>
      DD<SFSampler> getDD(Device *device, std::shared_ptr<SFSampler> sampler)
      {
        DD<SFSampler> dd;
        dd.sfCommon = sf->getDD(device);
        dd.sfSampler = sampler->getDD(device);
        dd.xf = xf.getDD(device);
        dd.emissionXf = emissionXf.getDD(device);
        dd.emissionXfActive = emissionXfActive ? 1 : 0;
        dd.userID = userID;
        dd.principled = principled.getDD(device);
        dd.anisotropy = anisotropy;
        dd.scatteringAlbedo = scatteringAlbedo;
        return dd;
      }

      typedef std::shared_ptr<Volume> SP;
    
      Volume(ScalarField::SP sf);

      /*! pretty-printer for printf-debugging */
      std::string toString() const override
      { return "Volume{}"; }

      static SP create(ScalarField::SP sf)
      {
        return std::make_shared<Volume>(sf);
      }
    
      /*! (re-)build the accel structure for this volume, probably after
        changes to transfer functoin (or later, scalar field) */
      virtual void build(bool full_rebuild);
    
      virtual void setXF(const range1f &domain,
                         const bn_float4 *values,
                         int numValues,
                         float baseDensity);
      void setEmissionXF(const range1f &domain,
                         const bn_float4 *values,
                         int numValues);
      void clearEmissionXF();
      void commit() override;
      bool set1i(const std::string &member,
                 const int   &value) override;
      bool set1f(const std::string &member,
                 const float &value) override;
      bool set2f(const std::string &member,
                 const vec2f &value) override;
      bool set3f(const std::string &member,
                 const vec3f &value) override;
               
      ScalarField::SP  sf;
      VolumeAccel::SP  accel;
      TransferFunction xf;
      TransferFunction emissionXf;
      bool emissionXfActive = false;
      PrincipledVolumeParams principled;
      float anisotropy = 0.0f;
      float scatteringAlbedo = 1.0f;
      bool needsMajorantRebuild = false;
      DevGroup::SP const devices;
      int userID = 0;
    
      /*! @{ the different intersectable objects that this geometry
          creates during 'build' */
      struct PLD {
        std::vector<rtc::Group *> generatedGroups;
        std::vector<rtc::Geom *>  generatedGeoms;
      };
      /*! a possible whole-wave kernel for this geom; this is not
          inside our own PLD, but can manage its own PLD by itself */
      WholeWaveIntersectKernel::SP wholeWaveKernel;
      /*! @} */
      
      PLD *getPLD(Device *device);
      std::vector<PLD> perLogical;
    };

    inline __rtc_device float fastLog(float f)
    {
      f = (f-1.f)/(f+1.f);
      float f2 = f*f;
      float s = f;
      f *= f2;
      s += (1.f/3.f)*f;
      f *= f2;
      s += (1.f/5.f)*f;
      return s+s;
    }
  
    /*! helper class that performs woodcock sampling over a given
      parameter range, for a given sample'able volume type */
    struct Woodcock {
      template<typename VolumeDD>
      static inline __rtc_device
      bool sampleRange(vec4f &sample,
                       const VolumeDD &sfSampler,
                       vec3f org, vec3f dir,
                       range1f &tRange,
                       float majorant,
                       Random &rand,
                       bool dbg=false) 
      {
        float t = tRange.lower;
        while (true) {
          float r = rand();
          float dt = - fastLog(1.f-r)/majorant;
          // float dt = - logf(1.f-r)/majorant;
          t += dt;
          if (t >= tRange.upper)
            return false;

          vec3f P = org+t*dir;
          sample = sfSampler.sampleAndMap(P,dbg);
          // if (dbg) printf("sample at t %f, P= %f %f %f -> %f %f %f : %f\n",
          //                 t,
          //                 P.x,P.y,P.z,
          //                 sample.x,
          //                 sample.y,
          //                 sample.z,
          //                 sample.w);
          if (sample.w >= rand()*majorant) {
            tRange.upper = t;
            return true;
          }
        }
      }
    };

    inline VolumeAccel::VolumeAccel(Volume *volume)
      : volume(volume),
        devices(volume->devices)
    {
      assert(volume);
      assert(volume->sf);
    }

    inline void VolumeAccel::rebuildMajorantsOnly() {}

    inline void VolumeAccel::refreshDeviceData() {}
  }
}
