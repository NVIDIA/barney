// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "barney/fb/FrameBuffer.h"
#include "barney/fb/TiledFB.h"
#include "barney/render/World.h"
#include "barney/render/DG.h"
#include "barney/render/Renderer.h"
#include "barney/GlobalModel.h"
#include "barney/render/RayQueue.h"
#include "rtcore/TraceInterface.h"

namespace BARNEY_NS {
  namespace render {

#define SCI_VIS_MODE 1
    
#define MAX_DIFFUSE_BOUNCES 1
    
#define ENV_LIGHT_SAMPLING 1

#define USE_MIS 1


#define CLAMP_F_R 13.f


#if RTC_DEVICE_CODE
    inline __rtc_device float square(float f) { return f*f; }
  
    
    enum { MAX_PATH_DEPTH = 10 };

    inline __rtc_device
    float safe_eps(float f, vec3f v)
    {
      // return 1e-4f;
      return max(f,1e-6f*reduce_max(abs(v)));
    }

    
    inline __rtc_device
    bool sampleAreaLights(Light::Sample &ls,
                          const render::World::DD &world,
                          const vec3f P,
                          const vec3f N,
                          Random &random,
                          bool dbg)
    {
      if (world.numQuadLights == 0) return false;
      static const int RESERVOIR_SIZE = 8;
      int   lID[RESERVOIR_SIZE];
      float u[RESERVOIR_SIZE];
      float v[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      QuadLight::DD light;
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numQuadLights),
                     world.numQuadLights-1);
        weights[i] = 0.f;
        light = world.quadLights[lID[i]];
        u[i] = random();
        v[i] = random();
        float lightArea = light.area;
// #ifndef NDEBUG
//         if (lightArea < 0.f)
//           printf("INVALID NEGATIVE LIGHT AREA on light %i/%i : %f\n",
//                  lID[i],world.numQuadLights,lightArea);
// #endif
        vec3f LN = light.normal;
        vec3f LP = light.corner + u[i]*light.edge0 + v[i]*light.edge1;
        vec3f lightDir = LP - P;
        float lightDist = length(lightDir);
        if (lightDist < 1e-3f) continue;
      
        lightDir *= 1.f/lightDist;

        // float weight = dot(lightDir,N);
        float weight = (N==vec3f(0.f)) ? 1.f : dot(lightDir,N);
        if (weight <= 1e-3f) continue;
        weight *= -dot(lightDir,LN);
        if (weight <= 1e-3f) continue;
// #ifndef NDEBUG
//         if (lightArea == 0.f || reduce_max(light.emission) == 0)
//           printf("invalid light! %f : %f %f %f\n",
//                  lightArea,
//                  light.emission.x,
//                  light.emission.y,
//                  light.emission.z);
// #endif
        weight *= (1.f/(lightDist*lightDist)) * lightArea * reduce_max(light.emission);
// #ifndef NDEBUG
//         if (isnan(sumWeights) || weight < 0.f)
//           printf("area lights: weight[%i:%i] is nan or negative: dist  %f area %f emission %f %f %f\n",
//                  i,lID[i],lightDist,lightArea,
//                  light.emission.x,
//                  light.emission.y,
//                  light.emission.z);
// #endif
        sumWeights += weight;
        weights[i] = weight;
      }
// #ifndef NDEBUG
//       if (isnan(sumWeights))
//         printf("area lights: sumWeights is nan!\n");
// #endif
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.quadLights[lID[i]];
      vec3f LP = light.corner + u[i]*light.edge0 + v[i]*light.edge1;
      vec3f LD = LP-P;
      ls.direction
        = normalize(LD);
      ls.distance
        = length(LD);
      ls.radiance
        = light.emission
        * (light.area * -dot(light.normal,ls.direction)
           / square(ls.distance));
      ls.pdf
        = weights[i]/sumWeights
        * (float(RESERVOIR_SIZE)/float(world.numQuadLights));
// #ifndef NDEBUG
//       if (ls.pdf <= 0.f)
//         printf("invalid area light PDF %f from i %i weight %f sum %f\n",
//                ls.pdf,i,weights[i],sumWeights);
// #endif
      return true;
    }

    inline __rtc_device
    bool sampleDirLights(Light::Sample &ls,
                         const World::DD &world,
                         const Renderer::DD &renderer,
                         const vec3f P,
                         const vec3f N,
                         Random &random,
                         bool dbg)
    {
      if (world.numDirLights == 0) return false;
      static const int RESERVOIR_SIZE = 2;
      int   lID[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      DirLight::DD light;
    
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numDirLights),
                     world.numDirLights-1);
        weights[i] = 0.f;
        light = world.dirLights[lID[i]];
        vec3f light_radiance
          = light.color
          * light.radiance;
        
        vec3f lightDir = -light.direction;
        float weight = (N==vec3f(0.f)) ? 1.f : dot(lightDir,N);
        if (dbg) printf("light #%i, dir %f %f %f weight %f\n",lID[i],lightDir.x,lightDir.y,lightDir.z,weight);
        if (weight <= 1e-3f) continue;
        weight *= reduce_max(light_radiance);
        if (weight <= 1e-3f) continue;
        weights[i] = weight;
        sumWeights += weight;
      }
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.dirLights[lID[i]];
      ls.direction
        = -light.direction;
      ls.distance
        = BARNEY_INF;
      ls.radiance
        = light.color*light.radiance
        * 1.f/(weights[i]/sumWeights
               * (float(RESERVOIR_SIZE)/float(world.numDirLights)));
      ls.pdf = BARNEY_INF;
      return weights[i] != 0.f;
    }


    inline __rtc_device
    bool samplePointLights(Light::Sample &ls,
                         const World::DD &world,
                         const Renderer::DD &renderer,
                         const vec3f P,
                         const vec3f N,
                         Random &random,
                         bool dbg)
    {
      struct {
        // constant term
        const float c0 = 0.f;
        // linear term
        const float c1 = 0.f;
        // square term
        const float c2 = 1.f;
      } /* fall-off costants */foc;
      if (world.numPointLights == 0) return false;
      static const int RESERVOIR_SIZE = 2;
      int   lID[RESERVOIR_SIZE];
      float weights[RESERVOIR_SIZE];
      float sumWeights = 0.f;
      PointLight::DD light;
    
      for (int i=0;i<RESERVOIR_SIZE;i++) {
        lID[i] = min(int(random()*world.numPointLights),
                     world.numPointLights-1);
        weights[i] = 0.f;
        light = world.pointLights[lID[i]];
        
        vec3f lightDirection = light.position-P;
        float dist = length(lightDirection);
        float falloff = foc.c0 + foc.c1*dist + foc.c2*dist*dist;
        lightDirection = normalize(lightDirection);

        vec3f light_radiance
          = light.radianceTowards(P)
          * (1.f/falloff);
        
        // light_radiance *= (1.f/dist);
        float weight = (N==vec3f(0.f)) ? 1.f : dot(lightDirection,N);
        // float weight = dot(lightDirection,N);
        if (weight <= 1e-3f) continue;
        weight *= reduce_max(light_radiance);
        if (weight <= 1e-3f) continue;
        weights[i] = weight;
        sumWeights += weight;
      }
      if (sumWeights == 0.f) return false;
      float r = random()*sumWeights;
      int i=0;
      while (i<RESERVOIR_SIZE && r >= weights[i]) { r-= weights[i]; ++i; }
      if (i == RESERVOIR_SIZE) return false;
    
      light = world.pointLights[lID[i]];
      ls.direction
        = light.position-P;
      float dist = length(ls.direction);
      ls.direction = normalize(ls.direction);
      float falloff = foc.c0 + foc.c1*dist + foc.c2*dist*dist;
      ls.distance
        = dist*.9999f;
      ls.radiance
        = light.radianceTowards(P)
        * (1.f/falloff)
        * 1.f/(weights[i]/sumWeights
               * (float(RESERVOIR_SIZE)/float(world.numPointLights)));
      ls.pdf = BARNEY_INF;
      return weights[i] != 0.f;
    }



    inline __rtc_device
    bool sampleEnvLight(Light::Sample &ls,
                        const World::DD &world,
                        const Renderer::DD &renderer,
                        const vec3f P,
                        const vec3f N,
                        Random &random,
                        bool dbg)
    {
      /* in barney, the environment is either a explicit hdri map (in
         EnvMapLight); or a uniform brightness of 'renderer.ambientRadiance' */
      if (world.envMapLight.texture) {
        ls = world.envMapLight.sample(random,dbg);
      } else {
        ls.direction = randomDirection(random);
        ls.distance  = BARNEY_INF;
        if (N != vec3f(0.f)) {
          ls.radiance  = TWO_PI*renderer.ambientRadiance;
          if (dot(ls.direction,N) < 0.f) ls.direction = -ls.direction;
          ls.pdf       = ONE_OVER_TWO_PI;
        } else {
          ls.radiance  = FOUR_PI*renderer.ambientRadiance;
          ls.pdf       = ONE_OVER_FOUR_PI;
        }
      }
      return true;
    }

    inline __rtc_device
    bool sampleLights(Light::Sample &ls,
                      const World::DD &world,
                      const Renderer::DD &renderer,
                      const vec3f P,
                      const vec3f Ng,
                      Random &random,
#if USE_MIS
                      bool &lightNeedsMIS,
                      bool &lightIsDirLight,
#endif
                      bool dbg)
    {
#if USE_MIS
# if 0
      // huh ... not sure this is correct; setting this to true means
      // we'll always compute MIS weights for shadow and bounce ray as
      // if there was only an env-map light; even though we may
      // acutally have sampled a dir-light. that _may_ be true because
      // even if we did sample a dirlight there still _is_ a pdf for
      // the env-map light... but it's a bit iffy.
      lightNeedsMIS = true;
# else
      lightNeedsMIS = false;
# endif
#endif

#if ENV_LIGHT_SAMPLING
      Light::Sample els;
      float elsWeight
        = (sampleEnvLight(els,world,renderer,P,Ng,random,dbg)
           ? (reduce_max(els.radiance)/*/els.pdf*/)
           : 0.f);
      if (dbg)
        printf("els rad %f %f %f pdf %f\n",
               els.radiance.x,
               els.radiance.y,
               els.radiance.z,
               els.pdf);
      // = world.envMapLight.sample(random,dbg);
#else
      float elsWeight = 0.f;
#endif

      Light::Sample als;
      float alsWeight
        = (sampleAreaLights(als,world,P,Ng,random,dbg)
           ? (reduce_max(als.radiance)/als.pdf)
           : 0.f);
      Light::Sample dls;
      float dlsWeight
        = (sampleDirLights(dls,world,renderer,P,Ng,random,dbg)
           ? (reduce_max(dls.radiance)/* /dls.pdf*/)
           : 0.f);
      Light::Sample pls;
      float plsWeight
        = (samplePointLights(pls,world,renderer,P,Ng,random,dbg)
           ? (reduce_max(pls.radiance)/* /dls.pdf*/)
           : 0.f);

      if (dbg) printf("sampling lights dls %f els %f\n",
                      dlsWeight,elsWeight);
      
      float sumWeights
        = alsWeight+dlsWeight+elsWeight+plsWeight;
      if (sumWeights == 0.f) return false;

      elsWeight *= 1.f/sumWeights;
      alsWeight *= 1.f/sumWeights;
      dlsWeight *= 1.f/sumWeights;
      plsWeight *= 1.f/sumWeights;
      
      float r = random();
      if (dbg) printf(" light sample %f in cdf %f %f %f\n",
                      r,alsWeight,elsWeight,dlsWeight);
      if (r <= alsWeight) {
        ls = als;
        //   ls.pdf *= alsWeight;
        ls.radiance *= (1.f/alsWeight);
#if ENV_LIGHT_SAMPLING
      } else if (r <= alsWeight+elsWeight) {
        ls = els;
        ls.radiance *= (1.f/elsWeight);
        // ls.pdf *= elsWeight;
        if (dbg) printf(" ->  picked env light sample\n");
# if USE_MIS
        lightNeedsMIS = true;
# endif
#endif
      } else if (r <= alsWeight+elsWeight+plsWeight) {
        ls = pls;
        ls.radiance *= (1.f/plsWeight);
        // ls.pdf *= dlsWeight;
# if USE_MIS
        lightIsDirLight = true;
# endif
        if (dbg) printf(" ->  picked POINT light sample, pls weight %f pdf %f\n",plsWeight,ls.pdf);
      } else {
        ls = dls;
        ls.radiance *= (1.f/dlsWeight);
        // ls.pdf *= dlsWeight;
# if USE_MIS
        lightIsDirLight = true;
# endif
        if (dbg) printf(" ->  picked DIR light sample, dls weight %f pdf %f\n",dlsWeight,ls.pdf);
      }
      if (isnan(ls.pdf) || (ls.pdf <= 0.f)) return false;
      
      return true;
    }




    inline __rtc_device
    float schlick(float cosine,
                  float ref_idx)
    {
      float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
      r0 = r0 * r0;
      return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
    }
  
  

    inline __rtc_device
    bool refract(const vec3f& v,
                 const vec3f& n,
                 float ni_over_nt,
                 vec3f &refracted)
    {
      vec3f uv = normalize(v);
      float dt = dot(uv, n);
      float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
      if (discriminant > 0.f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
      }
      else
        return false;
    }
  
    inline __rtc_device
    vec3f radianceFromEnv(const World::DD &world,
                          const Renderer::DD &renderer,
                          Ray &ray)
    {
      auto &env = world.envMapLight;
      if (env.texture) {
        vec3f d = xfmVector(env.toLocal,normalize(ray.dir));
        float theta = pbrtSphericalTheta(d);
        float phi   = pbrtSphericalPhi(d);
        const float invPi  = 1.f/(float)M_PI;
        const float inv2Pi = 1.f/(2.f* (float)M_PI);
        vec2f uv(phi * inv2Pi, theta * invPi);

        vec4f color = rtc::tex2D<vec4f>(env.texture,uv.x,uv.y);
        return env.scale*vec3f(color.x,color.y,color.z);
      } else {
        return renderer.ambientRadiance;
      }
    }

    /*! if there _is_ a dedicated env-map light specified, this looks
      up the background color from that map; otherwise, it returns
      the 'ray.misscolor' that the primary ray generation has set as
      default color for this ray */
    inline __rtc_device
    vec3f primaryRayMissColor(const World::DD &world,
                              const Renderer::DD &renderer,
                              Ray &ray)
    {
      if (world.envMapLight.texture)
        return radianceFromEnv(world,renderer,ray);
      return
        // primary rays do store a default misscolor in the ray itself
        // - we simply return this if there's no env-map.
        (const vec3f&)ray.missColor;
    }

    /*! ugh - that should all go into material::AnariPhysical .... */
    inline __rtc_device
    void bounce(int rayID,
                const World::DD &world,
                const Renderer::DD &renderer,
                vec3f &fragment,
                Ray &ray,
                PathState &state,
                Ray &shadowRay,
                PathState &shadowState,
                int pathDepth)
    {
      
      const float EPS = 1e-4f;

      const bool  hadNoIntersection  = !ray.hadHit();
      const vec3f incomingThroughput = state.throughput;
      
      bool dbg = 0 && ray.dbg();
      if (dbg)
        printf("(%i) ------------------------------------------------------------------\n -> incoming %f %f %f dir %f %f %f t %f\n  tp %f %f %f ismiss %i, bsdf %i\n",
               pathDepth,
               ray.org.x,
               ray.org.y,
               ray.org.z,
               (float)ray.dir.x,
               (float)ray.dir.y,
               (float)ray.dir.z,
               ray.tMax,
               (float)state.throughput.x,
               (float)state.throughput.y,
               (float)state.throughput.z,
               int(hadNoIntersection),(int)ray.bsdfType);
      
      if (ray.isShadowRay) {
        // ==================================================================
        // shadow ray = all we have to do is add carried radiance if it
        // reached the light, and discards
        // ==================================================================
                 
        if (hadNoIntersection) {
          fragment =
# if USE_MIS
            (float)state.misWeight *
#endif
            (vec3f)state.throughput;
          if (dbg)
            printf("_shadow_ ray reaches light: tp %f %f %f misweight %f frag %f %f %f\n",
                   (float)state.throughput.x,
                   (float)state.throughput.y,
                   (float)state.throughput.z,
                   (float)state.misWeight,
                   fragment.x,
                   fragment.y,
                   fragment.z);
          if (dbg) printf("shadow miss, frag %f %f %f\n",
                          fragment.x,
                          fragment.y,
                          fragment.z);
        }

        // this path is done.
        shadowRay.tMax = -1.f;
        ray.tMax = -1.f;
        return;
      }

      vec3f Ng = ray.getN();
      if (dbg) printf("ray.N %f %f %f\n",Ng.x,Ng.y,Ng.z);
      const bool  isVolumeHit        = (Ng == vec3f(0.f));
      if (!isVolumeHit)
        Ng = normalize(Ng);
      const bool  hitWasOnFront      = dot((vec3f)ray.dir,Ng) < 0.f;
      vec3f Ngff
        = hitWasOnFront
        ?   Ng
        : - Ng;

      if (hadNoIntersection) {
        // ==================================================================
        // regular ray that did NOT hit ANYTHING 
        // ==================================================================
        if (pathDepth == 0) {
          // ----------------------------------------------------------------
          // PRIMARY ray that didn't hit anything -> background
          // ----------------------------------------------------------------
          fragment = primaryRayMissColor(world,renderer,ray);
        } else {
          // ----------------------------------------------------------------
          // SECONDARY ray that didn't hit anything -> env-light
          // ----------------------------------------------------------------
          // this path had at least one bounce, but now bounced into
          // nothingness - compute env-light contribution, and weigh it
          // with the path's carried throughput.
#if ENV_LIGHT_SAMPLING
# if USE_MIS
          const vec3f fromEnv = radianceFromEnv(world,renderer,ray);
          fragment = (vec3f)state.throughput * fromEnv * (float)state.misWeight;

          if (dbg)
            printf("bounce ray hits env light: tp %f %f %f misweight %f fromEnv %f %f %f\n",
                   (float)state.throughput.x,
                   (float)state.throughput.y,
                   (float)state.throughput.z,
                   (float)state.misWeight,
                   fromEnv.x,
                   fromEnv.y,
                   fromEnv.z);
# else
          fragment
            = ray.isSpecular
            ? radianceFromEnv(world,renderer,path)
            : vec3f(0.f);
# endif
#else
          const vec3f fromEnv = radianceFromEnv(world,renderer,ray);
          if (dbg)
            printf("fromenv %f %f %f\n",
                   fromEnv.x,
                   fromEnv.y,
                   fromEnv.z);
          fragment = state.throughput * fromEnv;
#endif
        }
        // no outgoing rays; this path is done.
        ray.tMax = -1.f;
        return;
      }
    

      // ==================================================================
      // this ray DID hit something: compute its local frame buffer
      // contribution at this hit point (if any), and generate secondary
      // ray and shadow ray (if applicable), with proper weights.
      // ==================================================================    
      Random random(ray.rngSeed,(const uint32_t&)ray.tMax);//rayID,ray.rngSeed);
      // Random random(ray.rngSeed.next((const uint32_t&)ray.tMax));//rayID,ray.rngSeed);
      const PackedBSDF bsdf = ray.getBSDF();
      // bool doTransmission = false;
      // =  ((float)ray.mini.transmission > 0.f)
      // && (random() < (float)ray.mini.transmission);
      render::DG dg;
      dg.P  = ray.P;
      dg.Ng = Ng;
      dg.Ns = Ng;
      dg.wo = -normalize((vec3f)ray.dir);
      dg.insideMedium = ray.isInMedium;

      // if the ray is a volume hit we want it offset it into the
      // direction the ray came from (otherwise we have a chance of
      // the shadow ray or boucne ray to terminate right where the
      // original ray ended; for others we want to offset based on
      // normal */
      const float offsetEpsilon = safe_eps(EPS,dg.P);
      vec3f frontFacingSurfaceOffset
        = (isVolumeHit?dg.wo:Ngff);

      // ==================================================================
      // FIRST, let us look at generating any shadow rays, if
      // applicable; this way we can later modify the incoming ray in
      // place when we generate the outgoing ray.
      // ==================================================================
      Light::Sample ls;
      // todo check if BSDF is perfectly specular
#if USE_MIS
      bool lightNeedsMIS = false;
      bool lightIsDirLight = false;
#endif
      if (dbg)
        printf("sampling lights with N %f %f %f\n",Ngff.x,Ngff.y,Ngff.z);
      if (sampleLights(ls,world,renderer,dg.P,Ngff,random,
#if USE_MIS
                       lightNeedsMIS,
                       lightIsDirLight,
#endif
                       dbg)) {
        if (dbg)
          printf("sample light dir %f %f %f rad %f %f %f pdf %f spike %f\n",
                 ls.direction.x,
                 ls.direction.y,
                 ls.direction.z,
                 ls.radiance.x,
                 ls.radiance.y,
                 ls.radiance.z,
                 ls.pdf,
                 reduce_max(ls.radiance)/(isinf(ls.pdf)?1.f:ls.pdf));
        EvalRes f_r
          = bsdf.eval(dg,ls.direction,dbg);
        if (dbg) printf("eval light res %f %f %f: %f\n",
                        f_r.value.x,
                        f_r.value.y,
                        f_r.value.z,
                        f_r.pdf);
          
        if (!f_r.valid() || reduce_max(f_r.value) < 1e-4f) {
          if (dbg)
            printf(" no f_r, killing shadow ray\n");
          shadowRay.tMax = -1.f;
        } else {
#ifdef CLAMP_F_R
          f_r.value = min(f_r.value,vec3f(CLAMP_F_R));
#endif
          vec3f tp_sr
            = (incomingThroughput)
            * (isinf(ls.pdf)?1.f:rcp(ls.pdf))
            * f_r.value
            * ls.radiance
            
            * (isVolumeHit?1.f:(ONE_OVER_PI*fabsf(dot(dg.Ng,ls.direction))))
            ;

          
          if (dbg) {
            printf(" -> inc tp %f %f %f, dot %f\n",
                   incomingThroughput.x,
                   incomingThroughput.y,
                   incomingThroughput.z,
                   (isVolumeHit?1.f:fabsf(dot(dg.Ng,ls.direction))));
            printf(" -> shadow f_r %f %f %f ls.rad %f %f %f pdf %f\n",
                   f_r.value.x,
                   f_r.value.y,
                   f_r.value.z,
                   ls.radiance.x,
                   ls.radiance.y,
                   ls.radiance.z,
                   ls.pdf);
            printf(" -> shadow ray tp %f %f %f\n",
                   tp_sr.x,
                   tp_sr.y,
                   tp_sr.z);
          }
          makeShadowRay
            (shadowRay,shadowState,
             /* thrghhpt */tp_sr,
             /* surface: */dg.P + offsetEpsilon*frontFacingSurfaceOffset,
             /* to light */ls.direction,
             /* length   */ls.distance * (1.f-2.f*offsetEpsilon));
          shadowRay.rngSeed = ray.rngSeed;// + 1; random();
          ray.rngSeed.next((const uint32_t&)ray.tMax);
          // Random rng(ray.rngSeed.next(hash(ti.getRTCInstanceIndex(),
          //                        ti.getGeometryIndex(),
          //                        ti.getPrimitiveIndex())));

          shadowRay._dbg = ray._dbg;
          shadowState.pixelID = state.pixelID;
            
          shadowState.misWeight = 1.f;
#if USE_MIS
          if (!lightIsDirLight && lightNeedsMIS) {
            float pdf_lightRay_lightDir
              = world.envMapLight.pdf(ls.direction);
            float pdf_scatterRay_lightDir
              = bsdf.pdf(dg,ls.direction);
            // compute MIS weight weight that shadow direction
            shadowState.misWeight
              = pdf_lightRay_lightDir
              / (pdf_lightRay_lightDir + pdf_scatterRay_lightDir + 1e-10f);
            // and if it's too small for any reason, kill the shadow
            // ray
            if ((float)shadowState.misWeight < 1e-5f)
              shadowRay.tMax  = -1.f;
          }
#endif
        }
      }
      
      // ==================================================================
      // now, let's decide what to do with the ray itself
      // ==================================================================

      // if we exceeded max depth we die, one way or another.
      if (pathDepth >= MAX_PATH_DEPTH) {
        ray.tMax = -1.f;
        return;
      }
      // now per default, "create" a valid scatter ray.
      ray.clearHit();
      ray.tMax = BARNEY_INF;
      
      ScatterResult scatterResult;
      bsdf.scatter(scatterResult,dg,random,dbg);
#ifndef NDEBUG
      if (scatterResult.type == ScatterResult::INVALID)
        printf("broken BSDF, doesn't set scatter type!\n");
#endif 
      if (dbg)
        printf("scatter result.valid ? %i\n",
               int(scatterResult.valid()));
      if (!scatterResult.valid() || scatterResult.pdf <= 1e-6f)
        return;
      
      if (scatterResult.type == ScatterResult::VOLUME) {
#if SCI_VIS_MODE
        // sci vis mode: volumes do shadow, but nothing more
        ray.tMax = -1.f;
        return;
#else
        // treat volume scatter like a diffuse scatter.
        scatterResult.type = ScatterResult::DIFFUSE;
#endif
      }
      
      if (scatterResult.type == ScatterResult::DIFFUSE ||
          scatterResult.type == ScatterResult::VOLUME) {
        if (state.numDiffuseBounces >= MAX_DIFFUSE_BOUNCES) {
          ray.tMax = -1.f;
          return;
        } else
          state.numDiffuseBounces = state.numDiffuseBounces + 1;
      }
      ray.isSpecular = (scatterResult.type == ScatterResult::SPECULAR);
      
      if (dbg)
        printf("offsetting into sign %f, direction %f %f %f\n",
               scatterResult.offsetDirection,
               frontFacingSurfaceOffset.x,
               frontFacingSurfaceOffset.y,
               frontFacingSurfaceOffset.z); 
      ray.org
        = dg.P + scatterResult.offsetDirection * offsetEpsilon*frontFacingSurfaceOffset;
      // #ifdef CLAMP_F_R
      //       scatterResult.f_r = min(scatterResult.f_r,vec3f(100.f));
      // #endif

      if (dbg)
        printf("path scattered, bsdf in scatter dir is %f %f %f, pdf %f\n",
               (float)scatterResult.f_r.x, 
               (float)scatterResult.f_r.y, 
               (float)scatterResult.f_r.z,
               scatterResult.pdf);
      ray.dir
        = normalize(scatterResult.dir);
      
      vec3f scatterFactor
        = scatterResult.f_r
        * (isVolumeHit?1.f:fabsf(dot(dg.Ng,normalize(ray.dir))))
        // * ONE_OVER_PI
        / (isinf(scatterResult.pdf)? 1.f : (//ONE_PI*
                                            scatterResult.pdf + 1e-10f));
      
#if 1
      // uhhhh.... this is TOTALLY wrong, but let's limit how much
      // each bounce can increase the throughput of a ray ... this
      // makes fireflies go away (well, makes them go 'less', but can
      // lose a lot of envergy if the brdf sample code isn't close to
      // the actual brdf.
      scatterFactor = min(scatterFactor,vec3f(2.5f));
#endif

      state.throughput
        = state.throughput * scatterFactor;
      if (dbg && scatterResult.changedMedium)
        printf("path DID change medium\n");
      if (scatterResult.changedMedium)
        ray.isInMedium = !ray.isInMedium;
      
      if (dbg)
        printf("scatter dir %f %f %f tp %f %f %f\n",
               (float)ray.dir.x,
               (float)ray.dir.y,
               (float)ray.dir.z,
               (float)state.throughput.x,
               (float)state.throughput.y,
               (float)state.throughput.z);
      
      
#if USE_MIS
      if (lightNeedsMIS && !isinf(scatterResult.pdf)) {
        float pdf_scatterRay_scatterDir = bsdf.pdf(dg,ray.dir);
        float pdf_lightRay_scatterDir   = world.envMapLight.pdf(ray.dir);
        
        state.misWeight
          = pdf_scatterRay_scatterDir
          / (pdf_scatterRay_scatterDir + pdf_lightRay_scatterDir);
      } else {
        state.misWeight = 1.f;
      }
#endif
    }
#endif // device code  

#if RTC_DEVICE_CODE
    __rtc_global void _shadeRays(const rtc::ComputeInterface &rt,
                                 World::DD world,
                                 Renderer::DD renderer,
                                 AccumTile *accumTiles,
                                 AuxTiles   auxTiles,
                                 int accumID,
                                 SingleQueue readQueue,
                                 int numRays,
                                 SingleQueue writeQueue,
                                 int *d_nextWritePos,
                                 int generation
                                 )
    {
      int tid = rt.getThreadIdx().x + rt.getBlockIdx().x*rt.getBlockDim().x;
      if (tid >= numRays) return;

      Ray ray = readQueue.rays[tid];
      PathState state = readQueue.states[tid];
#ifdef NDEBUG
      enum { dbg = false };
#else
      bool dbg = ray.dbg();
#endif
      /* note(iw): IMHO pixels that did _not_ hit any geometry should
         have an alpha value of 0, even if they did comptue a 'color'
         from the env map. However, TSD blends its renderd image over
         a black background, in which case that blends away background
         pixels, so for now turn that off. The "right" fix for this
         would be for TSD to handle alpha properly. */
#define COMPUTE_PROPER_ALPHA_CHANNEL 1
      float alpha 
        = (generation == 0)
        ?
#if COMPUTE_PROPER_ALPHA_CHANNEL
        (ray.hadHit()? 1.f : ray.missColor.w)
#else
        1.f
#endif
        : 0.f;
      float incomingZ = ray.tMax;
      vec3f incomingN
        = ray.hadHit()
        ? ray.getN()
        : vec3f(0.f);

      // if (incomingN == vec3f(0.f))
      //   incomingN = vec3f(1.f,0.f,0.f);
      // what we'll add into the frame buffer
      vec3f fragment = 0.f;
      // create a (potential) shadow ray, and init to 'invalid'
      Ray shadowRay;
      PathState shadowState;
      shadowRay.tMax = -1.f;
      
      // bounce that ray on the scene, possibly generating a) a fragment
      // to add to frame buffer; b) a outgoing ray (in-place
      // modification of 'path'); and/or c) a shadow ray
      bounce(tid,
             world,renderer,
             fragment,
             ray,state,
             shadowRay,shadowState,
             generation);

#ifndef NDEBUG
      if (ray.crosshair && !dbg) {
        fragment = vec3f(1.f,0.f,0.f);
      }
#endif
      
      // write shadow and bounce ray(s), if any were generated
      if (dbg)
        printf("ray.tmax %f shadowray.tmax %f frag %f %f %f\n",
               ray.tMax,shadowRay.tMax,
               fragment.x,fragment.y,fragment.z);
      if (shadowRay.tMax > 0.f) {
        int pos = rt.atomicAdd(d_nextWritePos,1);
        writeQueue.rays[pos] = shadowRay;
        writeQueue.states[pos] = shadowState;
      }
      if (ray.tMax > 0.f) {
        int pos = rt.atomicAdd(d_nextWritePos,1);
        writeQueue.rays[pos] = ray;
        writeQueue.states[pos] = state;
      }

      // and write the shade fragment, if generated
      int tileID  = int(state.pixelID / pixelsPerTile);
      int tileOfs = int(state.pixelID % pixelsPerTile);
      vec4f &valueToAccumInto
        = accumTiles[tileID].accum[tileOfs];

#if DENOISE
      vec3f &valueToAccumNormalInto
        = accumTiles[tileID].normal[tileOfs];
#endif
      
      // ==================================================================
      // add to accum buffer. be careful of two things:
      //
      // a) since each pixel could have two DIFFERENT rays in the
      // queue (shadow ray and bounce ray) we cannot simply 'add', but
      // have to use an atomic add, because these could be in the same
      // warp.
      //
      // b) since we don't have an explicit frame buffer clear we
      // still have to make sure each pixel is written - not added -
      // exactly once in the first generation of the first frame.
      // ==================================================================

      // clamping ...
      float clampMax = 10.f*(1+accumID);
      fragment = min(fragment,vec3f(clampMax));
      
      if (accumID == 0 && generation == 0) {
        // first gen of first frame HAS to plain-write a value so later frames can
        valueToAccumInto = vec4f(fragment.x,fragment.y,
                                 fragment.z,alpha);
      
        // write aux buffers (depth, normal, hitIDs
        accumTiles[tileID].normal[tileOfs] = incomingN;
        if (auxTiles.depth) 
          auxTiles.depth[tileID] . f[tileOfs] = incomingZ;
        if (auxTiles.primID)
          auxTiles.primID[tileID].ui[tileOfs] = readQueue.hitIDs[tid].primID;
        if (auxTiles.objID)
          auxTiles.objID[tileID] .ui[tileOfs] = readQueue.hitIDs[tid].objID;
        if (auxTiles.instID)
          auxTiles.instID[tileID].ui[tileOfs] = readQueue.hitIDs[tid].instID;
        
      } else {
#if 1
        if (generation == 0) {
          if (alpha > 0.f) {
            rt.atomicAdd(&valueToAccumInto.w,alpha);
          }
          if (auxTiles.depth && incomingZ < auxTiles.depth[tileID] . f[tileOfs]) {
            auxTiles.depth[tileID] . f[tileOfs] = incomingZ;
          }
        }
#else
        if (generation == 0 && alpha > 0.f) 
          rt.atomicAdd(&valueToAccumInto.w,alpha);
#endif
        // we're either an accumulated frame, or a non-primary bounce
        // of the first frame; either way we'll accumulate color and
        // ignore anything else.

        if (fragment.x > 0.f)
          rt.atomicAdd(&valueToAccumInto.x,fragment.x);
        if (fragment.y > 0.f)
          rt.atomicAdd(&valueToAccumInto.y,fragment.y);
        if (fragment.z > 0.f)
          rt.atomicAdd(&valueToAccumInto.z,fragment.z);
      }
    }
#endif
  }  
  
  using namespace render;
  
  void Context::shadeRaysLocally(Renderer *renderer,
                                 GlobalModel *model,
                                 FrameBuffer *fb,
                                 int generation,
                                 uint32_t rngSeed)
  {
    int slotIdx = 0;
    for (auto slotModel : model->modelSlots) {
      World *world = slotModel->world.get();
      for (auto device : *world->devices) {
        SetActiveGPU forDuration(device);
        RayQueue *rayQueue = device->rayQueue;
        device->rayQueue->resetWriteQueue();
        
        TiledFB *devFB     = fb->getFor(device);
        int numRays        = rayQueue->numActive;
        if (numRays == 0) continue;
        int bs = 128;
        int nb = divRoundUp(numRays,bs);
        World::DD devWorld
          = world->getDD(device);
        Renderer::DD devRenderer
          = renderer->getDD(device);
        if (FromEnv::get()->logQueues) {
          std::stringstream ss;
          ss << "#bn" << myRank() << ": ## ray queue kernel SHADE " << std::endl
             << "  from " << rayQueue->traceAndShadeReadQueue.rays
             << " + " << rayQueue->traceAndShadeReadQueue.states << std::endl
             << "  to   " << rayQueue->receiveAndShadeWriteQueue.rays
             << " + " << rayQueue->receiveAndShadeWriteQueue.states << std::endl;
          std::cout << ss.str();
        }
        
        __rtc_launch(//device
                     device->rtc,
                     //kernel
                     _shadeRays,
                     //config
                     nb,bs,
                     //args
                     devWorld,devRenderer,
                     devFB->accumTiles,
                     devFB->auxTiles,
                     (int)fb->accumID,
                     rayQueue->traceAndShadeReadQueue,
                     numRays,
                     rayQueue->receiveAndShadeWriteQueue,
                     rayQueue->_d_nextWritePos,
                     generation
                     );
      }
      slotIdx++;
    }

    // ------------------------------------------------------------------
    // wait for kernel to complete, and swap queues
    // ------------------------------------------------------------------
    for (auto device : *devices) {
      SetActiveGPU forDuration(device);
      device->rtc->sync();
      device->rayQueue->swapAfterShade();
      device->rayQueue->numActive = device->rayQueue->readNumActive();
    }
  }
  
}

