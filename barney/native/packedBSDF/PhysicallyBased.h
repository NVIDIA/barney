// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "native/render/DG.h"
#include "native/render/HitAttributes.h"
#include "native/common/half.h"
#include <limits>

namespace BARNEY_NS {
  namespace native {
    namespace packedBSDF {
      struct PhysicallyBasedShadingState {
        PhysicallyBasedShadingState() = default;

        rtc::float3 baseColor;
        rtc::float3 normal;
        float metallic;
        float roughness;
        float transmission;
        float ior;
        rtc::float3 emissive;

        float occlusion;
        float specular;
        rtc::float3 specularColor;
        float clearcoat;
        float clearcoatRoughness;
        rtc::float3 clearcoatNormal;
        float thickness;
        float attenuationDistance;
        rtc::float3 attenuationColor;
        rtc::float3 sheenColor;
        float sheenRoughness;
        float iridescence;
        float iridescenceIor;
        float iridescenceThickness;
      };

      namespace physicallybased {
        // Clearcoat is fixed to IOR 1.5 per glTF KHR_materials_clearcoat (F0 = 0.04).
        // A macro (not constexpr) so it is usable from __device__ code under
        // -rdc=true; the pi constants come from native/common/math.h
        // (ONE_PI/TWO_PI/ONE_OVER_PI), which DG.h already pulls in.
#define CLEARCOAT_F0 0.04f

        inline __rtc_device float pow2(float v)
        {
          return v * v;
        }

        inline __rtc_device vec2f pow2(vec2f v)
        {
          return vec2f(v.x * v.x, v.y * v.y);
        }

        inline __rtc_device vec3f pow2(vec3f v)
        {
          return vec3f(v.x * v.x, v.y * v.y, v.z * v.z);
        }

        inline __rtc_device float pow5(float v)
        {
          return pow2(pow2(v)) * v;
        }

        inline __rtc_device vec2f pow5(vec2f v)
        {
          return pow2(pow2(v)) * v;
        }

        inline __rtc_device vec3f pow5(vec3f v)
        {
          return pow2(pow2(v)) * v;
        }

        inline __rtc_device vec2f exp(vec2f v)
        {
          return vec2f(expf(v.x), expf(v.y));
        }

        inline __rtc_device vec3f exp(vec3f v)
        {
          return vec3f(expf(v.x), expf(v.y), expf(v.z));
        }

        //-----------------------------------------------------------------------------
        // Helpers
        //-----------------------------------------------------------------------------

        // Transform a tangent-space normal (e.g. from a normal-map sampler,
        // already remapped to [-1,1]) into world space around the shading
        // normal `N`. This N-only overload is the fallback used when the
        // geometry carries no per-vertex tangent: the frame is built
        // deterministically from `N` (arbitrary but stable tangent
        // directions) — this preserves the perturbation magnitude, which is
        // what matters without UV-aligned tangents.
        inline __rtc_device vec3f
        applyNormalMap(const vec3f &tangentSpaceNormal, const vec3f &N)
        {
          const owl::common::LinearSpace3f f = owl::common::frame(N);
          const vec3f result = normalize(tangentSpaceNormal.x * f.vx
                                         + tangentSpaceNormal.y * f.vy
                                         + tangentSpaceNormal.z * f.vz);
          // Negated comparison catches NaN (zero-decoded texels) as well as
          // zero-length results — fall back to the shading normal so the
          // shading frame is always usable.
          return (dot(result, result) > 1e-12f) ? result : N;
        }

        // Transform a tangent-space normal into world space using a
        // UV-aligned TBN frame built from an interpolated per-vertex
        // tangent. `worldTangent.xyz` is the world-space tangent and
        // `worldTangent.w` its glTF handedness (+/-1); when w==0 no tangent
        // was supplied and we fall back to the deterministic N-only frame.
        inline __rtc_device vec3f
        applyNormalMap(const vec3f &tangentSpaceNormal,
                       const vec3f &N,
                       const vec4f &worldTangent)
        {
          if (worldTangent.w == 0.f)
            return applyNormalMap(tangentSpaceNormal, N);

          const vec3f T0 = (const vec3f&)worldTangent;
          // Gram-Schmidt: orthogonalize the tangent against the shading
          // normal, then derive the bitangent honoring handedness.
          const vec3f T = T0 - N * dot(N, T0);
          if (dot(T, T) <= 1e-12f)
            return applyNormalMap(tangentSpaceNormal, N);
          const vec3f Tn = normalize(T);
          const vec3f B = cross(N, Tn) * worldTangent.w;
          const vec3f result = normalize(tangentSpaceNormal.x * Tn
                                         + tangentSpaceNormal.y * B
                                         + tangentSpaceNormal.z * N);
          return (dot(result, result) > 1e-12f) ? result : N;
        }

        inline __rtc_device vec3f

        computeVolumeTransmission(const PhysicallyBasedShadingState *state)
        {
          if (!(state->thickness > 0.0f && state->attenuationDistance > 0.0f
                && isfinite(state->attenuationDistance)))
            return vec3f(1.0f);

          const float k = state->thickness / state->attenuationDistance;
          return vec3f(powf(fmaxf(state->attenuationColor.x, 1e-6f), k),
                       powf(fmaxf(state->attenuationColor.y, 1e-6f), k),
                       powf(fmaxf(state->attenuationColor.z, 1e-6f), k));
        }

        inline __rtc_device vec3f

        computeTransmissionFilter(const PhysicallyBasedShadingState *state)
        {
          const float transmission =
            fmaxf(0.0f, (1.0f - state->metallic) * state->transmission);
          return vec3f(
                   state->baseColor.x, state->baseColor.y, state->baseColor.z)
            * transmission * computeVolumeTransmission(state);
        }

        // Smith Lambda for GGX (common subterm of G1 / G2).
        inline __rtc_device float smithLambdaGGX(float NdotX, float alpha2)
        {
          const float NdotX2 = NdotX * NdotX;
          // Floor only guards the exactly-zero case; callers clamp cosines to
          // 1e-6, so x2 >= 1e-12 and Lambda still grows toward the grazing
          // singularity (G1 -> 0) instead of being capped at the alpha2 scale.
          const float safe = fmaxf(NdotX2, 1e-12f);
          return 0.5f
            * (-1.0f
               + sqrtf(fmaxf(0.0f, 1.0f + alpha2 * (1.0f - safe) / safe)));
        }

        inline __rtc_device float
        smithG2GGX(float NdotV, float NdotL, float alpha2)
        {
          return 1.0f
            / (1.0f + smithLambdaGGX(NdotV, alpha2)
               + smithLambdaGGX(NdotL, alpha2));
        }

        inline __rtc_device float smithG1GGX(float NdotV, float alpha2)
        {
          return 1.0f / (1.0f + smithLambdaGGX(NdotV, alpha2));
        }

        inline __rtc_device float ggxD(float NdotH, float alpha2)
        {
          // The textbook denom `x·(α²−1) + 1` cancels catastrophically in fp32 once
          // α² is below eps(1) ≈ 1.19e-7 (our α² floor is 1e-8): `α²−1` rounds to
          // exactly −1, and at x=1 the whole denom collapses to 0. The algebraically
          // equivalent `α²·x + (1−x)` has no near-1 subtraction so it stays exact.
          // The fminf clamp keeps `1−x ≥ 0` against dot-product rounding above 1.
          const float NdotH2 = fminf(NdotH * NdotH, 1.0f);
          const float denom = alpha2 * NdotH2 + (1.0f - NdotH2);
          return alpha2 / (ONE_PI * denom * denom);
        }

        // Heitz 2018 (https://jcgt.org/published/0007/04/01/) visible-normal sampling
        // for GGX. Ve is the view direction in local tangent space (+z = normal).
        inline __rtc_device vec3f sampleGGXVNDF(const vec3f &Ve,
                                                float alpha,
                                                float u1,
                                                float u2)
        {
          const vec3f Vh = normalize(vec3f(alpha * Ve.x, alpha * Ve.y, Ve.z));
          const float lensq = Vh.x * Vh.x + Vh.y * Vh.y;
          const vec3f T1 = lensq > 0.0f
            ? vec3f(-Vh.y, Vh.x, 0.0f) * (1.0f / sqrtf(lensq))
            : vec3f(1.0f, 0.0f, 0.0f);
          const vec3f T2 = cross(Vh, T1);
          const float r = sqrtf(u1);
          const float phi = TWO_PI * u2;
          const float t1 = r * cosf(phi);
          float t2 = r * sinf(phi);
          const float s = 0.5f * (1.0f + Vh.z);
          t2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - t1 * t1)) + s * t2;
          const vec3f Nh = t1 * T1 + t2 * T2
            + sqrtf(fmaxf(0.0f, 1.0f - t1 * t1 - t2 * t2)) * Vh;
          return normalize(
            vec3f(alpha * Nh.x, alpha * Nh.y, fmaxf(0.0f, Nh.z)));
        }

        // Charlie distribution (Estevez-Kulla 2017) for sheen.
        inline __rtc_device float charlieD(float NdotH, float alpha)
        {
          const float invAlpha = 1.0f / fmaxf(alpha, 1e-4f);
          const float sin2 = fmaxf(0.0f, 1.0f - NdotH * NdotH);
          return (2.0f + invAlpha) * powf(sin2, 0.5f * invAlpha) / (TWO_PI);
        }

        // Ashikhmin visibility term (Neubelt-Pettineo variant) used with Charlie D.
        inline __rtc_device float charlieV(float NdotV, float NdotL)
        {
          return 1.0f / (4.0f * (NdotV + NdotL - NdotV * NdotL) + 1e-6f);
        }

        // glTF KHR_materials_iridescence thin-film Fresnel (port of the reference
        // implementation at github.com/KhronosGroup/glTF-Sample-Renderer). Returns a
        // per-channel Fresnel reflectance for a thin film of thickness T sitting on a
        // base with Schlick F0. See the spec's Appendix B for the math.
        inline __rtc_device vec3f fresnel0ToIor(vec3f F0)
        {
          const vec3f s = sqrt(clamp(F0, vec3f(0.0f), vec3f(0.9999f)));
          return (vec3f(1.0f) + s) / (vec3f(1.0f) - s);
        }

        inline __rtc_device vec3f iorToFresnel0(vec3f transmittedIor,
                                                float incidentIor)
        {
          const vec3f t = (transmittedIor - vec3f(incidentIor))
            / (transmittedIor + vec3f(incidentIor));
          return t * t;
        }

        inline __rtc_device float iorToFresnel0(float transmittedIor,
                                                float incidentIor)
        {
          const float t =
            (transmittedIor - incidentIor) / (transmittedIor + incidentIor);
          return t * t;
        }

        inline __rtc_device vec3f evalSensitivity(float opd, vec3f shift)
        {
          // Approximate spectral sensitivity of the standard observer as three
          // Gaussians (Belcour & Barla 2017, simplified) so the result stays in RGB.
          const float phase = TWO_PI * opd * 1e-9f;
          const vec3f val = vec3f(5.4856e-13f, 4.4201e-13f, 5.2481e-13f);
          const vec3f pos = vec3f(1.6810e+06f, 1.7953e+06f, 2.2084e+06f);
          const vec3f var = vec3f(4.3278e+09f, 9.3046e+09f, 6.6121e+09f);

          vec3f xyz = val * sqrt(TWO_PI * var) * cos(pos * phase + shift)
            * exp(-var * phase * phase);
          xyz.x += 9.7470e-14f * sqrtf(TWO_PI * 4.5282e+09f)
            * cosf(2.2399e+06f * phase + shift.x)
            * expf(-4.5282e+09f * phase * phase);
          xyz /= 1.0685e-7f;

          // sRGB conversion (D65).
          return vec3f(
            3.2404542f * xyz.x - 1.5371385f * xyz.y - 0.4985314f * xyz.z,
            -0.9692660f * xyz.x + 1.8760108f * xyz.y + 0.0415560f * xyz.z,
            0.0556434f * xyz.x - 0.2040259f * xyz.y + 1.0572252f * xyz.z);
        }

        inline __rtc_device vec3f evalIridescence(float outsideIor,
                                                  float iridescenceIor,
                                                  float cosTheta1,
                                                  float thickness,
                                                  vec3f baseF0)
        {
          // Handle the case where thin-film IOR is close to the outside IOR: return
          // the base Fresnel to avoid division by zero and phase artifacts.
          const float iridescenceIorSafe =
            fmaxf(iridescenceIor, outsideIor + 1e-4f);

          // Force iridescenceIor > outsideIor (otherwise Snell's law cannot refract).
          const float sinTheta2Sq = pow2(outsideIor / iridescenceIorSafe)
            * (1.0f - cosTheta1 * cosTheta1);
          const float cosTheta2Sq = 1.0f - sinTheta2Sq;
          if (cosTheta2Sq < 0.0f)
            return vec3f(1.0f); // Total internal reflection.
          const float cosTheta2 = sqrtf(cosTheta2Sq);

          // First interface: Fresnel between outside and thin film.
          const float R0_12 = iorToFresnel0(iridescenceIorSafe, outsideIor);
          const float R12 = R0_12 + (1.0f - R0_12) * pow5(1.0f - cosTheta1);
          const float T121 = 1.0f - R12;
          const float phi12 = iridescenceIorSafe < outsideIor ? ONE_PI : 0.0f;
          const float phi21 = ONE_PI - phi12;

          // Second interface: film to base.
          const vec3f baseIor =
            fresnel0ToIor(clamp(baseF0, vec3f(0.f), vec3f(0.9999f)));
          const vec3f R1 = iorToFresnel0(baseIor, iridescenceIorSafe);
          const vec3f R23 = R1 + (vec3f(1.0f) - R1) * pow5(1.0f - cosTheta2);
          const vec3f phi23 =
            vec3f(baseIor.x < iridescenceIorSafe ? ONE_PI : 0.0f,
                  baseIor.y < iridescenceIorSafe ? ONE_PI : 0.0f,
                  baseIor.z < iridescenceIorSafe ? ONE_PI : 0.0f);

          const float opd = 2.0f * iridescenceIorSafe * thickness * cosTheta2;
          const vec3f phi = vec3f(phi21) + phi23;

          const vec3f R123 = clamp(R12 * R23, vec3f(1e-5f), vec3f(0.9999f));
          const vec3f r123 = sqrt(R123);
          const vec3f Rs = pow2(T121) * R23 / (vec3f(1.0f) - R123);

          // DC term.
          vec3f C0 = R12 + Rs;
          vec3f I = C0;

          // Higher-order terms.
          vec3f Cm = Rs - T121;
          for (int m = 1; m <= 2; ++m) {
            Cm *= r123;
            const vec3f Sm =
              2.0f * evalSensitivity(float(m) * opd, float(m) * phi);
            I += Cm * Sm;
          }

          return max(I, vec3f(0.0f));
        }

        //-----------------------------------------------------------------------------
        // glTF PBR Fresnel / BRDF helpers (port of VisRTX shadeSurface &
        // pbrBsdfPdf). `eta` is n_i/n_t, computed by the caller from the shading
        // state's `ior` and which side of the surface the ray is on — the barney
        // shading state stores `ior` but not the front/back-relative `eta` that
        // VisRTX bakes into its shading state at hit time.
        //-----------------------------------------------------------------------------

        inline __rtc_device vec3f mix(vec3f a, vec3f b, float t)
        {
          return a + (b - a) * t;
        }

        inline __rtc_device vec3f mix(vec3f a, vec3f b, vec3f t)
        {
          return a + (b - a) * t;
        }

        inline __rtc_device vec3f
        computeDielectricF0(const PhysicallyBasedShadingState *state, float eta)
        {
          // KHR_materials_specular: `specular` scales the dielectric reflection,
          // so the ANARI default specular = 0 yields a pure-diffuse dielectric.
          const float iorF0 = pow2((1.0f - eta) / (1.0f + eta));
          return min(vec3f(iorF0) * rtc::load(state->specularColor),
                     vec3f(1.0f))
            * state->specular;
        }

        inline __rtc_device vec3f
        computeF0(const PhysicallyBasedShadingState *state, float eta)
        {
          return mix(computeDielectricF0(state, eta),
                     rtc::load(state->baseColor),
                     state->metallic);
        }

        inline __rtc_device vec3f
        computeF90(const PhysicallyBasedShadingState *state)
        {
          return mix(vec3f(state->ior == 1.f ? 0.f : state->specular),
                     vec3f(1.0f), state->metallic);
        }

        inline __rtc_device vec3f schlickFresnel(vec3f F0, vec3f F90, float VdotH)
        {
          return F0 + (F90 - F0) * pow5(1.0f - fabsf(VdotH));
        }

        inline __rtc_device vec3f evalFresnelWithIridescence(
          const PhysicallyBasedShadingState *state,
          const vec3f &F0,
          const vec3f &F90,
          float cosTheta)
        {
          vec3f F = schlickFresnel(F0, F90, cosTheta);
          if (state->iridescence > 0.0f && state->iridescenceThickness > 0.0f) {
            const vec3f iridescent = evalIridescence(1.0f,
                                                     state->iridescenceIor,
                                                     cosTheta,
                                                     state->iridescenceThickness,
                                                     F0);
            F = mix(F, iridescent, state->iridescence);
          }
          return F;
        }

        // Value and marginal density must share the sharp GGX factors. Even
        // one ULP in independently reconstructed half-vectors can prevent D
        // from cancelling in value/pdf at the roughness floor.
        inline __rtc_device EvalRes
        pbrEvaluate(const PhysicallyBasedShadingState *state,
                    float eta,
                    const vec3f &N,
                    const vec3f &Nc,
                    const vec3f &V,
                    const vec3f &L,
                    bool reflection,
                    bool projected)
        {
          const float NdotV = dot(N, V);
          const float NdotL = dot(N, L);
          const bool baseSupported = NdotV > 0.f
            && (reflection ? NdotL > 0.f : NdotL < 0.f);
          if (!reflection && (eta == 1.f || !baseSupported))
            return EvalRes::zero();

          vec3f Q = V + L;
          if (!reflection) {
            // Explicit contraction keeps reconstruction identical when pdf()
            // inlining eliminates the value-only work from this evaluator.
            Q = vec3f(fmaf(eta, V.x, L.x),
                      fmaf(eta, V.y, L.y),
                      fmaf(eta, V.z, L.z));
          }
          const float length2 = dot(Q, Q);
          if (!(length2 > 0.f) || !isfinite(length2))
            return EvalRes::zero();
          vec3f H = Q / sqrtf(length2);
          if (!reflection && dot(H, N) < 0.f) H = -H;
          const float VdotH = dot(V, H);
          const float LdotH = dot(L, H);
          if (!(VdotH > 0.f) || (!reflection && !(LdotH < 0.f)))
            return EvalRes::zero();

          const vec3f F0 = computeF0(state, eta);
          const vec3f F90 = computeF90(state);
          const vec3f Fv = evalFresnelWithIridescence(state, F0, F90, NdotV);
          const vec3f transmissionFilter = computeTransmissionFilter(state);
          vec3f Fh = evalFresnelWithIridescence(state, F0, F90, VdotH);
          const vec3f Ltrans = refract(neg(V), H, eta);
          // Sampling folds all transmission energy into reflection at TIR.
          const bool tir = luminance(transmissionFilter) > 0.f
            && dot(Ltrans, Ltrans) == 0.f;
          if (tir) Fh = vec3f(1.f);
          const vec3f transmitEnergy =
            max(vec3f(1.f) - Fh, vec3f(0.f)) * transmissionFilter;
          const float reflW = fmaxf(luminance(Fh), 0.f);
          const float transW = fmaxf(luminance(transmitEnergy), 0.f);
          const float reflectGivenSpec =
            reflW + transW > 0.f ? reflW / (reflW + transW) : 1.f;

          const float specSelW = fmaxf(luminance(Fv), 0.f)
            + fmaxf(luminance(max(vec3f(1.f) - Fv, vec3f(0.f))
                              * transmissionFilter), 0.f);
          // Sheen has no dedicated proposal; keep this diffuse selection
          // weight in sync with scatter().
          const float diffSelW =
            fmaxf(luminance(max(vec3f(1.f) - Fv, vec3f(0.f))
                            * rtc::load(state->baseColor)
                            * (1.f - state->metallic)
                            * (1.f - state->transmission)
                            * state->occlusion), 0.f)
            + fmaxf(luminance(rtc::load(state->sheenColor)), 0.f);
          const float baseSel = specSelW + diffSelW;

          EvalRes result;
          if (baseSupported) {
            const float nv = fmaxf(NdotV, 1e-6f);
            const float nl = fmaxf(fabsf(NdotL), 1e-6f);
            const float NdotH = fmaxf(dot(N, H), 0.f);
            const float alpha = fmaxf(pow2(state->roughness), 1e-4f);
            const float alpha2 = alpha * alpha;
            const float D = ggxD(NdotH, alpha2);
            const float lambdaV = smithLambdaGGX(nv, alpha2);
            const float G1 = 1.f / (1.f + lambdaV);
            const float G2 = 1.f / (1.f + lambdaV + smithLambdaGGX(nl, alpha2));
            const float pSpec = baseSel > 0.f ? specSelW / baseSel : 0.f;
            const float pDiff = baseSel > 0.f ? diffSelW / baseSel : 0.f;

            if (reflection) {
              result.pdf = pDiff * NdotL * ONE_OVER_PI
                + pSpec * reflectGivenSpec * D * G1 / (4.f * nv);
              const vec3f Fdiff = evalFresnelWithIridescence(state, F0, F90, nv);
              const vec3f diffuseColor =
                mix(rtc::load(state->baseColor), vec3f(0.f), state->metallic);
              result.value = (vec3f(1.f) - Fdiff) * ONE_OVER_PI * diffuseColor
                * state->occlusion * (1.f - state->transmission)
                + Fh * D * G2 / (4.f * nv * nl);

              if (state->sheenColor.x > 0.f || state->sheenColor.y > 0.f
                  || state->sheenColor.z > 0.f) {
                const float alphaS = fmaxf(pow2(state->sheenRoughness), 1e-4f);
                const vec3f sheenColor = rtc::load(state->sheenColor);
                const float sheenComp = 1.f - 0.157f
                  * fmaxf(sheenColor.x, fmaxf(sheenColor.y, sheenColor.z));
                result.value = result.value * sheenComp
                  + sheenColor * charlieD(NdotH, alphaS) * charlieV(nv, nl);
              }
              if (projected) result.value *= NdotL;
            } else {
              const float denom = eta * VdotH + LdotH;
              if (denom == 0.f) return EvalRes::zero();
              // VNDF density times the refractive half-vector Jacobian.
              const float common = D * VdotH * fabsf(LdotH)
                / (denom * denom);
              result.pdf = pSpec * (1.f - reflectGivenSpec) * common * G1 / nv;
              result.value = transmitEnergy
                * (common * G2 * eta * eta / NdotV);
              if (!projected) result.value /= -NdotL;
            }
          }

          const float NcDotV = fmaxf(dot(Nc, V), 0.f);
          const float NcDotL = dot(Nc, L);
          const float FcV = CLEARCOAT_F0
            + (1.f - CLEARCOAT_F0) * pow5(1.f - NcDotV);
          const float ccProb = saturate(state->clearcoat * FcV);
          result.pdf *= 1.f - ccProb;
          if (!reflection) {
            const float FcL = CLEARCOAT_F0
              + (1.f - CLEARCOAT_F0) * pow5(1.f - fabsf(NcDotL));
            result.value *= (1.f - state->clearcoat * FcV)
              * (1.f - state->clearcoat * FcL);
          } else if (state->clearcoat > 0.f) {
            const float cv = fmaxf(NcDotV, 1e-6f);
            const float FcVSafe = CLEARCOAT_F0
              + (1.f - CLEARCOAT_F0) * pow5(1.f - cv);
            result.value *= 1.f - state->clearcoat * FcVSafe;
            // Coat support and projection are independent of the base normal.
            if (NcDotV > 0.f && NcDotL > 0.f) {
              const float cl = fmaxf(NcDotL, 1e-6f);
              const float NcDotH = fmaxf(dot(Nc, H), 0.f);
              const float alphaC = fmaxf(pow2(state->clearcoatRoughness), 1e-4f);
              const float alphaC2 = alphaC * alphaC;
              const float Dc = ggxD(NcDotH, alphaC2);
              const float lambdaC = smithLambdaGGX(cv, alphaC2);
              const float G1c = 1.f / (1.f + lambdaC);
              const float G2c = 1.f
                / (1.f + lambdaC + smithLambdaGGX(cl, alphaC2));
              const float FcH = CLEARCOAT_F0
                + (1.f - CLEARCOAT_F0) * pow5(1.f - VdotH);
              result.value += vec3f(state->clearcoat * FcH * Dc * G2c
                                   / (4.f * cv * cl))
                * (projected ? NcDotL : 1.f);
              if (NcDotV > 1e-6f)
                result.pdf += ccProb * Dc * G1c / (4.f * cv);
            }
          }
          return result;
        }

        //-----------------------------------------------------------------------------
        // Initialize shading state from material parameters
        //-----------------------------------------------------------------------------

      } // namespace physicallybased

      // glTF-style physically-based material BSDF: diffuse + GGX specular
      // (metallic/roughness), plus clearcoat, sheen, iridescence and
      // transmission lobes; see setDefaults() for the parameter set.
      //
      // This struct rides in the ray payload (PackedBSDF::Data ->
      // Ray::hitBSDF), so parameters are packed in half precision to keep
      // the payload registers and traffic low; unpack() widens them into
      // the float PhysicallyBasedShadingState the eval/pdf/scatter math
      // runs on. `emissive` stays float so HDR emission is not clamped to
      // the half range.
      struct PhysicallyBased {
        inline PhysicallyBased() = default;

        rtc::float3 emissive;
        vec3h baseColor;
        vec3h normal;
        half  metallic;
        half  roughness;
        half  transmission;
        half  ior;

        half  occlusion;
        half  specular;
        vec3h specularColor;
        half  clearcoat;
        half  clearcoatRoughness;
        vec3h clearcoatNormal;
        half  thickness;
        half  attenuationDistance;
        vec3h attenuationColor;
        vec3h sheenColor;
        half  sheenRoughness;
        half  iridescence;
        half  iridescenceIor;
        half  iridescenceThickness;

        inline __rtc_device PhysicallyBasedShadingState unpack() const
        {
          PhysicallyBasedShadingState st;
          st.emissive = rtc::float3(emissive.x, emissive.y, emissive.z);
          const vec3f bc = (vec3f)baseColor;
          st.baseColor = rtc::float3(bc.x, bc.y, bc.z);
          // Half packing changes length; GGX and its sampling frame require
          // unit normals, especially when the specular lobe is very narrow.
          const vec3f n = normalize((vec3f)normal);
          st.normal = rtc::float3(n.x, n.y, n.z);
          st.metallic = metallic;
          st.roughness = roughness;
          st.transmission = transmission;
          st.ior = ior;
          st.occlusion = occlusion;
          st.specular = specular;
          const vec3f sc = (vec3f)specularColor;
          st.specularColor = rtc::float3(sc.x, sc.y, sc.z);
          st.clearcoat = clearcoat;
          st.clearcoatRoughness = clearcoatRoughness;
          const vec3f ccn = normalize((vec3f)clearcoatNormal);
          st.clearcoatNormal = rtc::float3(ccn.x, ccn.y, ccn.z);
          st.thickness = thickness;
          st.attenuationDistance = attenuationDistance;
          const vec3f ac = (vec3f)attenuationColor;
          st.attenuationColor = rtc::float3(ac.x, ac.y, ac.z);
          const vec3f shc = (vec3f)sheenColor;
          st.sheenColor = rtc::float3(shc.x, shc.y, shc.z);
          st.sheenRoughness = sheenRoughness;
          st.iridescence = iridescence;
          st.iridescenceIor = iridescenceIor;
          st.iridescenceThickness = iridescenceThickness;
          return st;
        }

        inline __rtc_device void setDefaults()
        {
          baseColor = vec3f(1.0f, 1.0f, 1.0f);
          metallic = 1.0f;
          roughness = 1.0f;
          normal = vec3f(0.0f, 0.0f, 1.0f);
          emissive = rtc::float3(0.0f, 0.0f, 0.0f);
          occlusion = 1.0f;
          specular = 0.0f;
          specularColor = vec3f(1.0f, 1.0f, 1.0f);
          clearcoat = 0.0f;
          clearcoatRoughness = 0.0f;
          clearcoatNormal = vec3f(0.0f, 0.0f, 1.0f);
          transmission = 0.0f;
          ior = 1.5f;
          thickness = 0.0f;
          attenuationDistance = std::numeric_limits<float>::infinity();
          attenuationColor = vec3f(1.0f, 1.0f, 1.0f);
          sheenColor = vec3f(0.0f, 0.0f, 0.0f);
          sheenRoughness = 0.0f;
          iridescence = 0.0f;
          iridescenceIor = 1.3f;
          iridescenceThickness = 0.0f;
        }

        // Density in solid angle, excluding discrete index-matched transmission.
        inline __rtc_device float pdf(DG dg, const vec3f &wi, bool dbg) const
        {
          return evaluate(dg,wi,dbg,true).pdf;
        }

        inline __rtc_device EvalRes evalProjected(DG dg, const vec3f &wi, bool dbg) const
        { return evaluate(dg,wi,dbg,true); }

        inline __rtc_device EvalRes eval(DG dg, const vec3f &wi, bool dbg) const
        { return evaluate(dg,wi,dbg,false); }

        inline __rtc_device EvalRes evaluate(DG dg, const vec3f &wi, bool dbg,
                                              bool projected) const
        {
          const PhysicallyBasedShadingState st = unpack();
          const vec3f V = dg.wo;
          const vec3f N = dg.orientShadingNormal(rtc::load(st.normal));
          const float eta = dg.insideMedium ? st.ior : (1.0f / st.ior);
          const vec3f Nc = dg.orientShadingNormal(rtc::load(st.clearcoatNormal));
          const float side = dot(dg.Ng,wi)*dot(dg.Ng,V);
          if (side == 0.f) return EvalRes::zero();
          return physicallybased::pbrEvaluate(&st, eta, N, Nc, V, wi,
                                               side > 0.f, projected);
        }

        // Importance-sample the BSDF (port of VisRTX nextRay): clearcoat lobe
        // picked with probability clearcoat*FcV, then a V-only diffuse/specular
        // split, then a Fresnel-driven reflect/transmit split inside specular.
        //
        // Every continuous sample carries full projected evaluation divided
        // by the marginal density, including rough transmission. Index-matched
        // refraction is the only discrete event in this roughness-floor model.
        inline __rtc_device void scatter(ScatterResult &scatter,
                                         const DG &dg,
                                         Random &random,
                                         bool dbg) const
        {
          scatter = {};
          const PhysicallyBasedShadingState st = unpack();
          const vec3f V = dg.wo;
          const vec3f N = dg.orientShadingNormal(rtc::load(st.normal));
          const vec3f Nc = dg.orientShadingNormal(rtc::load(st.clearcoatNormal));
          const float eta = dg.insideMedium ? st.ior : (1.0f / st.ior);

          // Clearcoat lobe: pick with probability clearcoat*FcV(NcDotV). This
          // weight makes the entry-side attenuation cancel the (1-pick) divisor
          // of the base path (see VisRTX nextRay).
          const float NcDotV_world = fmaxf(dot(Nc, V), 0.0f);
          const float FcV_world = CLEARCOAT_F0
            + (1.0f - CLEARCOAT_F0)
              * physicallybased::pow5(1.0f - NcDotV_world);
          const float clearcoatPick = saturate(st.clearcoat * FcV_world);

          auto finish = [&](vec3f L, ScatterResult::Type type, bool transmission=false) {
            const float side = dot(dg.Ng,L)*dot(dg.Ng,V);
            if (transmission ? side >= 0.f : side <= 0.f) return;
            const EvalRes e = evalProjected(dg,L,dbg);
            if (!(e.pdf > 0.f) || !isfinite(e.pdf)) return;
            scatter.dir = L;
            scatter.pdf = e.pdf;
            scatter.weight = e.value/e.pdf;
            scatter.type = type;
            scatter.event = dot(dg.Ng,L)*dot(dg.Ng,V) > 0.f ? ScatterResult::REFLECTION : ScatterResult::TRANSMISSION;
            scatter.eta = scatter.event == ScatterResult::TRANSMISSION ? eta : 1.f;
          };

          if (clearcoatPick > 0.0f && random() < clearcoatPick) {
            const owl::common::LinearSpace3f toWorldC = owl::common::frame(Nc);
            const vec3f VlocalC =
              owl::common::xfmVector(toWorldC.transposed(), V);
            if (VlocalC.z <= 0.0f) {
              scatter.type = ScatterResult::NONE;
              return;
            }
            const float alphaC =
              fmaxf(physicallybased::pow2(st.clearcoatRoughness), 1e-4f);
            const vec3f HlocalC = physicallybased::sampleGGXVNDF(
              VlocalC, alphaC, random(), random());
            const vec3f LlocalC = reflect(neg(VlocalC), HlocalC);
            if (LlocalC.z <= 0.0f) {
              scatter.type = ScatterResult::NONE;
              return;
            }
            const vec3f Lworld =
              normalize(owl::common::xfmVector(toWorldC, LlocalC));
            finish(Lworld,ScatterResult::GLOSSY);
            return;
          }

          // Base path: diffuse vs specular, split V-only so the sampling
          // density is the closed form pbrEvaluate evaluates.
          const owl::common::LinearSpace3f toWorld = owl::common::frame(N);
          const vec3f Vlocal = owl::common::xfmVector(toWorld.transposed(), V);
          if (Vlocal.z <= 0.0f) {
            scatter.type = ScatterResult::NONE;
            return;
          }
          const float NdotV = Vlocal.z;

          const vec3f F0 = physicallybased::computeF0(&st, eta);
          const vec3f F90 = physicallybased::computeF90(&st);
          const vec3f Fv =
            physicallybased::evalFresnelWithIridescence(&st, F0, F90, NdotV);
          const vec3f transmissionFilter =
            physicallybased::computeTransmissionFilter(&st);
          const bool hasTransmission = luminance(transmissionFilter) > 0.0f;

          const float specSelW = fmaxf(luminance(Fv), 0.0f)
            + fmaxf(luminance(max(vec3f(1.0f) - Fv, vec3f(0.0f))
                              * transmissionFilter),
                    0.0f);
          const vec3f diffuseEnergy = max(vec3f(1.0f) - Fv, vec3f(0.0f))
            * rtc::load(st.baseColor) * (1.0f - st.metallic)
            * (1.0f - st.transmission) * st.occlusion;
          // Sheen is folded into the diffuse selection weight (see
          // pbrEvaluate); keep the two computations in sync.
          const float sheenSelW
            = fmaxf(luminance(rtc::load(st.sheenColor)), 0.0f);
          const float diffSelW = fmaxf(luminance(diffuseEnergy), 0.0f)
            + sheenSelW;
          const float baseSel = specSelW + diffSelW;
          if (baseSel <= 0.0f) {
            scatter.type = ScatterResult::NONE;
            return;
          }
          const float pSpec = specSelW / baseSel;

          // Diffuse lobe: cosine-weighted around the shading normal.
          if (random() >= pSpec) {
            const vec2f s(random(), random());
            const vec3f localDir = cosineSampleHemisphere(s);
            const vec3f Lworld =
              normalize(owl::common::xfmVector(toWorld, localDir));
            finish(Lworld,ScatterResult::DIFFUSE);
            return;
          }

          // Specular lobe: VNDF-sample the microfacet, then split
          // reflect/transmit by the microfacet Fresnel F(VdotH). TIR folds all
          // energy into reflection.
          const float alpha = fmaxf(physicallybased::pow2(st.roughness), 1e-4f);
          const vec3f Hlocal =
            physicallybased::sampleGGXVNDF(Vlocal, alpha, random(), random());
          const float VdotH = fmaxf(dot(Vlocal, Hlocal), 0.0f);
          const vec3f Fh =
            physicallybased::evalFresnelWithIridescence(&st, F0, F90, VdotH);

          const vec3f Lrefl = reflect(neg(Vlocal), Hlocal);
          const vec3f Ltrans = refract(neg(Vlocal), Hlocal, eta);
          const bool tir =
            hasTransmission && dot(Ltrans,Ltrans) == 0.f;
          const vec3f reflectEnergy = tir ? vec3f(1.0f) : Fh;
          const vec3f transmitEnergy =
            tir ? vec3f(0.0f)
                : max(vec3f(1.0f) - Fh, vec3f(0.0f)) * transmissionFilter;
          const float reflW = fmaxf(luminance(reflectEnergy), 0.0f);
          const float transW = fmaxf(luminance(transmitEnergy), 0.0f);
          const float specTotal = reflW + transW;
          const float reflectGivenSpec =
            specTotal > 0.0f ? reflW / specTotal : 1.0f;

          if (random() < reflectGivenSpec) {
            if (Lrefl.z <= 0.0f) {
              scatter.type = ScatterResult::NONE;
              return;
            }
            const vec3f Lworld =
              normalize(owl::common::xfmVector(toWorld, Lrefl));
            finish(Lworld,ScatterResult::GLOSSY);
            return;
          }

          const vec3f Ltworld =
            normalize(owl::common::xfmVector(toWorld, Ltrans));
          if (eta == 1.f) {
            // Index matching collapses refraction to a discrete straight ray,
            // independently of the microfacet roughness.
            scatter.dir = -V;
            const float cl = fabsf(dot(Nc,-V));
            const float exitAttenuation = 1.f-st.clearcoat*(CLEARCOAT_F0
              +(1.f-CLEARCOAT_F0)*physicallybased::pow5(1.f-cl));
            scatter.weight = transmitEnergy*exitAttenuation
              / (pSpec*(1.f-reflectGivenSpec));
            scatter.delta = true;
            scatter.event = ScatterResult::TRANSMISSION;
            scatter.type = ScatterResult::SPECULAR;
          } else {
            finish(Ltworld,ScatterResult::GLOSSY,true);
          }
        }
      };
    }
  }
}
