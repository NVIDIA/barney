/*
 * Copyright (c) 2019-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

// Adapted from NVIDIA/VisRTX ComputeTangent.cu at
// 0e793258d227ceef1aee10f7c9f559ae01e6a05f. Compatible position/normal/UV/sign
// buckets preserve soup welding and orientation seams instead of vertex IDs.
#include "native/geometry/TangentFrames.h"
#include <climits>

namespace BARNEY_NS {
namespace native {
namespace tangent_detail {

struct Corner
{
  vec3f position, normal;
  vec2f uv;
  float sign, weight;
  int bucket;
};
struct FaceFrame { vec3f t, b; };

#if RTC_DEVICE_CODE
inline __rtc_device bool finite3(vec3f v)
{
  return isfinite(v.x) && isfinite(v.y) && isfinite(v.z);
}
inline __rtc_device vec3f safeNormal(vec3f v, vec3f fallback)
{
  const float l2 = dot(v,v);
  return finite3(v) && isfinite(l2) && l2 > 1e-30f ? v / sqrtf(l2) : fallback;
}
inline __rtc_device vec3f fallbackTangent(vec3f n)
{
  const vec3f axis = fabsf(n.x) <= fabsf(n.y) && fabsf(n.x) <= fabsf(n.z)
      ? vec3f(1,0,0) : (fabsf(n.y) <= fabsf(n.z) ? vec3f(0,1,0) : vec3f(0,0,1));
  return safeNormal(axis - n*dot(n,axis), vec3f(1,0,0));
}
inline __rtc_device int components(int type)
{
  switch (type) {
#define INTEGER_WIDTHS(T) \
  case BN_##T##_VEC2: return 2; \
  case BN_##T##_VEC3: return 3; \
  case BN_##T##_VEC4: return 4;
  INTEGER_WIDTHS(INT8)
  INTEGER_WIDTHS(UINT8)
  INTEGER_WIDTHS(INT32)
  INTEGER_WIDTHS(UINT32)
  INTEGER_WIDTHS(INT64)
  INTEGER_WIDTHS(UINT64)
#undef INTEGER_WIDTHS
  case BN_FLOAT2: case BN_FLOAT16_VEC2:
  case BN_UFIXED8_VEC2: case BN_UFIXED16_VEC2: return 2;
  case BN_FLOAT3: case BN_FLOAT16_VEC3: return 3;
  case BN_FLOAT4: case BN_FLOAT16_VEC4: case BN_UFIXED8_RGBA: return 4;
  default: return 0;
  }
}
inline __rtc_device vec4f attribute(const GeometryAttribute::DD &a,
                                    int corner, int vertex, int width,
                                    vec4f fallback)
{
  const int i = a.scope == GeometryAttribute::FACE_VARYING ? corner : vertex;
  if ((a.scope != GeometryAttribute::FACE_VARYING && a.scope != GeometryAttribute::PER_VERTEX)
      || components(a.fromArray.type) < width || i < 0 || i >= a.fromArray.size)
    return fallback;
  // Native PODData also accepts integer arrays; typedRead's material-facing
  // formats are narrower. Preserve their numerical (not UNORM) conversion.
  const void *ptr = a.fromArray.ptr;
  if (!ptr) return fallback;
  const int n = components(a.fromArray.type);
  switch (a.fromArray.type) {
#define INTEGER_READ(BN, T) \
  case BN_##BN##_VEC2: case BN_##BN##_VEC3: case BN_##BN##_VEC4: { \
    const T *p = static_cast<const T *>(ptr) + size_t(i)*n; \
    return vec4f(float(p[0]),float(p[1]),n > 2 ? float(p[2]) : 0.f, \
                 n > 3 ? float(p[3]) : 1.f); \
  }
  INTEGER_READ(INT8,int8_t)
  INTEGER_READ(UINT8,uint8_t)
  INTEGER_READ(INT32,int32_t)
  INTEGER_READ(UINT32,uint32_t)
  INTEGER_READ(INT64,int64_t)
  INTEGER_READ(UINT64,uint64_t)
#undef INTEGER_READ
  default: return a.fromArray.valueAt(i);
  }
}
inline __rtc_device uint32_t mix(uint32_t x)
{
  x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15;
  x *= 0x846ca68bu; return x ^ (x >> 16);
}
inline __rtc_device uint32_t bits(float x)
{
  // Float equality welds +0 and -0; their hash must agree too.
  if (x == 0.f) return 0;
  uint32_t result;
  memcpy(&result, &x, sizeof(result));
  return result;
}
inline __rtc_device int bucketFor(const Corner &c, int count)
{
  const float key[] = {c.position.x,c.position.y,c.position.z,
    c.normal.x,c.normal.y,c.normal.z,c.uv.x,c.uv.y,c.sign};
  uint32_t h = 0;
  for (float v : key) h = mix(h ^ bits(v));
  return h % uint32_t(count);
}
inline __rtc_device bool compatible(const Corner &a, const Corner &b)
{
  return a.position == b.position && a.normal == b.normal
      && a.uv == b.uv && a.sign == b.sign;
}
#endif

__rtc_global void clearCounts(const rtc::ComputeInterface &ci, int *counts, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i < n) counts[i] = 0;
}
#endif

__rtc_global void computeFaces(const rtc::ComputeInterface &ci, TangentMesh mesh,
                              FaceFrame *faces, Corner *corners, int *counts)
#if !RTC_DEVICE_CODE
;
#else
{
  const int f = ci.launchIndex().x;
  if (f >= mesh.numTriangles) return;
  const vec3i idx = mesh.indices ? mesh.indices[f] : vec3i(3*f,3*f+1,3*f+2);
  vec3f p[3];
  vec2f uv[3];
  bool valid = true;
  for (int k = 0; k < 3; ++k) {
    const int v = idx[k];
    valid &= v >= 0 && v < mesh.numVertices;
    p[k] = v >= 0 && v < mesh.numVertices ? mesh.positions[v] : vec3f(0.f);
    valid &= finite3(p[k]);
    const vec4f st = attribute(mesh.uvs, 3*f+k, v, 2, vec4f(NAN));
    uv[k] = vec2f(st.x,st.y);
  }
  const vec3f e1 = p[1]-p[0], e2 = p[2]-p[0];
  const vec3f crossN = cross(e1,e2);
  const vec3f faceN = safeNormal(crossN,vec3f(0,0,1));
  const vec2f s = uv[1]-uv[0], r = uv[2]-uv[0];
  const double det = double(s.x)*r.y - double(s.y)*r.x;
  const double scale = double(dot(s,s))*dot(r,r);
  const bool usable = valid && isfinite(det) && isfinite(scale)
      && det*det > 1e-20*scale && dot(crossN,crossN) > 1e-30f;
  vec3f t = usable ? (e1*r.y-e2*s.y)/float(det) : vec3f(0.f);
  vec3f b = usable ? (e2*s.x-e1*r.x)/float(det) : vec3f(0.f);
  if (!finite3(t) || !finite3(b)) t = b = vec3f(0.f);
  faces[f] = {t,b};
  for (int k = 0; k < 3; ++k) {
    const vec4f authoredN = attribute(mesh.normals,3*f+k,idx[k],3,vec4f(0.f));
    Corner c;
    c.position = p[k];
    c.normal = safeNormal(vec3f(authoredN.x,authoredN.y,authoredN.z),faceN);
    c.uv = usable ? uv[k] : vec2f(0.f);
    c.sign = usable ? (dot(cross(c.normal,t),b) < 0.f ? -1.f : 1.f) : 0.f;
    const vec3f a = safeNormal(p[(k+1)%3]-p[k],vec3f(0.f));
    const vec3f d = safeNormal(p[(k+2)%3]-p[k],vec3f(0.f));
    c.weight = atan2f(length(cross(a,d)),dot(a,d));
    if (!usable || !isfinite(c.weight)) c.weight = 0.f;
    c.bucket = bucketFor(c,3*mesh.numTriangles);
    corners[3*f+k] = c;
    ci.atomicAdd(&counts[c.bucket],1);
  }
}
#endif

// A hierarchical exclusive scan with bounded local work. Both the local
// scans and recursive block-sum scans execute on device, with no readback.
constexpr int scanBlock = 128;
__rtc_global void scanBlocks(const rtc::ComputeInterface &ci, int *values,
                            int *sums, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int block = ci.launchIndex().x;
  const int begin = block*scanBlock;
  if (begin >= n) return;
  int sum = 0;
  const int end = begin + min(scanBlock,n-begin);
  for (int i = begin; i < end; ++i) {
    const int v = values[i]; values[i] = sum; sum += v;
  }
  sums[block] = sum;
}
#endif

__rtc_global void addBlockSums(const rtc::ComputeInterface &ci, int *values,
                              const int *sums, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i < n) values[i] += sums[i/scanBlock];
}
#endif

__rtc_global void bucketCorners(const rtc::ComputeInterface &ci,
                               const Corner *corners, int n,
                               int *cursors, int *bucketed)
#if !RTC_DEVICE_CODE
;
#else
{
  const int c = ci.launchIndex().x;
  if (c < n) bucketed[ci.atomicAdd(&cursors[corners[c].bucket],1)] = c;
}
#endif

__rtc_global void finalizeTangents(const rtc::ComputeInterface &ci,
                                  const Corner *corners, const FaceFrame *faces,
                                  const int *offsets, int *bucketed,
                                  int n, vec4f *out)
#if !RTC_DEVICE_CODE
;
#else
{
  const int bucket = ci.launchIndex().x;
  if (bucket >= n) return;
  const int begin = offsets[bucket], end = offsets[bucket+1];
  // One owner per bucket reduces a compatible group once, not once per
  // corner. High-valence vertices and repeated/degenerate soup stay linear
  // unless many distinct keys collide. Negative IDs mark completed groups.
  for (int i = begin; i < end; ++i) {
    const int c = bucketed[i];
    if (c < 0) continue;
    const Corner &corner = corners[c];
    if (corner.sign == 0.f) {
      const vec3f t = fallbackTangent(corner.normal);
      out[c] = vec4f(t.x,t.y,t.z,1.f);
      bucketed[i] = -c-1;
      continue;
    }
    vec3f t(0.f), b(0.f);
    for (int j = i; j < end; ++j) {
      const int other = bucketed[j];
      if (other < 0 || !compatible(corner,corners[other])) continue;
      t += faces[other/3].t*corners[other].weight;
      b += faces[other/3].b*corners[other].weight;
    }
    const vec3f normal = corner.normal;
    t = safeNormal(t-normal*dot(normal,t),fallbackTangent(normal));
    const vec4f frame(t.x,t.y,t.z,dot(cross(normal,t),b) < 0.f ? -1.f : 1.f);
    for (int j = i; j < end; ++j) {
      const int other = bucketed[j];
      if (other < 0 || (j != i && !compatible(corner,corners[other]))) continue;
      out[other] = frame;
      bucketed[j] = -other-1;
    }
  }
}
#endif

struct Scratch
{
  rtc::Device *device;
  std::vector<rtc::Buffer *> buffers;
  ~Scratch() { for (auto b : buffers) device->freeBuffer(b); }
  template<typename T> T *allocate(size_t count)
  {
    auto b = device->createBuffer(count*sizeof(T));
    buffers.push_back(b);
    return static_cast<T *>(b->getDD());
  }
};

void scan(rtc::Device *device, Scratch &scratch, int *values, int n)
{
  const int blocks = int(dru(n,scanBlock));
  int *sums = scratch.allocate<int>(blocks);
  __rtc_launch(device,scanBlocks,int(dru(blocks,128)),128,values,sums,n);
  if (blocks > 1) {
    scan(device,scratch,sums,blocks);
    __rtc_launch(device,addBlockSums,int(dru(n,128)),128,values,sums,n);
  }
}

} // namespace tangent_detail

rtc::Buffer *generateTangentFrames(rtc::Device *device, const TangentMesh &mesh)
{
  using namespace tangent_detail;
  if (mesh.numTriangles <= 0) return nullptr;
  if (mesh.numTriangles > (INT_MAX-1)/3)
    throw std::runtime_error("too many triangle corners for tangent generation");
  const int n = 3*mesh.numTriangles;
  Scratch scratch{device,{}};
  auto corners = scratch.allocate<Corner>(n);
  auto faces = scratch.allocate<FaceFrame>(mesh.numTriangles);
  auto offsets = scratch.allocate<int>(size_t(n)+1);
  auto cursors = scratch.allocate<int>(n);
  auto bucketed = scratch.allocate<int>(n);
  __rtc_launch(device,clearCounts,int(dru(size_t(n)+1,128)),128,offsets,n+1);
  __rtc_launch(device,computeFaces,int(dru(mesh.numTriangles,128)),128,
               mesh,faces,corners,offsets);
  scan(device,scratch,offsets,n+1);
  device->copyAsync(cursors,offsets,size_t(n)*sizeof(int));
  __rtc_launch(device,bucketCorners,int(dru(n,128)),128,corners,n,cursors,bucketed);
  auto output = device->createBuffer(size_t(n)*sizeof(vec4f));
  try {
    __rtc_launch(device,finalizeTangents,int(dru(n,128)),128,
                 corners,faces,offsets,bucketed,n,static_cast<vec4f *>(output->getDD()));
    device->sync();
  } catch (...) {
    device->freeBuffer(output);
    throw;
  }
  return output;
}

} // namespace native
} // namespace BARNEY_NS
