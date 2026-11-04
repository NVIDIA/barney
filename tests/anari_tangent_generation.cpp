// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Device.h"
#include "anari/Surface.h"
#include "native/geometry/Triangles.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace ba = BARNEY_NS::anari;
namespace bn = BARNEY_NS::native;
using bn::vec2f;
using bn::vec3f;
using bn::vec4f;
using BARNEY_NS::BN_FLOAT3;
using BARNEY_NS::BN_FLOAT4;

static void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}
static void expect(vec4f t, vec4f e, const char *message)
{
  require(std::isfinite(t.x) && std::isfinite(t.y) && std::isfinite(t.z)
    && std::abs(t.x-e.x) < 2e-5f && std::abs(t.y-e.y) < 2e-5f
    && std::abs(t.z-e.z) < 2e-5f && t.w == e.w,message);
}

static void meshCase(ba::BarneyDevice &device, bool quad, bool indexed)
{
  const auto geometry = device.newGeometry(quad ? "quad" : "triangle");
  const auto material = device.newMaterial("matte");
  const auto surface = device.newSurface();
  auto *surfaceObject = reinterpret_cast<ba::Surface *>(surface);
  auto *state = surfaceObject->deviceState();
  std::vector<ANARIArray1D> arrays;
  auto array = [&](const char *name, ANARIDataType type, const void *values, size_t n) {
    const auto a = device.newArray1D(values,nullptr,nullptr,type,n);
    arrays.push_back(a);
    device.setParameter(geometry,name,ANARI_ARRAY1D,&a);
    return a;
  };
  auto flush = [&] { state->commitBuffer.flush(); };
  auto native = [&] {
    flush();
    return reinterpret_cast<bn::Triangles *>(surfaceObject->getBarneyGeom());
  };
  auto output = [&] {
    auto *mesh = native();
    auto *dev = mesh->devices->get(0);
    bn::Triangles::DD dd;
    mesh->writeDD(dd,dev);
    const auto &a = dd.attributes.tangentAttribute;
    require(a.scope == bn::GeometryAttribute::FACE_VARYING
      || a.scope == bn::GeometryAttribute::PER_VERTEX,"no ANARI tangent data");
    require(a.fromArray.type == BN_FLOAT4,"ANARI tangent output type");
    std::vector<vec4f> result(a.fromArray.size);
    dev->rtc->copyAsync(result.data(),a.fromArray.ptr,result.size()*sizeof(vec4f));
    dev->sync();
    return result;
  };
  const size_t count = quad ? 4 : 3;
  double extraUV[] = {.25,.75,.25,.75,.25,.75,.25,.75};
  const auto extraArray = array("vertex.attribute1",ANARI_FLOAT64_VEC2,extraUV,count);
  ANARIArray1D extraFaceArray = nullptr;
  if (!quad)
    extraFaceArray = array("faceVarying.attribute2",ANARI_FLOAT64_VEC2,extraUV,count);
  const vec3f positions[] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
  const vec2f vertexUV[] = {{0,0},{1,0},{1,1},{0,1}};
  vec2f faceUV[] = {{0,0},{0,-1},{1,-1},{1,0}};
  const uint32_t index[] = {0,1,2,3};
  array("vertex.position",ANARI_FLOAT32_VEC3,positions,count);
  array("vertex.attribute0",ANARI_FLOAT32_VEC2,vertexUV,count);
  const auto faceArray = array("faceVarying.attribute0",ANARI_FLOAT32_VEC2,faceUV,count);
  if (indexed)
    array("primitive.index",quad ? ANARI_UINT32_VEC4 : ANARI_UINT32_VEC3,index,1);
  device.commitParameters(geometry);
  device.commitParameters(material);
  device.setParameter(surface,"geometry",ANARI_GEOMETRY,&geometry);
  device.setParameter(surface,"material",ANARI_MATERIAL,&material);
  device.commitParameters(surface);
  for (auto t : output()) expect(t,{0,1,0,1},"face UVs must take precedence");
  require(faceUV[1] == vec2f(0,-1) && vertexUV[1] == vec2f(1,0),
          "generator mutated input UV arrays");

  // Native input identities must survive traversal after an instance-only
  // change, including quad triangulation and expanded face-varying arrays.
  auto *mesh = native();
  const auto indicesBefore = mesh->indices;
  bn::Triangles::DD beforeDD;
  mesh->writeDD(beforeDD,mesh->devices->get(0));
  const void *uvBefore = beforeDD.attributes.attribute[0].fromArray.ptr;
  const void *tangentsBefore = beforeDD.attributes.tangentAttribute.fromArray.ptr;
  auto instance = device.newInstance("transform");
  const float transform[] = {1,0,0,0,1,0,0,0,1,2,3,4};
  device.setParameter(instance,"transform",ANARI_FLOAT32_MAT3x4,transform);
  device.commitParameters(instance);
  flush();
  require(native()->indices == indicesBefore,"instance-only traversal replaced mesh inputs");
  // Caching tangent inputs must not suppress ordinary attribute uploads.
  auto *extraMap = static_cast<double *>(device.mapArray(extraArray));
  extraMap[0] = .5;
  device.unmapArray(extraArray);
  mesh = native();
  bn::Triangles::DD extraDD;
  auto *extraDevice = mesh->devices->get(0);
  mesh->writeDD(extraDD,extraDevice);
  vec2f extraValue;
  extraDevice->rtc->copyAsync(&extraValue,extraDD.attributes.attribute[1].fromArray.ptr,sizeof(extraValue));
  extraDevice->sync();
  require(extraValue.x == .5f,"mesh cache suppressed an unrelated converted attribute update");
  if (extraFaceArray) {
    auto *faceMap = static_cast<double *>(device.mapArray(extraFaceArray));
    faceMap[0] = .875;
    device.unmapArray(extraFaceArray);
    mesh = native();
    mesh->writeDD(extraDD,extraDevice);
    require(extraDD.attributes.attribute[2].scope == bn::GeometryAttribute::FACE_VARYING,
            "material-only update lost face-varying attributes");
    extraDevice->rtc->copyAsync(&extraValue,extraDD.attributes.attribute[2].fromArray.ptr,sizeof(extraValue));
    extraDevice->sync();
    require(extraValue.x == .875f,"material-only update did not refresh converted face attributes");
  }
  require(mesh->indices == indicesBefore
      && extraDD.attributes.attribute[0].fromArray.ptr == uvBefore
      && extraDD.attributes.tangentAttribute.fromArray.ptr == tangentsBefore,
      "material-only update rebound tangent-generation inputs or output");
  for (auto v : output()) expect(v,{0,1,0,1},"material-only updates changed generated tangents");

  // A real unmap (no geometry recommit) must notify the mesh observer and
  // invalidate generated data, even though the native UV handle may be reused.
  auto *mapped = static_cast<vec2f *>(device.mapArray(faceArray));
  for (size_t i = 0; i < count; ++i) mapped[i] = vertexUV[i];
  device.unmapArray(faceArray);
  for (auto t : output()) expect(t,{1,0,0,1},"unmap must update generated frames");

  vec4f authored[] = {{1,0,0,1},{0,1,0,1},{-1,0,0,-1},{0,-1,0,-1}};
  const vec4f vertexT[] = {{0,1,0,-1},{0,1,0,-1},{0,1,0,-1},{0,1,0,-1}};
  array("vertex.tangent",ANARI_FLOAT32_VEC4,vertexT,count);
  const auto faceT = array("faceVarying.tangent",ANARI_FLOAT32_VEC4,authored,count);
  device.commitParameters(geometry);
  auto t = output();
  const int corners[] = {0,1,2,0,2,3};
  require(t.size() == (quad ? 6 : 3),"triangulated tangent count");
  for (size_t i = 0; i < t.size(); ++i)
    expect(t[i],authored[corners[i]],"authored face corner mapping must be preserved");

  auto *updated = static_cast<vec4f *>(device.mapArray(faceT));
  updated[0] = vec4f(0.f);
  device.unmapArray(faceT);
  authored[0] = vec4f(0.f);
  t = output();
  for (size_t i = 0; i < t.size(); ++i)
    expect(t[i],authored[corners[i]],"authored values must not trigger host validation or fallback");

  array("faceVarying.tangent",ANARI_FLOAT32_VEC4,authored,count-1);
  device.commitParameters(geometry);
  for (auto v : output()) expect(v,{0,1,0,-1},"short authored face array falls back to vertex");
  array("faceVarying.tangent",ANARI_FLOAT32_VEC2,vertexUV,count);
  device.commitParameters(geometry);
  for (auto v : output()) expect(v,{0,1,0,-1},"unsupported authored face type falls back to vertex");
  device.unsetParameter(geometry,"vertex.tangent");
  device.unsetParameter(geometry,"faceVarying.tangent");
  device.commitParameters(geometry);
  for (auto v : output()) expect(v,{1,0,0,1},"authored removal restores generation");

  // Normal formats not carried by native typedRead must use the same upload
  // conversion policy as UVs, rather than losing the old host generator's N.
  int16_t fixedNormals[] = {32767,0,0,32767,0,0,32767,0,0,32767,0,0};
  const auto normalArray = array("vertex.normal",ANARI_FIXED16_VEC3,fixedNormals,count);
  device.commitParameters(geometry);
  for (auto v : output()) expect(v,{0,1,0,1},"SNORM normal must reach generator");
  auto *normalMap = static_cast<int16_t *>(device.mapArray(normalArray));
  for (size_t i = 0; i < count; ++i) {
    normalMap[3*i] = 0; normalMap[3*i+2] = 32767;
  }
  device.unmapArray(normalArray);
  for (auto v : output()) expect(v,{1,0,0,1},"converted normal unmap must refresh frames");
  device.unsetParameter(geometry,"vertex.normal");
  device.commitParameters(geometry);
  for (auto v : output()) expect(v,{1,0,0,1},"normal removal must refresh frames");

  vec3f cornerNormals[] = {{1,0,0},{0,0,1},{0,0,1},{0,0,1}};
  const auto cornerNormalArray = array("faceVarying.normal",ANARI_FLOAT32_VEC3,cornerNormals,count);
  device.commitParameters(geometry);
  t = output();
  for (size_t i = 0; i < t.size(); ++i)
    expect(t[i],corners[i] == 0 ? vec4f(0,1,0,1) : vec4f(1,0,0,1),
           "face normal corner mapping must survive triangulation");
  auto *cornerNormalMap = static_cast<vec3f *>(device.mapArray(cornerNormalArray));
  cornerNormalMap[0] = vec3f(0,0,1);
  device.unmapArray(cornerNormalArray);
  for (auto v : output()) expect(v,{1,0,0,1},"face normal unmap must refresh frames");
  device.unsetParameter(geometry,"faceVarying.normal");
  device.commitParameters(geometry);

  // VEC3 authored tangents retain the native typed layout and implicit +1.
  const vec3f authored3[] = {{0,1,0},{0,1,0},{0,1,0},{0,1,0}};
  array("vertex.tangent",ANARI_FLOAT32_VEC3,authored3,count);
  device.commitParameters(geometry);
  mesh = native();
  bn::Triangles::DD dd;
  mesh->writeDD(dd,mesh->devices->get(0));
  require(dd.attributes.tangentAttribute.scope == bn::GeometryAttribute::PER_VERTEX
    && dd.attributes.tangentAttribute.fromArray.type == BN_FLOAT3,
    "authored VEC3 input must remain untouched");
  const int16_t authoredFixed[] = {0,32767,0,-32767,0,32767,0,-32767,
                                  0,32767,0,-32767,0,32767,0,-32767};
  array("faceVarying.tangent",ANARI_FIXED16_VEC4,authoredFixed,count);
  device.commitParameters(geometry);
  for (auto v : output()) expect(v,{0,1,0,-1},"supported SNORM tangents retain authored precedence");
  array("vertex.tangent",ANARI_FIXED16_VEC3,fixedNormals,count);
  device.unsetParameter(geometry,"faceVarying.tangent");
  device.commitParameters(geometry);
  mesh = native();
  mesh->writeDD(dd,mesh->devices->get(0));
  require(dd.attributes.tangentAttribute.scope == bn::GeometryAttribute::PER_VERTEX
    && dd.attributes.tangentAttribute.fromArray.type == BN_FLOAT3,
    "supported SNORM VEC3 tangents retain authored precedence");
  device.release(instance);
  device.release(surface);
  device.release(material);
  device.release(geometry);
  for (auto a : arrays) device.release(a);
  flush();
}

int main()
{
  try {
    ba::BarneyDevice device;
    for (bool quad : {false,true})
      for (bool indexed : {false,true}) meshCase(device,quad,indexed);
    std::cout << "ANARI tangent generation tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
