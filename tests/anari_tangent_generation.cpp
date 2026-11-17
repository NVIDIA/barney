// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Device.h"
#include "anari/Surface.h"
#include "anari/World.h"
#include "tangent_hits.h"
#include <iostream>

namespace ba = BARNEY_NS::anari;
using namespace tangent_test;

static void meshCase(ba::BarneyDevice &device, bool quad, bool indexed)
{
  const auto geometry = device.newGeometry(quad ? "quad" : "triangle");
  const auto material = device.newMaterial("physicallyBased");
  const auto surface = device.newSurface();
  const auto sampler = device.newSampler("transform");
  const float zero[16] = {};
  const float tilt[] = {.6f,0,.8f,1.f};
  device.setParameter(sampler,"outTransform",ANARI_FLOAT32_MAT4,zero);
  device.setParameter(sampler,"outOffset",ANARI_FLOAT32_VEC4,tilt);
  device.commitParameters(sampler);
  device.setParameter(material,"normal",ANARI_SAMPLER,&sampler);
  auto *surfaceObject = reinterpret_cast<ba::Surface *>(surface);
  auto *state = surfaceObject->deviceState();
  std::vector<ANARIArray1D> arrays;
  auto array = [&](const char *name, ANARIDataType type, const void *values, size_t n) {
    const auto a = device.newArray1D(values,nullptr,nullptr,type,n);
    arrays.push_back(a);
    device.setParameter(geometry,name,ANARI_ARRAY1D,&a);
    return a;
  };
  auto native = [&] {
    state->commitBuffer.flush();
    return reinterpret_cast<Triangles *>(surfaceObject->getBarneyGeom());
  };
  auto expect = [&](vec3f expected, const char *message) {
    auto *mesh = native();
    // Trace inside each triangulated half, not along the quad diagonal.
    const vec3f origins[] = {{.7f,.2f,1.f},{.2f,.7f,1.f}};
    for (int i = 0; i < (quad ? 2 : 1); ++i)
      for (const auto &hit : trace(*mesh->context,mesh,origins[i]))
        expectNormal(mappedNormal(hit),expected,message);
  };
  const size_t count = quad ? 4 : 3;
  const vec3f positions[] = {{0,0,0},{1,0,0},{1,1,0},{0,1,0}};
  const vec2f vertexUV[] = {{0,0},{1,0},{1,1},{0,1}};
  vec2f faceUV[] = {{0,0},{0,-1},{1,-1},{1,0}};
  const uint32_t index[] = {0,1,2,3};
  double extraUV[] = {.25,.75,.25,.75,.25,.75,.25,.75};
  const auto extraArray = array("vertex.attribute1",ANARI_FLOAT64_VEC2,extraUV,count);
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
  expect({0,.6f,.8f},"ANARI face UVs take precedence at hits");
  require(faceUV[1] == vec2f(0,-1) && vertexUV[1] == vec2f(1,0),
          "hit-time evaluation mutated input UV arrays");

  auto *mesh = native();
  const auto indicesBefore = mesh->indices;
  auto *extraMap = static_cast<double *>(device.mapArray(extraArray));
  extraMap[0] = .5;
  device.unmapArray(extraArray);
  mesh = native();
  Triangles::DD dd;
  auto *dev = mesh->devices->get(0);
  mesh->writeDD(dd,dev);
  vec2f extraValue;
  dev->rtc->copyAsync(&extraValue,dd.attributes.attribute[1].fromArray.ptr,sizeof(extraValue));
  dev->sync();
  require(extraValue.x == .5f,"converted unrelated attribute update lost");
  require(mesh->indices == indicesBefore,"attribute update replaced geometry indices");
  expect({0,.6f,.8f},"unrelated update preserves normal mapping");

  auto *mapped = static_cast<vec2f *>(device.mapArray(faceArray));
  for (size_t i = 0; i < count; ++i) mapped[i] = vertexUV[i];
  device.unmapArray(faceArray);
  expect({.6f,0,.8f},"UV unmap updates hit-time frame without geometry recommit");

  const vec4f vertexT[] = {{0,-1,0,1},{0,-1,0,1},{0,-1,0,1},{0,-1,0,1}};
  vec4f authored[] = {{0,1,0,1},{0,1,0,1},{0,1,0,1},{0,1,0,1}};
  array("vertex.tangent",ANARI_FLOAT32_VEC4,vertexT,count);
  const auto faceT = array("faceVarying.tangent",ANARI_FLOAT32_VEC4,authored,count);
  device.commitParameters(geometry);
  expect({0,.6f,.8f},"authored face tangent overrides derivation and vertex tangent");

  // Distinct corners catch incorrect quad expansion even when the shading
  // checks above use a uniform authored direction.
  auto *cornerMap = static_cast<vec4f *>(device.mapArray(faceT));
  const vec4f distinct[] = {{1,0,0,1},{0,1,0,1},{-1,0,0,-1},{0,-1,0,-1}};
  for (size_t i = 0; i < count; ++i) cornerMap[i] = distinct[i];
  device.unmapArray(faceT);
  mesh = native();
  mesh->writeDD(dd,dev);
  const auto &a = dd.attributes.tangentAttribute;
  require(a.scope == GeometryAttribute::FACE_VARYING && a.fromArray.type == BN_FLOAT4
      && a.fromArray.size == (quad ? 6 : 3),"authored face tangent layout");
  std::vector<vec4f> corners(a.fromArray.size);
  dev->rtc->copyAsync(corners.data(),a.fromArray.ptr,corners.size()*sizeof(vec4f));
  dev->sync();
  const int cornerIndex[] = {0,1,2,0,2,3};
  for (size_t i = 0; i < corners.size(); ++i)
    require(corners[i] == distinct[cornerIndex[i]],"authored corner expansion");
  auto *updated = static_cast<vec4f *>(device.mapArray(faceT));
  for (size_t i = 0; i < count; ++i) updated[i] = vec4f(-1,0,0,-1);
  device.unmapArray(faceT);
  expect({-.6f,0,.8f},"authored tangent unmap updates intersections");

  array("faceVarying.tangent",ANARI_FLOAT32_VEC4,authored,count-1);
  device.commitParameters(geometry);
  expect({0,-.6f,.8f},"short face tangents fall back to authored vertex tangents");
  array("faceVarying.tangent",ANARI_FLOAT32_VEC2,vertexUV,count);
  device.commitParameters(geometry);
  expect({0,-.6f,.8f},"unsupported face tangents fall back to vertex tangents");
  device.unsetParameter(geometry,"vertex.tangent");
  device.unsetParameter(geometry,"faceVarying.tangent");
  device.commitParameters(geometry);
  expect({.6f,0,.8f},"authored tangent removal restores hit-time derivation");

  const vec3f cornerNormals[] = {{0,0,1},{0,.6f,.8f},{0,-.6f,.8f},{0,.8f,.6f}};
  const auto faceNormals = array("faceVarying.normal",ANARI_FLOAT32_VEC3,cornerNormals,count);
  device.commitParameters(geometry);
  mesh = native();
  mesh->writeDD(dd,dev);
  const auto &normalAttribute = dd.attributes.normalAttribute;
  require(normalAttribute.scope == GeometryAttribute::FACE_VARYING
      && normalAttribute.fromArray.type == BN_FLOAT3
      && normalAttribute.fromArray.size == (quad ? 6 : 3),"face normal layout");
  std::vector<vec3f> expandedNormals(normalAttribute.fromArray.size);
  dev->rtc->copyAsync(expandedNormals.data(),normalAttribute.fromArray.ptr,
                      expandedNormals.size()*sizeof(vec3f));
  dev->sync();
  for (size_t i = 0; i < expandedNormals.size(); ++i)
    require(expandedNormals[i] == cornerNormals[cornerIndex[i]],"face normal corner expansion");
  auto *faceNormalMap = static_cast<vec3f *>(device.mapArray(faceNormals));
  for (size_t i = 0; i < count; ++i) faceNormalMap[i] = vec3f(0,.6f,.8f);
  device.unmapArray(faceNormals);
  expect({.6f,.48f,.64f},"face normal unmap refreshes hit-time shading");
  device.unsetParameter(geometry,"faceVarying.normal");
  device.commitParameters(geometry);
  expect({.6f,0,.8f},"face normal removal restores geometric shading");

  // Formats not supported directly by typedRead are converted on upload.
  int16_t fixedNormals[] = {0,0,32767,0,0,32767,0,0,32767,0,0,32767};
  const auto normalArray = array("vertex.normal",ANARI_FIXED16_VEC3,fixedNormals,count);
  device.commitParameters(geometry);
  expect({.6f,0,.8f},"converted SNORM normals reach hit-time frame");
  auto *normalMap = static_cast<int16_t *>(device.mapArray(normalArray));
  for (size_t i = 0; i < count; ++i) {
    normalMap[3*i+1] = 19660;
    normalMap[3*i+2] = 26214;
  }
  device.unmapArray(normalArray);
  expect({.6f,.48f,.64f},"converted normal unmap refreshes hit-time shading");
  device.unsetParameter(geometry,"vertex.normal");
  device.commitParameters(geometry);
  expect({.6f,0,.8f},"normal removal restores geometric normal");

  const vec3f authored3[] = {{0,1,0},{0,1,0},{0,1,0},{0,1,0}};
  array("vertex.tangent",ANARI_FLOAT32_VEC3,authored3,count);
  device.commitParameters(geometry);
  expect({0,.6f,.8f},"authored VEC3 tangents retain implicit positive sign");
  const int16_t authoredFixed[] = {0,-32767,0,-32767,0,-32767,0,-32767,
                                  0,-32767,0,-32767,0,-32767,0,-32767};
  array("faceVarying.tangent",ANARI_FIXED16_VEC4,authoredFixed,count);
  device.commitParameters(geometry);
  expect({0,-.6f,.8f},"converted SNORM authored tangent precedence");

  // Same generated frame must shade like its explicit authored equivalent.
  device.unsetParameter(geometry,"faceVarying.tangent");
  const vec4f equivalent[] = {{1,0,0,1},{1,0,0,1},{1,0,0,1},{1,0,0,1}};
  array("vertex.tangent",ANARI_FLOAT32_VEC4,equivalent,count);
  device.commitParameters(geometry);
  expect({.6f,0,.8f},"authored equivalent agrees with derived normal mapping");
  array("vertex.normal",ANARI_FLOAT32_VEC3,cornerNormals,count);
  device.commitParameters(geometry);
  // Exercise a real ANARI world update, not an unattached instance commit.
  const auto group = device.newGroup();
  const auto instance = device.newInstance("transform");
  const auto world = device.newWorld();
  const auto surfaces = device.newArray1D(&surface,nullptr,nullptr,ANARI_SURFACE,1);
  const auto instances = device.newArray1D(&instance,nullptr,nullptr,ANARI_INSTANCE,1);
  device.setParameter(group,"surface",ANARI_ARRAY1D,&surfaces);
  device.commitParameters(group);
  device.setParameter(instance,"group",ANARI_GROUP,&group);
  device.commitParameters(instance);
  device.setParameter(world,"instance",ANARI_ARRAY1D,&instances);
  device.commitParameters(world);
  state->commitBuffer.flush();
  auto *worldObject = reinterpret_cast<ba::World *>(world);
  worldObject->makeCurrent();
  mesh = native();
  mesh->writeDD(dd,dev);
  const auto indicesSnapshot = mesh->indices;
  const auto before = dd;
  // Instance::commitParameters consumes MAT4, not MAT3x4.
  const float transform[] = {1,0,0,0,0,1,0,0,0,0,1,0,2,3,4,1};
  device.setParameter(instance,"transform",ANARI_FLOAT32_MAT4,transform);
  device.commitParameters(instance);
  state->commitBuffer.flush();
  worldObject->makeCurrent();
  BNTransform applied;
  reinterpret_cast<ba::Instance *>(instance)->writeTransform(&applied);
  require(applied.p.x == 2.f && applied.p.y == 3.f && applied.p.z == 4.f,
          "instance-only transform was not applied");
  require(native() == mesh && mesh->indices == indicesSnapshot,
          "instance-only update replaced native mesh inputs");
  mesh->writeDD(dd,dev);
  require(dd.vertices == before.vertices
      && dd.attributes.attribute[0].fromArray.ptr == before.attributes.attribute[0].fromArray.ptr
      && dd.attributes.normalAttribute.fromArray.ptr == before.attributes.normalAttribute.fromArray.ptr
      && dd.attributes.tangentAttribute.fromArray.ptr == before.attributes.tangentAttribute.fromArray.ptr,
      "instance-only update rebound position/UV/normal/authored tangent inputs");
  device.release(world);
  device.release(instances);
  device.release(instance);
  device.release(group);
  device.release(surfaces);
  device.release(surface);
  device.release(material);
  device.release(sampler);
  device.release(geometry);
  for (auto a : arrays) device.release(a);
  state->commitBuffer.flush();
}

int main()
{
  try {
    ba::BarneyDevice device;
    for (bool quad : {false,true})
      for (bool indexed : {false,true}) meshCase(device,quad,indexed);
    std::cout << "ANARI hit-time tangent tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
