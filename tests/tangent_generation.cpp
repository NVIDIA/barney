// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/LocalContext.h"
#include "native/geometry/TangentFrames.h"
#include "tangent_hits.h"
#include <iostream>

using namespace tangent_test;

struct Mesh
{
  Context &context;
  std::shared_ptr<Triangles> triangle;
  HostMaterial::SP material;
  Sampler::SP normal, coat;

  Mesh(Context &context) : context(context)
  {
    triangle = std::dynamic_pointer_cast<Triangles>(context.createGeometry(0,"triangles"));
    material = context.createMaterial(0,"physicallyBased");
    normal = constantNormal(context,vec3f(.6f,0,.8f));
    coat = constantNormal(context,vec3f(0,.6f,.8f));
    material->setObject("normal",normal);
    material->setObject("clearcoatNormal",coat);
    material->commit();
    material->hasBeenCommittedAtLeastOnce = true;
    triangle->setMaterial(material);
    set("vertices",BN_FLOAT3,std::vector<vec3f>{{0,0,0},{1,0,0},{0,1,0}});
    set("indices",BN_INT3,std::vector<vec3i>{{0,1,2}});
    set("vertex.attribute0",BN_FLOAT2,std::vector<vec2f>{{0,0},{1,0},{0,1}});
  }

  template<typename T>
  Data::SP set(const char *name, BNDataType type, const std::vector<T> &values)
  {
    auto data = context.createData(0,type);
    data->set(values.data(),values.size());
    require(triangle->setData(name,data),"setData rejected input");
    return data;
  }

  void expect(vec3f base, vec3f clearcoat, const char *message,
              vec3f origin = vec3f(.2f,.3f,1.f),
              affine3f transform = affine3f(owl::common::one))
  {
    triangle->commit();
    for (const auto &hit : trace(context,triangle.get(),origin,transform)) {
      expectNormal(mappedNormal(hit),base,message);
      expectNormal(mappedNormal(hit,true),clearcoat,message);
    }
  }
};

static void scopesAndUpdates(Context &context)
{
  Mesh mesh(context);
  mesh.expect({.6f,0,.8f},{0,.6f,.8f},"unit UV frame");
  auto uv = mesh.set("faceVarying.attribute0",BN_FLOAT2,
    std::vector<vec2f>{{0,0},{0,1},{1,0}});
  mesh.expect({0,.6f,.8f},{.6f,0,.8f},"face UV precedence and mirrored sign");
  const vec2f ordinary[] = {{0,0},{1,0},{0,1}};
  uv->set(ordinary,3);
  mesh.expect({.6f,0,.8f},{0,.6f,.8f},"in-place UV update");
  mesh.triangle->setData("faceVarying.attribute0",nullptr);
  mesh.expect({.6f,0,.8f},{0,.6f,.8f},"removing face UVs exposes vertex UVs");

  auto normals = mesh.set("vertex.normal",BN_FLOAT3,
    std::vector<vec3f>(3,vec3f(0,0,1)));
  const vec3f tilted[] = {{0,.6f,.8f},{0,.6f,.8f},{0,.6f,.8f}};
  normals->set(tilted,3);
  mesh.expect({.6f,.48f,.64f},{0,.96f,.28f},"in-place normal update");
  mesh.set("faceVarying.normal",BN_FLOAT3,std::vector<vec3f>(3,vec3f(0,0,1)));
  mesh.expect({.6f,0,.8f},{0,.6f,.8f},"face normal precedence");
  mesh.triangle->setData("faceVarying.normal",nullptr);
  mesh.triangle->setData("vertex.normal",nullptr);

  auto positions = mesh.set("vertices",BN_FLOAT3,
    std::vector<vec3f>{{0,0,0},{1,0,0},{0,1,0}});
  mesh.set("vertex.normal",BN_FLOAT3,std::vector<vec3f>(3,vec3f(0,0,1)));
  const vec3f swapped[] = {{0,0,0},{0,1,0},{1,0,0}};
  positions->set(swapped,3);
  mesh.expect({0,.6f,.8f},{.6f,0,.8f},"position update changes hit-time frame");

  mesh.set("indices",BN_INT3,std::vector<vec3i>{{0,2,1}});
  mesh.expect({0,.6f,.8f},{.6f,0,.8f},"index update preserves vertex association");
}

static void authoredFrames(Context &context)
{
  Mesh mesh(context);
  mesh.set("vertex.tangent",BN_FLOAT4,std::vector<vec4f>(3,vec4f(0,-1,0,1)));
  mesh.expect({0,-.6f,.8f},{.6f,0,.8f},"authored vertex tangent wins");
  mesh.set("faceVarying.tangent",BN_FLOAT4,std::vector<vec4f>(3,vec4f(0,1,0,1)));
  mesh.expect({0,.6f,.8f},{-.6f,0,.8f},"authored face tangent wins");
  mesh.triangle->setData("faceVarying.tangent",nullptr);
  mesh.expect({0,-.6f,.8f},{.6f,0,.8f},"removing face tangent exposes vertex");
  mesh.triangle->setData("vertex.tangent",nullptr);
  mesh.expect({.6f,0,.8f},{0,.6f,.8f},"authored removal enables derivation");
  mesh.set("vertex.tangent",BN_FLOAT4,std::vector<vec4f>{
    {1,0,0,1},{0,1,0,1},{1,0,0,1}});
  mesh.expect({.5820855f,.1455214f,.8f},{-.1455214f,.5820855f,.8f},
              "authored tangents interpolate at the hit");
  mesh.set("vertex.tangent",BN_FLOAT4,std::vector<vec4f>(3,vec4f(0.f)));
  mesh.expect({.6f,0,.8f},{0,.6f,.8f},"zero-w retains PR derivation sentinel");
}

static void materialGating(Context &context)
{
  Mesh mesh(context);
  require(mesh.material->set3f("normal",vec3f(0,0,1)),"constant normal rejected");
  mesh.material->commit();
  mesh.expect({0,0,1},{0,.6f,.8f},"clearcoat-only normal map derives frame");
  require(mesh.material->setObject("clearcoatNormal",nullptr),"coat removal rejected");
  mesh.material->commit();
  mesh.expect({0,0,1},{0,0,1},"PBR without normal maps");
  auto matte = context.createMaterial(0,"matte");
  matte->commit();
  matte->hasBeenCommittedAtLeastOnce = true;
  mesh.triangle->setMaterial(matte);
  mesh.triangle->commit();
  for (const auto &hit : trace(context,mesh.triangle.get()))
    require(hit.bsdfType == PackedBSDF::TYPE_Lambertian,"matte hit requires no tangent array");
  mesh.triangle->setMaterial(mesh.material);
  mesh.material->setObject("normal",mesh.normal);
  mesh.material->commit();
  mesh.expect({.6f,0,.8f},{0,0,1},"material-only change enables base mapping");
}

static void fallbackAndTransforms(Context &context)
{
  Mesh mesh(context);
  const std::vector<std::vector<vec2f>> invalid = {
    {{0,0},{0,0},{0,0}}, {{NAN,0},{1,0},{0,1}}, {{0,0}}};
  for (const auto &uv : invalid) {
    mesh.set("faceVarying.attribute0",BN_FLOAT2,uv);
    mesh.triangle->commit();
    for (const auto &hit : trace(context,mesh.triangle.get())) {
      const auto n = mappedNormal(hit);
      require(std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z)
          && std::abs(length(n)-1.f) < 1e-3f,"invalid UV fallback must stay finite/unit");
    }
  }
  mesh.triangle->setData("faceVarying.attribute0",nullptr);
  mesh.triangle->setData("vertex.attribute0",nullptr);
  mesh.triangle->commit();
  for (const auto &hit : trace(context,mesh.triangle.get())) {
    const auto n = mappedNormal(hit);
    require(std::isfinite(n.x) && std::isfinite(n.y) && std::isfinite(n.z)
        && std::abs(length(n)-1.f) < 1e-3f,"missing UV fallback must stay finite/unit");
  }
  mesh.set("vertex.attribute0",BN_FLOAT2,std::vector<vec2f>{{0,0},{1,0},{0,1}});
  affine3f mirrored(owl::common::one);
  mirrored.l.vx.x = -2.f;
  mirrored.l.vy.y = 3.f;
  mesh.expect({-.6f,0,.8f},{0,.6f,.8f},"mirrored instance frame orientation",
              {-.4f,.9f,1.f},mirrored);
}

// The standalone generator is still a supported helper, but its output is
// not attached to meshes by commit(). Keep its direct coverage separate.
static void standaloneGenerator(Context &context)
{
  Mesh mesh(context);
  mesh.triangle->commit();
  for (auto *device : *mesh.triangle->devices) {
    Triangles::DD dd;
    mesh.triangle->writeDD(dd,device);
    TangentMesh input{dd.vertices,nullptr,3,1,dd.attributes.normalAttribute,
                      dd.attributes.attribute[0]};
    auto buffer = generateTangentFrames(device->rtc,input);
    vec4f output[3];
    device->rtc->copyAsync(output,buffer->getDD(),sizeof(output));
    device->sync();
    device->rtc->freeBuffer(buffer);
    for (auto t : output) {
      expectNormal(vec3f(t.x,t.y,t.z),{1,0,0},"standalone nonindexed generator");
      require(t.w == 1.f,"standalone generator handedness");
    }
  }
}

int main()
{
  try {
    const int rank = 0, gpu = 0;
    std::unique_ptr<Context> context(LocalContext::create(1,&rank,1,&gpu));
    scopesAndUpdates(*context);
    authoredFrames(*context);
    materialGating(*context);
    fallbackAndTransforms(*context);
    standaloneGenerator(*context);
    const int gpus[] = {0,0};
    std::unique_ptr<Context> replicated(LocalContext::create(1,&rank,2,gpus));
    scopesAndUpdates(*replicated);
    authoredFrames(*replicated);
    std::cout << "hit-time tangent tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
