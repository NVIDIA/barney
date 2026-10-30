// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/LocalContext.h"
#include "native/geometry/Triangles.h"
#include "native/geometry/TangentFrames.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace BARNEY_NS;
using namespace BARNEY_NS::native;

static void require(bool condition, const char *message)
{
  if (!condition) throw std::runtime_error(message);
}

struct Mesh
{
  Context &context;
  std::shared_ptr<Triangles> triangle;
  Mesh(Context &context) : context(context)
  {
    triangle = std::dynamic_pointer_cast<Triangles>(context.createGeometry(0,"triangles"));
    auto material = context.createMaterial(0,"matte");
    material->commit();
    material->hasBeenCommittedAtLeastOnce = true;
    triangle->setMaterial(material);
  }
  template<typename T> auto set(const char *name, BNDataType type, const std::vector<T> &values)
  {
    auto data = context.createData(0,type);
    data->set(values.data(),values.size());
    require(triangle->setData(name,data),"setData rejected input");
    return data;
  }
  std::vector<vec4f> tangents()
  {
    triangle->commit();
    std::vector<vec4f> result;
    for (auto device : *triangle->devices) {
      Triangles::DD dd;
      triangle->writeDD(dd,device);
      const auto &a = dd.attributes.tangentAttribute;
      require(a.scope == GeometryAttribute::FACE_VARYING
              || a.scope == GeometryAttribute::PER_VERTEX,"missing tangent output");
      require(a.fromArray.type == BN_FLOAT4,"unexpected tangent type");
#if defined(BARNEY_RTC_OPTIX) || defined(BARNEY_RTC_CUDA)
      cudaPointerAttributes pointer{};
      require(cudaPointerGetAttributes(&pointer,a.fromArray.ptr) == cudaSuccess
        && pointer.type == cudaMemoryTypeDevice,"GPU tangents must remain device-resident");
#endif
      std::vector<vec4f> current(a.fromArray.size);
      device->rtc->copyAsync(current.data(),a.fromArray.ptr,current.size()*sizeof(vec4f));
      device->sync();
      if (!result.empty()) {
        require(result.size() == current.size(),"device output sizes disagree");
        for (size_t i = 0; i < result.size(); ++i)
          for (int k = 0; k < 4; ++k)
            require(std::abs(result[i][k]-current[i][k]) < 2e-5f,
                    "per-device tangent results disagree");
      }
      result = std::move(current);
    }
    return result;
  }
};

// Hand-derived directions, not a second generator. 2e-5 accommodates ordinary
// CPU/GPU roundoff and bucket summation order; bit identity is not required.
static void expect(vec4f t, vec4f expected, const char *message)
{
  require(std::isfinite(t.x) && std::isfinite(t.y) && std::isfinite(t.z)
          && std::isfinite(t.w) && std::abs(t.x-expected.x) < 2e-5f
          && std::abs(t.y-expected.y) < 2e-5f && std::abs(t.z-expected.z) < 2e-5f
          && t.w == expected.w,message);
}

static void unitTriangle(Context &context)
{
  Mesh mesh(context);
  mesh.set("vertices",BN_FLOAT3,std::vector<vec3f>{{0,0,0},{1,0,0},{0,1,0}});
  mesh.set("indices",BN_INT3,std::vector<vec3i>{{0,1,2}});
  mesh.set("vertex.attribute0",BN_FLOAT2,std::vector<vec2f>{{0,0},{1,0},{0,1}});
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"unit triangle must have +X tangent");

  // A non-unit integer normal directed +Y projects the +X tangent unchanged;
  // its fallback frame must be orthogonal to Y even when UV is removed.
  mesh.set("vertex.normal",BN_INT32_VEC3,std::vector<vec3i>(3,vec3i(0,2,0)));
  mesh.triangle->setData("vertex.attribute0",nullptr);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"typed normal fallback");
  auto normal = mesh.set("vertex.normal",BN_INT32_VEC3,std::vector<vec3i>(3,vec3i(2,0,0)));
  for (auto t : mesh.tangents()) expect(t,{0,1,0,1},"integer normal must control fallback");
  const vec3i changed[] = {{0,0,2},{0,0,2},{0,0,2}};
  normal->set(changed,3);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"in-place normal update invalidates frames");
  mesh.set("vertex.attribute0",BN_INT32_VEC2,std::vector<vec2i>{{0,0},{0,1},{1,0}});
  for (auto t : mesh.tangents()) expect(t,{0,1,0,-1},"native integer UV conversion");
  mesh.triangle->setData("vertex.normal",nullptr);
  for (auto t : mesh.tangents()) expect(t,{0,1,0,-1},"normal removal uses geometric normal");
  // IEEE binary16: 0x3c00 is 1, 0x4000 is 2.
  const std::vector<vec2us> halfUV = {{0,0},{0x3c00,0},{0,0x3c00}};
  mesh.set("vertex.attribute0",BN_FLOAT16_VEC2,halfUV);
  mesh.set("vertex.normal",BN_FLOAT16_VEC4,
    std::vector<vec4us>(3,vec4us(0,0,0x4000,0)));
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"half UVs and vec4 normals");
  mesh.set("faceVarying.normal",BN_FLOAT3,std::vector<vec3f>{{NAN,0,0}});
  mesh.set("faceVarying.attribute0",BN_FLOAT2,std::vector<vec2f>{{0,0}});
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"short face arrays and invalid normals stay finite");
}

static void seamsAndWeighting(Context &context)
{
  Mesh mesh(context);
  // At the common origin the first face contributes 90 degrees * (2,0),
  // the second 45 degrees * (0,1), hence normalize(4,1), not normalize(2,1).
  mesh.set("vertices",BN_FLOAT3,std::vector<vec3f>{
    {0,0,0},{2,0,0},{0,1,0},{-0.f,0,0},{0,1,0},{-1,1,0}});
  mesh.set("indices",BN_INT3,std::vector<vec3i>{{0,1,2},{3,4,5}});
  mesh.set("vertex.normal",BN_FLOAT3,std::vector<vec3f>(6,vec3f(0,0,1)));
  auto uvs = mesh.set("faceVarying.attribute0",BN_FLOAT2,
    std::vector<vec2f>{{0,0},{1,0},{0,1},{0,0},{1,0},{0,1}});
  auto t = mesh.tangents();
  const vec4f weighted(0.9701425f,0.2425356f,0,1);
  expect(t[0],weighted,"angle weighting across soup positions");
  expect(t[3],weighted,"matching soup corners must weld including signed zero");

  const vec2f seamUV[] = {{0,0},{1,0},{0,1},{.5f,0},{1,0},{0,1}};
  uvs->set(seamUV,6);
  t = mesh.tangents();
  expect(t[0],{1,0,0,1},"UV seam must isolate first frame");
  expect(t[3],{0,1,0,1},"UV seam must isolate second frame");

  const vec2f continuousUV[] = {{0,0},{1,0},{0,1},{0,0},{1,0},{0,1}};
  uvs->set(continuousUV,6);
  mesh.set("faceVarying.normal",BN_FLOAT3,std::vector<vec3f>{
    {0,0,1},{0,0,1},{0,0,1},{0,1,1},{0,0,1},{0,0,1}});
  t = mesh.tangents();
  expect(t[0],{1,0,0,1},"normal seam must isolate first frame");
  expect(t[3],{0,.70710678f,-.70710678f,1},"normal seam projects second tangent");

  mesh.triangle->setData("faceVarying.normal",nullptr);
  mesh.set("vertices",BN_FLOAT3,std::vector<vec3f>{{0,0,0},{1,0,0},{0,1,0}});
  mesh.set("indices",BN_INT3,std::vector<vec3i>{{0,1,2},{0,1,2}});
  mesh.set("vertex.normal",BN_FLOAT3,std::vector<vec3f>(3,vec3f(0,0,1)));
  mesh.set("faceVarying.attribute0",BN_FLOAT2,
    std::vector<vec2f>{{0,0},{1,0},{0,1},{0,0},{-1,0},{0,1}});
  t = mesh.tangents();
  for (int c = 0; c < 3; ++c) {
    expect(t[c],{1,0,0,1},"positive orientation must not cancel with mirrored UVs");
    expect(t[c+3],{-1,0,0,-1},"negative orientation must retain handedness");
  }
}

static void authoredAndUpdates(Context &context)
{
  Mesh mesh(context);
  auto vertices = mesh.set("vertices",BN_FLOAT3,
    std::vector<vec3f>{{0,0,0},{1,0,0},{0,1,0}});
  auto indices = mesh.set("indices",BN_INT3,std::vector<vec3i>{{0,1,2}});
  auto uv = mesh.set("vertex.attribute0",BN_UFIXED16_VEC2,
    std::vector<vec2us>{{0,0},{65535,0},{0,65535}});
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"UNORM16 UV conversion");
  mesh.set("vertex.tangent",BN_FLOAT4,std::vector<vec4f>(3,vec4f(0,1,0,-1)));
  for (auto t : mesh.tangents()) expect(t,{0,1,0,-1},"authored vertex tangent wins");
  mesh.set("faceVarying.tangent",BN_FLOAT4,std::vector<vec4f>(3,vec4f(0,-1,0,1)));
  for (auto t : mesh.tangents()) expect(t,{0,-1,0,1},"authored face tangent wins over vertex");
  mesh.triangle->setData("faceVarying.tangent",nullptr);
  for (auto t : mesh.tangents()) expect(t,{0,1,0,-1},"removing face tangent exposes vertex");
  mesh.triangle->setData("vertex.tangent",nullptr);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"removing authored input regenerates");
  const vec2us mirrored[] = {{0,0},{0,65535},{65535,0}};
  uv->set(mirrored,3);
  for (auto t : mesh.tangents()) expect(t,{0,1,0,-1},"in-place UV update invalidates frames");
  const vec3f changed[] = {{0,0,0},{1,0,0},{0,0,1}};
  vertices->set(changed,3);
  for (auto t : mesh.tangents()) expect(t,{0,0,1,-1},"in-place positions update invalidates frames");
  mesh.triangle->setData("vertex.attribute0",nullptr);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"missing UVs give finite normal-orthogonal fallback");
  mesh.set("faceVarying.attribute0",BN_FLOAT2,
    std::vector<vec2f>{{NAN,0},{0,0},{0,0}});
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"NaN/degenerate UV fallback");
  const vec3f degenerate[] = {{0,0,0},{0,0,0},{INFINITY,0,0}};
  vertices->set(degenerate,3);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"degenerate/nonfinite positions fallback");
  const vec3i invalid[] = {{0,-1,1000}};
  indices->set(invalid,1);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"invalid indices must not read out of bounds");
}

static void nonindexedGenerator(Context &context)
{
  Mesh mesh(context);
  mesh.set("vertices",BN_FLOAT3,std::vector<vec3f>{
    {0,0,0},{1,0,0},{0,1,0},{0,0,0},{1,0,0},{0,1,0}});
  mesh.set("indices",BN_INT3,std::vector<vec3i>{{0,1,2},{3,4,5}});
  mesh.set("vertex.attribute0",BN_FLOAT2,std::vector<vec2f>{
    {0,0},{1,0},{0,1},{0,0},{1,0},{0,1}});
  mesh.triangle->commit();
  for (auto *device : *mesh.triangle->devices) {
    Triangles::DD dd;
    mesh.triangle->writeDD(dd,device);
    TangentMesh input{dd.vertices,nullptr,6,2,dd.attributes.normalAttribute,dd.attributes.attribute[0]};
    auto buffer = generateTangentFrames(device->rtc,input);
    vec4f output[6];
    device->rtc->copyAsync(output,buffer->getDD(),sizeof(output));
    device->sync();
    device->rtc->freeBuffer(buffer);
    for (auto t : output) expect(t,{1,0,0,1},"nonindexed generator output");
  }
}

static void highValence(Context &context)
{
  Mesh mesh(context);
  auto vertices = mesh.set("vertices",BN_FLOAT3,std::vector<vec3f>{{0,0,0},{1,0,0},{0,1,0}});
  mesh.set("indices",BN_INT3,std::vector<vec3i>(10000,vec3i(0,1,2)));
  mesh.set("vertex.attribute0",BN_FLOAT2,std::vector<vec2f>{{0,0},{1,0},{0,1}});
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"high-valence compatible group");
  const vec3f invalid[] = {{NAN,0,0},{NAN,0,0},{NAN,0,0}};
  vertices->set(invalid,3);
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"repeated nonfinite soup fallback");
}

static void benchmark(Context &context)
{
  constexpr int side = 256;
  std::vector<vec3f> positions;
  std::vector<vec2f> uvs;
  std::vector<vec3i> indices;
  for (int y = 0; y <= side; ++y)
    for (int x = 0; x <= side; ++x) {
      positions.emplace_back(float(x),float(y),0.f);
      uvs.emplace_back(float(x),float(y));
    }
  for (int y = 0; y < side; ++y)
    for (int x = 0; x < side; ++x) {
      const int v = x+(side+1)*y;
      indices.emplace_back(v,v+1,v+side+2);
      indices.emplace_back(v,v+side+2,v+side+1);
    }
  using Clock = std::chrono::steady_clock;
  auto ms = [](auto a, auto b) { return std::chrono::duration<double,std::milli>(b-a).count(); };
  Mesh mesh(context);
  const auto start = Clock::now();
  mesh.set("vertices",BN_FLOAT3,positions);
  mesh.set("indices",BN_INT3,indices);
  mesh.set("vertex.attribute0",BN_FLOAT2,uvs);
  const auto uploaded = Clock::now();
  mesh.triangle->commit();
  const auto committed = Clock::now();
  for (int i = 0; i < 20; ++i) mesh.triangle->commit();
  const auto cached = Clock::now();
  auto *device = mesh.triangle->devices->get(0);
  Triangles::DD dd;
  mesh.triangle->writeDD(dd,device);
  TangentMesh input{dd.vertices,dd.indices,int(positions.size()),int(indices.size()),
                    dd.attributes.normalAttribute,dd.attributes.attribute[0]};
  const auto generatedStart = Clock::now();
  for (int i = 0; i < 5; ++i) {
    auto buffer = generateTangentFrames(device->rtc,input);
    device->rtc->freeBuffer(buffer);
  }
  const auto generatedEnd = Clock::now();
  for (auto t : mesh.tangents()) expect(t,{1,0,0,1},"grid scan/bucket numerical check");
  std::cout << "benchmark warm context: triangles=" << indices.size()
    << " vertices=" << positions.size() << " input_bytes="
    << positions.size()*sizeof(vec3f)+uvs.size()*sizeof(vec2f)+indices.size()*sizeof(vec3i)
    << " upload_allocation_ms=" << ms(start,uploaded)
    << " first_commit_ms=" << ms(uploaded,committed)
    << " setup_plus_commit_ms=" << ms(start,committed)
    << " cached_commit_ms=" << ms(committed,cached)/20
    << " generator_allocation_sync_free_ms=" << ms(generatedStart,generatedEnd)/5 << '\n';
}

int main(int argc, char **argv)
{
  try {
    const int rank = 0, gpu = 0;
    std::unique_ptr<Context> context(LocalContext::create(1,&rank,1,&gpu));
#if defined(BARNEY_RTC_OPTIX) || defined(BARNEY_RTC_CUDA)
    cudaDeviceProp properties{};
    require(cudaGetDeviceProperties(&properties,0) == cudaSuccess,"cannot query CUDA execution device");
    std::cout << "CUDA execution device: " << properties.name << '\n';
#endif
    unitTriangle(*context);
    seamsAndWeighting(*context);
    authoredAndUpdates(*context);
    highValence(*context);
    nonindexedGenerator(*context);
    // Two logical devices on the available physical GPU exercise separate
    // ownership/cleanup without requiring a multi-GPU machine.
    {
      const int gpus[] = {0,0};
      std::unique_ptr<Context> replicated(LocalContext::create(1,&rank,2,gpus));
      unitTriangle(*replicated);
      authoredAndUpdates(*replicated);
    }
    if (argc > 1 && std::string(argv[1]) == "--benchmark") benchmark(*context);
    std::cout << "tangent generation tests passed\n";
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
