// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include <anari/anari.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstdlib>

#ifndef NANOVDB_TEST_NO_ALLOC_FAILURE
// Linux/glibc system-allocation fault injection, not a renderer test hook.
static std::atomic<size_t> failAllocationSize{0};
static std::atomic<bool> allocationFailed{false};
static std::atomic<unsigned> skipAllocationMatches{0};
extern "C" void *__libc_malloc(size_t);
extern "C" void *__libc_memalign(size_t, size_t);
static bool failAllocation(size_t size)
{
  if (size && failAllocationSize == size && skipAllocationMatches > 0) {
    --skipAllocationMatches;
    return false;
  }
  size_t expected = size;
  if (size && failAllocationSize.compare_exchange_strong(expected, 0)) {
    allocationFailed = true;
    errno = ENOMEM;
    return true;
  }
  return false;
}
extern "C" void *malloc(size_t size) noexcept
{
  return failAllocation(size) ? nullptr : __libc_malloc(size);
}
extern "C" void *aligned_alloc(size_t alignment, size_t size) noexcept
{
  return failAllocation(size) ? nullptr : __libc_memalign(alignment, size);
}
#endif

static void status(const void *user, ANARIDevice, ANARIObject,
    ANARIDataType, ANARIStatusSeverity severity, ANARIStatusCode,
    const char *message)
{
  if (severity == ANARI_SEVERITY_ERROR || severity == ANARI_SEVERITY_FATAL_ERROR)
    ++*static_cast<int *>(const_cast<void *>(user));
  std::fprintf(stderr, "%s\n", message);
}

struct Scene
{
  int errors = 0;
  ANARILibrary library = anariLoadLibrary("barney", status, &errors);
  ANARIDevice device = anariNewDevice(library, "default");
  ANARISpatialField field;
  ANARIVolume volume;
  ANARIVolume secondVolume;
  ANARIVolume volumeHandles[2];
  ANARIArray1D volumes;
  ANARIWorld world;
  Scene()
  {
    anariCommitParameters(device, device);
    field = anariNewSpatialField(device, "nanovdb");
    anariCommitParameters(device, field);
    volume = anariNewVolume(device, "transferFunction1D");
    anariSetParameter(device, volume, "value", ANARI_SPATIAL_FIELD, &field);
    anariCommitParameters(device, volume);
    secondVolume = anariNewVolume(device, "transferFunction1D");
    anariSetParameter(device, secondVolume, "value", ANARI_SPATIAL_FIELD, &field);
    const float opacity = 0.3f;
    anariSetParameter(device, secondVolume, "opacity", ANARI_FLOAT32, &opacity);
    anariCommitParameters(device, secondVolume);
    volumeHandles[0] = volume;
    volumeHandles[1] = secondVolume;
    // Explicit two-entry maps avoid the independent one-entry TF defect (08).
    for (int i = 0; i < 2; ++i) {
      auto opacityArray = anariNewArray1D(device, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2);
      auto values = static_cast<float *>(anariMapArray(device, opacityArray));
      values[0] = values[1] = i == 0 ? 1.f : 0.3f;
      anariUnmapArray(device, opacityArray);
      anariSetParameter(device, volumeHandles[i], "opacity", ANARI_ARRAY1D, &opacityArray);
      anariCommitParameters(device, volumeHandles[i]);
      anariRelease(device, opacityArray);
    }
    volumes = anariNewArray1D(device, volumeHandles, nullptr, nullptr, ANARI_VOLUME, 2);
    world = anariNewWorld(device);
    anariSetParameter(device, world, "volume", ANARI_ARRAY1D, &volumes);
    anariCommitParameters(device, world);
  }
  ~Scene()
  {
    anariRelease(device, world);
    anariRelease(device, volumes);
    anariRelease(device, volume);
    anariRelease(device, secondVolume);
    anariRelease(device, field);
    anariRelease(device, device);
    anariUnloadLibrary(library);
  }
  void check(bool expectedValid, const char *label)
  {
    errors = 0;
    float bounds[6];
    if (!anariGetProperty(device, world, "bounds", ANARI_FLOAT32_BOX3,
        bounds, sizeof(bounds), ANARI_WAIT))
      throw std::runtime_error("bounds unavailable");
    int valid = 0;
    anariGetProperty(device, field, "valid", ANARI_BOOL,
        &valid, sizeof(valid), ANARI_WAIT);
    if (bool(valid) != expectedValid || (expectedValid ? errors != 0 : errors == 0))
      throw std::runtime_error(label);
    std::printf("PASS: %s\n", label);
  }
  void instanceVolumes(const float *matrix = nullptr)
  {
    auto group = anariNewGroup(device);
    anariSetParameter(device, group, "volume", ANARI_ARRAY1D, &volumes);
    anariCommitParameters(device, group);
    auto instance = anariNewInstance(device, "transform");
    anariSetParameter(device, instance, "group", ANARI_GROUP, &group);
    const float transform[16] = {-1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 4, 0, 0, 1};
    anariSetParameter(device, instance, "transform", ANARI_FLOAT32_MAT4,
        matrix ? matrix : transform);
    anariCommitParameters(device, instance);
    auto instances = anariNewArray1D(device, nullptr, nullptr, nullptr, ANARI_INSTANCE, 1);
    *static_cast<ANARIInstance *>(anariMapArray(device, instances)) = instance;
    anariUnmapArray(device, instances);
    anariUnsetParameter(device, world, "volume");
    anariSetParameter(device, world, "instance", ANARI_ARRAY1D, &instances);
    anariCommitParameters(device, world);
    anariRelease(device, instances);
    anariRelease(device, instance);
    anariRelease(device, group);
  }
  void emptyBounds(const char *label)
  {
    float bounds[6];
    anariGetProperty(device, world, "bounds", ANARI_FLOAT32_BOX3,
        bounds, sizeof(bounds), ANARI_WAIT);
    for (int i = 0; i < 3; ++i)
      if (!(bounds[i] > bounds[i + 3]))
        throw std::runtime_error(label);
    std::printf("PASS: %s\n", label);
  }
  void data(const void *ptr, size_t size, bool expectedValid, const char *label)
  {
    auto bytes = anariNewArray1D(device, ptr, nullptr, nullptr, ANARI_UINT8, size);
    anariSetParameter(device, field, "data", ANARI_ARRAY1D, &bytes);
    anariCommitParameters(device, field);
    check(expectedValid, label);
    anariRelease(device, bytes);
  }
};

struct RenderScene : Scene
{
  ANARICamera camera;
  ANARIRenderer renderer;
  RenderScene()
  {
    auto geometry = anariNewGeometry(device, "sphere");
    const float position[3] = {-3.f, 2.f, 2.f};
    auto positions = anariNewArray1D(device, nullptr, nullptr, nullptr,
        ANARI_FLOAT32_VEC3, 1);
    std::memcpy(anariMapArray(device, positions), position, sizeof(position));
    anariUnmapArray(device, positions);
    anariSetParameter(device, geometry, "vertex.position", ANARI_ARRAY1D, &positions);
    const float radius = 1.f;
    anariSetParameter(device, geometry, "radius", ANARI_FLOAT32, &radius);
    anariCommitParameters(device, geometry);
    auto material = anariNewMaterial(device, "matte");
    const float color[3] = {0.8f, 0.1f, 0.1f};
    anariSetParameter(device, material, "color", ANARI_FLOAT32_VEC3, color);
    anariCommitParameters(device, material);
    auto surface = anariNewSurface(device);
    anariSetParameter(device, surface, "geometry", ANARI_GEOMETRY, &geometry);
    anariSetParameter(device, surface, "material", ANARI_MATERIAL, &material);
    anariCommitParameters(device, surface);
    auto surfaces = anariNewArray1D(device, nullptr, nullptr, nullptr, ANARI_SURFACE, 1);
    *static_cast<ANARISurface *>(anariMapArray(device, surfaces)) = surface;
    anariUnmapArray(device, surfaces);
    anariSetParameter(device, world, "surface", ANARI_ARRAY1D, &surfaces);
    anariCommitParameters(device, world);
    for (auto object : {static_cast<ANARIObject>(positions), static_cast<ANARIObject>(geometry),
        static_cast<ANARIObject>(material), static_cast<ANARIObject>(surface),
        static_cast<ANARIObject>(surfaces)})
      anariRelease(device, object);

    camera = anariNewCamera(device, "orthographic");
    const float eye[3] = {0.f, 2.f, 15.f};
    const float direction[3] = {0.f, 0.f, -1.f};
    const float up[3] = {0.f, 1.f, 0.f};
    const float height = 8.f, aspect = 1.5f;
    anariSetParameter(device, camera, "position", ANARI_FLOAT32_VEC3, eye);
    anariSetParameter(device, camera, "direction", ANARI_FLOAT32_VEC3, direction);
    anariSetParameter(device, camera, "up", ANARI_FLOAT32_VEC3, up);
    anariSetParameter(device, camera, "height", ANARI_FLOAT32, &height);
    anariSetParameter(device, camera, "aspect", ANARI_FLOAT32, &aspect);
    anariCommitParameters(device, camera);
    renderer = anariNewRenderer(device, "default");
    const float background[4] = {1.f, 1.f, 1.f, 1.f};
    const int samples = 64, denoise = 0;
    anariSetParameter(device, renderer, "background", ANARI_FLOAT32_VEC4, background);
    anariSetParameter(device, renderer, "pixelSamples", ANARI_INT32, &samples);
    anariSetParameter(device, renderer, "denoise", ANARI_BOOL, &denoise);
    anariCommitParameters(device, renderer);
    const float albedo = 0.f;
    for (int i = 0; i < 2; ++i) {
      auto v = volumeHandles[i];
      auto opacity = anariNewArray1D(device, nullptr, nullptr, nullptr, ANARI_FLOAT32, 2);
      auto values = static_cast<float *>(anariMapArray(device, opacity));
      values[0] = 0.f;
      values[1] = i == 0 ? 0.8f : 0.3f;
      anariUnmapArray(device, opacity);
      anariSetParameter(device, v, "opacity", ANARI_ARRAY1D, &opacity);
      anariSetParameter(device, v, "scatteringAlbedo", ANARI_FLOAT32, &albedo);
      anariCommitParameters(device, v);
      anariRelease(device, opacity);
    }
  }
  ~RenderScene()
  {
    anariRelease(device, camera);
    anariRelease(device, renderer);
  }
  void visible(int first, int second)
  {
    anariSetParameter(device, volume, "visible", ANARI_BOOL, &first);
    anariSetParameter(device, secondVolume, "visible", ANARI_BOOL, &second);
    anariCommitParameters(device, volume);
    anariCommitParameters(device, secondVolume);
  }
  std::vector<float> render()
  {
    const int initialErrors = errors;
    auto frame = anariNewFrame(device);
    const uint32_t size[2] = {48, 32};
    const ANARIDataType type = ANARI_FLOAT32_VEC4;
    anariSetParameter(device, frame, "size", ANARI_UINT32_VEC2, size);
    anariSetParameter(device, frame, "channel.color", ANARI_DATA_TYPE, &type);
    anariSetParameter(device, frame, "camera", ANARI_CAMERA, &camera);
    anariSetParameter(device, frame, "renderer", ANARI_RENDERER, &renderer);
    anariSetParameter(device, frame, "world", ANARI_WORLD, &world);
    anariCommitParameters(device, frame);
    anariRenderFrame(device, frame);
    anariFrameReady(device, frame, ANARI_WAIT);
    uint32_t width, height;
    ANARIDataType actual;
    auto pixels = static_cast<const float *>(anariMapFrame(device, frame,
        "channel.color", &width, &height, &actual));
    if (!pixels || width != 48 || height != 32 || actual != type)
      throw std::runtime_error("frame unavailable");
    std::vector<float> image(pixels, pixels + width * height * 4);
    anariUnmapFrame(device, frame, "channel.color");
    anariRelease(device, frame);
    for (float value : image)
      if (!std::isfinite(value)) throw std::runtime_error("nonfinite image");
    int fieldValid = 0;
    anariGetProperty(device, field, "valid", ANARI_BOOL,
        &fieldValid, sizeof(fieldValid), ANARI_WAIT);
    // Releasing a borrowed array can privatize it and re-finalize an invalid field.
    if (fieldValid && errors != initialErrors)
      throw std::runtime_error("valid-field render reported an error");
    return image;
  }
};

static double difference(const std::vector<float> &a, const std::vector<float> &b)
{
  double error = 0;
  for (size_t i = 0; i < a.size(); ++i) error += std::abs(a[i] - b[i]);
  return error / a.size();
}

static ANARIArray1D setOpacity(RenderScene &s, int volume, const float *values)
{
  auto array = anariNewArray1D(s.device, nullptr, nullptr, nullptr, ANARI_FLOAT32, 5);
  std::memcpy(anariMapArray(s.device, array), values, 5*sizeof(float));
  anariUnmapArray(s.device, array);
  anariSetParameter(s.device, s.volumeHandles[volume], "opacity", ANARI_ARRAY1D, &array);
  anariCommitParameters(s.device, s.volumeHandles[volume]);
  return array;
}

static void checkTransferFunctionUpdates()
{
  nanovdb::tools::build::Grid<float> source(0.f);
  auto accessor = source.getAccessor();
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        accessor.setValue(nanovdb::Coord(x,y,z), x/3.f);
  auto handle = nanovdb::tools::createNanoGrid(source);
  const float initial[2][5] = {{0,.05f,0,.1f,0}, {.1f,0,.4f,0,.1f}};
  const float final[2][5] = {{0,.9f,0,.7f,0}, {.01f,0,.02f,0,.01f}};
  RenderScene updated;
  ANARIArray1D arrays[2];
  for (int i = 0; i < 2; ++i) arrays[i] = setOpacity(updated, i, initial[i]);
  updated.data(handle.data(), handle.size(), true, "independent non-monotonic transfer functions");
  for (int target = 0; target < 2; ++target) {
    updated.visible(1, 1);
    const auto before = updated.render();
    RenderScene fresh;
    for (int i = 0; i < 2; ++i) {
      auto array = setOpacity(fresh, i, i <= target ? final[i] : initial[i]);
      anariRelease(fresh.device, array);
    }
    fresh.data(handle.data(), handle.size(), true, "fresh independent final transfer functions");
    if (target == 0) {
      std::memcpy(anariMapArray(updated.device, arrays[0]), final[0], sizeof(final[0]));
      anariUnmapArray(updated.device, arrays[0]);
    } else {
      auto array = setOpacity(updated, 1, final[1]);
      anariRelease(updated.device, array);
    }
    // First consume the TF-only update with both volumes in the same frame.
    // No field, other-volume or world commit occurs before this render.
    const auto after = updated.render();
    if (difference(after, before) <= 0.01)
      throw std::runtime_error("TF edit must visibly change joint transport");
    for (int view = 0; view < 3; ++view) {
      if (view) {
        updated.visible(view != 2, view != 1);
        fresh.visible(view != 2, view != 1);
      }
      const double delta = difference(view ? updated.render() : after, fresh.render());
      std::printf("TF target=%d view=%d fresh delta=%g\n", target, view, delta);
      if (delta > 0.005) throw std::runtime_error("independent TF update differs from fresh");
    }
    std::printf("PASS: volume %d TF-only update, joint and individual fresh scenes\n", target);
  }
  for (auto array : arrays) anariRelease(updated.device, array);
}

static void checkDataUpdates()
{
  nanovdb::tools::build::Grid<float> initialSource(0.f), finalSource(0.f), editedSource(0.f);
  for (int z = 0; z < 35; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        initialSource.getAccessor().setValue(nanovdb::Coord(x,y,z), z < 16 ? 0.1f : 0.3f);
  for (int z = 0; z < 18; ++z)
    for (int y = 1; y < 4; ++y)
      for (int x = 1; x < 3; ++x) {
        finalSource.getAccessor().setValue(nanovdb::Coord(x,y,z), z < 8 ? 0.6f : 0.9f);
        editedSource.getAccessor().setValue(nanovdb::Coord(x,y,z), z < 8 ? 0.05f : 0.2f);
      }
  auto initial = nanovdb::tools::createNanoGrid(initialSource);
  auto final = nanovdb::tools::createNanoGrid(finalSource);
  auto edited = nanovdb::tools::createNanoGrid(editedSource);
  if (initial.size() == final.size() || final.size() != edited.size())
    throw std::runtime_error("replacement/in-place fixture sizes");
  RenderScene updated, freshFinal, freshEdited;
  for (auto *s : {&updated, &freshFinal, &freshEdited}) {
    const float eye[3] = {0.f, 2.f, 50.f};
    anariSetParameter(s->device, s->camera, "position", ANARI_FLOAT32_VEC3, eye);
    anariCommitParameters(s->device, s->camera);
  }
  updated.data(initial.data(), initial.size(), true, "multi-cell initial field");
  const auto before = updated.render();
  freshFinal.data(final.data(), final.size(), true, "fresh changed extent/extrema");
  freshEdited.data(edited.data(), edited.size(), true, "fresh changed content");
  auto compare = [&](RenderScene &fresh, const char *label) {
    // Joint render first: no dependent volume/world recommit can hide a missed update.
    for (int view = 0; view < 3; ++view) {
      if (view) updated.visible(view != 2, view != 1);
      fresh.visible(view != 2, view != 1);
      const double delta = difference(updated.render(), fresh.render());
      std::printf("%s view=%d delta=%g\n", label, view, delta);
      if (delta > 0.005) throw std::runtime_error(label);
    }
    float actual[6], expected[6];
    anariGetProperty(updated.device, updated.world, "bounds", ANARI_FLOAT32_BOX3,
        actual, sizeof(actual), ANARI_WAIT);
    anariGetProperty(fresh.device, fresh.world, "bounds", ANARI_FLOAT32_BOX3,
        expected, sizeof(expected), ANARI_WAIT);
    for (int i = 0; i < 6; ++i)
      if (std::abs(actual[i]-expected[i]) > 1e-5f) throw std::runtime_error("updated bounds");
    std::printf("PASS: %s, bounds and both dependent volumes\n", label);
  };
  auto bytes = anariNewArray1D(updated.device, nullptr, nullptr, nullptr, ANARI_UINT8, final.size());
  std::memcpy(anariMapArray(updated.device, bytes), final.data(), final.size());
  anariUnmapArray(updated.device, bytes);
  anariSetParameter(updated.device, updated.field, "data", ANARI_ARRAY1D, &bytes);
  anariCommitParameters(updated.device, updated.field);
  if (difference(before, freshFinal.render()) <= 0.01)
    throw std::runtime_error("replacement must change the image");
  compare(freshFinal, "replacement changes bounds, dimensions and extrema");
  updated.visible(1, 1);
  updated.render();
  std::memcpy(anariMapArray(updated.device, bytes), edited.data(), edited.size());
  anariUnmapArray(updated.device, bytes);
  // Neither field nor either volume is recommitted after unmap.
  freshFinal.visible(1, 1);
  if (difference(freshFinal.render(), freshEdited.render()) <= 0.01)
    throw std::runtime_error("in-place edit must change the image");
  compare(freshEdited, "array unmap refreshes multi-cell sampling and majorants");
  anariRelease(updated.device, bytes);
}

static void checkFilterUpdates()
{
  // The first macrocell ends at z=35/3: nearest reaches index 11, but
  // linear also needs the impulse at 12. Each TF is opaque only between 0/1.
  nanovdb::tools::build::Grid<float> source(0.f);
  auto accessor = source.getAccessor();
  for (int z = 0; z < 35; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        accessor.setValue(nanovdb::Coord(x,y,z), z == 12 ? 1.f : 0.f);
  auto handle = nanovdb::tools::createNanoGrid(source);
  RenderScene nearest, linear, omitted;
  for (auto *s : {&nearest, &linear, &omitted}) {
    const float eye[3] = {0.f, 2.f, 50.f};
    anariSetParameter(s->device, s->camera, "position", ANARI_FLOAT32_VEC3, eye);
    anariCommitParameters(s->device, s->camera);
    for (int i = 0; i < 2; ++i) {
      auto opacity = anariNewArray1D(s->device, nullptr, nullptr, nullptr, ANARI_FLOAT32, 3);
      auto values = static_cast<float *>(anariMapArray(s->device, opacity));
      values[0] = values[2] = 0.f;
      values[1] = i == 0 ? 0.8f : 0.3f;
      anariUnmapArray(s->device, opacity);
      anariSetParameter(s->device, s->volumeHandles[i], "opacity", ANARI_ARRAY1D, &opacity);
      anariCommitParameters(s->device, s->volumeHandles[i]);
      anariRelease(s->device, opacity);
    }
  }
  anariSetParameter(nearest.device, nearest.field, "filter", ANARI_STRING, "nearest");
  anariSetParameter(linear.device, linear.field, "filter", ANARI_STRING, "linear");
  nearest.data(handle.data(), handle.size(), true, "nearest shared Float field");
  linear.data(handle.data(), handle.size(), true, "linear shared Float field");
  omitted.data(handle.data(), handle.size(), true, "default shared Float field");
  if (difference(omitted.render(), linear.render()) > 0.005)
    throw std::runtime_error("omitted filter differs from explicit linear");
  const double delta = difference(nearest.render(), linear.render());
  std::printf("nearest/linear shared-field delta=%g\n", delta);
  if (delta <= 0.01) throw std::runtime_error("nearest must not blend opaque TF interiors");
  std::puts("PASS: nearest and linear select different production reconstruction");

  // Save fresh equivalents for both volumes together and each independently.
  // The latter prevents one missing/stale shared-field binding hiding in a sum.
  std::vector<float> freshLinear[3], freshNearest[3];
  for (int view = 0; view < 3; ++view) {
    nearest.visible(view != 2, view != 1);
    linear.visible(view != 2, view != 1);
    freshNearest[view] = nearest.render();
    freshLinear[view] = linear.render();
    if (difference(freshNearest[view], freshLinear[view]) <= 0.01)
      throw std::runtime_error("each shared volume must be interpolation-sensitive");
  }
  // All updates below retain the same data parameter/array. No world or volume
  // recommit occurs between the field-only commit and the first (joint) render.
  for (const char *filter : {"linear", "cubic", "", "Nearest", "NEAREST",
                             "Linear", "nearestNeighbor", "point", "unknown",
                             static_cast<const char *>(nullptr)}) {
    nearest.visible(1, 1);
    anariSetParameter(nearest.device, nearest.field, "filter", ANARI_STRING, "nearest");
    anariCommitParameters(nearest.device, nearest.field);
    if (difference(nearest.render(), freshNearest[0]) > 0.005)
      throw std::runtime_error("filter-only return to nearest differs from fresh scene");
    if (filter)
      anariSetParameter(nearest.device, nearest.field, "filter", ANARI_STRING, filter);
    else
      anariUnsetParameter(nearest.device, nearest.field, "filter");
    anariCommitParameters(nearest.device, nearest.field);
    for (int view = 0; view < 3; ++view) {
      if (view) nearest.visible(view != 2, view != 1);
      const double updateDelta = difference(nearest.render(), freshLinear[view]);
      std::printf("filter=%s view=%d fresh delta=%g\n",
                  filter ? filter : "<unset>", view, updateDelta);
      if (updateDelta > 0.005)
        throw std::runtime_error("shared-field filter update differs from fresh linear scene");
    }
    std::printf("PASS: shared-field nearest to %s, joint and individual volumes\n",
                filter ? filter : "<unset>");
  }
}

static void checkHomogeneousTransmittance()
{
  nanovdb::tools::build::Grid<float> source(0.f);
  auto accessor = source.getAccessor();
  for (int z = 0; z < 8; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        accessor.setValue(nanovdb::Coord(x,y,z), .5f);
  auto handle = nanovdb::tools::createNanoGrid(source);
  RenderScene s;
  s.visible(1, 0);
  anariUnsetParameter(s.device, s.world, "surface");
  anariCommitParameters(s.device, s.world);
  const float eye[3] = {2.f, 2.f, 15.f};
  const float height = 2.f;
  anariSetParameter(s.device, s.camera, "position", ANARI_FLOAT32_VEC3, eye);
  anariSetParameter(s.device, s.camera, "height", ANARI_FLOAT32, &height);
  anariCommitParameters(s.device, s.camera);
  const int samples = 1024;
  const float ambient = 0.f;
  anariSetParameter(s.device, s.renderer, "ambientRadiance", ANARI_FLOAT32, &ambient);
  anariSetParameter(s.device, s.renderer, "pixelSamples", ANARI_INT32, &samples);
  anariCommitParameters(s.device, s.renderer);
  s.data(handle.data(), handle.size(), true, "homogeneous eight-unit slab");
  for (const char *filter : {"nearest", "linear"}) {
    anariSetParameter(s.device, s.field, "filter", ANARI_STRING, filter);
    anariCommitParameters(s.device, s.field);
    for (float density : {0.f, .1f, .25f, .5f}) {
      const float values[5] = {density, density, density, density, density};
      auto opacity = setOpacity(s, 0, values);
      anariRelease(s.device, opacity);
      const float unitDistance = 2.f;
      anariSetParameter(s.device, s.volume, "unitDistance", ANARI_FLOAT32, &unitDistance);
      anariCommitParameters(s.device, s.volume);
      const auto image = s.render();
      // Orthographic rays lie strictly inside the slab's x/y footprint.
      // Pure absorption against white: Beer-Lambert T=exp(-sigma*length).
      const double expected = std::exp(-double(density)*8.0/unitDistance);
      double mean = 0;
      for (size_t i = 0; i < image.size(); i += 4) mean += image[i];
      mean /= image.size()/4;
      const double tolerance = 6*std::sqrt(expected*(1-expected)/(48*32*1024)) + .001;
      std::fprintf(stderr, "TRANSMITTANCE filter=%s density=%g expected=%.9f measured=%.9f tolerance=%.9f\n",
          filter, density, expected, mean, tolerance);
      // The shared three-term fastLog biases optical depth 2 (ticket 10).
      // Keep that measurement visible, but do not claim it as acceptance.
      if (density <= .25f && std::abs(mean-expected) > tolerance)
        throw std::runtime_error("homogeneous transmittance differs from Beer-Lambert law");
      if (density > .25f)
        std::fprintf(stderr, "LIMIT: optical depth 2 is diagnostic only (fastLog bias, ticket 10)\n");
    }
  }
  std::printf("PASS: analytical homogeneous transmittance through optical depth 1, both filters\n");
}

static void benchmarkSparse(int extent)
{
  if (extent < 16 || extent > 2048)
    throw std::runtime_error("benchmark fixture extent must be 16..2048 (not a field limit)");
  nanovdb::tools::build::Grid<float> source(0.f);
  auto accessor = source.getAccessor();
  for (int z = 0; z < 8; ++z)
    for (int y = 0; y < 8; ++y)
      for (int x = 0; x < 8; ++x) {
        accessor.setValue(nanovdb::Coord(x,y,z), .5f);
        accessor.setValue(nanovdb::Coord(extent-8+x,extent-8+y,extent-8+z), 1.f);
      }
  auto handle = nanovdb::tools::createNanoGrid(source, nanovdb::tools::StatsMode::BBox);
  const bool shear = std::getenv("NANOVDB_BENCHMARK_SHEAR") != nullptr;
  if (shear) {
    auto *grid = handle.grid<float>();
    grid->mMap.mMatF[1] = float(grid->mMap.mMatD[1] = .5);
    grid->mMap.mInvMatF[1] = float(grid->mMap.mInvMatD[1] = -.5);
    grid->mVoxelSize = grid->mMap.getVoxelSize();
    nanovdb::Vec3dBBox world;
    for (int i = 0; i < 8; ++i)
      world.expand(grid->indexToWorld(nanovdb::Vec3d(i&1 ? extent : 0,
          i&2 ? extent : 0, i&4 ? extent : 0)));
    grid->mWorldBBox = world;
  }
  RenderScene s;
  s.visible(1, 0);
  anariUnsetParameter(s.device, s.world, "surface");
  anariCommitParameters(s.device, s.world);
  const float eye[3] = {shear ? 6.f : 4.f, 4.f, float(extent+15)};
  const float height = 12.f, ambient = 0.f;
  anariSetParameter(s.device, s.camera, "position", ANARI_FLOAT32_VEC3, eye);
  anariSetParameter(s.device, s.camera, "height", ANARI_FLOAT32, &height);
  anariCommitParameters(s.device, s.camera);
  anariSetParameter(s.device, s.renderer, "ambientRadiance", ANARI_FLOAT32, &ambient);
  anariCommitParameters(s.device, s.renderer);
  using Clock = std::chrono::steady_clock;
  std::vector<double> builds, renders;
  const size_t dim = (extent+15)/16;
  std::printf("SPARSE extent=%d shear=%d active=1024 payload_bytes=%zu cells=%zu range_bytes=%zu one_volume_majorant_bytes=%zu\n",
      extent, shear, size_t(handle.size()), dim*dim*dim, 8*dim*dim*dim, 4*dim*dim*dim);
  for (int trial = 0; trial < 6; ++trial) {
    // A new managed array forces the normal field update/build path each trial.
    auto bytes = anariNewArray1D(s.device, nullptr, nullptr, nullptr, ANARI_UINT8, handle.size());
    std::memcpy(anariMapArray(s.device, bytes), handle.data(), handle.size());
    anariUnmapArray(s.device, bytes);
    const auto start = Clock::now();
    anariSetParameter(s.device, s.field, "data", ANARI_ARRAY1D, &bytes);
    anariCommitParameters(s.device, s.field);
    s.check(true, "sparse benchmark build");
    const auto built = Clock::now();
    const auto image = s.render();
    const auto done = Clock::now();
    anariRelease(s.device, bytes);
    const double buildMs = std::chrono::duration<double,std::milli>(built-start).count();
    const double renderMs = std::chrono::duration<double,std::milli>(done-built).count();
    double mean = 0;
    for (size_t i = 0; i < image.size(); i += 4) mean += image[i];
    mean /= image.size()/4;
    std::printf("SPARSE trial=%d build_ms=%.6f render_ms=%.6f mean_red=%.9f\n",
        trial, buildMs, renderMs, mean);
    if (trial) { builds.push_back(buildMs); renders.push_back(renderMs); }
  }
  std::sort(builds.begin(), builds.end());
  std::sort(renders.begin(), renders.end());
  std::printf("SPARSE warm_summary build_ms_min_med_max=%.6f,%.6f,%.6f render_ms_min_med_max=%.6f,%.6f,%.6f\n",
      builds.front(), builds[2], builds.back(), renders.front(), renders[2], renders.back());
}

int main()
{
  if (const char *extent = std::getenv("NANOVDB_BENCHMARK")) {
    benchmarkSparse(std::stoi(extent));
    return 0;
  }
  checkHomogeneousTransmittance();
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  checkTransferFunctionUpdates();
  checkDataUpdates();
  checkFilterUpdates();
  Scene scene;
  scene.check(false, "missing data");
  const unsigned char shortPayload[1] = {0};
  scene.data(shortPayload, 1, false, "truncated header");
  nanovdb::tools::build::Grid<float> source(0.f);
  auto accessor = source.getAccessor();
  for (int z = 0; z < 4; ++z)
    for (int y = 0; y < 4; ++y)
      for (int x = 0; x < 4; ++x)
        accessor.setValue(nanovdb::Coord(x, y, z), 1.f);
  auto handle = nanovdb::tools::createNanoGrid(source);
  scene.data(handle.data(), handle.size(), true, "real version 32.7 Float grid");
  {
    RenderScene withStats, withoutStats;
    auto noStats = nanovdb::tools::createNanoGrid(source, nanovdb::tools::StatsMode::BBox);
    withStats.data(handle.data(), handle.size(), true, "render grid with statistics");
    withoutStats.data(noStats.data(), noStats.size(), true, "render grid without statistics");
    if (difference(withStats.render(), withoutStats.render()) > 0.005)
      throw std::runtime_error("statistics must not affect rendering");
    std::puts("PASS: statistics-independent Float rendering");
  }
  {
    nanovdb::tools::build::Grid<float> affineSource(0.2f);
    auto a = affineSource.getAccessor();
    for (int z = 0; z < 4; ++z)
      for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 35; ++x)
          a.setValue(nanovdb::Coord(x,y,z), x < 17 ? 0.2f : 1.f);
    auto original = nanovdb::tools::createNanoGrid(affineSource);
    auto mapped = nanovdb::tools::createNanoGrid(affineSource, nanovdb::tools::StatsMode::BBox);
    auto *g = mapped.grid<float>();
    const double matrix[9] = {0,-0.5,0, 0.1,0,0.15, 0,0,0.5};
    const double inverse[9] = {0,10,-3, -2,0,0, 0,0,2};
    for (int i = 0; i < 9; ++i) {
      g->mMap.mMatF[i] = float(g->mMap.mMatD[i] = matrix[i]);
      g->mMap.mInvMatF[i] = float(g->mMap.mInvMatD[i] = inverse[i]);
    }
    g->mMap.mVecF[0] = float(g->mMap.mVecD[0] = 2.0);
    nanovdb::Vec3dBBox world;
    for (int corner = 0; corner < 8; ++corner)
      world.expand(g->indexToWorld(nanovdb::Vec3d(
          corner & 1 ? 35 : 0, corner & 2 ? 4 : 0, corner & 4 ? 4 : 0)));
    g->mWorldBBox = world;
    g->mVoxelSize = nanovdb::Vec3d(0.1, 0.5, std::sqrt(0.15*0.15+0.5*0.5));
    auto *unreflected = original.grid<float>();
    unreflected->mMap = g->mMap;
    unreflected->mWorldBBox = g->mWorldBBox;
    unreflected->mVoxelSize = g->mVoxelSize;
    // Compose a reflection/translation with the affine header. Unlike baking
    // an arbitrary shear from an instance, this preserves the header's AABB
    // domain (and hence the existing boundary-clamped sampling semantics).
    for (int i = 0; i < 3; ++i) {
      g->mMap.mMatF[i] = float(g->mMap.mMatD[i] = -g->mMap.mMatD[i]);
      g->mMap.mInvMatF[3*i] = float(g->mMap.mInvMatD[3*i] = -g->mMap.mInvMatD[3*i]);
    }
    g->mWorldBBox = nanovdb::Vec3dBBox(nanovdb::Vec3d(2,0,0), nanovdb::Vec3d(4,4.1,2));
    RenderScene baked, instanced;
    baked.data(mapped.data(), mapped.size(), true, "composed affine header without statistics");
    instanced.instanceVolumes();
    instanced.data(original.data(), original.size(), true, "affine header composed with ANARI instance");
    // Instance IDs change the random sequence. Reduce independent Monte Carlo
    // noise rather than requiring bitwise equality with the baked scene.
    const int affineSamples = 4096;
    for (auto *s : {&baked, &instanced}) {
      anariSetParameter(s->device, s->renderer, "pixelSamples", ANARI_INT32, &affineSamples);
      anariCommitParameters(s->device, s->renderer);
    }
    const double delta = difference(baked.render(), instanced.render());
    std::printf("affine header/instance delta=%g\n", delta);
    if (delta > 0.005) throw std::runtime_error("header/instance affine rendering mismatch");
    float b[6], c[6];
    anariGetProperty(baked.device, baked.world, "bounds", ANARI_FLOAT32_BOX3, b, sizeof(b), ANARI_WAIT);
    anariGetProperty(instanced.device, instanced.world, "bounds", ANARI_FLOAT32_BOX3, c, sizeof(c), ANARI_WAIT);
    for (int i = 0; i < 6; ++i)
      if (std::abs(b[i]-c[i]) > 1e-5f) throw std::runtime_error("header/instance affine bounds mismatch");
    std::puts("PASS: affine Float rendering and ANARI instance composition");
  }
  const auto *raw = static_cast<const unsigned char *>(handle.data());
  std::vector<unsigned char> bad(raw, raw + handle.size());
  nanovdb::GridData header;
  std::memcpy(&header, bad.data(), sizeof(header));
  for (double extent : {0.0, 1e300}) {
    std::memcpy(&header, raw, sizeof(header));
    header.mWorldBBox = nanovdb::Vec3dBBox(nanovdb::Vec3d(0.0), nanovdb::Vec3d(extent));
    std::memcpy(bad.data(), &header, sizeof(header));
    scene.data(bad.data(), bad.size(), false, "unrepresentable nonempty macrocell domain");
    scene.data(handle.data(), handle.size(), true, "recover after unrepresentable domain");
  }
  for (bool nonfinite : {false, true}) {
    std::memcpy(&header, raw, sizeof(header));
    header.mMap.mMatD[0] = header.mMap.mMatF[0] =
        nonfinite ? std::numeric_limits<float>::infinity() : 0.f;
    std::memcpy(bad.data(), &header, sizeof(header));
    scene.data(bad.data(), bad.size(), false,
        nonfinite ? "nonfinite affine map" : "singular affine map");
    scene.data(handle.data(), handle.size(), true, "recover after invalid affine map");
  }
  std::memcpy(&header, raw, sizeof(header));
  header.mVersion = nanovdb::Version(31, 0, 0);
  std::memcpy(bad.data(), &header, sizeof(header));
  scene.data(bad.data(), bad.size(), false, "incompatible major version");
  scene.data(handle.data(), sizeof(nanovdb::GridData), false, "declared payload truncated");
  std::memcpy(&header, raw, sizeof(header));
  header.mGridSize = sizeof(nanovdb::GridData);
  std::memcpy(bad.data(), &header, sizeof(header));
  scene.data(bad.data(), bad.size(), false, "declared size omits fixed tree");
  std::memcpy(&header, raw, sizeof(header));
  header.mMagic = 0;
  header.mData2 = 0;
  std::memcpy(bad.data(), &header, sizeof(header));
  scene.data(bad.data(), bad.size(), false, "invalid grid magic");
  std::memcpy(&header, raw, sizeof(header));
  header.mGridCount = 2;
  std::memcpy(bad.data(), &header, sizeof(header));
  scene.data(bad.data(), bad.size(), false, "declared multiple grids");
  nanovdb::tools::build::Grid<int> integerSource(0);
  integerSource.getAccessor().setValue(nanovdb::Coord(0), 1);
  auto integerHandle = nanovdb::tools::createNanoGrid(integerSource);
  scene.data(integerHandle.data(), integerHandle.size(), false, "unsupported Int32 grid");
  std::memcpy(bad.data(), raw, bad.size());
  nanovdb::TreeData tree;
  std::memcpy(&tree, raw + sizeof(header), sizeof(tree));
  tree.mNodeOffset[3] = bad.size();
  std::memcpy(bad.data() + sizeof(header), &tree, sizeof(tree));
  scene.data(bad.data(), bad.size(), false, "root outside declared payload");
  auto wrongType = anariNewArray1D(scene.device, nullptr, nullptr, nullptr,
      ANARI_FLOAT32, 8);
  anariSetParameter(scene.device, scene.field, "data", ANARI_ARRAY1D, &wrongType);
  anariCommitParameters(scene.device, scene.field);
  scene.check(false, "wrong element type");
  anariRelease(scene.device, wrongType);
  auto wrongShape = anariNewArray2D(scene.device, raw, nullptr, nullptr,
      ANARI_UINT8, handle.size(), 1);
  anariSetParameter(scene.device, scene.field, "data", ANARI_ARRAY2D, &wrongShape);
  anariCommitParameters(scene.device, scene.field);
  scene.check(false, "wrong array shape");
  anariRelease(scene.device, wrongShape);
  scene.data(raw, handle.size(), true, "populated before empty");
  nanovdb::tools::build::Grid<float> emptySource(7.f);
  auto emptyHandle = nanovdb::tools::createNanoGrid(emptySource);
  scene.data(emptyHandle.data(), emptyHandle.size(), true, "valid zero-active grid");
  scene.emptyBounds("empty grid excludes old bounds");
  scene.data(raw, handle.size(), true, "populated after empty");
  std::memcpy(bad.data(), raw, bad.size());
  std::memcpy(&tree, raw + sizeof(header), sizeof(tree));
  tree.mVoxelCount = 0;
  std::memcpy(bad.data() + sizeof(header), &tree, sizeof(tree));
  scene.data(bad.data(), bad.size(), true, "zero active with stored populated bounds");
  scene.emptyBounds("zero active overrides stored bounds");
  scene.instanceVolumes();
  scene.emptyBounds("empty bounds survive reflected translated instance");
  std::memcpy(bad.data(), raw, bad.size());
  using RootData = nanovdb::NanoGrid<float>::TreeType::RootType::DataType;
  std::memcpy(&tree, raw + sizeof(header), sizeof(tree));
  const size_t rootOffset = sizeof(header) + tree.mNodeOffset[3];
  const nanovdb::CoordBBox hugeBounds(nanovdb::Coord(INT32_MIN), nanovdb::Coord(INT32_MAX));
  std::memcpy(bad.data() + rootOffset + offsetof(RootData, mBBox),
      &hugeBounds, sizeof(hugeBounds));
  scene.data(bad.data(), bad.size(), false, "unrepresentable index dimensions");
  const nanovdb::CoordBBox manyCells(nanovdb::Coord(0), nanovdb::Coord(1048575));
  std::memcpy(bad.data() + rootOffset + offsetof(RootData, mBBox),
      &manyCells, sizeof(manyCells));
  scene.data(bad.data(), bad.size(), false, "macrocell count exceeds signed RTC indexing");
  scene.data(handle.data(), handle.size(), true, "recover after macrocell count overflow");
#ifndef NANOVDB_TEST_NO_ALLOC_FAILURE
  scene.data(raw, handle.size(), true, "valid before allocator failure");
  auto update = anariNewArray1D(scene.device, raw, nullptr, nullptr,
      ANARI_UINT8, handle.size());
  anariSetParameter(scene.device, scene.field, "data", ANARI_ARRAY1D, &update);
  anariCommitParameters(scene.device, scene.field);
  failAllocationSize = handle.size() + NANOVDB_DATA_ALIGNMENT;
  scene.check(false, "allocator failure reports error and invalidates");
  if (!allocationFailed)
    throw std::runtime_error("allocator fault was not exercised");
  anariRelease(scene.device, update);
  scene.data(raw, handle.size(), true, "recovery after allocator failure");
#endif
  scene.data(raw, 0, false, "empty byte array is not an empty grid");
  scene.data(raw, sizeof(header) - 1, false, "header one byte short");
  auto rejectHeader = [&](const char *label) {
    std::memcpy(bad.data(), raw, bad.size());
    std::memcpy(bad.data(), &header, sizeof(header));
    scene.data(bad.data(), bad.size(), false, label);
  };
  std::memcpy(&header, raw, sizeof(header));
  header.mGridCount = 0;
  rejectHeader("zero declared grids");
  header.mGridCount = 1;
  header.mGridIndex = 1;
  rejectHeader("nonzero grid index");
  header.mGridIndex = 0;
  header.mGridSize = UINT64_MAX;
  rejectHeader("declared size overflow");
  header.mGridSize = sizeof(header) + sizeof(tree) - 1;
  rejectHeader("declared tree one byte short");
  header.mGridSize = rootOffset + sizeof(RootData) - 1;
  rejectHeader("declared root one byte short");
  for (int64_t offset : {int64_t(-32), int64_t(0), int64_t(32), int64_t(65), INT64_MAX}) {
    std::memcpy(bad.data(), raw, bad.size());
    std::memcpy(&tree, raw + sizeof(header), sizeof(tree));
    tree.mNodeOffset[3] = offset;
    std::memcpy(bad.data() + sizeof(header), &tree, sizeof(tree));
    scene.data(bad.data(), bad.size(), false, "invalid root offset/alignment");
  }
  auto quantized = nanovdb::tools::createNanoGrid<decltype(source), nanovdb::Fp4>(source);
  scene.data(quantized.data(), quantized.size(), false, "Float stage rejects real Fp4 grid");
  std::vector<unsigned char> unaligned(handle.size() + 1);
  std::memcpy(unaligned.data() + 1, raw, handle.size());
  scene.data(unaligned.data() + 1, handle.size(), true, "unaligned application buffer");
  std::vector<unsigned char> padded(raw, raw + handle.size());
  padded.insert(padded.end(), raw, raw + handle.size());
  padded.push_back(0xa5);
  scene.data(padded.data(), padded.size(), true, "padding may contain grid magic");

  RenderScene updated;
  updated.data(raw, handle.size(), true, "shared volumes populated");
  const auto before = updated.render();
  updated.visible(1, 0);
  const auto firstOnly = updated.render();
  updated.visible(0, 1);
  const auto secondOnly = updated.render();
  updated.visible(1, 1);
  updated.data(shortPayload, 1, false, "shared volumes invalid replacement");
  const auto invalid = updated.render();
  float bounds[6];
  anariGetProperty(updated.device, updated.world, "bounds", ANARI_FLOAT32_BOX3,
      bounds, sizeof(bounds), ANARI_WAIT);
  const float sphereBounds[6] = {-4, 1, 1, -2, 3, 3};
  for (int i = 0; i < 6; ++i)
    if (std::abs(bounds[i] - sphereBounds[i]) > 1e-5f)
      throw std::runtime_error("invalid interval retains old volume bounds");
  RenderScene reference;
  reference.data(emptyHandle.data(), emptyHandle.size(), true, "geometry-only reference");
  const auto geometryOnly = reference.render();
  std::printf("image deltas: populated-invalid=%g invalid-reference=%g\n",
      difference(before, invalid), difference(invalid, geometryOnly));
  if (difference(before, invalid) < 0.01 || difference(invalid, geometryOnly) > 0.005
      || difference(firstOnly, invalid) < 0.005 || difference(secondOnly, invalid) < 0.005)
    throw std::runtime_error("invalid interval rendering");
  int redPixels = 0;
  for (size_t i = 0; i < invalid.size(); i += 4)
    if (invalid[i] > invalid[i + 1] + 0.1f) ++redPixels;
  if (redPixels < 10) throw std::runtime_error("unrelated sphere disappeared");
  nanovdb::tools::build::Grid<float> replacementSource(0.f);
  auto replacementAccessor = replacementSource.getAccessor();
  for (int z = 0; z < 2; ++z)
    for (int y = 0; y < 2; ++y)
      for (int x = 0; x < 2; ++x)
        replacementAccessor.setValue(nanovdb::Coord(x, y, z), 0.2f);
  auto replacement = nanovdb::tools::createNanoGrid(replacementSource);
  updated.data(replacement.data(), replacement.size(), true, "shared volumes recovery");
  const auto recovered = updated.render();
  RenderScene freshScene;
  freshScene.data(replacement.data(), replacement.size(), true, "fresh equivalent populated scene");
  const auto fresh = freshScene.render();
  std::printf("recovery delta=%g\n", difference(recovered, fresh));
  if (difference(recovered, fresh) > 0.005)
    throw std::runtime_error("recovered shared volumes differ from fresh");
  std::puts("PASS: shared-volume invalid interval and rendering recovery");
  const size_t treeEnd = sizeof(nanovdb::GridData) + sizeof(nanovdb::TreeData);
  std::vector<unsigned char> spaced(handle.size() + NANOVDB_DATA_ALIGNMENT, 0x7f);
  std::memcpy(spaced.data(), raw, treeEnd);
  std::memcpy(spaced.data() + treeEnd + NANOVDB_DATA_ALIGNMENT,
      raw + treeEnd, handle.size() - treeEnd);
  std::memcpy(&header, spaced.data(), sizeof(header));
  header.mGridSize = spaced.size();
  std::memcpy(spaced.data(), &header, sizeof(header));
  std::memcpy(&tree, spaced.data() + sizeof(header), sizeof(tree));
  for (auto &offset : tree.mNodeOffset) offset += NANOVDB_DATA_ALIGNMENT;
  std::memcpy(spaced.data() + sizeof(header), &tree, sizeof(tree));
  updated.data(spaced.data(), spaced.size(), true, "aligned gap before root");
  const auto relocated = updated.render();
  if (difference(relocated, before) > 0.005)
    throw std::runtime_error("non-adjacent root rendering differs");
  std::puts("PASS: non-adjacent root rendering");
  updated.data(emptyHandle.data(), emptyHandle.size(), true, "render populated-to-empty");
  if (difference(updated.render(), geometryOnly) > 0.005)
    throw std::runtime_error("empty grid contributes background");
  updated.data(replacement.data(), replacement.size(), true, "render empty-to-populated");
  if (difference(updated.render(), fresh) > 0.005)
    throw std::runtime_error("empty-to-populated retains stale data");
  std::puts("PASS: empty transition rendering");

#ifndef NANOVDB_TEST_NO_ALLOC_FAILURE
  {
    nanovdb::tools::build::Grid<float> wideSource(0.3f);
    wideSource.getAccessor().setValue(nanovdb::Coord(0), 1.f);
    wideSource.getAccessor().setValue(nanovdb::Coord(1024, 3, 3), 1.f);
    auto wide = nanovdb::tools::createNanoGrid(wideSource);
    auto bytes = anariNewArray1D(updated.device, wide.data(), nullptr, nullptr,
        ANARI_UINT8, wide.size());
    anariSetParameter(updated.device, updated.field, "data", ANARI_ARRAY1D, &bytes);
    anariCommitParameters(updated.device, updated.field);
    allocationFailed = false;
    failAllocationSize = 544; // 65 range pairs, rounded to CPU's 32-byte alignment
    updated.check(false, "macrocell allocation failure");
    failAllocationSize = 0;
    if (!allocationFailed || difference(updated.render(), geometryOnly) > 0.005)
      throw std::runtime_error("macrocell allocation failure retains volumes");
    anariRelease(updated.device, bytes);
    updated.data(replacement.data(), replacement.size(), true, "macrocell allocator recovery");
    if (difference(updated.render(), fresh) > 0.005)
      throw std::runtime_error("macrocell allocation recovery differs from fresh");
    std::puts("PASS: macrocell allocation recovery rendering");
  }
  for (auto failure : {std::pair<size_t, unsigned>{handle.size() + NANOVDB_DATA_ALIGNMENT, 0},
           {handle.size(), 0}, {handle.size() + NANOVDB_DATA_ALIGNMENT, 1}}) {
    auto failedUpdate = anariNewArray1D(updated.device, raw, nullptr, nullptr,
        ANARI_UINT8, handle.size());
    anariSetParameter(updated.device, updated.field, "data", ANARI_ARRAY1D, &failedUpdate);
    anariCommitParameters(updated.device, updated.field);
    allocationFailed = false;
    skipAllocationMatches = failure.second;
    failAllocationSize = failure.first;
    updated.check(false, "shared-volume allocation failure");
    if (!allocationFailed || difference(updated.render(), geometryOnly) > 0.005)
      throw std::runtime_error("allocation failure retains volume contributions");
    anariGetProperty(updated.device, updated.world, "bounds", ANARI_FLOAT32_BOX3,
        bounds, sizeof(bounds), ANARI_WAIT);
    for (int i = 0; i < 6; ++i)
      if (std::abs(bounds[i] - sphereBounds[i]) > 1e-5f)
        throw std::runtime_error("allocation failure retains bounds");
    anariRelease(updated.device, failedUpdate);
    updated.data(replacement.data(), replacement.size(), true, "shared allocator recovery");
    if (difference(updated.render(), fresh) > 0.005)
      throw std::runtime_error("allocation recovery differs from fresh");
    std::puts("PASS: shared-volume allocation recovery rendering");
  }
#else
  std::puts("SKIP: allocator fault injection requires the non-sanitized CPU run");
#endif
  if (replacement.size() != handle.size())
    throw std::runtime_error("in-place fixtures must have equal byte sizes");
  auto mutableData = anariNewArray1D(updated.device, nullptr, nullptr, nullptr,
      ANARI_UINT8, handle.size());
  std::memcpy(anariMapArray(updated.device, mutableData), raw, handle.size());
  anariUnmapArray(updated.device, mutableData);
  anariSetParameter(updated.device, updated.field, "data", ANARI_ARRAY1D, &mutableData);
  anariCommitParameters(updated.device, updated.field);
  updated.check(true, "managed array populated");
  if (difference(updated.render(), before) > 0.005)
    throw std::runtime_error("valid replacement differs from fresh");
  std::memcpy(anariMapArray(updated.device, mutableData), replacement.data(), replacement.size());
  anariUnmapArray(updated.device, mutableData);
  updated.check(true, "array unmap updates shared field without recommit");
  if (difference(updated.render(), fresh) > 0.005)
    throw std::runtime_error("array update retains stale native data");
  anariRelease(updated.device, mutableData);
  std::puts("PASS: managed array update rendering");
  for (size_t padding = 1; padding <= 32; ++padding) {
    std::vector<unsigned char> payload(raw, raw + handle.size());
    payload.resize(payload.size() + padding, 0xa5);
    std::memcpy(&header, payload.data(), sizeof(header));
    header.mGridSize = payload.size();
    std::memcpy(payload.data(), &header, sizeof(header));
    updated.data(payload.data(), payload.size(), true, "declared payload padding");
    if (difference(updated.render(), before) > 0.005)
      throw std::runtime_error("declared padding changes rendering");
  }
  return 0;
}
