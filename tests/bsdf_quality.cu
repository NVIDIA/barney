// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/packedBSDF/Lambertian.h"
#include "native/packedBSDF/PhysicallyBased.h"
#include "native/packedBSDF/NVisii.h"
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace BARNEY_NS;
using namespace BARNEY_NS::native;

namespace bsdf_quality {
constexpr double pi = 3.14159265358979323846;
constexpr int trials = 262144;
constexpr double roundoff = 2e-5;
constexpr int bins = 16;

enum Model { LAMBERT, PBR, NVISII };
struct Parameters
{
  Model model = LAMBERT;
  ShadingInteraction dg{};
  vec3f color = vec3f(.125f,.5f,.875f);
  vec3f normal = vec3f(0,0,1);
  vec3f coatNormal = vec3f(0.f); // Zero keeps the base normal.
  float roughness = .5f, metallic = 0.f, specular = 0.f;
  float coat = 0.f, transmission = 0.f, ior = 1.5f;
};
struct Case
{
  std::string name;
  Parameters p;
  bool metal = false, quadrature = true;
  double sampleBound = 0.;
};

void require(bool ok, const std::string &message)
{
  if (!ok) throw std::runtime_error(message);
}
bool close(double a, double b, double relative = 5e-5)
{
  return std::isfinite(a) && std::isfinite(b)
      && std::abs(a-b) <= 1e-7 + relative * std::max(std::abs(a),std::abs(b));
}
void nonnegative(double x, const std::string &message)
{
  require(std::isfinite(x) && x >= 0., message);
}

struct Buffer
{
  rtc::Device *device;
  rtc::Buffer *buffer;
  Buffer(rtc::Device *device, size_t bytes, const void *data = nullptr)
      : device(device), buffer(device->createBuffer(bytes, data)) {}
  Buffer(const Buffer &) = delete;
  ~Buffer() { device->freeBuffer(buffer); }
  template<typename T> T *ptr() { return static_cast<T *>(buffer->getDD()); }
  template<typename T> std::vector<T> read(size_t n)
  {
    std::vector<T> result(n);
    device->copyAsync(result.data(), ptr<T>(), n * sizeof(T));
    device->sync();
    return result;
  }
};

#if RTC_DEVICE_CODE
inline __rtc_device void setup(packedBSDF::Lambertian &b, Parameters p)
{
  b.albedo = rtc::float3(p.color.x,p.color.y,p.color.z);
}
inline __rtc_device void setup(packedBSDF::PhysicallyBased &b, Parameters p)
{
  b.setDefaults();
  b.baseColor = p.color;
  b.normal = p.normal;
  b.clearcoatNormal = dot(p.coatNormal,p.coatNormal) > 0.f
      ? p.coatNormal : p.normal;
  b.roughness = p.roughness;
  b.metallic = p.metallic;
  b.specular = p.specular;
  b.clearcoat = p.coat;
  b.clearcoatRoughness = .5f;
  b.transmission = p.transmission;
  b.ior = p.ior;
}
inline __rtc_device void setup(packedBSDF::NVisii &b, Parameters p)
{
  b.setDefaults();
  b.baseColor = p.color;
  b.roughness = b.transmissionRoughness = p.roughness;
  b.metallic = p.metallic;
  b.specular = p.specular;
  b.clearcoat = p.coat;
  b.clearcoatGloss = 0.f;
  b.specularTransmission = p.transmission;
  b.ior = p.ior;
}
#endif

// Separate entry points prevent CSE from making pdf-only and value-only
// invocation contexts identical (particularly important at the roughness floor).
#define DISPATCH(OP) \
  if (p.model == LAMBERT) { packedBSDF::Lambertian b; setup(b,p); OP; } \
  else if (p.model == PBR) { packedBSDF::PhysicallyBased b; setup(b,p); OP; } \
  else { packedBSDF::NVisii b; setup(b,p); OP; }

__rtc_global void projected(const rtc::ComputeInterface &ci, Parameters p,
                           const vec3f *directions, EvalRes *out, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i >= n) return;
  DISPATCH(out[i] = b.evalProjected(p.dg,directions[i],false));
}
#endif

__rtc_global void raw(const rtc::ComputeInterface &ci, Parameters p,
                     const vec3f *directions, EvalRes *out, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i >= n) return;
  DISPATCH(out[i] = b.eval(p.dg,directions[i],false));
}
#endif

__rtc_global void density(const rtc::ComputeInterface &ci, Parameters p,
                         const vec3f *directions, float *out, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i >= n) return;
  DISPATCH(out[i] = b.pdf(p.dg,directions[i],false));
}
#endif

__rtc_global void sample(const rtc::ComputeInterface &ci, Parameters p,
                        ScatterResult *out, vec3f *directions, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i >= n) return;
  Random rng(uint32_t(i),0x51b5dfu);
  DISPATCH(b.scatter(out[i],p.dg,rng,false));
  directions[i] = out[i].dir;
}
#endif
#undef DISPATCH

struct NormalInput
{
  vec3f geometric, view, shading;
};

__rtc_global void adjustNormals(const rtc::ComputeInterface &ci,
                               const NormalInput *in, vec3f *out, int n)
#if !RTC_DEVICE_CODE
;
#else
{
  const int i = ci.launchIndex().x;
  if (i < n)
    out[i] = adjustNormalForView(in[i].geometric, in[i].view, in[i].shading);
}
#endif

__rtc_global void resolvedNormals(const rtc::ComputeInterface &ci,
                                 Parameters p, vec3f *out)
#if !RTC_DEVICE_CODE
;
#else
{
  if (ci.launchIndex().x != 0) return;
  packedBSDF::PhysicallyBased b;
  setup(b,p);
  const auto normals = b.resolveShadingNormals(p.dg,b.unpack());
  out[0] = normals.base;
  out[1] = normals.clearcoat;
}
#endif

void checkNormalAdjustment(rtc::Device *device)
{
  const vec3f ng(0,0,1);
  const std::vector<NormalInput> inputs = {
    {ng,ng,ng},
    {ng,ng,vec3f(.6f,0,.8f)},
    {ng,ng,vec3f(.8f,0,.6f)},
    {ng,ng,vec3f(-.8f,0,.6f)},
    {ng,normalize(vec3f(1,0,.001f)),ng},
    {ng,ng,-ng},
    {ng,ng,normalize(vec3f(1e-8f,0,-1))},
    {ng,vec3f(.6f,0,.8f),vec3f(-.8f,0,.6f)},
    {ng,normalize(vec3f(1,0,.001f)),vec3f(-.8f,0,.6f)}
  };
  // For V=Ng, 2*Nz^2-1=.01 gives Nz=sqrt(.505), Nx=sqrt(.495).
  const std::vector<vec3f> expected = {
    ng,inputs[1].shading,vec3f(.703562364f,0,.710633520f),
    vec3f(-.703562364f,0,.710633520f),ng,-ng,inputs[6].shading,
    // VisRTX rotation formula evaluated in double for an oblique view.
    vec3f(-.442735813f,0,.896652106f)
  };
  Buffer in(device,inputs.size()*sizeof(NormalInput),inputs.data());
  Buffer out(device,inputs.size()*sizeof(vec3f));
  __rtc_launch(device,adjustNormals,1,128,in.ptr<NormalInput>(),
               out.ptr<vec3f>(),int(inputs.size()));
  const auto normals = out.read<vec3f>(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    require(std::isfinite(dot(normals[i],normals[i]))
            && std::abs(dot(normals[i],normals[i])-1.f) < 2e-5f,
            "view adjustment must return a finite unit normal");
    if (i < expected.size())
      for (int c = 0; c < 3; ++c)
        require(close(normals[i][c],expected[i][c]),
                "view adjustment disagrees with reference case "+std::to_string(i));
    if (i == 2 || i == 3 || i >= 7) {
      const vec3f view = inputs[i].view;
      const vec3f mirror = 2.f*dot(normals[i],view)*normals[i]-view;
      const float threshold = std::min(.9f*dot(ng,view),.01f);
      require(dot(ng,mirror) >= threshold-2e-6f,
              "adjusted mirror direction must clear geometric surface");
    }
  }
  std::cout << "View normal adjustment: " << inputs.size() << " cases passed\n";
}

struct Evaluation
{
  std::vector<EvalRes> projected, raw;
  std::vector<float> pdf;
  vec3f baseNormal;
  vec3f coatNormal;
};
Evaluation evaluate(rtc::Device *device, Parameters p, Buffer &directions, int n)
{
  Buffer projectedOut(device,n*sizeof(EvalRes));
  Buffer rawOut(device,n*sizeof(EvalRes));
  Buffer pdfOut(device,n*sizeof(float));
  __rtc_launch(device,projected,(n+127)/128,128,p,
               directions.ptr<vec3f>(),projectedOut.ptr<EvalRes>(),n);
  __rtc_launch(device,raw,(n+127)/128,128,p,
               directions.ptr<vec3f>(),rawOut.ptr<EvalRes>(),n);
  __rtc_launch(device,density,(n+127)/128,128,p,
               directions.ptr<vec3f>(),pdfOut.ptr<float>(),n);
  vec3f baseNormal = dot(p.dg.geometricNormal,p.dg.shadingNormal) < 0.f
      ? -p.dg.shadingNormal : p.dg.shadingNormal;
  vec3f coatNormal = baseNormal;
  if (p.model == PBR) {
    Buffer normals(device,2*sizeof(vec3f));
    __rtc_launch(device,resolvedNormals,1,1,p,normals.ptr<vec3f>());
    const auto resolved = normals.read<vec3f>(2);
    baseNormal = resolved[0];
    coatNormal = resolved[1];
  }
  return {projectedOut.read<EvalRes>(n),rawOut.read<EvalRes>(n),
          pdfOut.read<float>(n),baseNormal,coatNormal};
}

void checkEvaluation(const Case &test, const std::vector<vec3f> &directions,
                     const Evaluation &e)
{
  const vec3f normal = e.baseNormal;
  for (size_t i = 0; i < directions.size(); ++i) {
    nonnegative(e.pdf[i],test.name+": invalid density");
    require(close(e.pdf[i],e.projected[i].pdf)
            && close(e.pdf[i],e.raw[i].pdf),
            test.name+": standalone/evaluation density disagree");
    // Raw base and coat lobes project independently. Reconstruct the coat
    // term in double rather than asking production evaluation for a cosine.
    double coatProjectionCorrection = 0.;
    const vec3f wi = directions[i], view = test.p.dg.wo;
    const vec3f nc = e.coatNormal;
    const double nv = dot(nc,view), nl = dot(nc,wi);
    if (test.p.model == PBR && test.p.coat > 0.f && normal != nc
        && dot(test.p.dg.geometricNormal,wi)
               * dot(test.p.dg.geometricNormal,view) > 0.f
        && nv > 0. && nl > 0.) {
      const vec3f h = normalize(view+wi);
      const double nh = std::max(0.,double(dot(nc,h)));
      const double vh = dot(view,h);
      const double alpha2 = .0625; // Packed clearcoat roughness is .5.
      const double q = std::min(nh*nh,1.);
      const double denom = alpha2*q + (1.-q);
      const double d = alpha2/(pi*denom*denom);
      const auto lambda = [&](double cosine) {
        const double safe = std::max(cosine*cosine,1e-12);
        return .5*(-1.+std::sqrt(1.+alpha2*(1.-safe)/safe));
      };
      const double cv = std::max(nv,1e-6), cl = std::max(nl,1e-6);
      const double g = 1./(1.+lambda(cv)+lambda(cl));
      const double f = .04+.96*std::pow(1.-vh,5);
      const double coatRaw = float(to_half(test.p.coat))*f*d*g/(4.*cv*cl);
      coatProjectionCorrection = coatRaw*(nl-std::abs(dot(normal,wi)));
    }
    for (int c = 0; c < 3; ++c) {
      nonnegative(e.projected[i].value[c],test.name+": invalid projected value");
      nonnegative(e.raw[i].value[c],test.name+": invalid raw value");
      require(close(e.projected[i].value[c],
                    e.raw[i].value[c]*std::abs(dot(normal,directions[i]))
                        + coatProjectionCorrection),
              test.name+": raw/projected per-lobe cosine mismatch");
      require(e.projected[i].value[c] == 0.f || e.pdf[i] > 0.f,
              test.name+": nonzero continuous value without proposal support");
    }
  }
}

// Gauss-Legendre in mu, periodic midpoint in phi, dOmega=dmu*dphi.
// Separate hemispheres and phi quadrants; even mu rules straddle the
// |mu|=.5 bin boundary symmetrically. Bin integrals must also converge.
// Nodes/weights/sums are double; values are actual backend evaluations.
std::vector<std::pair<double,double>> gaussLegendre(int n)
{
  std::vector<std::pair<double,double>> nodes;
  for (int i = 0; i < n; ++i) {
    double x = std::cos(pi*(i+.75)/(n+.5)), derivative = 0.;
    for (int iteration = 0; iteration < 20; ++iteration) {
      double p0 = 1., p1 = x;
      for (int j = 2; j <= n; ++j) {
        const double next = ((2*j-1)*x*p1-(j-1)*p0)/j;
        p0 = p1;
        p1 = next;
      }
      derivative = n*(x*p1-p0)/(x*x-1);
      const double step = p1/derivative;
      x -= step;
      if (std::abs(step) < 1e-15) break;
    }
    nodes.emplace_back((x+1)/2,1/((1-x*x)*derivative*derivative));
  }
  return nodes;
}
struct Integral
{
  std::array<double,3> value{};
  std::array<double,bins> mass{};
};
int bin(vec3f d)
{
  return (d.z < 0.f ? 8 : 0) + (std::abs(d.z) < .5f ? 4 : 0)
      + (d.x < 0.f ? 2 : 0) + (d.y < 0.f ? 1 : 0);
}
Integral integrate(rtc::Device *device, const Case &test, int nz)
{
  const int nphi = 2*nz;
  const int n = 2*nz*nphi;
  std::vector<vec3f> directions;
  directions.reserve(n);
  std::vector<double> weights;
  weights.reserve(n);
  for (int side : {1,-1})
    for (auto node : gaussLegendre(nz)) {
      const double mu = side*node.first;
      for (int a = 0; a < nphi; ++a) {
        const double phi = 2*pi*(a+.5)/nphi;
        directions.emplace_back(std::sqrt(1-mu*mu)*std::cos(phi),
                                std::sqrt(1-mu*mu)*std::sin(phi),mu);
        weights.push_back(node.second*2*pi/nphi);
      }
    }
  Buffer input(device,n*sizeof(vec3f),directions.data());
  const auto e = evaluate(device,test.p,input,n);
  checkEvaluation(test,directions,e);
  Integral integral;
  for (int i = 0; i < n; ++i) {
    const double w = weights[i];
    for (int c = 0; c < 3; ++c) integral.value[c] += w*e.projected[i].value[c];
    integral.mass[bin(directions[i])] += w*e.pdf[i];
  }
  return integral;
}

struct Moments
{
  double sum = 0., squares = 0.;
  void add(double x) { sum += x; squares += x*x; }
  double mean(int n) const { return sum/n; }
  double se(int n) const
  {
    return std::sqrt(std::max(0.,(squares-sum*sum/n)/(n-1)/n));
  }
};

void runCase(rtc::Device *device, const Case &test)
{
  Integral coarse, fine;
  std::array<double,3> error{};
  std::array<double,bins> massError{};
  int resolution = 256;
  if (test.quadrature) {
    coarse = integrate(device,test,128);
    fine = integrate(device,test,resolution);
    auto resolved = [&]() {
      for (int c = 0; c < 3; ++c)
        if (2*std::abs(fine.value[c]-coarse.value[c]) >= 5e-4) return false;
      for (int b = 0; b < bins; ++b)
        if (2*std::abs(fine.mass[b]-coarse.mass[b]) >= 5e-4) return false;
      return true;
    };
    while (!resolved() && resolution < 1024) {
      coarse = fine;
      resolution *= 2;
      fine = integrate(device,test,resolution);
    }
    for (int c = 0; c < 3; ++c) {
      error[c] = 2*std::abs(fine.value[c]-coarse.value[c]);
      require(error[c] < 5e-4,test.name+": quadrature unresolved; refine, do not relax tolerance");
      if (test.p.model == LAMBERT)
        require(std::abs(fine.value[c]-test.p.color[c]) < 2e-6,
                test.name+": furnace must equal albedo");
      if (test.metal)
        require(fine.value[c] <= 1+error[c]+roundoff,
                test.name+": integrated white metal creates energy");
    }
    for (int b = 0; b < bins; ++b) {
      massError[b] = 2*std::abs(fine.mass[b]-coarse.mass[b]);
      require(massError[b] < 5e-4,test.name+": density quadrature unresolved");
    }
  }

  Buffer samples(device,trials*sizeof(ScatterResult));
  Buffer directions(device,trials*sizeof(vec3f));
  __rtc_launch(device,sample,(trials+127)/128,128,test.p,
               samples.ptr<ScatterResult>(),directions.ptr<vec3f>(),trials);
  const auto s = samples.read<ScatterResult>(trials);
  const auto dirs = directions.read<vec3f>(trials);
  const auto e = evaluate(device,test.p,directions,trials);
  checkEvaluation(test,dirs,e);
  std::array<Moments,3> weights{}, prefix{};
  std::array<int,bins> counts{};
  int nulls = 0, deltas = 0;
  for (int i = 0; i < trials; ++i) {
    if (s[i].event == ScatterResult::NO_EVENT) {
      ++nulls;
      require(s[i].weight == vec3f(0.f) && s[i].pdf == 0.f,
              test.name+": rejected proposal must carry zero weight/density");
      continue;
    }
    require(std::isfinite(dot(s[i].dir,s[i].dir))
            && std::abs(dot(s[i].dir,s[i].dir)-1.f) < 2e-5,
            test.name+": sample direction is not unit length");
    const bool reflection = dot(test.p.dg.geometricNormal,s[i].dir)*dot(test.p.dg.geometricNormal,test.p.dg.wo) > 0.f;
    require(reflection == (s[i].event == ScatterResult::REFLECTION),
            test.name+": sample event disagrees with geometric side");
    if (s[i].delta) {
      ++deltas;
      require(s[i].pdf == 0.f && e.pdf[i] == 0.f,
              test.name+": delta must not masquerade as solid-angle density");
    } else {
      ++counts[bin(s[i].dir)];
      require(s[i].pdf > 0.f && close(s[i].pdf,e.pdf[i]),
              test.name+": returned sample density differs from full density");
    }
    for (int c = 0; c < 3; ++c) {
      nonnegative(s[i].weight[c],test.name+": invalid sampled weight");
      if (test.sampleBound > 0.)
        require(s[i].weight[c] <= test.sampleBound+roundoff,
                test.name+": sampled weight exceeds analytic bound");
      if (!s[i].delta)
        require(close(s[i].weight[c],e.projected[i].value[c]/e.pdf[i]),
                test.name+": sample is not complete projected value / full density");
      weights[c].add(s[i].weight[c]);
      if (i < trials/4) prefix[c].add(s[i].weight[c]);
    }
  }
  // Null proposals contribute zero, but remain in EVERY denominator.
  for (int c = 0; c < 3; ++c) {
    const double mean = weights[c].mean(trials), se = weights[c].se(trials);
    if (test.quadrature) {
      require(deltas == 0,test.name+": continuous quadrature cannot include delta energy");
      require(std::abs(mean-fine.value[c]) <= 6*se+error[c]+roundoff,
              test.name+": sampler mean disagrees with independent quadrature: MC="
                  +std::to_string(mean)+" Q="+std::to_string(fine.value[c])
                  +" SE="+std::to_string(se)+" Qerr="+std::to_string(error[c]));
      require(std::abs(prefix[c].mean(trials/4)-fine.value[c])
                  <= 6*prefix[c].se(trials/4)+error[c]+roundoff,
              test.name+": quarter-count sampler disagrees with quadrature");
    }
    if (test.metal)
      require(mean <= 1+6*se+roundoff,test.name+": sampled white metal creates energy");
  }
  if (test.quadrature)
    for (int b = 0; b < bins; ++b) {
      const double probability = double(counts[b])/trials;
      const double se = std::sqrt(probability*(1-probability)/trials);
      require(std::abs(probability-fine.mass[b]) <= 6*se+massError[b]+roundoff,
              test.name+": sampler bin "+std::to_string(b)
                  +" frequency="+std::to_string(probability)
                  +" disagrees with integrated full PDF="+std::to_string(fine.mass[b]));
    }
  double mass = 0.;
  for (double m : fine.mass) mass += m;
  std::cout << test.name;
  if (test.quadrature)
    std::cout << " Q=" << fine.value[0] << ',' << fine.value[1] << ',' << fine.value[2]
              << " Qerr=" << error[0] << " nz=" << resolution << " pdf_mass=" << mass;
  else
    std::cout << " Q=not-integrated";
  std::cout << " MC=";
  for (int c = 0; c < 3; ++c)
    std::cout << weights[c].mean(trials) << "+/-" << weights[c].se(trials) << ' ';
  std::cout << " MC(N/4)=" << prefix[0].mean(trials/4)
            << "+/-" << prefix[0].se(trials/4)
            << " continuous_frequency=" << double(trials-nulls-deltas)/trials
            << " null=" << nulls << " delta=" << deltas << '\n';
}

float fromBits(uint32_t bits)
{
  float f;
  std::memcpy(&f,&bits,sizeof(f));
  return f;
}
Case makeCase(Model model, float mu, const std::string &name)
{
  Case test;
  test.name = name;
  test.p.model = model;
  test.p.dg.geometricNormal = test.p.dg.shadingNormal = vec3f(0,0,1);
  test.p.dg.wo = vec3f(std::sqrt(1-mu*mu),0,mu);
  return test;
}

void capturedRegression(rtc::Device *device)
{
  auto test = makeCase(PBR,1.f,"captured PBR transmission");
  test.p.dg.wo = vec3f(fromBits(0x3f0a8d26),fromBits(0x3eba28bf),fromBits(0x3f421a32));
  test.p.normal = vec3f(fromBits(0x3f56e0ce),fromBits(0x3da4609e),fromBits(0x3f09a084));
  test.p.dg.shadingNormal = test.p.dg.geometricNormal = test.p.normal;
  test.p.dg.insideMedium = true;
  test.p.color = vec3f(1.f);
  test.p.roughness = 0.f;
  test.p.transmission = test.p.specular = 1.f;
  const std::vector<vec3f> directions = {
    vec3f(fromBits(0xbe9c0815),fromBits(0xbefe6d17),fromBits(0xbf500178))};
  Buffer in(device,sizeof(vec3f),directions.data());
  const auto e = evaluate(device,test.p,in,1);
  std::cout << test.name << " projected=" << e.projected[0].value.x
            << " pdf=" << e.pdf[0] << " weight=" << e.projected[0].value.x/e.pdf[0] << '\n';
  checkEvaluation(test,directions,e);
  for (int c = 0; c < 3; ++c)
    require(close(e.projected[0].value[c]/e.pdf[0],2.25,2e-5),
            "captured exiting transmission must have eta^2=2.25, not 375.638");
  // Real samples around the captured packed frame run in a fourth invocation
  // context; their directions then go through the three independent kernels.
  test.quadrature = false;
  test.sampleBound = 2.25;
  runCase(device,test);
}

void tiltedNormals(rtc::Device *device)
{
  auto tilted = makeCase(PBR,.5f,"PBR view-adjusted base");
  tilted.p.normal = vec3f(-.8f,0,.6f);
  tilted.p.roughness = .7f;
  tilted.p.specular = .5f;
  auto coat = tilted;
  coat.name = "PBR independent view-adjusted coat";
  coat.p.normal = vec3f(0,0,1);
  coat.p.coatNormal = tilted.p.normal;
  coat.p.coat = .75f;
  auto back = coat;
  back.name = "PBR back-face view-adjusted coat";
  back.p.dg.shadingNormal = vec3f(0,0,-1);
  back.p.dg.insideMedium = true;
  back.p.normal = -coat.p.normal;
  back.p.coatNormal = -coat.p.coatNormal;
  auto transmission = tilted;
  transmission.name = "PBR view-adjusted rough transmission";
  transmission.p.transmission = .75f;
  auto grazing = tilted;
  grazing.name = "PBR grazing view-adjusted base";
  grazing.p.dg.wo = normalize(vec3f(1,0,.001f));
  auto valid = tilted;
  valid.name = "PBR already-valid tilted base";
  valid.p.normal = normalize(vec3f(.2f,0,1.f));

  for (const auto &test : {tilted,coat,back,transmission,grazing,valid}) {
    Buffer normals(device,2*sizeof(vec3f));
    __rtc_launch(device,resolvedNormals,1,1,test.p,normals.ptr<vec3f>());
    const auto resolved = normals.read<vec3f>(2);
    for (int lobe = 0; lobe < 2; ++lobe) {
      vec3f original = lobe == 1 && dot(test.p.coatNormal,test.p.coatNormal) > 0.f
          ? test.p.coatNormal : test.p.normal;
      original = normalize((vec3f)to_half(original));
      if (dot(test.p.dg.geometricNormal,test.p.dg.shadingNormal) < 0.f)
        original = -original;
      const vec3f view = test.p.dg.wo, ng = test.p.dg.geometricNormal;
      const float threshold = std::min(.9f*dot(ng,view),.01f);
      const vec3f mirror = 2.f*dot(original,view)*original-view;
      if (dot(ng,mirror) >= threshold)
        require(length(resolved[lobe]-original) < 2e-6f,
                test.name+": already-valid normal changed");
      else {
        require(length(resolved[lobe]-original) > .01f,
                test.name+": tilted normal was not adjusted");
        const vec3f adjustedMirror =
            2.f*dot(resolved[lobe],view)*resolved[lobe]-view;
        require(dot(ng,adjustedMirror) >= threshold-2e-6f,
                test.name+": resolved mirror fails geometric clearance");
      }
    }
    runCase(device,test);
  }
}

void deltaEvents(rtc::Device *device, Model model)
{
  auto test = makeCase(model,1.f,model == PBR ? "PBR index matched" : "NVisii index matched");
  test.p.color = vec3f(.5f);
  test.p.transmission = 1.f;
  test.p.ior = 1.f;
  Buffer samples(device,trials*sizeof(ScatterResult));
  Buffer directions(device,trials*sizeof(vec3f));
  __rtc_launch(device,sample,(trials+127)/128,128,test.p,
               samples.ptr<ScatterResult>(),directions.ptr<vec3f>(),trials);
  const auto s = samples.read<ScatterResult>(trials);
  const auto dirs = directions.read<vec3f>(trials);
  const auto e = evaluate(device,test.p,directions,trials);
  checkEvaluation(test,dirs,e);
  Moments energy, probability;
  for (int i = 0; i < trials; ++i) {
    if (!s[i].delta) continue;
    require(s[i].event == ScatterResult::TRANSMISSION && s[i].pdf == 0.f
            && e.pdf[i] == 0.f && s[i].dir == -test.p.dg.wo,
            test.name+": invalid straight-ray delta event");
    for (int c = 0; c < 3; ++c)
      require(close(s[i].weight[c],model == PBR ? .5 : 1.),
              test.name+": incorrect delta selection compensation");
    energy.add(s[i].weight.x);
    probability.add(1.);
  }
  require(std::abs(energy.mean(trials)-.5) < 6*energy.se(trials)+roundoff,
          test.name+": discrete energy must be albedo");
  std::cout << test.name << " delta_mass=" << probability.mean(trials)
            << " delta_energy=" << energy.mean(trials) << '\n';
}
} // namespace bsdf_quality

int main()
{
  using namespace bsdf_quality;
  try {
    const auto start = std::chrono::steady_clock::now();
    rtc::Device device(0);
    std::cout << std::setprecision(8) << "BSDF compute device: " << device.toString() << '\n';
#if defined(BARNEY_RTC_OPTIX) || defined(BARNEY_RTC_CUDA)
    cudaDeviceProp properties{};
    require(cudaGetDeviceProperties(&properties,0) == cudaSuccess,"CUDA device query failed");
    std::cout << "GPU: " << properties.name << '\n';
#endif
    checkNormalAdjustment(&device);
    capturedRegression(&device);
    tiltedNormals(&device);
    for (float mu : {1.f,.25f})
      runCase(&device,makeCase(LAMBERT,mu,"Lambertian mu="+std::to_string(mu)));
    for (Model model : {PBR,NVISII}) {
      const std::string name = model == PBR ? "PBR" : "NVisii";
      for (float roughness : {0.f,.35f,.7f,1.f})
        for (float mu : {1.f,.5f,.1f}) {
          auto test = makeCase(model,mu,name+" white metal r="+std::to_string(roughness)
                                       +" mu="+std::to_string(mu));
          test.p.color = vec3f(1.f);
          test.p.metallic = 1.f;
          test.p.roughness = roughness;
          test.metal = true;
          // White F=1 with visible-normal sampling has G2/G1 <= 1.
          // NVisii uses NDF sampling, so only its integral has this bound.
          if (model == PBR) test.sampleBound = 1.;
          test.quadrature = roughness >= .35f;
          runCase(&device,test);
        }
      auto diffuse = makeCase(model,.5f,name+" colored diffuse");
      runCase(&device,diffuse);
      auto mixture = makeCase(model,.5f,name+" base/coat mixture");
      mixture.p.metallic = .25f;
      mixture.p.specular = .5f;
      mixture.p.coat = .75f;
      runCase(&device,mixture);
      auto transmission = makeCase(model,.8f,name+" rough transmission");
      transmission.p.roughness = .7f;
      transmission.p.transmission = .75f;
      transmission.p.specular = .5f;
      runCase(&device,transmission);
      deltaEvents(&device,model);
    }
    std::cout << "BSDF quality: PASS seconds="
              << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count() << '\n';
    return 0;
  } catch (const std::exception &e) {
    std::cerr << "BSDF quality: FAIL: " << e.what() << '\n';
    return 1;
  }
}
