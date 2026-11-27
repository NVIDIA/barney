# Volume lighting regression

Enable `BARNEY_BUILD_VOLUME_LIGHTING_TESTS` in any backend build:

```sh
cmake -S . -B _out/volume-lighting -DBARNEY_BUILD_VOLUME_LIGHTING_TESTS=ON
cmake --build _out/volume-lighting --parallel
ctest --test-dir _out/volume-lighting -R '^volume_lighting_' --output-on-failure
```

The option is off by default and does not change backend selection. Each enabled
backend gets a `volume_lighting_<backend>` executable and CTest entry. The test
uses device 0; OptiX is qualified, other backends are unverified.

Synthetic volume hits call production `bounce()` through rtc compute launches.
The test checks direct and unoccluded continuation contributions against literal
isotropic energies and independent double-precision per-direction references.
It covers disabled scattering, zero/exhausted budgets, the last allowed volume
and global bounces, the global depth limit (including a Lambertian surface),
a Lambertian surface with a spent diffuse budget,
SimplePhase and HG with positive/negative anisotropy, colored HDR throughput,
emission, and ambient/directional/area/mixed/no-light cases. It also checks
continuation validity, bounce counts, and that shadow MIS is applied only once.
A rough dielectric surface checks that spending the diffuse bounce budget keeps
the mean environment energy within Monte Carlo tolerance.

There is no ray traversal or scene geometry: visibility is explicitly synthesized
as an unoccluded miss. This regression does not test environment textures/CDFs,
volume tracing, indirect light lost to the diffuse bounce budget, or final image
appearance. The older optional packed-BSDF suite is independent and is not
required by this test.

# Hit-time tangent tests

Enable `BARNEY_BUILD_TANGENT_TESTS` in a CPU or GPU backend build:

```sh
cmake -S . -B _out/tangent-cpu \
  -DBARNEY_BACKEND_CPU=ON -DBARNEY_BACKEND_OPTIX=OFF \
  -DBARNEY_BUILD_TANGENT_TESTS=ON
cmake --build _out/tangent-cpu --parallel
ctest --test-dir _out/tangent-cpu -R tangent --output-on-failure
```

For native CUDA, use a separate build directory with
`-DBARNEY_BACKEND_CPU=OFF -DBARNEY_BACKEND_CUDA=ON`. Each enabled backend
registers two tests:

- `tangent_generation_*`: hit-time frame behavior, authored tangent precedence
  and interpolation, vertex/face-varying attributes, input updates, material
  changes, clearcoat-only mapping, finite fallback for invalid/missing UVs, and
  a mirrored nonuniform instance transform. It also checks two logical devices.
- `anari_tangent_generation_*`: ANARI triangle/quad uploads, indexed/nonindexed
  meshes, authored tangent and face-normal corner expansion, format conversion,
  and array-unmap updates. An ANARI world instance-only transform update checks
  that native mesh input identities remain unchanged.
  Rays traverse the production native intersection/material path after ANARI
  translation, and mapped normals are checked against hand-derived values and
  equivalent authored frames. This does not exercise ANARI frame rendering.

These tests inspect the packed BSDF normals from actual hits, not persistent
mesh-generated tangent arrays: mesh commit no longer creates those arrays.
The `1e-3` normal tolerance accommodates half-precision BSDF storage. No images,
lighting, path integration, performance or smooth tangent continuity across
neighbouring faces are validated. GPU tests use device 0;
HIP/HIPRT and MPI have not been verified.

# Numerical packed-BSDF tests

`bsdf_quality.cu` exercises the production Lambertian, PhysicallyBased (PBR),
and NVisii packed evaluators and complete samplers through **rtc compute
launches**. A focused integrator check also calls the production `bounce()`
with synthetic hits to verify medium transitions. No images or scene assets
are involved. These tests do **not** validate renderer MIS, lights, visibility,
normal-map construction, complete path integration, or final image appearance.

## Run

Enable `BARNEY_BUILD_BSDF_TESTS` in any backend build; the option does not
change backend selection:

```sh
cmake -S . -B _out/bsdf -DBARNEY_BUILD_BSDF_TESTS=ON
cmake --build _out/bsdf --parallel
ctest --test-dir _out/bsdf -R '^bsdf_quality_' --output-on-failure
```

For a differently named build directory, substitute its path. Each enabled
backend gets its own executable/test. `ctest -V` shows numerical results and
elapsed time; executables can also be run directly. Device 0 is used. CUDA
builds print the actual GPU name and retain default FMA contraction. The suite
runtime depends on the backend and quadrature refinement, with
roughly half a GiB peak working storage for the finest quadrature. CTest has a
180-second timeout. HIP, HIPRT, MPI and multiple physical GPUs are unverified.

## Checks and numerical rationale

* **Integrator medium transitions:** synthetic hits call production `bounce()`
  through an outside-to-inside transmission, total internal reflection, and
  inside-to-outside transmission. Both boundary windings must preserve the same
  crossing parity. Diffuse proposals rejected by the bounce limit must leave
  both possible incoming medium states unchanged.
* **Lambertian furnace:** unit incident radiance must integrate to RGB albedo
  `(0.125, 0.5, 0.875)` at normal and oblique views (`N.V=1, 0.25`). The absolute
  tolerance is `2e-6`; missing/extra pi or projection cosines cannot hide in it.
* **White metal furnace:** PBR and NVisii at roughness `0, .35, .7, 1` and
  `N.V=1, .5, .1` must remain finite/nonnegative and reflect no more than unit
  radiance. Single-scattering GGX may lose energy: equality is not required.
  PBR white-metal VNDF weights also satisfy the analytic `G2/G1 <= 1` bound.
  NVisii samples the NDF instead, so that *per-sample* bound does not apply.
  Roughness-floor lobes use sampling/bounds only, **not angular quadrature**.
* **Independent integration versus sampling:** host double-precision
  Gauss–Legendre integration in `mu=cos(theta)` and periodic midpoint integration
  in azimuth sum actual backend **projected** evaluator outputs over both
  hemispheres (`dOmega=dmu*dphi`). They do not integrate `f/pdf` or reuse sampled
  directions. Start with 128/256 mu nodes per hemisphere and twice as many
  azimuth points; refine to at most 1024 mu nodes. Both RGB energy and each PDF
  bin must have `2*abs(fine-coarse) < 5e-4`, otherwise the test fails as unresolved.
  The measured difference (not the `5e-4` ceiling) enters each comparison.
  This is a numerical convergence check, not a rigorous quadrature error proof.
* **Full sampler/density:** 262144 deterministically seeded production trials
  include **all null/rejected proposals in the denominator**. RGB means are
  compared to quadrature at N and N/4, within six estimated standard errors,
  the measured quadrature error above, and `2e-5` absolute float-roundoff slack.
  Finite means alone are insufficient: 16 directional bins (hemisphere,
  `abs(mu)<.5`, x/y signs) compare empirical probability against integrated
  full PDF. This catches a wrong sampling distribution even if `f/pdf` agrees
  everywhere. Neither total continuous PDF mass nor acceptance is assumed to
  equal one. Cases include diffuse, white metal, colored base/coat mixtures,
  and rough transmission for both PBR and NVisii. The six-SE margin handles
  multiple checks; fixed seeds make failures reproducible, not a proof for all
  seeds or rare tails. Output `+/-` denotes **one** standard error, not tolerance.
* **Contract and delta checks:** separate compute entry points evaluate raw
  value, projected value, standalone PDF, and real scattering. Returned sample
  PDF must match full density, and weight must be complete projected value/PDF.
  Density/value comparisons allow `1e-7 + 5e-5*max(abs(a),abs(b))`; projection
  uses the actual half-packed normal. Index-matched straight transmission is
  checked separately against analytic discrete energy `.5`, with zero
  solid-angle PDF, never infinity or continuous-quadrature normalization.
* **View-adjusted normals:** nine compute cases check unchanged valid normals,
  analytic/reference rotations, geometric mirror clearance, grazing views, and
  degenerate rotation planes. PBR base, independent clearcoat, back-face,
  rough-transmission, grazing, and already-valid tilted frames run through the
  full quadrature/sampling checks above. Normal resolution must preserve valid
  normals and adjust invalid ones after the shared orientation flip. When base
  and coat differ, the raw/projected check independently reconstructs the coat
  term in double precision and applies each lobe's cosine, not one shared cosine.
  These tests supply mapped normals directly; they do not test texture decoding
  or normal-map construction. Geometric rejection still counts as a null sample.
* **Captured GPU regression:** exact captured V/L/N bit patterns are fed to
  the actual half-packed PBR material (inside, IOR 1.5, roughness 0, white,
  transmission/specular 1). Projected value/PDF must be `eta^2=2.25` within
  `2e-5` relative tolerance; the old faulty GPU header yielded `375.63834`.
  Actual sampled weights in that frame must also stay at or below `2.25`.

The suite's normalization, complete-density, and sampling-distribution checks
were observed to fail with temporary real-header overlays (missing Lambertian
`1/pi`; halved density with compensating doubled weight; uniform rather than
cosine sampling). The captured regression fails against the historical faulty
PBR header. No fault-injection switches or production model changes are needed.
