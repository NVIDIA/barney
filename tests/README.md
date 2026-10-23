# Numerical packed-BSDF tests

`bsdf_quality.cu` exercises the production Lambertian, PhysicallyBased (PBR),
and NVisii packed evaluators and complete samplers through **rtc compute
launches**. No images, scene assets, or renderer are involved. These tests do
**not** validate renderer MIS, lights, visibility, normal-map construction,
path integration, or final image appearance.

## Run

Enable the optional suite in an existing backend build, then build and use
CTest (from the source workspace):

```sh
# OptiX, existing multi-config build
cmake -S . -B _out/_cmake -DBARNEY_BUILD_BSDF_TESTS=ON
just build
cmake --build _out/_cmake --config RelWithDebInfo --target bsdf_quality_optix
ctest --test-dir _out/_cmake -C RelWithDebInfo -R '^bsdf_quality_' --output-on-failure

# Existing CPU and native-CUDA builds; do not change their backend selection
cmake -S . -B _out/cpu-tangents -DBARNEY_BUILD_BSDF_TESTS=ON
cmake --build _out/cpu-tangents --target bsdf_quality_cpu
ctest --test-dir _out/cpu-tangents -R '^bsdf_quality_' --output-on-failure
cmake -S . -B _out/cuda-tangents -DBARNEY_BUILD_BSDF_TESTS=ON
cmake --build _out/cuda-tangents --target bsdf_quality_cuda
ctest --test-dir _out/cuda-tangents -R '^bsdf_quality_' --output-on-failure
```

For a differently named build directory, substitute its path. Each enabled
backend gets its own executable/test. `ctest -V` shows numerical results and
elapsed time; executables can also be run directly. Device 0 is used. CUDA
builds print the actual GPU name and retain default FMA contraction. The suite
runtime depends on the backend and quadrature refinement, with
roughly half a GiB peak working storage for the finest quadrature. CTest has a
180-second timeout. HIP, HIPRT, MPI and multiple physical GPUs are unverified.

## Checks and numerical rationale

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
