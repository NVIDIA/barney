# Float acceptance evidence (ticket 05)

Measured 2026-10-24 on integration `7a6804c5`, plus this ticket's **tests and
measurement harness only**. No production algorithm, safety boundary, filter
policy, or extent limit changed. Quantized restoration can use this Float
baseline, subject to the explicit limits below. This is not a claim of complete
KHR extension conformance or general renderer correctness.

## Machine, configurations, and backend matrix

- CPU: Intel Core Ultra 9 285K, 24 cores; Embree 4.4.1.
- GPU: CUDA runtime reports **one** NVIDIA RTX 5880 Ada Generation, compute 8.9.
  Real kernels and rendering succeed; the previously observed `nvidia-smi`
  segfault is not evidence of GPU absence. Multi-GPU unavailable.
- GCC 16.2.0, CUDA 13.4, OptiX 9.1; RelWithDebInfo, CUDA architecture 75
  (PTX works on the 8.9 device). ANARI SDK from the development environment.
- Nsight Systems 2026.3.2.476 and Compute Sanitizer 2026.3.0.0 are available.
  No clock locking, exclusive-device reservation, or thermal stabilization was
  applied. Runs were sequential, not concurrent benchmarks.
- Three separate builds enable only CPU, only OptiX, or only native CUDA RT.
  `BARNEY_USE_MULTI_SCATTERING=OFF` (default); alternate integrator configurations
  and MPI are not covered. Denoising is disabled on every tested renderer.

| Backend | Build and complete API suite | Numerical seam | Sanitizers |
|---|---|---|---|
| CPU/Embree | pass, 138 checkpoints | six groups pass | UBSan and ASan+UBSan: 123 API checkpoints each, existing ticket 08 suppressions; numerical checks unsuppressed |
| OptiX | pass, 123 checkpoints | four map/statistics combinations pass | standalone CUDA numerical memcheck: 0 errors; full OptiX memcheck remains ticket 09's baseline hardware-exception limitation, not rerun here |
| Native CUDA RT | pass, 123 checkpoints | four combinations pass with native-CUDA headers/configuration | standalone numerical memcheck: 0 errors; full API assertions complete but memcheck exits 1 with 10 duplicate-kernel registration errors (ticket 11), **not a clean pass** |
| HIP | configure unavailable | not run | no HIP compiler/root; `check_language(HIP)` reports NOTFOUND and cuBQL enable_language fails |
| HIPRT | unavailable prerequisite | not run | same required HIP compiler absent; no HIPRT result inferred from CUDA |

`hipcc`, `rocminfo`, and `shellcheck` are not on PATH; no `/dev/kfd` exists.
CMake, rather than GPU utility output alone, established the missing HIP compiler.
No new package/tool provisioning was attempted. Device allocation-failure
injection remains untested. CPU's 15 extra checkpoints exercise system allocator
failures (including range allocation), invalidation of both volumes and recovery;
they are explicitly skipped on GPUs and under sanitizer allocation interceptors.

The 123 common PASS checkpoint labels are identical as multisets across all three
normal runs. Rejected envelopes/types/maps report ANARI errors, invalidate the
field, exclude old bounds and both dependent volumes, and recover on all three;
exact diagnostic text and counts of repeated warnings are not contracts.
Numerical tests compare production samples with the same independently worked
Float values, and check conservative ranges/majorants, not framebuffer bits.

| Observable | CPU | OptiX | Native CUDA |
|---|---:|---:|---:|
| nearest vs linear joint mean RGBA difference | 0.114831 | 0.113807 | 0.114964 |
| affine header/instance vs equivalent baked header, 4096 spp | 0.00225099 | 0.00207173 | 0.00225027 |
| six independent TF-update/fresh comparisons, maximum | < 1e-9 | < 1e-9 | < 1e-9 |

The existing 0.005 equivalent-scene mean-RGBA tolerance and 1e-5 bounds tolerance
are unchanged. Different backends need not choose identical ray/random sequences;
the observed filter effect differs by at most 0.001157, while all backends satisfy
the same numerical and equivalent-scene criteria. The following independent
analytical observable also agrees across backends without asserting bit equality.

## Analytical homogeneous transmission

The API test constructs a real version-32.7.0 constant Float grid on
`[0,3] x [0,3] x [0,7]`, world extent `[0,4] x [0,4] x [0,8]`.
One volume, no surfaces/lights, white background, **ambientRadiance=0**, and
scatteringAlbedo=0 isolate transmission. Orthographic eye `(2,2,15)`, direction
`(0,0,-1)`, height 2 and aspect 1.5 put every jittered ray strictly inside the
x/y footprint, with path length 8. Both nearest and linear are exercised.
Five identical opacity entries avoid the known one-entry TF defect (08);
unitDistance=2 gives sigma=opacity/2. Each image is 48x32 linear float RGBA,
1024 spp, a new frame resetting the sample sequence, denoising off. There is no
public seed parameter in this harness; repeatability comes from fresh frames.

The independent oracle is Beer-Lambert `T=exp(-sigma*8)`. The test compares mean
red (white background transmission) against it. Tolerance is
`6*sqrt(T*(1-T)/(48*32*1024)) + 0.001`: six Bernoulli standard errors plus a
bounded numerical/approximation allowance. This is a statistical engineering
tolerance, not proof of independence of the renderer's pseudorandom streams.
For optical depth <=1, the omitted atanh-series tail of the production `fastLog`
can shift transmission by less than 0.001; it is not negligible at larger depth.
All image values must also be finite.

| Opacity | Optical depth | Exact T | CPU / OptiX / CUDA observed mean (both filters) | Tolerance | Disposition |
|---|---:|---:|---:|---:|---|
| 0 | 0 | 1 | 1 | 0.001 | pass |
| 0.1 | 0.4 | 0.670320042 | 0.670241674 | 0.003249019 | pass |
| 0.25 | 1 | 0.367879441 | 0.367260615 | 0.003307057 | pass |
| 0.5 | 2 | 0.135335283 | 0.122198741 | 0.002636572 | **diagnostic only, outside tolerance** |

The last row is deliberately still printed, explicitly marked LIMIT, and is not
included in the analytical PASS claim. `Woodcock::sampleRange` uses a three-term
atanh-series logarithm shared by all volume types; its tail biases long flights.
Ticket **10-homogeneous-transport-log-bias** records that general integrator
limitation and the reversible decision to establish analytical acceptance only
through optical depth 1. No tolerance was enlarged to hide optical-depth-2 error.
This ticket does not change general transport.

TDD/sensitivity: the initial fixture failed because default ambient illumination
was not disabled (mean 1.95447 instead of 0.67032). Setting ambientRadiance=0
isolated transmission, then exposed the high-depth approximation above.
A temporary production mutation forcing unitDistance=1 made the retained
optical-depth-0.4 assertion fail (mean 0.449107488, expected 0.670320042).
Reverting it restores the pass. No mutation or production change is committed.

## Sparse performance experiment

`NANOVDB_BENCHMARK=N` selects the measurement mode of the **same ANARI client**;
it does not enable a different renderer/range pipeline. It generates two active
8x8x8 islands, values 0.5 and 1, at opposite corners of `[0,N-1]^3`, background 0,
1024 active voxels total, and **BBox-only statistics**. Stored payload is only
309376 bytes at N=64, and 343232 bytes at N=256/512/1024, despite 262144 through
1073741824 index coordinates. No external data download is needed.
`NANOVDB_BENCHMARK_SHEAR=1` applies x'=x+0.5*y with its inverse and recomputed
world AABB; N=256 then has world extent 384x256x256 but the same 16^3 MC counts.

A single visible volume uses the existing ramp opacity `[0,0.8]`, linear filter,
zero albedo/ambient, white background, no unrelated sphere. Camera `(4,4,N+15)`
(or `(6,4,N+15)` for shear), height 12, aspect 1.5, looks down -z through the
near-origin island and a long empty index region. This is not a full-volume
image where the tiny islands disappear below pixel resolution. Images are
48x32, 64 spp, new frame per trial, no denoiser. Sparse means are nontrivial
(~0.685 identity; ~0.607 shear), not all-background timings.

Six trials each supply a fresh managed byte array and force the ordinary field
update. Trial 0 warms initialization; the next five report min/median/max.
Fixture creation, ANARI array allocation/copy, and device/context initialization
are outside the build timer. The **API build** timer includes field commit and
world-bounds flush: validation, upload/download, production range construction
(which synchronizes), volume majorants and backend acceleration setup. It is
**not** described as kernel-only time. **Render** times include fresh-frame
setup, rendering, wait, map/download, finite-image checks and frame release;
these small-image results include substantial dispatch/host overhead and do
not predict large-image throughput. No speedup threshold or old singleton
comparison is introduced.

### Unprofiled timings (milliseconds, min / median / max)

| Backend | N / map | API build | Render |
|---|---|---|---|
| OptiX | 64 identity | 1.250 / 1.283 / 1.311 | 19.580 / 22.864 / 23.597 |
| OptiX | 256 identity | 1.646 / 1.686 / 1.732 | 19.632 / 22.868 / 23.211 |
| OptiX | 512 identity | 1.723 / 1.758 / 1.765 | 19.874 / 23.068 / 23.288 |
| OptiX | 1024 identity | 4.768 / 4.786 / 4.799 | 20.202 / 23.567 / 23.760 |
| OptiX | 256 shear | 2.518 / 2.559 / 2.577 | 20.106 / 23.076 / 23.513 |
| Native CUDA | 64 identity | 1.098 / 1.120 / 1.432 | 5.480 / 5.503 / 5.570 |
| Native CUDA | 256 identity | 1.504 / 1.511 / 1.775 | 5.805 / 5.867 / 5.907 |
| Native CUDA | 512 identity | 1.556 / 1.571 / 1.806 | 6.282 / 6.332 / 6.346 |
| Native CUDA | 1024 identity | 4.558 / 4.598 / 4.897 | 7.214 / 7.217 / 7.263 |
| Native CUDA | 256 shear | 2.382 / 2.386 / 2.640 | 5.869 / 5.878 / 6.002 |
| CPU (fixture smoke only) | 64 identity | 1.008 / 1.591 / 1.796 | 15.921 / 20.762 / 33.218 |

Native CUDA and OptiX mean-red observables differ by at most 0.001455 for these
fixtures (64 spp); these are not bitwise-image comparisons or analytical claims
for the nonuniform scene. CPU's 64 fixture mean is 0.685729980.

### Production GPU construction and memory

Separate Nsight Systems runs trace the actual
`barney_optix::native::NanoVDBData_computeMCs<float>` kernel, not a reconstructed
benchmark kernel. Exactly six launches appear per scene; discard the first for
the five warm kernel durations below. Profiling includes memory tracing and may
add overhead, so these are kept separate from unprofiled API/render times.

| N / map | MC count | Range buffer bytes | One volume majorant bytes | Kernel ms min / median / max | Peak traced device bytes (whole client) |
|---|---:|---:|---:|---|---:|
| 64 identity | 64 | 512 | 256 | 0.668806 / 0.668838 / 0.670982 | 38463140 |
| 256 identity | 4096 | 32768 | 16384 | 1.063722 / 1.065002 / 1.065418 | 38545380 |
| 512 identity | 32768 | 262144 | 131072 | 1.153450 / 1.156843 / 1.159914 | 38889444 |
| 1024 identity | 262144 | 2097152 | 1048576 | 4.071523 / 4.075140 / 4.084356 | 41641956 |
| 256 shear | 4096 | 32768 | 16384 | 1.926575 / 1.926638 / 1.928463 | 38545380 |

Range/majorant sizes follow production's 8/4 bytes per cell and are **confirmed
by actual CUDA allocation events**: six allocations and six frees of each size
in every profile. Scalar ranges are per field/device; majorants are per
volume/device. These byte counts exclude NanoVDB storage, geometry/BVH metadata,
allocator padding, context and framebuffers. The separate whole-client peak
sums traced device allocations minus frees and includes those other traced
allocations; it is not an isolated BVH size, physical VRAM measurement, or a
claim to account for driver-internal memory. Traced device balance returns to
zero at shutdown. No isolated backend-BVH memory attribution was available.

The growing extent costs memory even though payload stays fixed. Shear increases
inverse-mapped scan work/overlap without changing cell counts. These data support
future optimization discussion, not a performance guarantee: more extreme
extents/affines, dense payloads, other hardware, and production-size images were
not measured. No arbitrary field work/extent cap, additional validation scan,
root-statistics prerequisite, or tree-based acceleration was added. Checked
allocation arithmetic and controlled CPU failure/recovery tests remain intact.

## Reproduction

From the ticket source workspace (all build directories are workspace-local):

```sh
just submodule # requires the explicit provisioning authorization
cmake -S "$PWD" -B "$PWD/_out/_cmake" -G Ninja \
  -DBARNEY_BACKEND_OPTIX=OFF -DBARNEY_BACKEND_CPU=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$PWD/_out" \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build _out/_cmake -j 6
just build # confirmed source/build/install all point to this workspace
for backend in optix cuda; do
  build=_out/_gpu; optix=ON; cuda=OFF
  if [ "$backend" = cuda ]; then build=_out/_cuda; optix=OFF; cuda=ON; fi
  cmake -S "$PWD" -B "$build" -G Ninja -DBARNEY_BACKEND_CPU=OFF \
    -DBARNEY_BACKEND_OPTIX=$optix -DBARNEY_BACKEND_CUDA=$cuda \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=75
  cmake --build "$build" -j 6
  NANOVDB_TEST_GPU=1 bash tests/test_nanovdb_ingestion.sh "$PWD/$build"
  NANOVDB_TEST_BACKEND=$backend bash tests/test_nanovdb_ranges_gpu.sh "$PWD/$build"
  NANOVDB_TEST_BACKEND=$backend COMPUTE_SANITIZER=1 \
    bash tests/test_nanovdb_ranges_gpu.sh "$PWD/$build"
done
bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cmake"
bash tests/test_nanovdb_ranges.sh "$PWD/_out/_cmake"
```

Host sanitizer configure/build/commands and the exact existing suppressions are
in [README.nanovdb.md](README.nanovdb.md#sanitizers-and-known-unrelated-issues).
Both configurations were rerun for this ticket, then normal flags restored and
`just build` and all three normal API suites rerun. The new benchmark mode is
not a sanitizer stress test. New CPU sanitizer runs do not remove ticket 08's
qualifications; external SDK/Embree libraries remain uninstrumented.

```sh
# Unprofiled: repeat with _out/_cuda for native CUDA.
for n in 64 256 512 1024; do
  NANOVDB_TEST_GPU=1 NANOVDB_BENCHMARK=$n \
    bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_gpu"
done
NANOVDB_TEST_GPU=1 NANOVDB_BENCHMARK=256 NANOVDB_BENCHMARK_SHEAR=1 \
  bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_gpu"

# Profiling: one file per extent; use the same shear environment for that case.
for n in 64 256 512 1024; do
  NANOVDB_TEST_GPU=1 NANOVDB_BENCHMARK=$n nsys profile --trace=cuda \
    --cuda-memory-usage=true --sample=none --cpuctxsw=none \
    --force-overwrite=true -o /tmp/nvdb05-profile-$n \
    bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_gpu"
  nsys stats --report cuda_gpu_kern_sum --format csv \
    /tmp/nvdb05-profile-$n.nsys-rep
done
```

`nsys stats` exports a SQLite file. Reproduce the kernel/memory extraction with
`python3 tests/summarize_nanovdb_profile.py /tmp/nvdb05-profile-256.sqlite 256`.
The script validates event counts/sizes and prints measured warm timings and
whole-client traced device peak. Profiles themselves are machine-local generated
artifacts, not repository test inputs. Benchmark fixture argument guards apply
only to this synthetic workload generator, never to accepted field extents.

## Remaining acceptance qualifications

- Float only. Real bundled 32.7.0 fixtures verified; no other genuine compatible
  reader-version fixtures available. Cubic/unknown/omitted filters select linear.
- Internal topology and scalar content remain trusted; no arbitrary-corruption
  or NaN/Inf-content safety promise. No new validation traversal.
- CPU sanitizer suppressions (08), full OptiX memcheck (09), dense analytical
  transport (10), and native CUDA kernel registration (11) remain explicit.
- No multi-GPU, HIP/HIPRT, alternate integrators, MPI, device allocator injection,
  other-GPU performance, or unsuppressed full-renderer memory-safety claim.
- Normal Float functionality, measured analytical range, numerical conservation,
  lifecycle/error behavior, and sparse GPU construction measurements are verified;
  unavailable coverage is not silently promoted to a pass.
