# Final review evidence (ticket 12)

Verified 2026-10-24 in `/home/tarcila/Code/ANARI/barney/nvdb-review-fixes`, based
on integration `1ecb5a4b`. This supplements, not replaces, the Float and quantized
evidence and their remaining limitations.

## Reproductions and fixes

- **Dependent allocation failure:** the new ANARI regression failed before the
  fix with `TERMINATING DUE TO UNCAUGHT ANARI EXCEPTION ... std::bad_alloc`
  (exit 134). A 65-cell field allocates 544 aligned bytes of scalar ranges;
  system `aligned_alloc` injection targets a subsequent 288-byte majorant
  allocation, skipping zero or one matches to fail either shared-field volume.
  This is after field range construction, not the earlier range-buffer test.
  Native majorant resize now allocates before freeing its owned buffer and only
  publishes dimensions after success. The constructor no longer makes an
  unnecessary placeholder allocation, also avoiding partial-constructor buffer
  ownership on multiple devices. NanoVDB accelerator creation prepares majorant
  storage during ANARI volume finalization. Its failure uses the field's common
  error/invalidation handler before any group is built. Both dependent volumes
  therefore disappear, including a sibling whose allocation already succeeded.
  Both tests require an error, invalid field, sphere-only image/bounds and
  recovery matching fresh volumes. Existing field-allocation tests still pass.
- **Finite affine overflow:** the review example is valid: x'=1e38*x-3e38,
  index x=5 gives finite world x in [2e38,3e38], but float inverse subtraction
  exceeds FLT_MAX. Off-axis zero coefficients multiply infinity, producing NaN.
  Before the fix, the production numerical test with
  `SANITIZERS=undefined,float-cast-overflow UBSAN_OPTIONS=halt_on_error=1`
  failed at `nanovdb/math/Math.h:185` (NaN converted to int). Sampling now widens
  the *same stored float coefficients* when a mapped component is nonfinite;
  finite float positions, translations and coefficient products fit double.
  Ordinary float sampling is retained. No map rejection threshold was added.
  Existing outward-rounded range intervals enclose widened evaluation as well,
  using their full-axis fallback for nonfinite interval intermediates.
  CPU/CUDA tests check the original x=5 case and x=4..5: at world
  (2.25e38, .75, .75), linear gives 25 and 20 respectively, nearest gives 10.
  The latter checks interior x interpolation, so merely clamping infinity is
  insufficient. Tolerance is 1e-4 for float-coefficient representation. API
  checks accept and replace the original extreme map on all three backends.
- **Fixture duplication:** `nanovdb_fixtures.h` shares only host map/header,
  world-box and voxel-size setup, including the composed affine matrix.
  CPU, quantized, CUDA and API fixtures use it; independent sample/range oracles
  remain separate. No test-only production API was introduced.
- **Warning evidence:** corrected the quantized report's blanket claim. Only
  the ingestion client uses `-Wall -Wextra -Werror -Wno-template-body`. CPU
  numerical uses `-O2 -Wno-template-body`; CUDA uses
  `-O2 -Xcompiler=-Wno-template-body`, neither with warning-as-error flags.
- **DCO:** `CONTRIBUTING.md` requires real contributor sign-offs. The six
  implementation commits and this fix are not made submission-ready by an
  agent inventing a certification. A human contributor must review/certify and
  supply the required sign-offs before upstream submission. None were added.

## Commands and results

All commands below ran from the workspace above. Provisioning was explicitly
authorized; these are fresh workspace-local builds, not borrowed binaries.

```sh
just submodule
cmake -S "$PWD" -B "$PWD/_out/_cmake" -G Ninja \
  -DBARNEY_BACKEND_OPTIX=OFF -DBARNEY_BACKEND_CPU=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_INSTALL_PREFIX="$PWD/_out" \
  -DCMAKE_INSTALL_LIBDIR=lib
cmake --build _out/_cmake -j 6
just build
cmake -S "$PWD" -B "$PWD/_out/_gpu" -G Ninja \
  -DBARNEY_BACKEND_OPTIX=ON -DBARNEY_BACKEND_CPU=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build _out/_gpu -j 6
cmake -S "$PWD" -B "$PWD/_out/_cuda" -G Ninja \
  -DBARNEY_BACKEND_OPTIX=OFF -DBARNEY_BACKEND_CPU=OFF -DBARNEY_BACKEND_CUDA=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build _out/_cuda -j 6
bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cmake"
NANOVDB_TEST_GPU=1 bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_gpu"
NANOVDB_TEST_GPU=1 bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cuda"
bash tests/test_nanovdb_ranges.sh "$PWD/_out/_cmake"
COMPUTE_SANITIZER=1 bash tests/test_nanovdb_ranges_gpu.sh "$PWD/_out/_gpu"
NANOVDB_TEST_BACKEND=cuda COMPUTE_SANITIZER=1 \
  bash tests/test_nanovdb_ranges_gpu.sh "$PWD/_out/_cuda"
```

All passed. CPU API: **321 checkpoints** (21 allocator-related); OptiX and
native-CUDA API: **300 each**. CPU numerical: **15 groups**. Each CUDA numerical
configuration: **20 type/statistics/map combinations plus two extreme cases,
zero memcheck errors**. Real kernel execution reports one RTX 5880 Ada, compute
8.9; no reliance on the known crashing `nvidia-smi`. Final normal rebuilds and
API runs passed again; `just build` installed into this workspace's `_out/lib`
and completed its cleanup successfully.

For each `sanitizers=undefined` and `sanitizers=address,undefined`:

```sh
cmake -S "$PWD" -B "$PWD/_out/_cmake" \
  -DCMAKE_CXX_FLAGS="-fsanitize=$sanitizers -fno-omit-frame-pointer" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=$sanitizers"
cmake --build _out/_cmake -j 6
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
LSAN_OPTIONS="suppressions=$PWD/tests/nanovdb_lsan.supp:print_suppressions=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:suppressions=$PWD/tests/nanovdb_ubsan.supp" \
SANITIZERS=$sanitizers bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cmake"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
SANITIZERS=$sanitizers,float-cast-overflow \
  bash tests/test_nanovdb_ranges.sh "$PWD/_out/_cmake"
```

Both full API runs passed **300 checkpoints** with only the existing ticket-08
suppressions (LSan: 9151 allocations / 338676 bytes from InstanceGroup).
Both numerical runs passed **15 groups unsuppressed**, including
float-cast-overflow. Allocation fault injection remains disabled under
sanitizers and on GPUs; these checks do not claim device allocator injection or
instrumented allocation-failure execution. SDK/Embree are uninstrumented.

The *additional* full API build/client run with `undefined,float-cast-overflow`
stopped in bundled `CreateNanoGrid.h:1376`, converting NaN to unsigned char
while generating an Fp4 fixture. This is **not a pass** and has no new
suppression. Ticket **13** records the builder diagnostic and the reversible
assumption to retain the existing decoder-trust boundary. The focused numerical
seam retains the stronger sanitizer and passes. This is distinct from the
production sampler's reproduced and fixed NaN-to-int defect.

Finally:

```sh
cmake -S "$PWD" -B "$PWD/_out/_cmake" \
  -DCMAKE_CXX_FLAGS= -DCMAKE_SHARED_LINKER_FLAGS=
just build
bash -n tests/test_nanovdb_ingestion.sh tests/test_nanovdb_ranges.sh tests/test_nanovdb_ranges_gpu.sh
bash tests/test_image_region.sh
bash tests/test_to_string_datatype.sh
```

All passed; normal CPU flags are restored. Added-line trailing-whitespace check
also passed. No numerical/image tolerance or existing suppression was weakened.

## Retained limitations

Tickets **08–11 are unchanged**: general CPU sanitizer/one-entry TF defects,
full OptiX Compute Sanitizer baseline failure, dense-slab fastLog bias, and
native-CUDA duplicate kernel registration. Known failing full GPU renderer
memchecks were not retried; passing standalone kernels are not a full renderer
safety claim. Optical-depth-2 remains a printed LIMIT, not a passing assertion.
No new sparse performance benchmark, multi-GPU, HIP/HIPRT, MPI, alternate
integrator, other genuine compatible-version fixture, exhaustive codec/topology
validation or full extension conformance claim. Extreme-domain verification is
API acceptance plus production numerical sampling/ranges, not a huge-world
camera/rendering accuracy claim. DCO sign-offs remain a human submission step.
