# NanoVDB Float regression tests

Ticket 05's [Float acceptance evidence](nanovdb_float_evidence.md) records the
current CPU/OptiX/native-CUDA backend matrix, analytical homogeneous transmission,
reproducible sparse build/memory/render measurements, and explicit remaining
integrator, sanitizer and hardware limits. The sections below describe the
accumulated tests and earlier implementation evidence.

Run from a Barney source workspace with a configured CPU build:

```sh
cmake --build "$PWD/_out/_cmake" -j 4
bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cmake"
```

The runner compiles a small ANARI client against the installed SDK and loads
**this build's** Barney library. The test generates real NanoVDB fixtures with
the bundled construction API; it does not relabel an older-version payload.
GCC's `-Wno-template-body` works around the bundled GridBuilder's unused iterator
referencing a nonexistent `LeafNode::isActive`. Other client warnings are errors.

## Contract and envelope inventory

The Float reader uses `Version::isCompatible()`: in the bundled reader this is
`major == 32`. The generated Float, Int32 and Fp4 fixtures are version **32.7.0**.
No other real compatible-version fixtures were available; modifying the major
to 31 tests rejection, not acceptance of another version.

Checks precede their corresponding reads:

1. Require `ARRAY1D` / `UINT8`. Helium's typed parameter lookup rejects ARRAY2D
   before the adapter treats it as Array1D. Even a parameterless first commit
   is finalized and reports missing data.
2. Require `sizeof(GridData)` (672 bytes here), then copy into an aligned local
   header. Check the reader's header validity predicate, version compatibility,
   Float type, count 1 and index 0.
3. Require the declared payload within the array and at least GridData plus
   TreeData (64 bytes). Bound its size by pointer-offset representation and
   `size_t - NANOVDB_DATA_ALIGNMENT` before HostBuffer's padded allocation.
4. Copy TreeData. Require its signed root offset to be past TreeData, aligned
   to 32 bytes, and within the declared payload, with a complete fixed Float
   RootNode available. Subtraction-based bounds checks avoid offset overflow.
5. Zero active voxels means valid/empty immediately, regardless of stored
   bounds, background or topology. No native field or macrocell launch is made.
6. For nonempty fields, copy the root index bbox; compute each dimension in
   int64 and require a positive, int-representable result. No extent-to-file-size
   restriction is imposed. Rounded-up 16-voxel-target dimensions and their
   product must fit the existing signed-int MC/DDA/majorant indexing, and range
   allocation bytes (including alignment padding) must fit `size_t`.
7. Copy only the declared grid into aligned HostBuffer storage. CPU RTC buffers
   also guarantee 32-byte alignment, with checked rounding and allocation
   failure exceptions. CUDA/other device execution is not covered by this test.
8. Native metadata is constructed through the typed grid, not a GridMetaData
   pointer cast: the root need not immediately follow TreeData.
9. Both stored float/double affine matrices and inverses must be finite and
   nonsingular; translations must be finite. Determinants use widened arithmetic
   and an exact zero test, not an epsilon. The eventual float MC origin, extent,
   spacing and reciprocal spacing must be representable (finite, positive
   spacing). These are representation checks, not a near-singular map cutoff.
   Zero-active grids still skip all extent/allocation/construction work.

Trailing padding is accepted even if it contains grid magic/a complete second
byte sequence: the **declared count**, not a signature scan, determines rejection.
Declared padding of 1–32 bytes and an aligned gap before the root are exercised.

This is NOT a topology validator. Root tiles, child references, variable node
storage, blind data, checksums and scalar NaN/Inf content are not traversed or
validated. Arbitrary corruption within the declared tree remains outside the
safe-rejection boundary.

## Lifecycle and image oracle

Each NanoVDB finalize drops the adapter's old native field. A successful update
creates a new one; dependent volumes detect the changed native binding and
recreate their acceleration. Invalid and empty fields are excluded from groups.
This intentionally avoids keeping stale shared macrocell data or volume-specific
majorants alive as the current scene. Normal array-unmap notifications use the
same path, without an unrelated world/volume recommit.

Tests render two volumes sharing a field with different, value-dependent
transfer functions. Each volume is also rendered alone to prove it contributed
before rejection. During the invalid interval, both disappear, world bounds
contain only an unrelated red sphere, and that sphere remains visible. Recovery
uses different bounds and values and compares against a newly constructed scene.
Populated/empty/populated transitions and in-place managed-array edits are also
compared against fresh equivalents.

Images are 48x32 linear float RGBA, 64 samples/pixel, denoising disabled. Every
comparison creates a fresh frame (reset accumulation/sample sequence); no
cross-backend bitwise equality is assumed. Mean absolute RGBA difference must
be <= 0.005 for equivalent scenes. A populated scene must differ from the invalid
reference by > 0.01, and each volume alone by > 0.005, so an all-background or
missing-volume result cannot pass. Bounds use 1e-5 absolute tolerance. The CPU
runs have produced effectively zero equivalent-scene differences.

On Linux/glibc the non-sanitized client interposes system malloc/aligned_alloc
for a one-shot failure, not an internal renderer mock. It exercises the adapter
HostBuffer, native upload buffer, and native download HostBuffer. Each failure
must produce an ANARI error, invalidate the field, remove both volume images
and bounds, and recover to the fresh image. Sanitizer runs disable this hook so
it cannot interfere with sanitizer allocation interceptors. A fourth shared-volume
case fails the CPU macrocell range-buffer allocation, checks that both volumes
vanish, and compares recovery to a fresh scene. GPU runs explicitly skip these
CPU/system-allocation hooks; device allocator failure is not claimed tested.

## Data and independent transfer-function updates

`checkDataUpdates` starts with a 4x4x35 grid (three macrocells along z,
values 0.1–0.3), replaces it with a differently sized byte array containing a
2x3x18 grid at index origin (1,1,0) (two macrocells along z, values 0.6–0.9),
then edits that managed array in place to values 0.05–0.2. The in-place fixtures
have identical byte lengths and topology. Only `anariUnmapArray` notifies the
last update: no field, volume or world recommit is used. Each transition is
compared against fresh final scenes, jointly first and then per volume, with
world bounds compared too. Both transitions must change the joint image by
more than 0.01, preventing an ignored update from passing by coincidence.

`checkTransferFunctionUpdates` uses two volumes sharing a ramp field. It first
unmaps only volume 0's managed opacity array, increasing its interior peaks
from [0,0.05,0,0.1,0] to [0,0.9,0,0.7,0]. It then replaces only volume 1's
opacity array, decreasing [0.1,0,0.4,0,0.1] to [0.01,0,0.02,0,0.01]. Each
update is consumed with both volumes in the very next frame, without committing
the field, world or other volume. Joint and individual images must agree with
fresh scenes; the joint before/after difference must exceed 0.01. There is no
assertion that pixels for one volume stay unchanged during joint transport.
The existing 64 spp, fresh-frame reset, disabled denoiser and 0.005 mean-RGBA
tolerance apply. The original invalid/empty/recovery and filter-only matrix
remains intact.

`nanovdb_majorants.h` extends the existing production numerical seam on CPU
and CUDA. Two independent non-monotonic `TransferFunction::DD` instances map
production reconstructed samples at affine-cell interiors, faces, edges and
corners; each density must be bounded by its own production majorant over the
shared scalar range (1e-6 absolute float tolerance). Updating only the first
map's peaks tests both maps again. Independently worked full-domain, partial
segment and clamped endpoint checks prevent an endpoint-only maximum from
passing: the updated first full majorant is 1.8, the second is 1.2, and their
[0.1,0.2] maxima are 1.44 and 0.18 with density scales 2 and 3.

Ticket 04 required no production change: existing field replacement/rebinding
and per-volume majorant ownership satisfy these tests. Sensitivity experiments
were temporary and reverted: ignoring subsequent native `setXF` calls makes
the API test fail at the first TF update; omitting interior TF knots makes the
numerical test fail. Disabling only `setXF`'s dirty flag did **not** fail: the
ANARI path also sets `principledVolume` (marking majorants dirty) and performs
structural builds. The tests intentionally assert observables, not a particular
invalidation path or rebuild count.

Verified on CPU/Embree and one RTX 5880 Ada OptiX device: 136 and 121 API
checkpoints respectively; all 12 new updated/fresh image comparisons had mean
RGBA error below 1e-9. Six CPU numerical groups and all four CUDA map/statistics
combinations pass, including non-monotonic majorants. CUDA standalone memcheck
reports zero errors. UBSan-only and ASan+UBSan API runs each pass 121 checkpoints
with the existing ticket 08 suppressions; numerical sanitizer runs are
unsuppressed. The sanitizer/backend qualifications below still apply, notably
no passing full OptiX Compute Sanitizer claim (ticket 09).

## Nearest and linear filtering

The ANARI `filter` parameter matches exact, case-sensitive `nearest` and
`linear`. Omitted/unset, empty, `cubic`, differently cased, alias and unknown
strings select linear. **Cubic is deliberately unsupported**; this fallback
is a declared extension limitation, not full filtering conformance. The
adapter carries the effective mode into each production sampler. Both modes
retain the existing world-to-index/half-voxel shift and boundary clamping;
nearest selects the closer voxel (halfway ties select the upper index).

Filter-only commits retain the application's data array and use the existing
field-finalization/rebinding lifecycle. Both dependent volumes receive the new
sampler and their own transfer-function majorants by the next render. Current
finalization recreates the native field; ranges already conservatively cover
both reconstruction footprints, so rebuilding them is not a correctness
requirement or a test assertion.

The API regression uses a 35-voxel extent with one impulse slab at index 12
and two independent non-monotonic opacity maps: [0, 0.8, 0] and [0, 0.3, 0].
The first macrocell ends at world z=35/3: its nearest footprint reaches index
11, while linear also reaches the impulse at 12. Thus a nearest-only local
range would need widening on the update. A numerical production-sampler/range
check confirms the first-cell sample at z=11.625 changes from 0 to 0.125 and
that the shared range already includes [0,1].
Nearest samples transparent endpoints, while linear reaches opaque intermediate
values. Thus simply comparing updated and fresh scenes cannot mask an ignored
filter: their nearest/linear images must also differ by > 0.01, jointly and
for each volume alone. Fresh omitted and explicit-linear fields must agree.
After each return to nearest, commits change only the filter to `linear`,
`cubic`, empty, `Nearest`, `NEAREST`, `Linear`, `nearestNeighbor`, `point`,
`unknown`, or unset. Both volumes are rendered together immediately (without a
world/volume recommit), then individually, against fresh linear equivalents.
The existing 64 spp / 0.005 mean-RGBA tolerance and fresh-frame reset apply.
CPU/OptiX nearest-versus-linear differences were 0.114831/0.113807; equivalent
scene deltas were below 1e-9 on both backends. No rebuild counts are asserted.

CPU and CUDA numerical tests check both production modes against hand-calculated
samples: corners 1, -7, and six 9s yield linear quarter-weight values 3.375,
8.125, 7.125 and 1.125, versus discrete nearest values 1, -7 or 9. Mixed-axis
positions, negative origins, inactive values and clamping are included. Both
modes are checked against the same conservative ranges at cell interiors,
faces, edges and corners under affine maps. The CPU large-origin regression
also runs both modes. These tests use no second sampler as an oracle.

## Sanitizers and known unrelated issues

Instrument the existing build and the client (restore the flags afterwards):

```sh
cmake -S "$PWD" -B "$PWD/_out/_cmake" \
  -DCMAKE_CXX_FLAGS='-fsanitize=address,undefined -fno-omit-frame-pointer' \
  -DCMAKE_SHARED_LINKER_FLAGS='-fsanitize=address,undefined'
cmake --build "$PWD/_out/_cmake" -j 4
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
LSAN_OPTIONS="suppressions=$PWD/tests/nanovdb_lsan.supp:print_suppressions=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:suppressions=$PWD/tests/nanovdb_ubsan.supp" \
SANITIZERS=address,undefined \
  bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cmake"
```

Also configure/build/run with **undefined only** using the same UBSAN_OPTIONS.
This matters: ASan's allocator can mask ordinary-allocator alignment defects.
That separate run caught the CPU storage alignment bug fixed by this ticket.

Suppressions are restricted to existing general CPU defects tracked by
nanovdb-support ticket 08: zero-byte null memcpy arguments in CPU Buffer/Device,
a deliberately null unused TraceInterface reference in CPU bounds callbacks,
and leaked instance-geometry references allocated by InstanceGroup::buildAccel.
No NanoVDB diagnostics or ASan access errors are suppressed. Explicit two-entry
transfer functions avoid the independent one-entry majorant out-of-bounds bug
also recorded in 08. These runs do not establish an unsuppressed, leak-free CPU
backend; external SDK/Embree libraries themselves are not instrumented.

Restore the normal build:

```sh
cmake -S "$PWD" -B "$PWD/_out/_cmake" \
  -DCMAKE_CXX_FLAGS= -DCMAKE_SHARED_LINKER_FLAGS=
cmake --build "$PWD/_out/_cmake" -j 4
```

## Conservative macrocell construction

The bundled `nanovdb/tools/cuda/GridStats.cuh` implements device statistics,
including `updateGridStats`, but iterates `cbeginValueOn()` and writes active-only
per-node extrema/bounds. It neither includes all stored inactive/background
values nor produces reconstruction-aware world-aligned local ranges. It is not
suitable for this task. Historical Barney (`c24899f0`'s parent) used rounded-up
16-voxel dimensions and one GPU thread per cell, but associated index blocks
with world cells and used unchecked int products. That association is not valid
under rotation/shear. VisRTX's interpolation margin is useful comparison
history, not a proof for Barney's transformed/clamped sampler.

Production now launches the Barney RTC compute kernel on every selected device
(one thread per macrocell). CPU RTC executes the same range helper on the host;
CPU results alone are not GPU evidence. Dimensions are `ceil(indexExtent/16)`;
the world header AABB is divided uniformly into those counts. Thus 16 is the
historical index-extent target, not a guarantee of an exact 16-sample footprint
under arbitrary affine maps. Small and partial blocks are retained.

For each world cell, `NanoVDBHelpers.h` encloses the inverse affine mapping using
outward-rounded intervals of the **stored float** coefficients. Subtraction,
products and the right-associated NanoVDB matrix sum are enclosed, whether
fused or unfused. It includes the existing -0.5 centering shift, clamps to the
true integer index bounds, and scans floor(lower) through floor(upper)+1
(inclusive, clipped to the index bounds). This covers both linear interpolation
neighbors and the nearest neighbor footprint. No filter-specific rebuild is
needed. The integer scan uses int64 loop counters and NanoVDB accessors, so
stored inactive values, tile values and sparse background all participate.
Nonfinite interval intermediates conservatively select the entire affected
index axis instead of imposing a determinant threshold. The final range is
outward widened for float interpolation rounding. No root extrema are read and
no separate topology/content-validation scan is added.

This is conservative but intentionally not topology-aware: construction work
is proportional to the sum of inverse-mapped cell AABB volumes, not stored node
count. Rotation/shear can increase overlap and loosen ranges. Huge sparse
extents can be expensive; there is no file-size-proportional cap or work timeout.
Large sparse-scene timing/memory measurements are now recorded in
[nanovdb_float_evidence.md](nanovdb_float_evidence.md); they establish no speedup
threshold or guarantee for arbitrary extents.

Ranges are built during native field commit so construction/allocation failures
reach the adapter's existing error/invalidation handler. The shared MC resize
allocates a replacement before freeing its old buffer, permitting exception-safe
cleanup. Scalar ranges remain shared; transfer-function majorants remain per
volume. Integer bounds are compared before the sampler's float-to-int
conversion, preventing float-rounded large origins from fetching outside the
clamped grid.

## Numerical and GPU checks

```sh
bash tests/test_nanovdb_ranges.sh "$PWD/_out/_cmake"
# Fresh GPU build, separate from the CPU build:
cmake -S "$PWD" -B "$PWD/_out/_gpu" -G Ninja \
  -DBARNEY_BACKEND_OPTIX=ON -DBARNEY_BACKEND_CPU=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build "$PWD/_out/_gpu" -j 4
bash tests/test_nanovdb_ranges_gpu.sh "$PWD/_out/_gpu"
NANOVDB_TEST_GPU=1 bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_gpu"
COMPUTE_SANITIZER=1 bash tests/test_nanovdb_ranges_gpu.sh "$PWD/_out/_gpu"
```

The numerical tests exercise the actual range helper and production sampler,
not a duplicate sampler. Hand-calculated center/interpolation values supplement
interior/face/edge/corner range assertions. Cases include negative/nonzero
origins, small/partial cells, dense-ramp local tightness and halos, an origin
beyond exactly representable float integers, inactive -7 and background 9
outside active extrema [1,2], statistics present/absent, translation,
nonuniform scaling, rotation, shear and a 1e-12 invertible scale. The standalone
CUDA test checks 225 cells (including a partial launch block) and 125 boundary
and interior samples per cell for four statistics/map combinations.

End-to-end tests compare statistics-present/absent scenes and exercise affine
headers with ANARI instances. The latter composes a reflection/translation with
a translated, scaled, rotated and sheared header. It compares against a baked
combined header while preserving its AABB sampling domain. Baking an arbitrary
shear from an instance is **not** an equivalent oracle: the header's world AABB
contains additional boundary-clamped regions. Instance IDs change random
sequences, so this comparison uses 4096 samples/pixel (other lifecycle tests use
64). The 0.005 mean RGBA tolerance is retained: observed CPU/OptiX differences
were about 0.00225/0.00207; at 64 independent samples they were about 0.018.
Bounds are also compared. This does not change the existing centering policy.

Verified hardware: one NVIDIA RTX 5880 Ada Generation, compute 8.9, CUDA 13.4,
OptiX 9.1; CPU uses Embree 4.4.1. `nvidia-smi -L` crashes in this environment,
but a direct CUDA runtime query and real device kernel/render execution succeed.
The unsuppressed standalone CUDA memcheck reports **zero errors**. Full OptiX
rendering under Compute Sanitizer is **not verified**: both this change and
unchanged integration 639c8aef fail with an internal sanitizer hardware-exception
error followed by CUDA error 700 on the first populated render. Ticket 09 tracks
that baseline/tooling limitation; no suppression or full device-safety claim is
made. Normal OptiX integration tests pass, including nearest/linear dispatch
and shared-field filter updates. Native CUDA RT was subsequently exercised in
ticket 05; see the current evidence matrix linked above. Multi-GPU, HIP and HIPRT
remain unverified. Quantized types remain a later ticket.

The repository's `just build` must be invoked from the ticket workspace. The
initial direct configure selected `lib64`, while the recipe removes a file in
`lib`; its install completed but cleanup failed. Setting
`-DCMAKE_INSTALL_LIBDIR=lib` in the CPU build fixed that local configuration
mismatch; the workspace-specific `just build` then passed. Direct CMake builds
and recipe results are separate checks.
