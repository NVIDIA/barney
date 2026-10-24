# NanoVDB ingestion and recovery regression

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
   restriction is imposed. The existing singleton macrocell allocation is
   fixed-size; finer-grid arithmetic belongs to the macrocell ticket.
7. Copy only the declared grid into aligned HostBuffer storage. CPU RTC buffers
   also guarantee 32-byte alignment, with checked rounding and allocation
   failure exceptions. CUDA/other device execution is not covered by this test.
8. Native metadata is constructed through the typed grid, not a GridMetaData
   pointer cast: the root need not immediately follow TreeData.

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
it cannot interfere with sanitizer allocation interceptors.

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

Only the enabled Embree CPU backend was verified. CUDA/OptiX/HIP, device
sanitizers and multi-GPU behavior are unverified. Filtering, finer conservative
ranges, affine-map validation and quantized acceptance are subsequent tickets,
not claims of this ingestion test.
