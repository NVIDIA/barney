# Quantized acceptance evidence (ticket 06)

Verified 2026-10-24 on Float integration `d29a0310` plus this change, in
`/home/tarcila/Code/ANARI/barney/nvdb-ticket-06`. Supported types are now Float,
Fp4, Fp8, Fp16 and FpN. Other types still reject (including the retained real
Int32 fixture). Cubic/unknown/omitted filter strings select linear, not cubic.
This is not complete extension conformance or a NanoVDB decoder validation.

## Implementation and validation boundary

One type table selects the adapter's fixed-root size, native typed metadata,
production macrocell kernel, and sampler/geometry dispatch. All five types use
NanoVDB accessors to decode values. The existing conservative range algorithm,
16-voxel target, filter footprint, affine policy and per-volume majorants are
unchanged. No host voxel scan, root-statistics prerequisite, extent cap,
independent codec or alternate acceleration pipeline was introduced.

Each supported type's fixed envelope in bundled version 32.7.0 is GridData
672 bytes, TreeData 64 bytes and fixed RootNode 64 bytes. Validation checks the
selected type's actual `sizeof(RootType)` rather than assuming Float layout.
The bounded checks and copy-before-read order documented in the README remain
in force. Variable nodes (including FpN leaf widths), internal references,
checksums and scalar contents remain trusted. This does not promise safe
rejection of arbitrary tree corruption. Real version-32 fixtures are generated
with `createNanoGrid`, including FpN's supported adaptive quantization overload;
no header relabeling substitutes for a valid payload.

OptiX initially failed when multiple supported geometry types used modules
created from the same PTX: externally visible non-entry `__boundsFunc__` symbols
collided at pipeline link (error 7251). NanoVDB's OptiX exports now keep those
helpers in an anonymous namespace; C entry points remain exported. CPU and
native-CUDA export behavior is unchanged. The mixed-type update tests then
passed, including switching back to Float in the same device.

## Tests and oracles

- The original Float API tests, six numerical groups, analytical transmission
  and allocation-failure cases remain. The old real-Fp4 rejection assertion is
  now acceptance. All four quantized types get real-grid acceptance, unaligned
  input with trailing padding, fixed header/tree/root truncation and recovery.
- For each quantized type, two dependent volumes start with Float (4x4x35),
  replace it with quantized data (2x3x18 at origin (1,1,0), changed extrema),
  legally edit the managed byte array in place, invalidate with empty bytes,
  recover, and replace with Float again. Joint images are consumed before any
  dependent-volume/world recommit, then compared individually too. Invalid
  images match the unrelated-sphere reference and bounds contain only that
  sphere. Final images and bounds match fresh equivalent scenes.
- The independent non-monotonic TF update tests now run for all five types:
  unmap only volume 0's opacity array, then replace only volume 1's opacity.
  Both volumes render in the next frame and agree with fresh scenes.
- Each type runs the full impulse-based nearest-to-linear/fallback/unset matrix.
  Both volumes individually must visibly distinguish nearest from linear;
  comparisons to fresh scenes cannot hide a no-op filter. Defaults, empty,
  cubic, aliases and case variants retain the established policy.
- All five types run translated/scaled/rotated/sheared header plus ANARI
  reflection-instance composition against equivalent baked headers, with and
  without statistics. The existing 4096 spp affine tolerance is unchanged.
- CPU numerical tests add all four quantized types with statistics present and
  absent, negative origins, inactive values below active extrema, background
  above active extrema, partial cells and affine maps. Nearest is compared to
  NanoVDB's decoded voxel; linear to independently worked decoded-corner weights.
  Local ranges and two independent non-monotonic TF majorants bound production
  samples at interiors, faces, edges and corners for both filters.
- The CUDA numerical client exercises all five types, two statistics modes,
  identity/composed affine maps, 225 cells (partial launch block), 125 positions
  per cell and both filters. Its decoded-corner oracle uses explicit weights,
  not a second Barney sampler. A Float grid containing a stored inactive -7
  value is converted through NanoVDB to each destination type; inactivity is
  asserted. Quantized values need not equal the original Float source.

Images remain 48x32 linear float RGBA, 64 spp, fresh frames/reset sample sequence,
no denoiser, mean-RGBA equivalent-scene tolerance 0.005 and bounds tolerance 1e-5.
No exact cross-type/source image equality or codec fidelity is required.

TDD: changing the real-Fp4 acceptance expectation first failed with `NanoVDB
requires a Float grid` and `accept real Fp4 grid`, before production changes.
The first OptiX run then exposed the duplicate bounds symbols above; localizing
helpers restored the pass. A temporary production sensitivity mutation forcing
only quantized samplers to linear failed the retained API nearest assertion:
nearest/linear delta 2.66785e-11 instead of >0.01. Reverting it restored all 313
CPU checkpoints. No mutation remains in the commit.

## Backend and sanitizer evidence

Same machine/toolchain as [ticket 05](nanovdb_float_evidence.md): CPU/Embree 4.4.1,
GCC 16.2.0; one RTX 5880 Ada (compute 8.9), CUDA 13.4, OptiX 9.1. Actual runtime
queries, kernels and frames establish GPU availability, not `nvidia-smi`.

| Backend/check | Result |
|---|---|
| CPU/Embree full API | 313 PASS checkpoints, including 15 allocator-failure checkpoints |
| OptiX full API | 298 PASS checkpoints |
| Native CUDA RT full API | 298 PASS checkpoints |
| CPU numerical | 14 groups: six original Float plus eight quantized type/statistics cases |
| CUDA numerical, OptiX configuration | 20 type/statistics/map combinations; memcheck 0 errors |
| CUDA numerical, native-CUDA configuration | same 20 combinations; memcheck 0 errors |
| CPU UBSan API / numerical | 298 API checkpoints with existing 08 suppressions; 14 numerical groups unsuppressed |
| CPU ASan+UBSan API / numerical | same; LSan uses existing InstanceGroup suppression (9264 allocations, 339572 bytes); numerical unsuppressed |

The 298 common checkpoint labels match as multisets across both GPUs and both
CPU sanitizer runs. CPU normal runs add only the 15 allocator checkpoints.
Nearest/linear joint deltas for every type: CPU 0.114831, OptiX 0.113807, native
CUDA 0.114964. Affine-instance deltas for every type: 0.00225099 / 0.00207173 /
0.00225027 respectively. These controlled fixtures encode their endpoints
consistently; this is not a claim that arbitrary quantized images equal Float.

Nsight traced the **actual production** `NanoVDBData_computeMCs<T>` kernels in
the complete OptiX API suite: Float 166 launches, Fp4 49, Fp8 48, Fp16 48, FpN
48. Thus GPU typed construction evidence is distinct from the standalone helper
tests. This is dispatch evidence, not a new representative performance benchmark;
Float sparse performance measurements and their limitations remain in ticket 05.

### Reproduction commands

All commands run from the workspace above. Fresh workspace provisioning was
explicitly authorized. Build outputs are local, not borrowed from integration.

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

The CPU configure/build/API/numerical sequence was repeated with each of
`undefined` and `address,undefined` as follows:

```sh
# Repeat with sanitizers=address,undefined
sanitizers=undefined
cmake -S "$PWD" -B "$PWD/_out/_cmake" \
  -DCMAKE_CXX_FLAGS="-fsanitize=$sanitizers -fno-omit-frame-pointer" \
  -DCMAKE_SHARED_LINKER_FLAGS="-fsanitize=$sanitizers"
cmake --build _out/_cmake -j 6
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
LSAN_OPTIONS="suppressions=$PWD/tests/nanovdb_lsan.supp:print_suppressions=1" \
UBSAN_OPTIONS="halt_on_error=1:print_stacktrace=1:suppressions=$PWD/tests/nanovdb_ubsan.supp" \
SANITIZERS=$sanitizers bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_cmake"
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 UBSAN_OPTIONS=halt_on_error=1 \
SANITIZERS=$sanitizers bash tests/test_nanovdb_ranges.sh "$PWD/_out/_cmake"
cmake -S "$PWD" -B "$PWD/_out/_cmake" \
  -DCMAKE_CXX_FLAGS= -DCMAKE_SHARED_LINKER_FLAGS=
just build
```

The first combined ASan build+API command exceeded its 300-second harness limit
after the build succeeded, during affine rendering. A separate 600-second API
and numerical run completed successfully (442 seconds). Normal flags were then
restored, `just build` passed, and normal CPU API/numerical checks passed again.
No new sanitizer suppression was added; SDK/Embree remain uninstrumented.

```sh
NANOVDB_TEST_GPU=1 nsys profile --trace=cuda --sample=none --cpuctxsw=none \
  --force-overwrite=true -o /tmp/nvdb06-dispatch \
  bash tests/test_nanovdb_ingestion.sh "$PWD/_out/_gpu"
nsys stats --report cuda_gpu_kern_sum --format csv /tmp/nvdb06-dispatch.nsys-rep
bash -n tests/test_nanovdb_ingestion.sh tests/test_nanovdb_ranges.sh tests/test_nanovdb_ranges_gpu.sh
bash tests/test_image_region.sh
bash tests/test_to_string_datatype.sh
```

All above checks passed. The ANARI ingestion client uses
`-Wall -Wextra -Werror -Wno-template-body`. The CPU numerical runner uses
`-O2 -Wno-template-body` without `-Werror`; the CUDA numerical runner uses
`-O2 -Xcompiler=-Wno-template-body` without host or device warning-as-error
flags. Thus the numerical runs are not warning-as-error evidence. Repository
`just build` confirmed this workspace's source/build/install paths and successful
`lib` cleanup. (Corrected during ticket 12's final review.)

## Explicit remaining limits

- Tickets **08–11 remain open qualifications**, not cleared by quantized support:
  existing CPU suppressions/one-entry TF defect; full OptiX sanitizer baseline
  hardware exception; general high-optical-depth fastLog bias; native CUDA
  duplicate cuBQL/CUB kernel registration under Compute Sanitizer.
- Full OptiX memcheck was not rerun. A full native-CUDA memcheck attempt
  (`NANOVDB_TEST_GPU=1 COMPUTE_SANITIZER=1 bash tests/test_nanovdb_ingestion.sh
  "$PWD/_out/_cuda"`) again reported ticket 11's duplicate-kernel errors, then
  hit the 240-second command bound after 179 checkpoints. **Incomplete, not a
  clean pass**; no final error-summary or full instrumented quantized-rendering
  claim. The passing standalone numerical memchecks are separate evidence.
- No multi-GPU, HIP/HIPRT, MPI, alternate integrator, device allocator injection,
  other genuine compatible NanoVDB-version fixture, exhaustive variable-width
  FpN codec coverage, arbitrary-topology validation or complete renderer memory
  safety claim. Prior unsupported-platform evidence is not promoted to a pass.
- The original analytical Float tests remain limited to optical depth <=1;
  the printed depth-2 LIMIT is not a passing analytical case. Quantized images
  are compared with equivalent final quantized scenes, not unquantized sources.
