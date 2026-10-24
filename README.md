![](samples/collage-triangles.jpg)


# Barney - A Multi-GPU (and optionally, Multi-Node) Implementation of the ANARI Rendering API

Build Status:
[![Windows](https://github.com/NVIDIA/barney/actions/workflows/Windows.yml/badge.svg)](https://github.com/NVIDIA/barney/actions/workflows/Windows.yml) 
[![Ubuntu](https://github.com/NVIDIA/barney/actions/workflows/Ubuntu.yml/badge.svg)](https://github.com/NVIDIA/barney/actions/workflows/Ubuntu.yml)


DISCLAIMER: Though Barney has by now reached a stage where it can be
expected to be reasonably stable and complete, it is still under
active development. If you run into bugs, missing features, or simply
broken/outdated documentation please report those at
https://github.com/NVIDIA/barney 

# What is Barney?

Barney is a renderer that implements that ANARI Cross Platform
Rendering API (https://www.khronos.org/anari/) primarily for NVIDIA
OptiX and CUDA capable GPUs.

### Multi-GPU and Multi-Node Parallel Rendering

Barney is highly scalable, and can be used for both local, single-GPU
rendering (as I do on my laptop on a daily basis), as well as for
parallel rendering on multi-GPU nodes, and even for MPI based
data-replicated and/or data-parallel rendering:

- Single-GPU usage: For single GPU usage Barney works like any other
  ANARI device; multiple GPus or MPI are *not* required to run Barney.

- Multi-GPU: Barney can also make use of more than one GPU. Barney can
  either be used in *explicit* multi-GPU mode (where the ANARI app
  explicitly creates different devices for different GPUs, and then
  "tethers" those using a specific ANARI extension we have introduced
  for this purpose); or it can simply use *automatic* multi-GPU, where
  Barney will simply grab all available GPUs and split the work across
  them.
  
- Multi-Node: For cluster, cloud, or HPC environments Barney also
  supports MPI-parallel rendering (if built with MPI support), in
  which case an MPI-parallel application can use Barney across
  multiple different GPUs and/or nodes.
  
### Data Parallel and/or Data Replicated Rendering

Barney also supports both *data parallel* as well as *data replicated*
rendering: In fully data replicated rendering each GPU (and/or each
node) gets the exact same copy of all the scene content, and different
GPUs render different portions of the final image (ie, this should
make rendering the *same* content *faster*). In fully data parallel
(also sometimes called "distributed") rendering the scene to be
rendered is "distributed" across the different GPUs/nodes, so
different GPUs get different parts of what is logically a single
model; then barney will make sure that each GPU "sees" all content
during rendering (in this mode, Barney will not get faster by adding
more GPUs, but it can render models much larger than what a single GPU
could have rendered). Barney also supports some intermediate modes
where, for example, different nodes work data parallel, but all GPUs
on a given node work data replicated, etc.

### Primarily Focussed on Sci-Vis Content

Barney is primarily intended for the type and size of data one could
encounter in a scientific visualization (sci-vis) content, when used
from tools such as, for example, ParaView. Barney supports all the
typical geometric types required by such applications (triangle
meshes, spheres, cylinders, curves, etc), and also supports the
typical scalar field/volume data types such as structured volumes as
well as Block-Structured AMR and unstructured data (as far as these
are currently supported by ANARI).

### NanoVDB Volume Filtering

The ANARI `nanovdb` spatial field supports Float grids with `filter` set to
exactly `nearest` or `linear` (case-sensitive). Omitting or unsetting the
parameter selects linear, as does any other string, including `cubic`, empty
strings, aliases, and differently cased spellings. Filter-only commits apply
to every volume sharing the field without replacing the application's data.

Cubic reconstruction is deliberately **not supported**: its fallback to linear
is a limitation of Barney's `KHR_SPATIAL_FIELD_NANOVDB` support, not full
extension filtering conformance. See [NanoVDB regression coverage and
limitations](tests/README.nanovdb.md) for verification details.

### Path Tracer

Though clearly focussed on Sci-Vis, Barney is still a pretty capable
ray/path tracer on its own, and will, if scene and material data is
properly set up, also do HRDI environment map lighting, indirect
illumiation, glossy and specular reflection/refraction; depth-of-field
cameras; point-, directional, and to some degree area lights;
volumetric scattering, etc. Barney will clearly not achieve the kind
of correctness or realism that pure global illumination renderers like
Mitsuba or PBRT will be able to achieve; but it is still expected to
behave creditably on typical non-Sci Vis rendering content.

# Building and Running

Barney is not a stand-alone "renderer" or "vis-tool"; it is a library
with an API, and needs other applications to build towards it. As
such, it is never "run" on its own; it also needs to be run from another
application (e.g., `hayStack`, at http://github.org/ingowald/hayStack),
or from any application that supports the ANARI API (see https://www.khronos.org/anari/).


## Dependencies for building Barney

Barney is primarily intended for interactive (multi-)GPU rendering,
but can also be built in a non-GPU configuration. Similarly, one of
barney's most important features is MPI-based data parallel rendering,
but can absolutely also be built---and used--without MPI. As such,
dependencies depend on what exactly needs to get built:

One way or another, barney requires:

- `cmake`, for building
-  a c++-20 compliant c++ compiler (gcc on linux, visual studio on windows, clang on mac)

For CUDA/OptiX Acceleration, it also requires:

- `CUDA`, version 12 and up.

- `OWL` (https://github.com/NVIDIA/owl). Note OWL gets pulled in as a git
   submodule, no need to externally get and install.

- `OptiX`, as part of OWL. See documentation in OWL (https://github.com/NVIDIA/owl) for 
   where to get, and how to best install for OWL to easily find it)

For AMD GPU (ROCm/HIP) Acceleration, it instead requires:

- `ROCm` (https://rocm.docs.amd.com/), version 7 and up, which provides `hipcc`
   and the HIP runtime. The same GPU code that builds for CUDA is compiled with
   HIP; OptiX is NVIDIA-only and is not used in this configuration.

- optionally, `HIPRT` (https://github.com/GPUOpen-LibrariesAndSDKs/HIPRT) for
   hardware-accelerated ray traversal. Without it, barney uses its own
   (cuBQL-based) software traversal.
   
For MPI-based data-parallel rendering:

- Building requires a working MPI install. *Running* barney
  requires a CUDA-aware MPI, for *building* this should not matter. We 
  typically develop under---and test with OpenMPI---4.1.6 or 5.0, but
  users have reported working with other MPI flavors such as  MPICH.

## Building Barney

The ANARI build of barney works in pretty much the same way (and with
the same options), but requires a pre-built and installed `ANARI-SDK` from
https://github.com/KhronosGroup/ANARI-SDK. As to the time of this writing,
you need ANARI SDK version 0.15 (or `next_release` branch)

First, build and install the ANARI SDK (https://github.com/KhronosGroup/ANARI-SDK):

``` bash
cd ANARI-SDK
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<install-dir>
cmake --build .. [ --config Release ]
cmake --install .. [ --config Release ]
```

Then, build barney, using same install dir:
``` bash
cd barney
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<same-install-dir-as-anari> [options]
cmake --build .. [ --config Release ]
cmake --install .. [ --config Release ]
```

By default Barney builds without MPI support; to enable this add
`-DBARNEY_MPI=ON` to the cmake config command.

To build for AMD GPUs with HIP/ROCm instead of CUDA/OptiX, add `-DUSE_HIP=ON`
and select the target architecture with `-DCMAKE_HIP_ARCHITECTURES=<arch>` (for
example `gfx90a` for CDNA2, or `gfx1100` for RDNA3); when left unset it defaults
to `gfx90a`. To use HIPRT hardware traversal, also add `-DBARNEY_BACKEND_HIPRT=ON
-Dhiprt_ROOT=<HIPRT-install>`. The CUDA/OptiX build is unchanged when `USE_HIP`
is off.



# Examples of Supported Geometry and Volume Types 

## Triangle Meshes (including Instances and Color- and Opacity Textures)

Example: PBRT landscape in `miniScene` (http://github.com/ingowald/miniScene) format, 
the 'embree headlight', and the TSDViewer 'TestORB'; all rendering
with HDRI env-map lighting.

![](samples/collage-triangles.jpg)

## Non-Triangular Surface Types for Sci-Vis

Supporting most(all?) of the ANARI non-triangular geometry types: Here
various examples with capsules, spheres, cylinders, cones, and curves.
![](samples/collage-usergeom.jpg)


## Volume Data

Structured Volume Data (`float`, `uint8` and `uint16` are supported,
and any volume can be distributed across different ranks by each rank
having different porions of that volume. `Barney` being intended for
sci-vis, every volume can have its own transfer function.

Here examples 'chest' and 'kingsnake' (both regular structuctured
data, in different input scalar types), and on the right,
'scivis2011', a unstructured mesh volume type inside a
semi-transparent triangular surface.

![](samples/collage-volumes.jpg)


# ANARI / BARNARI

Though `barney` is not *limited to* ANARI (it is its own library, with
its own API), it will also, by default build a (by now reasonably
complete!) `ANARI` "device" that exposes most of barney's
functionality to applications using the ANARI API. If enabled in the
cmake build (it's on by default)---and properly installed via `make
install` or `cmake --install`---this builds a implements an ANARI
device that any ANARI app can load as a ANARI device named
`"barney"`.

Note: To distinguish between the (general) ANARI *API* and the
specific barney-based implementation of this API we typically refer to
this implementation as the `(B)ANARI` device, or simply as `banari`.

## Building BANARI:

- dependencies: `libgtk-3-dev`

- need to get, build, *and install* the ANARI-SDK:
  `https://github.com/KhronosGroup/ANARI-SDK`. Note the SDK *must* be
  installed for barney to properly find it.

- build barney as described above. `BUILD_ANARI` should be on by
  default, so unless explictily disabled this should also build the
  banari device.

## Using BANARI:

The barney devices should be easily usably by any existing ANARI
application by simply loading the `"barney"` device. For those apps
that respect the `ANARI_LIBRARY` environment variable convention you
sohld also be able to just set `ANARI_LIBRARY=barney`, and have the
app load the `"default"` device. 

For data-parallel rendering across multiple collaborating ranks use
the `"mpi"` device subtype. Also see https://arxiv.org/abs/2407.00179
for the conventions on how to properly use data-parallel ANARI (which
barney implements).

# Version History

## v0.10

- major updated to MPI performance
- support for multi-GPU data parallel ANARI using device tethering
- more cuda-like kernel launches across both embree and CPU backends
- updates as required for ANARI 0.15 
- support for CUDA 13
- performance fixes for threading on embree backend
- various fixes throughout

## v0.9.2, 0.9.4, and 0.9.6: 

- various stability fixes and bug fixes in particular relating to
  materials, path tracing, and lighting, as well as on multi-device
  rendering
- closed various missing gaps wrt anari specs (missing formats,
  unsupported parameters, etc)

## v0.9.0

- major rework that allows 'rtcore' abstraction and multiple backends
- support both optix and embree (cpu only) backends
- completely reworked cmake build system (and in particular how stuff
  gets linked)
- reworked install/export/import system, exporting both `barney` and
  `barney_mpi`; should always get used through `find_package(barney)`
  and then linked as import target(s) `barney::barney` and (if found)
  `barney::barney_mpi`
- anari device split into `anari_library_barney` and
  `anari_library_barney_mpi`
- version used for pynari backend that works on all of
  linux/windows/mac and both cpu/gpu
  
Known limitations/issues:
- AMR support currently disabled
- umesh support only support cubql sampler

## v0.8.0

- updated to latest anari 0.12 SDK
- changed cuda arch detection to use 'native' by default, but allowing
  to override on cmdline.
- updated cuda arch detection in owl and cubql so as to allow barney
  to tell them which one to use (so it matches across all projects)
- reworked path tracing (and in particular MIS code) to (mostly-)pass
  a furnace test
