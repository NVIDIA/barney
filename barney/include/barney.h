// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#define BARNEY_VERSION_MAJOR @BARNEY_VERSION_MAJOR@
#define BARNEY_VERSION_MINOR @BARNEY_VERSION_MINOR@
#define BARNEY_VERSION_PATCH @BARNEY_VERSION_PATCH@

/*! whether the cuda/optix backend has been included in this build */
#cmakedefine01 BARNEY_BACKEND_OPTIX

/*! whether the native-cuda backend has been included in this build */
#cmakedefine01 BARNEY_BACKEND_CUDA

/*! whether the embree backend has been included in this build */
#cmakedefine01 BARNEY_BACKEND_EMBREE

/*! whether this build of barney has support for MPI based rendering;
    i.e., wether barney/barney_mpi.h and (if anari is enabled)
    anari_library_barney_mpi exist */
#cmakedefine01 BARNEY_HAVE_MPI

/*! whether this build of barney has support for the NanoVDB volume
    type. NanoVDB takes a long while to compile, so can be
    enabled/disabled by user */
#cmakedefine01 BARNEY_HAVE_NANOVDB

#include <cstdint>
#include <cstddef>

#ifdef _WIN32
# if defined(barney_STATIC) || defined(barney_mpi_STATIC)
#  define BARNEY_INTERFACE /* nothing */
# elif defined(barney_EXPORTS) || defined(barney_mpi_EXPORTS)
#  define BARNEY_INTERFACE __declspec(dllexport)
# else
#  define BARNEY_INTERFACE __declspec(dllimport)
# endif
#elif defined(__clang__) || defined(__GNUC__)
#  define BARNEY_INTERFACE __attribute__((visibility("default")))
#else
#  define BARNEY_INTERFACE
#endif

#ifdef __cplusplus
#  define BARNEY_API extern "C" BARNEY_INTERFACE
#  define BN_IF_CPP(a) a
#else
#  define BARNEY_API /* bla */
#  define BN_IF_CPP(a) /* ignore */
#endif

typedef struct _BNContext                           *BNContext;
typedef struct _BNObject                         {} *BNObject;
typedef struct _BNData         : public _BNObject{} *BNData;
typedef struct _BNTextureData  : public _BNObject{} *BNTextureData;
typedef struct _BNScalarField  : public _BNObject{} *BNScalarField;
typedef struct _BNGeom         : public _BNObject{} *BNGeom;
typedef struct _BNVolume       : public _BNObject{} *BNVolume;
typedef struct _BNGroup        : public _BNObject{} *BNGroup;
typedef struct _BNModel        : public _BNObject{} *BNModel;
typedef struct _BNRenderer     : public _BNObject{} *BNRenderer;
typedef struct _BNFrameBuffer  : public _BNObject{} *BNFrameBuffer;
// typedef struct _BNDataGroup    : public _BNObject{} *BNDataGroup;
typedef struct _BNTexture2D    : public _BNObject{} *BNTexture2D;
typedef struct _BNTexture3D    : public _BNObject{} *BNTexture3D;
typedef struct _BNLight        : public _BNObject{} *BNLight;
typedef struct _BNCamera       : public _BNObject{} *BNCamera;
typedef struct _BNMaterial     : public _BNObject{} *BNMaterial;
typedef struct _BNSampler     : public _BNObject{} *BNSampler;

typedef BNTexture2D BNTexture;

struct bn_float2 { float x,y; };
struct bn_float3 { float x,y,z; };
struct bn_float4 { float x,y,z,w; };
struct bn_int2 { int x,y; };
struct bn_int3 { int x,y,z; };
struct bn_int4 { int x,y,z,w; };

struct BNTransform {
  struct {
    bn_float3 vx;
    bn_float3 vy;
    bn_float3 vz;
  } l;
  bn_float3 p;
};

typedef enum {
  BN_FB_COLOR  = (1<<0),
  BN_FB_DEPTH  = (1<<1),
  BN_FB_PRIMID = (1<<2),
  BN_FB_INSTID = (1<<3),
  BN_FB_OBJID  = (1<<4),
  BN_FB_NORMAL = (1<<5),
} BNFrameBufferChannel;

typedef enum {
  /*! a undefined data type */
  BN_DATA_UNDEFINED,
  /*! BNData */
  BN_DATA,
  /*! BNObject */
  BN_OBJECT,
  /*! BNTexture */
  BN_TEXTURE,
  BN_TEXTURE_3D,
  /*! scalar types */
  BN_INT8=100,
  BN_INT8_VEC2,
  BN_INT8_VEC3,
  BN_INT8_VEC4,
  BN_UINT8=110,
  BN_UINT8_VEC2,
  BN_UINT8_VEC3,
  BN_UINT8_VEC4,
  BN_INT16=120,
  BN_INT16_VEC2,
  BN_INT16_VEC3,
  BN_INT16_VEC4,
  BN_UINT16=130,
  BN_UINT16_VEC2,
  BN_UINT16_VEC3,
  BN_UINT16_VEC4,
  BN_INT32=140,
  BN_INT32_VEC2,
  BN_INT32_VEC3,
  BN_INT32_VEC4,
  BN_UINT32=150,
  BN_UINT32_VEC2,
  BN_UINT32_VEC3,
  BN_UINT32_VEC4,
  BN_INT64=160,
  BN_INT64_VEC2,
  BN_INT64_VEC3,
  BN_INT64_VEC4,
  BN_UINT64=170,
  BN_UINT64_VEC2,
  BN_UINT64_VEC3,
  BN_UINT64_VEC4,
  BN_FLOAT32=180,
  BN_FLOAT32_VEC2,
  BN_FLOAT32_VEC3,
  BN_FLOAT32_VEC4,
  BN_FLOAT64=190,
  BN_FLOAT64_VEC2,
  BN_FLOAT64_VEC3,
  BN_FLOAT64_VEC4,
  /* DEPRECATED - USED BN_INT32_VEC<N> */BN_INT=BN_INT32,
  /* DEPRECATED - USED BN_INT32_VEC<N> */BN_INT2 = BN_INT32_VEC2, 
  /* DEPRECATED - USED BN_INT32_VEC<N> */BN_INT3 = BN_INT32_VEC3,
  /* DEPRECATED - USED BN_INT32_VEC<N> */BN_INT4 = BN_INT32_VEC4,
  /* DEPRECATED - USED BN_FLOAT32 */BN_FLOAT = BN_FLOAT32,
  /* DEPRECATED - USED BN_FLOAT32_VEC<N> */BN_FLOAT2 = BN_FLOAT32_VEC2,
  /* DEPRECATED - USED BN_FLOAT32_VEC<N> */BN_FLOAT3 = BN_FLOAT32_VEC3,
  /* DEPRECATED - USED BN_FLOAT32_VEC<N> */BN_FLOAT4 = BN_FLOAT32_VEC4,

  
  /*! int64_t */
    /* DEPRECATED - USED BN_INT64 */BN_LONG=BN_INT64,
  /*! int2 */
  /* DEPRECATED - USED BN_INT64_VEC<N> */BN_LONG2=BN_INT64_VEC2,
  /* DEPRECATED - USED BN_INT64_VEC<N> */BN_LONG3=BN_INT64_VEC3,
  /* DEPRECATED - USED BN_INT64_VEC<N> */BN_LONG4=BN_INT64_VEC4,
  
  BN_UFIXED8=300,
  BN_UFIXED8_RGBA,
  BN_UFIXED8_RGBA_SRGB,

  BN_UFIXED16,
  
  BN_RAW_DATA_BASE
} BNDataType;

/*! supported element types for unstructured mesh scalar field
    type. Currently this uses VTK numbering of element types; if or
    when anari decides to use some other numbering of its own this
    will change to anari numbering */
typedef enum {
  BN_UNSTRUCTURED_TET = 10,
  BN_UNSTRUCTURED_HEX = 12,
  BN_UNSTRUCTURED_PRISM = 13,
  BN_UNSTRUCTURED_PYRAMID = 14
} BNUnstructuredElementType;

/*! currently supported texture filter modes */
typedef enum {
  BN_TEXTURE_NEAREST,
  BN_TEXTURE_LINEAR
}
BNTextureFilterMode;

/*! currently supported texture filter modes */
typedef enum {
  BN_TEXTURE_WRAP,
  BN_TEXTURE_CLAMP,
  BN_TEXTURE_BORDER,
  BN_TEXTURE_MIRROR
}
BNTextureAddressMode;

/*! Indicates if a texture is linear or SRGB */
typedef enum {
  BN_COLOR_SPACE_LINEAR,
  BN_COLOR_SPACE_SRGB
}
BNTextureColorSpace;

struct BNGridlet {
  bn_float3 lower;
  bn_float3 upper;
  bn_int3   dims;
};


struct BNHardwareInfo {
  int numRanks;
  int numHosts;
  int numGPUsThisRank;
  int numGPUsThisHost;
  int numRanksThisHost;
  int localRank;
};



// ==================================================================
// creators for CONTEXT-owned objects
// ==================================================================

/*! create a new camera of given type. currently supported types:
  "pinhole" */
BARNEY_API
BNCamera      bnCameraCreate(BNContext context,
                             const char *type);

/*! creates a new frame buffer that can be used to render into.  In
  case of using MPI-parallel rendering only the 'owningRank' is
  allowed to read the frame buffer's content. For non-mpi rendering,
  owningRank should be 0 */
BARNEY_API
BNFrameBuffer bnFrameBufferCreate(BNContext context, int deprecated=0);

BARNEY_API
BNModel       bnModelCreate(BNContext ctx);

/*! create a new renderer object. Currently supported types:
    "pathTracer", "default" (same as pathtracer) */
BARNEY_API
BNRenderer    bnRendererCreate(BNContext ctx,
                               const char *type BN_IF_CPP(= "default"));


// ==================================================================
// general set/commit semantics
// ==================================================================

BARNEY_API
void bnCommit(BNObject target);
              
BARNEY_API
void bnSetString(BNObject target, const char *paramName, const char *value);

BARNEY_API
void bnSetData(BNObject target, const char *paramName, BNData value);

BARNEY_API
void bnSetObject(BNObject target, const char *paramName, const BNObject value);

BARNEY_API
void bnSetLight(BNObject target, const char *paramName, BNLight value);

BARNEY_API
void bnSet1i(BNObject target, const char *paramName, int value);

BARNEY_API
void bnSet2i(BNObject target, const char *paramName, int x, int y);

BARNEY_API
void bnSet3i(BNObject target, const char *paramName, int x, int y, int z);

BARNEY_API
void bnSet4i(BNObject target, const char *paramName, int x, int y, int z, int w);

BARNEY_API
void bnSet1f(BNObject target, const char *paramName, float value);

BARNEY_API
void bnSet2f(BNObject target, const char *paramName, float x, float y);

BARNEY_API
void bnSet3f(BNObject target, const char *paramName, float x, float y, float z);

BARNEY_API
void bnSet4f(BNObject target, const char *paramName, float x, float y, float z, float w);

BARNEY_API
void bnSet4x3fv(BNObject target, const char *paramName, const BNTransform *affineMatrix);

BARNEY_API
void bnSet4x4fv(BNObject target, const char *paramName, const bn_float4 *xfm);

BARNEY_API
BNContext bnContextCreate(/*! how many data slots this context is to
                              offer, and which part(s) of the
                              distributed model data these slot(s)
                              will hold */
                          const int *dataRanksOnThisContext=0,
                          int        numDataRanksOnThisContext=1,
                          /*! which gpu(s) to use for this
                            process. default is to distribute
                            node's GPUs equally over all ranks on
                             that given node */
                          const int *gpuIDs=nullptr,
                          int  numGPUs=-1);

/*! destroys a barney context, and all still-active objects aquired
    from this context. After calling bnCntextDestroy, all handles
    acquired through the given context may no longer be accessed or
    used in any form */
BARNEY_API
void bnContextDestroy(BNContext context);

/*! decreases (the app's) reference count of said object by one. if
    said refernce count falls to 0 the object handle gets destroyed
    and may no longer be used by the app, and the object referenced to
    by this handle may be removed (from the app's point of view). Note
    the object referenced by this handle may not get destroyed
    immediagtely if it had other indirect references, such as, for
    example, a group still holding a refernce to a geometry */
BARNEY_API
void  bnRelease(BNObject object);

/*! increases (the app's) reference count of said object by one. This
    will not interfere with barney's internal reference counting (the
    given object will not get deleted until no other barney objects
    use it any more); but will tell barney that the _app_ no longer
    has any claim on this object, and that it is free to remove it if
    it is no longer needed internally */
BARNEY_API
void  bnAddReference(BNObject object);

BARNEY_API
void  bnBuild(BNModel model,
              int whichDataSlot);

// ==================================================================
// render interface
// ==================================================================

BARNEY_API
void bnAccumReset(BNFrameBuffer fb);

BARNEY_API
void bnFrameBufferResize(BNFrameBuffer fb,
                         BNDataType colorFormat,
                         int sizeX, int sizeY,
                         uint32_t requiredChannels BN_IF_CPP( = BN_FB_COLOR));

BARNEY_API
void bnFrameBufferRead(BNFrameBuffer fb,
                       BNFrameBufferChannel channelToRead,
                       void *pointerToReadDataInto,
                       BNDataType requiredFormat);

/*! Return the actual framebuffer dimensions (may differ from resize
    when e.g. AI upscaling forces even dimensions). Use for buffer
    allocation and stride when mapping the color channel. */
BARNEY_API
void bnFrameBufferGetSize(BNFrameBuffer fb, int *sizeX, int *sizeY);

BARNEY_API
void bnRender(BNRenderer    renderer,
              BNModel       model,
              BNCamera      camera,
              BNFrameBuffer fb);

BARNEY_API
void bnSetInstances(BNModel model,
                    int whichSlot,
                    BNGroup *groupsToInstantiate,
                    BNTransform *instanceTransforms,
                    int numInstances);

/*! update only the instance transforms for the given slot, reusing
    existing group structures. This is much faster than bnSetInstances +
    bnBuild when only transforms change (e.g., animation). The number of
    instances must match what was previously set via bnSetInstances. */
BARNEY_API
void bnUpdateInstanceTransforms(BNModel model,
                                int whichSlot,
                                BNTransform *instanceTransforms,
                                int numInstances);

/*! allows for setting one of 5 attribute arrays for the given slot's
    model. */
BARNEY_API
void bnSetInstanceAttributes(BNModel model,
                             int whichSlot,
                             const char *attributeName,
                             BNData data);

// ==================================================================
// scene content
// ==================================================================

/*! a regular, one-dimensional array of numItems elements of given
  type. On the device the respective elemtns will appear as a plain
  CUDA array that lies in either device or managed memory.  Note data
  arrays of this type can _not_ be assigned to samplers because these
  need data to be put into cudaArray's (in order to create
  cudaTextures) */
BARNEY_API
BNData bnDataCreate(BNContext context,
                    int whichSlot,
                    BNDataType dataType,
                    size_t numItems,
                    const void *items);

BARNEY_API
void bnDataSet(BNData data,
               size_t numItems,
               const void *items);

/*! creates a cudaArray2D of specified size and texels. Can be passed
  to a sampler to create a matching cudaTexture2D, or as a background
  image to a renderer */
BARNEY_API
BNTextureData bnTextureData2DCreate(BNContext context,
                                    int whichSlot,
                                    BNDataType texelFormat,
                                    int width, int height,
                                    const void *items);
BARNEY_API
BNTextureData bnTextureData3DCreate(BNContext context,
                                    int whichSlot,
                                    BNDataType texelFormat,
                                    int width, int height, int depth,
                                    const void *items);

BARNEY_API
BNLight bnLightCreate(BNContext context,
                      int whichSlot,
                      const char *type);
                    
BARNEY_API
BNGroup bnGroupCreate(BNContext context,
                      int whichSlot,
                      BNGeom *geoms, int numGeoms,
                      BNVolume *volumes, int numVolumes);
BARNEY_API
void bnGroupBuild(BNGroup group);

BARNEY_API
BNTexture2D
bnTexture2DCreate(BNContext context,
                  int whichSlot,
                  BNDataType texelFormat,
                  /*! number of texels in x dimension */
                  uint32_t size_x,
                  /*! number of texels in y dimension */
                  uint32_t size_y,
                  const void *texels,
                  BNTextureFilterMode  filterMode    BN_IF_CPP(= BN_TEXTURE_LINEAR),
                  BNTextureAddressMode addressMode_x BN_IF_CPP(= BN_TEXTURE_WRAP),
                  BNTextureAddressMode addressMode_y BN_IF_CPP(= BN_TEXTURE_WRAP),
                  BNTextureColorSpace  colorSpace  = BN_IF_CPP(BN_COLOR_SPACE_LINEAR));

BARNEY_API
BNTexture3D
bnTexture3DCreate(BNContext context,
                  int whichSlot,
                  BNDataType texelFormat,
                  /*! number of texels in x dimension */
                  uint32_t size_x,
                  /*! number of texels in y dimension */
                  uint32_t size_y, 
                  /*! number of texels in z dimension */
                  uint32_t size_z,
                  const void *texels,
                  BNTextureFilterMode  filterMode  BN_IF_CPP(= BN_TEXTURE_LINEAR),
                  BNTextureAddressMode addressMode BN_IF_CPP(= BN_TEXTURE_CLAMP));

// ------------------------------------------------------------------
// object-"create" interface
// ------------------------------------------------------------------

/*! create a new geometry of given type. Currently supported types:
    "triangles", "spheres", "cylinders", "capsules" */
BARNEY_API
BNGeom bnGeometryCreate(BNContext context,
                        int whichSlot,
                        const char *type);


/*! create a new scalar field of given type. currently supported
    types: "structured" */
BARNEY_API
BNScalarField bnScalarFieldCreate(BNContext context,
                                  int whichSlot,
                                  const char *type);
                                     
/*! create a new material of given type. currently supported types:
  "matte", "glass" */
BARNEY_API
BNMaterial bnMaterialCreate(BNContext context,
                            int whichSlot,
                            const char *type);

BARNEY_API
BNSampler bnSamplerCreate(BNContext context,
                          int whichSlot,
                          const char *type);


BARNEY_API
BNVolume bnVolumeCreate(BNContext context,
                        int whichSlot,
                        BNScalarField sf);

BARNEY_API
void bnVolumeSetXF(BNVolume         volume,
                   bn_float2        domain,
                   const bn_float4 *colorMap,
                   int              numColorMapValues,
                   float            densityAt1);



#ifdef __cplusplus
/* just for c++ : polymorphic versions */
inline void bnSet(BNObject o, const char *n, bn_float2 v)
{ bnSet2f(o,n,v.x,v.y); }
inline void bnSet(BNObject o, const char *n, bn_float3 v)
{ bnSet3f(o,n,v.x,v.y,v.z); }
inline void bnSet(BNObject o, const char *n, bn_float4 v)
{ bnSet4f(o,n,v.x,v.y,v.z,v.w); }
inline void bnSet(BNObject o, const char *n, bn_int2 v)
{ bnSet2i(o,n,v.x,v.y); }
inline void bnSet(BNObject o, const char *n, bn_int3 v)
{ bnSet3i(o,n,v.x,v.y,v.z); }
inline void bnSet(BNObject o, const char *n, bn_int4 v)
{ bnSet4i(o,n,v.x,v.y,v.z,v.w); }
#endif

