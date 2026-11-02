// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "native/Context.h"
#include "native/FromEnv.h"
#include "native/GlobalModel.h"
#include "native/render/Renderer.h"
#include "native/light/Light.h"
#include "native/Camera.h"
#include "native/fb/FrameBuffer.h"
#include "native/LocalContext.h"

static_assert(sizeof(size_t) == 8, "Trying to compile in 32-bit mode ... this isn't going to work");

#if 0
# define LOG_API_ENTRY std::cout << OWL_TERMINAL_BLUE << "#bn: " << __FUNCTION__ << OWL_TERMINAL_DEFAULT << std::endl;
#else
# define LOG_API_ENTRY /**/ 
#endif


#ifdef NDEBUG 
# define BARNEY_ENTER(fct) /* nothing */
# define BARNEY_LEAVE(fct,retValue) /* nothing */
#else
# define BARNEY_ENTER(fct) try {                                \
  if (0) std::cout << "@bn.entering " << fct << std::endl;      \
  

# define BARNEY_LEAVE(fct,retValue)                             \
  } catch (std::exception &e) {                                 \
    std::cerr << OWL_TERMINAL_RED << "@" << fct << ": "         \
              << e.what() << OWL_TERMINAL_DEFAULT << std::endl; \
    return retValue ;                                           \
  }
#endif

namespace BARNEY_NS {
  using namespace native;

  extern
  Context *
  createLocalContext(/*! how many data slots this context is to
                       offer, and which part(s) of the
                       distributed model data these slot(s)
                       will hold */
                     const int *dataRanksOnThisContext,
                     int        numDataRanksOnThisContext,
                     /*! which gpu(s) to use for this
                       process. default is to distribute
                       node's GPUs equally over all ranks on
                       that given node */
                     const int *gpuIDs,
                     int  numGPUs);
  
  BARNEY_API
  BNContext bnContextCreate(/*! this is the NUMBER of different data
                              groups/slots inthat context. a data
                              *group* is one or more GPUs that share a
                              common set of geometric/volumetric
                              objects. */
                          int        numDataGroupsOnThisContext,
                          /*! tells which data rank / 'color' of data
                            will be stored in each of the
                            `numDataGroupsOnThisContext` data groups
                            of this context. This array *must* have
                            `numDataGroupsOnThisContext` entries */
                          const int *dataRanksInDataGroup,
                          /*! which gpu(s) to use for this
                            process. GPUs will be assigned to data
                            groups on a round-robin basis, so the i'th
                            GPU listed here will get assigned to data
                            group 'i%numDataGroupsOnThisContext' */
                          int numGPUs,
                          const int *gpuIDs)
  {
    LOG_API_ENTRY;
    FromEnv::init();
    assert(dataRanksInDataGroup);
    assert(gpuIDs);
    BNContext ctx
      = (BNContext)LocalContext::create(numDataGroupsOnThisContext,
                                        dataRanksInDataGroup,
                                        numGPUs,
                                        gpuIDs);
    if (!ctx) throw std::runtime_error("could not create barney context");
    return ctx;
  }  
  
  inline Context *checkGet(BNContext context)
  {
    assert(context);
    return (Context *)context;
  }
  
  inline Renderer *checkGet(BNRenderer renderer)
  {
    assert(renderer);
    return (Renderer *)renderer;
  }
  
  inline GlobalModel *checkGet(BNModel model)
  {
    assert(model);
    return (GlobalModel *)model;
  }
  
  inline std::string checkGet(const char *s)
  {
    assert(s != nullptr);
    return s;
  }
  
  inline Object *checkGet(BNObject object)
  {
    assert(object);
    if (!object) throw std::runtime_error
                   ("@barney: trying to use/access null object");
    return (Object *)object;
  }

  inline Data *checkGet(BNData data)
  {
    assert(data);
    return (Data *)data;
  }

  inline Camera *checkGet(BNCamera camera)
  {
    assert(camera);
    return (Camera *)camera;
  }
  
  // inline GlobalModel *checkGet(BNModel model)
  // {
  //   assert(model);
  //   return (GlobalModel *)model;
  // }
  
  // inline ModelSlot *checkGet(BNModel model, int slot)
  // {
  //   return checkGet(model)->getSlot(slot);
  // }
  
  inline Geometry *checkGet(BNGeom geom)
  {
    if (!geom) throw std::runtime_error("@barney: trying to use/access null geometry");
    assert(geom);
    return (Geometry *)geom;
  }
  
  inline Volume *checkGet(BNVolume volume)
  {
    if (!volume) throw std::runtime_error("@barney: trying to use/access null volume");
    assert(volume);
    return (Volume *)volume;
  }
  
  inline ScalarField *checkGet(BNScalarField sf)
  {
    assert(sf);
    return (ScalarField *)sf;
  }
  
  inline Group *checkGet(BNGroup group)
  {
    if (!group) throw std::runtime_error
                  ("@barney: trying to use/access null group");
    assert(group);
    return (Group *)group;
  }
  
  
  inline FrameBuffer *checkGet(BNFrameBuffer frameBuffer)
  {
    assert(frameBuffer);
    return (FrameBuffer *)frameBuffer;
  }

  // ------------------------------------------------------------------
  inline std::shared_ptr<Group> checkGetSP(BNGroup group)
  {
    if (!group) return {};
    return ((Object*)group)->shared_from_this()->as<Group>();
  }
  
  // ------------------------------------------------------------------
  inline std::shared_ptr<Data> checkGetSP(BNData data)
  {
    return checkGet(data)->shared_from_this()->as<Data>();
  }
  
  inline std::shared_ptr<ScalarField> checkGetSP(BNScalarField sf)
  {
    if (!sf) return {};
    return ((Object*)sf)->shared_from_this()->as<ScalarField>();
  }
  // ------------------------------------------------------------------

  /*! creates a cudaArray2D of specified size and texels. Can be passed
    to a sampler to create a matching cudaTexture2D, or as a background
    image to a renderer */
  BARNEY_API
  BNTextureData bnTextureData2DCreate(BNContext _context,
                                      int slot,
                                      BNDataType texelFormat,
                                      int width, int height,
                                      const void *texels)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,
                                   texelFormat,
                                   vec3i(width,height,0),
                                   texels);
    return (BNTextureData)context->initReference(td);
  }
  
  BARNEY_API
  BNTextureData bnTextureData3DCreate(BNContext _context,
                                      int slot,
                                      BNDataType texelFormat,
                                      int width, int height, int depth,
                                      const void *texels)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,
                                   texelFormat,
                                   vec3i(width,height,depth),
                                   texels);
    return (BNTextureData)context->initReference(td);
  }

  
  // ------------------------------------------------------------------
  BARNEY_API
  BNTexture2D bnTexture2DCreate(BNContext _context,
                                int slot,
                                BNDataType texelFormat,
                                /*! number of texels in x dimension */
                                uint32_t size_x,
                                /*! number of texels in y dimension */
                                uint32_t size_y,
                                const void *texels,
                                BNTextureFilterMode  filterMode,
                                BNTextureAddressMode addressMode_x,
                                BNTextureAddressMode addressMode_y,
                                BNTextureColorSpace  colorSpace)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,texelFormat,
                                   vec3i(size_x,size_y,0),texels);
    BNTextureAddressMode addressModes[3] = {
      addressMode_x,addressMode_y,(BNTextureAddressMode)0
    };
    std::shared_ptr<Texture> tex
      = context->createTexture(td,
                               filterMode,
                               addressModes,
                               colorSpace);
    return (BNTexture2D)context->initReference(tex);
  }

  BARNEY_API
  BNTexture3D bnTexture3DCreate(BNContext _context,
                                int slot,
                                BNDataType texelFormat,
                                uint32_t size_x,
                                uint32_t size_y,
                                uint32_t size_z,
                                const void *texels,
                                BNTextureFilterMode  filterMode,
                                BNTextureAddressMode addressMode)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<TextureData> td
      = context->createTextureData(slot,texelFormat,
                                   vec3i(size_x,size_y,size_z),texels);
    BNTextureAddressMode addressModes[3] = {
      addressMode,addressMode,addressMode
    };
    std::shared_ptr<Texture> tex
      = context->createTexture(td,
                               filterMode,
                               addressModes,
                               BN_COLOR_SPACE_LINEAR);
    return (BNTexture3D)context->initReference(tex);
  }
  
  // ------------------------------------------------------------------
  
  BARNEY_API
  BNModel bnModelCreate(BNContext _context)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    assert(context);
    std::shared_ptr<GlobalModel> model = context->createModel();
    return (BNModel)context->initReference(model);
  }

  BARNEY_API
  BNRenderer bnRendererCreate(BNContext _context,
                              const char *ignoreForNow)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Renderer> renderer = context->createRenderer();
    return (BNRenderer)context->initReference(renderer);
  }

  /*! allows for setting one of 5 attribute arrays for the given slot's
    model. */
  BARNEY_API
  void bnSetInstanceAttributes(BNModel model,
                               int slot,
                               const char *whichAttribute,
                               BNData value)
  {
    LOG_API_ENTRY;
    Data::SP data
      = value
      ? ((Data *)value)->shared_from_this()->as<Data>()
      : Data::SP{};
    checkGet(model)->setInstanceAttributes(slot,whichAttribute,data);
  }

  
  BARNEY_API
  void bnSetInstances(BNModel model,
                      int slot,
                      BNGroup *_groups,
                      BNTransform *xfms,
                      int numInstances)
  {
    LOG_API_ENTRY;
    checkGet(model)->setInstances(slot,
                                  (Group **)_groups,
                                  (const affine3f *)xfms,
                                  numInstances);
  }

  BARNEY_API
  void bnUpdateInstanceTransforms(BNModel model,
                                  int slot,
                                  BNTransform *xfms,
                                  int numInstances)
  {
    LOG_API_ENTRY;
    checkGet(model)->updateInstanceTransforms(slot,
                                              (const affine3f *)xfms,
                                              numInstances);
  }

  BARNEY_API
  void bnSetInstanceMotionDeltas(BNModel model,
                                 int slot,
                                 BNTransform *deltas,
                                 int numInstances)
  {
    LOG_API_ENTRY;
    checkGet(model)->setInstanceMotionDeltas(slot,
                                             (const affine3f *)deltas,
                                             numInstances);
  }

  BARNEY_API
  void  bnRelease(BNObject _object)
  {
    LOG_API_ENTRY;
    Object *object = checkGet(_object);
    assert(object);
    Context *context = object->getContext();
    assert(context);
    context->releaseHostReference(object);
  }
  
  BARNEY_API
  void bnContextDestroy(BNContext context)
  {
    LOG_API_ENTRY;
    delete (Context *)context;
  }

  BARNEY_API
  BNScalarField bnScalarFieldCreate(BNContext _context,
                                    int slot,
                                    const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<ScalarField> sf
      = context->createScalarField(slot,type);
    return (BNScalarField)context->initReference(sf);
  }
  
  
  BARNEY_API
  BNGeom bnGeometryCreate(BNContext _context,
                          int slot,
                          const char *type)
  {
    Context *context = checkGet(_context);
    std::shared_ptr<Geometry> geom
      = context->createGeometry(slot,type);
    return (BNGeom)context->initReference(geom);
  }

  BARNEY_API
  BNMaterial bnMaterialCreate(BNContext _context,
                              int slot,
                              const char *type)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<HostMaterial> material
      = context->createMaterial(slot,type);
    return (BNMaterial)context->initReference(material);
  }

  BARNEY_API
  BNSampler bnSamplerCreate(BNContext _context,
                            int slot,
                            const char *type)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Sampler> sampler
      = context->createSampler(slot,type);
    if (!sampler) return 0;
    return (BNSampler)context->initReference(sampler);
  }
  
  BARNEY_API
  BNCamera bnCameraCreate(BNContext _context,
                          const char *type)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Camera> camera
      = context->createCamera(type);
    if (!camera) return 0;
    return (BNCamera)context->initReference(camera);
  }

  BARNEY_API
  void bnVolumeSetXF(BNVolume volume,
                     bn_float2 domain,
                     const bn_float4 *_values,
                     int numValues,
                     float densityAt1)
  {
    LOG_API_ENTRY;
    assert(_values);

    checkGet(volume)->setXF(range1f(domain.x,domain.y),
                            _values,numValues,
                            densityAt1);
  }

  BARNEY_API
  void bnVolumeSetEmissionXF(BNVolume volume,
                             bn_float2 domain,
                             const bn_float4 *_values,
                             int numValues)
  {
    LOG_API_ENTRY;
    assert(_values);

    checkGet(volume)->setEmissionXF(range1f(domain.x,domain.y),
                                    _values,numValues);
  }

  BARNEY_API
  void bnVolumeClearEmissionXF(BNVolume volume)
  {
    LOG_API_ENTRY;
    checkGet(volume)->clearEmissionXF();
  }
  
  BARNEY_API
  BNVolume bnVolumeCreate(BNContext _context,
                          int slot,
                          BNScalarField _sf)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    
    if (!_sf)
      return {};
    std::shared_ptr<ScalarField> sf = checkGetSP(_sf);
    std::shared_ptr<Volume> volume
      = context->createVolume(checkGetSP(_sf));
    return (BNVolume)context->initReference(volume);
  }

  BARNEY_API
  BNLight bnLightCreate(BNContext _context,
                        int slot,
                        const char *type)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Light> light
      = context->createLight(slot,type);
    return (BNLight)context->initReference(light);
  }

  BARNEY_API
  BNData bnDataCreate(BNContext _context,
                      int slot,
                      BNDataType dataType,
                      size_t numItems,
                      const void *items)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<Data> data
      = context->createData(slot,dataType);
    data->set(items,numItems);
    return (BNData)context->initReference(data);
  }

  BARNEY_API
  void bnDataSet(BNData _data,
                 size_t numItems,
                 const void *items)
  {
    Data::SP data = checkGetSP(_data);
    data->set(items,(int)numItems);
  }

  


  BARNEY_API
  BNGroup bnGroupCreate(BNContext _context,
                        int slot,
                        BNGeom *geoms, int numGeoms,
                        BNVolume *volumes, int numVolumes)
  {
    LOG_API_ENTRY;
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    Context *context = checkGet(_context);
    std::shared_ptr<Group>
      group = context->createGroup(slot,
                                   (Geometry **)geoms,numGeoms,
                                   (Volume **)volumes,numVolumes);
    return (BNGroup)context->initReference(group);
    BARNEY_LEAVE(__PRETTY_FUNCTION__,0);
  }

  BARNEY_API
  void  bnGroupBuild(BNGroup group)
  {
    LOG_API_ENTRY;
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    if (!group) {
#ifndef NDEBUG
      std::cerr << "@barney(WARNING): bnGroupBuild with null group - ignoring this, but this is is an app error that should be fixed, and is only likely to cause issues later on" << std::endl;
#endif
      return;
    }
    checkGet(group)->build();
    BARNEY_LEAVE(__PRETTY_FUNCTION__,);
  }
  
  BARNEY_API
  void  bnBuild(BNModel model,
                int slot)
  {
    BARNEY_ENTER(__PRETTY_FUNCTION__);
    LOG_API_ENTRY;
    checkGet(model)->build(slot);
    BARNEY_LEAVE(__PRETTY_FUNCTION__,);
  }
  
  BARNEY_API
  void bnCommit(BNObject target)
  {
    LOG_API_ENTRY;
    checkGet(target)->commit();
  }
              
  BARNEY_API
  void bnSetString(BNObject target, const char *param, const char *value)
  {
    if (!checkGet(target)->setString(checkGet(param),value))
      checkGet(target)->warn_unsupported_member(param,"std::string");
  }

  BARNEY_API
  void bnSetData(BNObject target, const char *param, BNData value)
  {
    Data::SP data
      = value
      ? checkGetSP(value)
      : Data::SP{};
    if (!checkGet(target)->setData(checkGet(param),data))
      checkGet(target)->warn_unsupported_member(param,"BNData");
  }

  BARNEY_API
  void bnSetObject(BNObject target, const char *param, BNObject value)
  {
    Object::SP asObject
      = value 
      ? checkGet(value)->shared_from_this()
      : Object::SP{};
    bool accepted = checkGet(target)->setObject(checkGet(param),asObject);
    if (!accepted)
      checkGet(target)->warn_unsupported_member(param,"BNObject");
  }

  BARNEY_API
  void bnSet1i(BNObject target, const char *param, int x)
  {
    if (!checkGet(target)->set1i(checkGet(param),x))
      checkGet(target)->warn_unsupported_member(param,"int");
  }

  BARNEY_API
  void bnSet2i(BNObject target, const char *param, int x, int y)
  {
    if (!checkGet(target)->set2i(checkGet(param),vec2i(x,y)))
      checkGet(target)->warn_unsupported_member(param,"vec2i");
  }

  BARNEY_API
  void bnSet3i(BNObject target, const char *param, int x, int y, int z)
  {
    if (!checkGet(target)->set3i(checkGet(param),vec3i(x,y,z)))
      checkGet(target)->warn_unsupported_member(param,"vec3i");
  }

# ifdef __VECTOR_TYPES__
  BARNEY_API
  void bnSet3ic(BNObject target, const char *param, int3 value)
  {
    if (!checkGet(target)->set3i(checkGet(param),(const vec3i&)value))
      checkGet(target)->warn_unsupported_member(param,"vec3i");
  }
#endif
  
  BARNEY_API
  void bnSet4i(BNObject target, const char *param, int x, int y, int z, int w)
  {
    if (!checkGet(target)->set4i(checkGet(param),vec4i(x,y,z,w)))
      checkGet(target)->warn_unsupported_member(param,"vec4i");
  }

  BARNEY_API
  void bnSet1f(BNObject target, const char *param, float value)
  {
    if (!checkGet(target)->set1f(checkGet(param),value))
      checkGet(target)->warn_unsupported_member(param,"float");
  }

  BARNEY_API
  void bnSet2f(BNObject target, const char *param, float x, float y)
  {
    if (!checkGet(target)->set2f(checkGet(param),vec2f(x,y)))
      checkGet(target)->warn_unsupported_member(param,"vec2f");
  }

  BARNEY_API
  void bnSet3f(BNObject target, const char *param, float x, float y, float z)
  {
    if (!checkGet(target)->set3f(checkGet(param),vec3f(x,y,z)))
      checkGet(target)->warn_unsupported_member(param,"vec3f");
  }

  BARNEY_API
  void bnSet4f(BNObject target, const char *param, float x, float y, float z, float w)
  {
    if (!checkGet(target)->set4f(checkGet(param),vec4f(x,y,z,w)))
      checkGet(target)->warn_unsupported_member(param,"vec4f");
  }

  BARNEY_API
  void bnSet4x3fv(BNObject target, const char *param, const BNTransform *transform)
  {
    assert(transform);
    if (!checkGet(target)->set4x3f(checkGet(param),*(const affine3f*)transform))
      checkGet(target)->warn_unsupported_member(param,"affine3f");
  }

  BARNEY_API
  void bnSet4x4fv(BNObject target, const char *param, const bn_float4 *transform)
  {
    assert(transform);
    if (!checkGet(target)->set4x4f(checkGet(param),(const vec4f*)transform))
      checkGet(target)->warn_unsupported_member(param,"mat4f");
  }
  



  

  
  BARNEY_API
  BNFrameBuffer bnFrameBufferCreate(BNContext _context, int deprecated)
  {
    LOG_API_ENTRY;
    Context *context = checkGet(_context);
    std::shared_ptr<FrameBuffer> fb
      = context->createFrameBuffer();
    return (BNFrameBuffer)context->initReference(fb);
  }

  BARNEY_API
  void bnFrameBufferResize(BNFrameBuffer fb,
                           BNDataType colorFormat,
                           int sizeX, int sizeY,
                           uint32_t channels)
  {
    LOG_API_ENTRY;
    checkGet(fb)->resize(colorFormat,vec2i{sizeX,sizeY},channels,/*isaltframe*/0);
  }

  BARNEY_API
  void bnFrameBufferRead(BNFrameBuffer fb,
                         BNFrameBufferChannel channel,
                         void *hostPtr,
                         BNDataType requestedFormat)
  {
    LOG_API_ENTRY;
    checkGet(fb)->read(channel,hostPtr,requestedFormat);
  }

  BARNEY_API
  void bnFrameBufferGetSize(BNFrameBuffer fb, int *sizeX, int *sizeY)
  {
    if (!sizeX || !sizeY) return;
    vec2i np = checkGet(fb)->getNumPixels();
    *sizeX = np.x;
    *sizeY = np.y;
  }

  BARNEY_API
  int bnPhysicalDeviceCount()
  {
    return rtc::physicalDeviceCount();
  }
  
  BARNEY_API
  void bnAccumReset(BNFrameBuffer fb)
  {
    checkGet(fb)->resetAccumulation();
  }
  
  BARNEY_API
  void bnRender(BNRenderer renderer,
                BNModel    model,
                BNCamera   camera,
                BNFrameBuffer fb)
  {
    static int numCalls = 0;
    if (++numCalls < 10)
      LOG_API_ENTRY;
    checkGet(model)->render(checkGet(renderer),checkGet(camera),checkGet(fb));
  }

} 
