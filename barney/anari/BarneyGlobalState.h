// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <anari/anari_cpp.hpp>
#include <helium/array/Array1D.h>
#include "anari/barney_math.h"
#include "anari/common.h"
// helium
#include <memory>
#include <map>

#include <helium/array/Array2D.h>
#include <helium/array/Array3D.h>
#include <helium/array/ObjectArray.h>
#include "helium/BaseGlobalDeviceState.h"

#ifndef BARNEY_NS
# error BARNEY_NS not defined
#endif

namespace BARNEY_NS {
  namespace anari {

    struct Frame;
    struct World;

    struct BarneyDevice;
    struct Tether;

    struct Geometry;
    struct SpatialField;
    struct BarneyGlobalState;
    
    struct TetheredModel : public std::enable_shared_from_this<TetheredModel> {
      typedef std::shared_ptr<TetheredModel> SP;
      TetheredModel(Tether *tether, int uniqueID);
      ~TetheredModel();
      BNModel model = 0;
      Tether *const tether;
      int     const uniqueID;
    };

    /*! keeps info on multiple (banari-)devices that are tethered
      together onto a singel barney ncontext */
    struct Tether {
      ~Tether();
    
      BNContext context{nullptr};

      bool allDevicesPresent();

      TetheredModel::SP getOrCreateTetheredModel(int uniqueID);
      std::map<int,TetheredModel*> activeModels;
      std::mutex mutex;

      int numRenderCallsOutstanding = 0;
      struct {
        BNCamera camera;
        BNRenderer renderer;
        BNFrameBuffer fb;
        BNModel model;
      } deferredRenderCall;

      std::vector<BarneyDevice *> devices;
    };


    /*! plugin infrastructure. barney allows for conditionally
        including plugins at build type; these can then load and
        initialize themselves at runtime by having the plugin name
        encoded in the spatial field / geometry / etc type string. Ie,
        if the app calls `anariNewGeometry(myFancyGeom@myFancyPlugin)`
        barney will check if any plugin registerd itself - during
        build time - under the name of 'myFancyPlugin'; and if so,
        will ask this plugin to register whatever geometry, spatial
        field, etc, it will provide; then barney can look up this
        plugin and ask it create the respective object instance */
    struct PluginInfrastructure {

      /*! a singleton (per backend, of course, through namespace) in
          which plugins can register themselves, and device can look
          up registered backends */
      static PluginInfrastructure *get();
      
      // ------------------------------------------------------------------
      // called from REGISTER_PLUGIN()
      // ------------------------------------------------------------------
      static int registerPlugin(const std::string &name,
                                void (*initFct)());
      
      // ------------------------------------------------------------------
      // called from plugins' registerPlugin() functions, where they
      // register thetypes of geometries and fields, etc that they
      // provide
      // ------------------------------------------------------------------
      
      // called by plugin registry function to declare a new geometry type */
      void exportGeometry(const std::string &typeName,
                          Geometry*(*)(BarneyGlobalState*));
      
      // called by plugin registry function to declare a new spatial field type */
      void exportSpatialField(const std::string &typeName,
                              SpatialField*(*)(BarneyGlobalState*));

      // ------------------------------------------------------------------
      // called from device
      // ------------------------------------------------------------------
      /*! expects a string of the form '<geomtype>@<pluginname>'. may
        return null if either the plugin cannot be found (ie, it's not
        included in the build), or that plugin for some reason cannot
        create that geometry. */
      Geometry *newGeometry(const std::string_view &typeAtPlugin,
                            BarneyGlobalState *banari);
      
      /*! expects a string of the form '<geomtype>@<pluginname>' may
        return null if either the plugin cannot be found (ie, it's not
        included in the build), or that plugin for some reason cannot
        create that geometry. */
      SpatialField *newSpatialField(const std::string_view &typeAtPlugin,
                                    BarneyGlobalState *banari);

      /*! we defer initializing the plugins until the first device
          gets created, to make sure that all other stuff is already
          loaded and initialized */
      void initPlugins();
      int size() const { return registeredPluginInitFunctions.size(); }
    private:
      std::map<std::string,void (*)()> registeredPluginInitFunctions;
      
      std::map<std::string,
               Geometry*(*)(BarneyGlobalState*)
               > supportedGeometries;
      std::map<std::string,
               SpatialField*(*)(BarneyGlobalState*)
               > supportedSpatialFields;
    };
    
    struct BarneyGlobalState : public helium::BaseGlobalDeviceState
    {
      struct ObjectUpdates
      {
        helium::TimeStamp lastSceneChange{0};
        helium::TimeStamp lastStructuralChange{0};
      } objectUpdates;

      int slot = -1;

      std::shared_ptr<Tether> tether;

      /*! created models get consecutive IDs, which allows us for
        identifying which models created by which (tethered) device(s)
        belong togther. Ie, if two devices A and B are tethered, then
        the i'th model of A is always tethered with the i'th model of
        B (and vice versa) */
      int nextUniqueModelID = 0;

      bool hasBeenCommitted = false;

      // Helper methods //

      BarneyGlobalState(ANARIDevice d);
      ~BarneyGlobalState();

      void markSceneChanged();
      void markStructuralSceneChanged();
    };

    // Helper functions/macros ////////////////////////////////////////////////////

    inline BarneyGlobalState *asBarneyState(helium::BaseGlobalDeviceState *s)
    {
      return (BarneyGlobalState *)s;
    }
    
  }
}

#define BARNEY_ANARI_TYPEFOR_SPECIALIZATION(type, anari_type)   \
  namespace anari {                                             \
    ANARI_TYPEFOR_SPECIALIZATION(type, anari_type);             \
  }

#define BARNEY_ANARI_TYPEFOR_DEFINITION(type)   \
  namespace anari {                             \
    ANARI_TYPEFOR_DEFINITION(type);             \
  }

