// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define ANARI_BARNEY_MATH_DEFINITIONS 1

#include "anari/BarneyGlobalState.h"
#include "anari/Frame.h"
#include "anari/common.h"

// for dlsym
#ifdef _WIN32
# include <windows.h>
#else
# include <dlfcn.h>
#endif

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)

namespace BARNEY_NS {
  namespace anari {
    
    Tether::~Tether()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: tether destructing - "
                         "releasing barney context" << std::endl);
      if (context) { bnContextDestroy(context); context = 0; }
    }

    BarneyGlobalState::BarneyGlobalState(ANARIDevice d)
      : helium::BaseGlobalDeviceState(d)
    {
      PluginInfrastructure::get()->initPlugins();
    }

    BarneyGlobalState::~BarneyGlobalState()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: barneyglobalstate destructing"
                         " - releasing tether" << std::endl);
    }
  
    void BarneyGlobalState::markSceneChanged()
    {
      objectUpdates.lastSceneChange = helium::newTimeStamp();
    }

    void BarneyGlobalState::markStructuralSceneChanged()
    {
      auto ts = helium::newTimeStamp();
      objectUpdates.lastSceneChange = ts;
      objectUpdates.lastStructuralChange = ts;
    }

    bool Tether::allDevicesPresent()
    {
      for (auto dev : devices)
        if (dev == 0) return false;
      return true;
    }

    TetheredModel::SP Tether::getOrCreateTetheredModel(int uniqueID)
    {
      std::lock_guard<std::mutex> lock(mutex);
      if (activeModels.find(uniqueID) != activeModels.end()) {
        BANARI_TRACK_LEAKS(std::cout << "#banari returning already created model "
                           << uniqueID << std::endl);
        return activeModels[uniqueID]->shared_from_this();
      }

      BANARI_TRACK_LEAKS(std::cout << "#banari creating new tethered model "
                         << uniqueID << std::endl);
      TetheredModel::SP newModel = std::make_shared<TetheredModel>(this,uniqueID);
      return newModel;
    }

    TetheredModel::TetheredModel(Tether *tether, int uniqueID)
      : tether(tether),
        uniqueID(uniqueID)
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: creating new tetherd model ID "
                         << uniqueID << std::endl);
      model = bnModelCreate(tether->context);
      BANARI_TRACK_LEAKS(std::cout << "#banari: created new barney model "
                         << (int*)model << std::endl);
    
      // iw do NOT try to lock tether, it's already locked when it creates us!
      tether->activeModels[uniqueID] = this;
    }
  
    TetheredModel::~TetheredModel()
    {
      BANARI_TRACK_LEAKS(std::cout << "#banari: tethered model is dying" << std::endl);
      std::lock_guard<std::mutex> lock(tether->mutex);
      tether->activeModels.erase(tether->activeModels.find(uniqueID));
    
      if (model) {
        BANARI_TRACK_LEAKS(std::cout << "#banari: releasing barney model handle ID "
                           << uniqueID << std::endl);
        bnRelease(model);
      }
    }

    // ==================================================================
    // plugin infrastructure - should at some point more to a separate
    // compilation unit
    // ==================================================================

    PluginInfrastructure *PluginInfrastructure::get()
    {
      static PluginInfrastructure *singleton = nullptr;
      if (!singleton) singleton = new PluginInfrastructure;
      return singleton;
    }
    
    /*! expects a string of the form '<geomtype>@<pluginname>'. may
      return null if either the plugin cannot be found (ie, it's not
      included in the build), or that plugin for some reason cannot
      create that geometry. */
    Geometry *PluginInfrastructure::newGeometry(const std::string_view &name,
                                                BarneyGlobalState *banari)
    {
      auto creatorFromPlugin = supportedGeometries[std::string(name)];
      if (!creatorFromPlugin) return nullptr;
      
      return creatorFromPlugin(banari);
    }

    /*! expects a string of the form '<geomtype>@<pluginname>'. may
      return null if either the plugin cannot be found (ie, it's not
      included in the build), or that plugin for some reason cannot
      create that geometry. */
    SpatialField *PluginInfrastructure::newSpatialField(const std::string_view &name,
                                                BarneyGlobalState *banari)
    {
      auto creatorFromPlugin = supportedSpatialFields[std::string(name)];
      if (!creatorFromPlugin) return nullptr;

      return creatorFromPlugin(banari);
    }

    // called by plugin registry function to declare a new geometry type */
    void PluginInfrastructure
    ::exportGeometry(const std::string &typeName,
                     Geometry*(*creatorFunction)(BarneyGlobalState*))
    {
      supportedGeometries[typeName] = creatorFunction;
    }
    
    // called by plugin registry function to declare a new spatial field type */
    void PluginInfrastructure
    ::exportSpatialField(const std::string &typeName,
                         SpatialField*(*creatorFunction)(BarneyGlobalState*))
    {
      supportedSpatialFields[typeName] = creatorFunction;
    }

    /*! we defer initializing the plugins until the first device
      gets created, to make sure that all other stuff is already
      loaded and initialized */
    void PluginInfrastructure::initPlugins()
    {
      // in case there's more than one device/barneyglobalstate we
      // don't want to re-initialze
      static bool alreadyInitialized = false;
      if (alreadyInitialized)
        return;
      alreadyInitialized = true;
      
      for (auto pluginInit : registeredPluginInitFunctions)
        pluginInit.second();
    }
    
    int PluginInfrastructure::registerPlugin(const std::string &name,
                                             void (*initFct)())
    {
      std::cout << OWL_TERMINAL_LIGHT_BLUE;
      std::cout << "#banari: globally registering plugin '"
                << BARNEY_BACKEND_STRING << "::" << name << "'\n";
      std::cout << OWL_TERMINAL_DEFAULT;
      
      PluginInfrastructure *pi = PluginInfrastructure::get();
      pi->registeredPluginInitFunctions[name] = initFct;
      return pi->registeredPluginInitFunctions.size();
    }
    
  }
}
