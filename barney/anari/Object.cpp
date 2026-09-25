// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
// CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "anari/Object.h"
// std
#include <atomic>
#include <cstdarg>
#include <set>

namespace BARNEY_NS {
  namespace anari {
  
    // Object definitions /////////////////////////////////////////////////////////

    Object::Object(ANARIDataType type, BarneyGlobalState *s)
      : helium::BaseObject(type, s)
    {}

    void Object::commitParameters()
    {
      // no-op
    }

    void Object::finalize()
    {
      // no-op
    }

    bool Object::getProperty(const std::string_view &name,
                             ANARIDataType type,
                             void *ptr,
                             uint64_t size,
                             uint32_t flags)
    {
      if (name == "valid" && type == ANARI_BOOL) {
        helium::writeToVoidP(ptr, isValid());
        return true;
      }

      return false;
    }

    bool Object::isValid() const
    {
      return true;
    }

    BarneyGlobalState *Object::deviceState() const
    {
      return (BarneyGlobalState *)helium::BaseObject::m_state;
    }

    // UnknownObject definitions //////////////////////////////////////////////////

    UnknownObject::UnknownObject(ANARIDataType type,
                                 std::string_view _subType,
                                 BarneyGlobalState *s)
      : Object(type, s)
    {
      const std::string subType(_subType);
      static std::set<std::string> alreadyWarned;
      if (alreadyWarned.find(subType) != alreadyWarned.end())
        return;
      
      reportMessage(ANARI_SEVERITY_WARNING,
                    "banari object type '%s' of subtype '%s' not implemented",
                    anari::toString(type),
                    subType.c_str());
      alreadyWarned.insert(subType);
    }
  
    UnknownObject::~UnknownObject() = default;
  
    bool UnknownObject::isValid() const
    {
      return false;
    }

  }
}

BARNEY_ANARI_TYPEFOR_DEFINITION(BARNEY_NS::anari::Object *);
