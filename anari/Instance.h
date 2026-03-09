// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#pragma once

#include "Group.h"

namespace barney_device {

  struct Instance : public Object
  {
    struct Attributes {
      enum { count = 5 };
      // for sake of easier processing we store those as jsut five
      // unnamed attributes; at the time fo thise writing attibute[4]
      // is implicitly color (across all anari apps, not just banrey),
      // but that may of course change going forward
      math::float4 values[count];
    };
  
    Instance(BarneyGlobalState *s);
    ~Instance() override;

    void commitParameters() override;
    void finalize() override;
    void markFinalized() override;

    bool isValid() const override;

    const Group *group() const;

    /*! writes the anari 4x4 matrix out into a barney-style 4x3
        matrix, into the memory location indicated by the provided
        pointer */
    void writeTransform(BNTransform *out) const;

    box3 bounds() const;

    /*! attributes for that instance, if specified. if not specifies
        this will be a null pointer */
    Attributes *attributes = 0;
    int m_id = ~0;
  private:
    math::mat4 m_transform;
    helium::IntrusivePtr<Group> m_group;
    const Group *m_previousGroup = nullptr;
  };

} // namespace barney_device

BARNEY_ANARI_TYPEFOR_SPECIALIZATION(barney_device::Instance *, ANARI_INSTANCE);
