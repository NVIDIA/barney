#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
queries="${1:-$root/barney/anari/pre_generated_queries}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

cat > "$tmp/check.cpp" <<'CPP'
#include "generated_queries.h"
#include <cassert>
#include <cstdint>
#include <cstring>

static bool contains(const char *const *strings, const char *value)
{
  if (strings)
    for (; *strings; ++strings)
      if (std::strcmp(*strings, value) == 0)
        return true;
  return false;
}

int main()
{
  using namespace QUERY_NAMESPACE;
  assert(contains(query_extensions(), "ANARI_KHR_SPATIAL_FIELD_NANOVDB"));
  assert(contains(query_object_types(ANARI_SPATIAL_FIELD), "nanovdb"));
  const auto *params = static_cast<const ANARIParameter *>(query_object_info(
      ANARI_SPATIAL_FIELD, "nanovdb", "parameter", ANARI_PARAMETER_LIST));
  assert(params);
  bool data = false, filter = false;
  for (; params->name; ++params) {
    data |= std::strcmp(params->name, "data") == 0 && params->type == ANARI_ARRAY1D;
    filter |= std::strcmp(params->name, "filter") == 0 && params->type == ANARI_STRING;
  }
  assert(data && filter);
  const auto *required = static_cast<const int32_t *>(query_param_info(
      ANARI_SPATIAL_FIELD, "nanovdb", "data", ANARI_ARRAY1D, "required", ANARI_BOOL));
  assert(required && *required);
  const auto *types = static_cast<const ANARIDataType *>(query_param_info(
      ANARI_SPATIAL_FIELD, "nanovdb", "data", ANARI_ARRAY1D, "elementType", ANARI_DATA_TYPE_LIST));
  assert(types && types[0] == ANARI_UINT8 && types[1] == ANARI_UNKNOWN);
  const auto *def = static_cast<const char *>(query_param_info(
      ANARI_SPATIAL_FIELD, "nanovdb", "filter", ANARI_STRING, "default", ANARI_STRING));
  assert(def && std::strcmp(def, "linear") == 0);
  const auto *values = static_cast<const char *const *>(query_param_info(
      ANARI_SPATIAL_FIELD, "nanovdb", "filter", ANARI_STRING, "value", ANARI_STRING_LIST));
  assert(contains(values, "nearest") && contains(values, "linear"));
  // The SDK schema lists cubic; reconstruction fallback is tested separately.
  const auto *source = static_cast<const char *>(query_param_info(
      ANARI_SPATIAL_FIELD, "nanovdb", "data", ANARI_ARRAY1D, "sourceExtension", ANARI_STRING));
  assert(source && std::strcmp(source, "KHR_SPATIAL_FIELD_NANOVDB") == 0);
}
CPP

for backend in cpu cuda optix hip hiprt; do
  "${CXX:-c++}" -std=c++17 -DQUERY_NAMESPACE="barney_$backend" \
    -I"$queries/$backend" "$tmp/check.cpp" \
    "$queries/$backend/generated_queries.cpp" -o "$tmp/check"
  "$tmp/check"
  echo "PASS: $backend NanoVDB extension, subtype, parameters, and provenance"
done
