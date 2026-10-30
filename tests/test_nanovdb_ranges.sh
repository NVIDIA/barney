#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$root/_out/_cmake}"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
flags=()
if [[ -n "${SANITIZERS:-}" ]]; then
  flags+=(-fsanitize="$SANITIZERS" -fno-omit-frame-pointer -g)
fi
g++ -std=c++20 -O2 -Wno-template-body "${flags[@]}" \
  -DBARNEY_DEVICE_PROGRAM=1 -DBARNEY_HAVE_NANOVDB=1 -DBARNEY_EXTERNAL_EMBREE=1 \
  -I"$root/barney/rtcore/cpu" -I"$build/build_barney_cpu" \
  -I"$root" -I"$root/barney" -I"$root/3rdParty/glm/include" \
  -I"$root/submodules/owl/include" -I"$root/submodules/cuBQL" \
  -isystem "$root/3rdParty/nanovdb/include" \
  "$root/tests/nanovdb_ranges.cpp" -o "$tmp/test"
"$tmp/test"
