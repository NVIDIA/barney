#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
build="${1:-$root/_out/_gpu}"
backend="${NANOVDB_TEST_BACKEND:-optix}"
case "$backend" in
  optix|cuda) ;;
  *) echo "unsupported numerical CUDA backend: $backend" >&2; exit 2 ;;
esac
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
nvcc -std=c++20 -O2 -arch="${CUDA_ARCH:-sm_75}" -Xcompiler=-Wno-template-body \
  --extended-lambda --expt-relaxed-constexpr -DBARNEY_DEVICE_PROGRAM=0 -DBARNEY_HAVE_NANOVDB=1 \
  -I"$root/barney/rtcore/$backend" -I"$build/build_barney_$backend" \
  -I"$root" -I"$root/barney" -I"$root/3rdParty/glm/include" \
  -I"$root/submodules/owl/include" -I"$root/submodules/cuBQL" \
  -I"$root/submodules/optix/include" -I"$root/3rdParty/nanovdb/include" \
  "$root/tests/nanovdb_ranges_gpu.cu" -o "$tmp/test"
if [[ "${COMPUTE_SANITIZER:-0}" == 1 ]]; then
  compute-sanitizer --tool memcheck --error-exitcode 1 "$tmp/test"
else
  "$tmp/test"
fi
