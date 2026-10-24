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
  flags+=(-fsanitize="$SANITIZERS" -fno-omit-frame-pointer -g
          -DNANOVDB_TEST_SANITIZERS)
fi

# GCC 16 diagnoses an unused GridBuilder iterator template in bundled NanoVDB.
# Do not patch the dependency to generate the fixtures used by this test.
g++ -std=c++17 -Wall -Wextra -Werror -Wno-template-body "${flags[@]}" \
  -isystem "$root/3rdParty/nanovdb/include" \
  "$root/tests/nanovdb_ingestion.cpp" -lanari -pthread -o "$tmp/test"
LD_LIBRARY_PATH="$build${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$tmp/test"
