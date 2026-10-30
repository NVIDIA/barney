#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize Nsight Systems 2026.3 SQLite exports of the sparse ANARI fixture.

Allocation sizes/counts here validate attribution of a measurement, not a
renderer correctness contract. Update attribution if storage ownership changes.
"""

import argparse
import pathlib
import sqlite3
import statistics


def summarize(path, extent):
    if extent < 16:
        raise ValueError("extent must match the benchmark fixture")
    cells = ((extent + 15) // 16) ** 3
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        durations = [
            (end - start) / 1e6
            for start, end in db.execute(
                "SELECT k.start, k.end FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id "
                "WHERE s.value LIKE '%NanoVDBData_computeMCs%' ORDER BY k.start"
            )
        ]
        if len(durations) != 6:
            raise ValueError(f"expected six field builds, found {len(durations)}")
        warm = durations[1:]
        print("kernel_ms_min_med_max=" + ",".join(
            f"{value:.6f}" for value in (min(warm), statistics.median(warm), max(warm))
        ))
        # Resolve enums instead of assuming CUPTI's numeric values are stable.
        device_kind = db.execute(
            "SELECT id FROM ENUM_CUDA_MEM_KIND WHERE label='Device'"
        ).fetchone()[0]
        allocate = db.execute(
            "SELECT id FROM ENUM_CUDA_DEV_MEM_EVENT_OPER WHERE label='Allocation'"
        ).fetchone()[0]
        free = db.execute(
            "SELECT id FROM ENUM_CUDA_DEV_MEM_EVENT_OPER WHERE label='Deallocation'"
        ).fetchone()[0]
        for name, size in (("range", 8 * cells), ("one_volume_majorant", 4 * cells)):
            counts = dict(db.execute(
                "SELECT memoryOperationType, count(*) FROM CUDA_GPU_MEMORY_USAGE_EVENTS "
                "WHERE memKind=? AND bytes=? GROUP BY memoryOperationType",
                (device_kind, size),
            ))
            if counts != {allocate: 6, free: 6}:
                raise ValueError(f"cannot attribute {name} bytes={size}: events={counts}")
            print(f"{name}_bytes={size} allocations=6 frees=6")
        live = peak = 0
        for size, operation in db.execute(
            "SELECT bytes, memoryOperationType FROM CUDA_GPU_MEMORY_USAGE_EVENTS "
            "WHERE memKind=? ORDER BY start", (device_kind,)
        ):
            if operation not in (allocate, free):
                raise ValueError(f"unknown device memory operation {operation}")
            live += size if operation == allocate else -size
            if live < 0:
                raise ValueError("incomplete device allocation history")
            peak = max(peak, live)
        if live:
            raise ValueError(f"nonzero final traced device balance: {live}")
        print(f"whole_client_traced_device_peak_bytes={peak} final_balance_bytes={live}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=pathlib.Path)
    parser.add_argument("extent", type=int)
    args = parser.parse_args()
    summarize(args.sqlite, args.extent)
