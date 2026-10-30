#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate or check all backend query snapshots with an installed ANARI SDK."""

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("code_gen", type=Path, help="ANARI SDK share/anari/code_gen directory")
    parser.add_argument("--check", action="store_true", help="compare without changing snapshots")
    args = parser.parse_args()
    code_gen = args.code_gen.resolve()
    generator = code_gen / "generate_queries.py"
    if not generator.is_file():
        parser.error(f"missing generator: {generator}")

    anari = Path(__file__).resolve().parents[1] / "barney" / "anari"
    stale = []
    for device in sorted((anari / "json").glob("*/barney_device.json")):
        backend = device.parent.name
        destination = anari / "pre_generated_queries" / backend
        with tempfile.TemporaryDirectory(prefix=f"barney-queries-{backend}-") as tmp:
            subprocess.run([
                sys.executable, str(generator), "--prefix", "generated",
                "--json", str(code_gen), "--json", str(anari / "json" / "common"),
                "--json", str(device.parent), "--device", str(device),
                "--namespace", f"barney_{backend}", "--output", tmp,
            ], check=True)
            for name in ("generated_queries.cpp", "generated_queries.h"):
                content = (Path(tmp) / name).read_bytes()
                target = destination / name
                if target.is_file() and target.read_bytes() == content:
                    continue
                if args.check:
                    stale.append(str(target.relative_to(anari)))
                else:
                    destination.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(content)
                    print(f"Updated {target.relative_to(anari)}")
    if stale:
        print("Out-of-date queries:\n" + "\n".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
