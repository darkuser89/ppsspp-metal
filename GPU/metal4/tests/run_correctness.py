#!/usr/bin/env python3
# Copyright (c) 2012- PPSSPP Project.
# This program is free software, licensed under GPL-2.0-or-later.
"""Build/run the standalone Metal tests using an existing macOS CMake build.

Run after building PPSSPPUnitTest. All generated files go in the supplied scratch
directory, outside code/audit. No app, Git command, or configuration write is run.
"""
import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("build", type=Path)
parser.add_argument("scratch", type=Path)
args = parser.parse_args()
build = args.build.resolve()
scratch = args.scratch.resolve()
root = Path(__file__).resolve().parents[3]
scratch.mkdir(parents=True, exist_ok=True)
entries = json.loads((build / "compile_commands.json").read_text())


def compile_source(suffix, source, output, extra=()):
    entry = next(e for e in entries if e["file"].endswith(suffix))
    command = shlex.split(entry["command"])
    command = [str(source) if a == entry["file"] else a for a in command]
    command[command.index("-o") + 1] = str(output)
    command[1:1] = extra
    subprocess.run(command, cwd=entry["directory"], check=True)


compile_source("/GPU/metal4/TextureCacheMetal.mm", root / "GPU/metal4/tests/CorrectnessMetal.mm", scratch / "test.o", ["-mmacosx-version-min=26.0"])
compile_source("/unittest/UnitTest.cpp", root / "unittest/UnitTest.cpp", scratch / "stubs.o", ["-Dmain=UnusedUnitTestMain"])
link = shlex.split((build / "CMakeFiles/PPSSPPUnitTest.dir/link.txt").read_text())
link = [str(scratch / "stubs.o") if a.endswith("/unittest/UnitTest.cpp.o") else a for a in link]
link.insert(1, str(scratch / "test.o"))
link[link.index("-o") + 1] = str(scratch / "MetalCorrectness")
subprocess.run(link, cwd=build, check=True)
sys.exit(subprocess.run([str(scratch / "MetalCorrectness"), str(root)], cwd=scratch).returncode)
