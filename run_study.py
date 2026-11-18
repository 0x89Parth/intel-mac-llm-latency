#!/usr/bin/env python3
"""Run both paired experiments sequentially on the Intel Mac that owns the models."""

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, help="defaults to a new directory under runs/")
    parser.add_argument("--tuning-report", type=Path, default=PROJECT / "results/gemma-cpu-tuning.json")
    parser.add_argument("--resume", action="store_true",
                        help="resume a specific --output-dir using the original binaries, models and protocol")
    args = parser.parse_args()
    if args.resume and args.output_dir is None:
        parser.error("--resume needs the original --output-dir")
    if not args.tuning_report.is_file():
        parser.error("a completed Gemma tuning report is required; see tune_gemma_cpu.py")
    selection = json.loads(args.tuning_report.read_text())
    if "selected_profile" not in selection:
        parser.error("tuning report is incomplete")
    profile = selection["selected_profile"]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = (args.output_dir or PROJECT / "runs" / stamp).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    print(f"Saving the paired results to {destination}", flush=True)
    for model in ("qwen", "gemma"):
        command = [
            sys.executable, "-Bu", str(PROJECT / "compare_correct_answers.py"),
            "--model", model, "--rounds", "5", "--cooldown", "20",
            "--output", str(destination / f"correct-answers-{model}.json"),
        ]
        if args.resume:
            command.append("--resume")
        if model == "gemma":
            command.append("--no-repack")
            if profile and (profile["threads"], profile["gpu_layers"]) != (3, 3):
                command.extend(["--candidate-threads", str(profile["threads"]),
                                "--candidate-layers", str(profile["gpu_layers"])])
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
