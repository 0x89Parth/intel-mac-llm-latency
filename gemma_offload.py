#!/usr/bin/env python3
"""Compare two and three Radeon layers for the same small Gemma request."""

import argparse
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

from compare_inference import DEFAULT_ENGINE, LOCAL, PROJECT, cpu_speed_limit, free_port, request

DEFAULT_MODEL = LOCAL / "models/Huihui-gemma-4-26B-A4B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf"
PROMPT = "Copy this sentence exactly: The Moon orbits Earth."
EXPECTED = "The Moon orbits Earth."
CONTEXT = 256
BATCH = 16
THREADS = 3
MAX_TOKENS = 10
ORDER = (2, 3, 3, 2)


def run(engine, model, layers, timeout):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command = [
        str(engine), "-m", str(model), "--alias", "gemma-benchmark",
        "-dev", "MTL0", "-ngl", str(layers), "--fit", "off", "--load-mode", "mmap",
        "-c", str(CONTEXT), "-b", str(BATCH), "-ub", str(BATCH),
        "-t", str(THREADS), "-tb", str(THREADS), "-np", "1", "--cache-ram", "0",
        "--no-warmup", "--no-mmproj", "--reasoning", "off", "--offline", "--no-ui",
        "--cors-origins", "localhost", "--host", "127.0.0.1", "--port", str(port), "--verbose",
    ]
    env = {key: value for key, value in os.environ.items() if not key.startswith(("TOSH_", "LLAMA_ARG_"))}
    env["TOSH_FA_AMD"] = "1"
    row = {"requested_layers": layers, "cpu_speed_limit_before_pct": cpu_speed_limit()}
    with tempfile.TemporaryFile() as log:
        start = time.monotonic()
        process = subprocess.Popen(
            command, cwd=engine.parent, env=env, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            while time.monotonic() - start < timeout:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with status {process.returncode}")
                try:
                    if request(opener, url + "/health").get("status") == "ok":
                        break
                except urllib.error.URLError:
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("server startup timed out")
            ready = time.monotonic()
            row["load_wall_s"] = round(ready - start, 3)
            response = request(
                opener, url + "/v1/chat/completions",
                {
                    "model": "gemma-benchmark", "max_tokens": MAX_TOKENS,
                    "temperature": 0, "seed": 42, "stream": False,
                    "messages": [{"role": "user", "content": PROMPT}],
                },
                timeout=timeout,
            )
            finished = time.monotonic()
            answer = response["choices"][0]["message"]["content"] or ""
            row.update(
                answer=answer, matches_expected=answer.strip() == EXPECTED,
                usage=response.get("usage"), timings=response.get("timings"),
                request_wall_s=round(finished - ready, 3),
                end_to_end_wall_s=round(finished - start, 3),
            )
            if not row["matches_expected"]:
                row["error"] = "answer did not match expected text"
            elif not row["timings"]:
                row["error"] = "server did not expose inference timings"
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            row["error"] = str(error)
        finally:
            row["cpu_speed_limit_after_pct"] = cpu_speed_limit()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.seek(0)
            lines = log.read().decode(errors="replace").splitlines()

    row["offloaded_layers"] = None
    row["metal_buffers_mib"] = {}
    for line in lines:
        match = re.search(r"offloaded (\d+)/(\d+) layers to GPU", line)
        if match:
            row["offloaded_layers"] = int(match[1])
            row["total_layers_including_output"] = int(match[2])
        match = re.search(r"MTL0_Private (model|KV|compute) buffer size =\s*([\d.]+) MiB", line)
        if match:
            row["metal_buffers_mib"][match[1].lower()] = float(match[2])
    if row["offloaded_layers"] != layers or not row["metal_buffers_mib"].get("model"):
        row.setdefault("error", "requested layer count or Metal weights could not be verified")
    if "error" in row:
        row["log_tail"] = [line.replace(str(Path.home()), "~") for line in lines[-14:]]
    return row


def summarize(trials):
    result = {}
    for layers in sorted(set(ORDER)):
        rows = [row for row in trials if row["requested_layers"] == layers and "error" not in row]
        if rows:
            result[str(layers)] = {
                "successful_trials": len(rows),
                "request_wall_s": median(row["request_wall_s"] for row in rows),
                "load_wall_s": median(row["load_wall_s"] for row in rows),
                "prompt_tokens_per_s": median(row["timings"]["prompt_per_second"] for row in rows),
                "generation_tokens_per_s": median(row["timings"]["predicted_per_second"] for row in rows),
            }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=90)
    parser.add_argument("--cooldown", type=float, default=12, help="seconds between trials")
    parser.add_argument("--note", help="measurement conditions or background load")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/gemma-offload.json")
    args = parser.parse_args()
    if args.timeout <= 0 or args.cooldown < 0:
        parser.error("--timeout must be positive and --cooldown cannot be negative")
    engine, model = args.engine.resolve(), args.model.resolve()
    for path in (engine, model):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")

    report = {
        "engine": "ToshLLM llama-server",
        "model": model.name,
        "model_bytes": model.stat().st_size,
        "prompt": PROMPT,
        "expected": EXPECTED,
        "order": ORDER,
        "note": args.note,
        "settings": {
            "context": CONTEXT, "batch": BATCH, "ubatch": BATCH, "threads": THREADS,
            "max_output_tokens": MAX_TOKENS, "temperature": 0, "seed": 42,
            "warmup": False, "mmap": True, "tosh_fa_amd": 1,
            "cooldown_s": args.cooldown,
        },
        "trials": [],
    }
    for index, layers in enumerate(ORDER):
        if index:
            time.sleep(args.cooldown)
        print(f"Trial {index + 1}: {layers} GPU layers", flush=True)
        row = run(engine, model, layers, args.timeout)
        report["trials"].append(row)
        report["summary_medians"] = summarize(report["trials"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(row.get("error") or row["timings"], flush=True)
    return int(any("error" in row for row in report["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())
