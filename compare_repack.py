#!/usr/bin/env python3
"""Measure the startup and response cost of Gemma CPU weight repacking."""

import argparse
import json
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from compare_avx2 import (
    BATCH, CODING_EXPECTED, CODING_MAX_TOKENS, CODING_PROMPT, CONTEXT, ENVIRONMENT,
    LAYERS, SETTINGS, THREADS, chat, coding_answer_is_correct, stream, wait_for_cool_cpu,
)
from compare_inference import LOCAL, PROJECT, free_port, request
from gemma_offload import DEFAULT_MODEL

AVX2_ENGINE = LOCAL / "toshllm-engine-0.88.0-macos13-x86_64-avx2/bin/llama-server"


def memory_state(pid=None):
    swap = subprocess.check_output(["/usr/sbin/sysctl", "-n", "vm.swapusage"], text=True)
    counters = subprocess.check_output(["/usr/bin/vm_stat"], text=True)
    state = {
        "swap_usage": swap.strip(),
        **{
            name.lower(): int(count)
            for name, count in re.findall(r"^(Swapins|Swapouts):\s*(\d+)\.", counters, re.M)
        },
    }
    if pid is not None:
        rss = subprocess.check_output(["/bin/ps", "-p", str(pid), "-o", "rss="], text=True)
        state["server_rss_mib"] = round(int(rss.strip()) / 1024, 2)
    return state


def trial(engine, model, repack, timeout, stream_tokens):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command = [
        str(engine), "-m", str(model), "--alias", "gemma-benchmark", "-dev", "MTL0",
        "-ngl", str(LAYERS), "--fit", "off", "--load-mode", "mmap",
        "-c", str(CONTEXT), "-b", str(BATCH), "-ub", str(BATCH),
        "-t", str(THREADS), "-tb", str(THREADS), "-np", "1", "--cache-ram", "0",
        "--no-warmup", "--no-mmproj", "--reasoning", "off", "--offline", "--no-ui",
        "--cors-origins", "localhost", "--host", "127.0.0.1", "--port", str(port), "--verbose",
    ]
    if not repack:
        command.append("--no-repack")
    environment = ENVIRONMENT.copy()
    environment.pop("GGML_CPU_NO_REPACK", None)
    cooldown, thermal_before = wait_for_cool_cpu(50)
    row = {
        "repack": repack, "cooldown_s": cooldown, "thermal_before": thermal_before,
        "memory_before": memory_state(),
    }
    if thermal_before["thermal_warning"]:
        row["error"] = "macOS reported a thermal warning before startup"
        return row
    with tempfile.TemporaryFile() as log:
        start = time.monotonic()
        process = subprocess.Popen(
            command, cwd=engine.parent, env=environment, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            while time.monotonic() - start < timeout:
                if process.poll() is not None:
                    raise RuntimeError(f"server stopped during load ({process.returncode})")
                try:
                    if request(opener, url + "/health").get("status") == "ok":
                        break
                except urllib.error.URLError:
                    pass
                time.sleep(0.25)
            else:
                raise TimeoutError("server startup timed out")
            row["startup_s"] = round(time.monotonic() - start, 3)
            print(f"repack={repack}: ready in {row['startup_s']:.2f}s", flush=True)
            row["memory_ready"] = memory_state(process.pid)
            coding = chat(opener, url, CODING_PROMPT, CODING_MAX_TOKENS, timeout)
            row["startup_plus_coding_s"] = round(time.monotonic() - start, 3)
            row["coding"] = coding
            row["memory_after_coding"] = memory_state(process.pid)
            coding["correct"] = coding_answer_is_correct(coding["answer"])
            if not coding["correct"] or not coding["timings"]:
                raise RuntimeError("ordinary coding test failed correctness or timings check")
            if coding["thermal_after"]["thermal_warning"]:
                raise RuntimeError("macOS reported a thermal warning during coding")
            if stream_tokens:
                row["stream"] = stream(opener, url, stream_tokens, 0, 0, [], timeout)
                row["memory_after_stream"] = memory_state(process.pid)
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            row["error"] = str(error)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.seek(0)
            lines = log.read().decode(errors="replace").splitlines()
    totals = defaultdict(float)
    row["model_buffer_log"] = []
    for line in lines:
        if match := re.search(r"offloaded (\d+)/(\d+) layers to GPU", line):
            row["offloaded_layers"] = int(match[1])
        if match := re.search(r"(\S+) model buffer size\s*=\s*([\d.]+) MiB", line):
            totals[match[1]] += float(match[2])
            row["model_buffer_log"].append(line)
    row["model_buffer_mib_by_type"] = {name: round(size, 2) for name, size in totals.items()}
    if row.get("offloaded_layers") != LAYERS or not totals.get("MTL0_Private"):
        row.setdefault("error", "requested Metal placement was not verified")
    if "error" in row:
        row["log_tail"] = [line.replace(str(Path.home()), "~") for line in lines[-12:]]
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, default=AVX2_ENGINE)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=PROJECT / "results/gemma-repack.json")
    parser.add_argument("--timeout", type=float, default=110)
    parser.add_argument("--stream-tokens", type=int, choices=(0, 48), default=0)
    parser.add_argument("--repeat", action="store_true", help="include a second pair in reverse order")
    args = parser.parse_args()
    engine, model = args.engine.resolve(), args.model.resolve()
    for path in (engine, model):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")

    order = (False, True, True, False) if args.repeat else (False, True)
    report = {
        "engine": engine.parent.parent.name, "model": model.name, "order": order,
        "settings": SETTINGS, "coding_prompt": CODING_PROMPT, "coding_expected": CODING_EXPECTED,
        "stream_tokens": args.stream_tokens, "trials": [],
    }
    for repack in order:
        print(f"Starting with repack={repack}", flush=True)
        row = trial(engine, model, repack, args.timeout, args.stream_tokens)
        report["trials"].append(row)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        coding = row.get("coding", {})
        if error := row.get("error"):
            print(error, flush=True)
            if "thermal warning" in error:
                break
        else:
            rate = coding["timings"]["predicted_per_second"]
            print(
                f"repack={repack}: {coding['request_wall_s']:.2f}s request, "
                f"{rate:.2f} tok/s decode, {row['startup_plus_coding_s']:.2f}s total",
                flush=True,
            )
    return int(any("error" in row for row in report["trials"]))


if __name__ == "__main__":
    raise SystemExit(main())
