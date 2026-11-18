#!/usr/bin/env python3
"""Compare CPU and Radeon inference with the same ToshLLM engine and model."""

import argparse
import ast
import json
import os
import re
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path
from statistics import median

PROJECT = Path(__file__).resolve().parent
LOCAL = Path.home() / ".local/share/local-llm"
DEFAULT_ENGINE = LOCAL / "toshllm-engine-0.88.0-macos12-x86_64/bin/llama-server"
DEFAULT_MODEL = LOCAL / "models/qwen2.5-coder-0.5b-instruct-q4_k_m.gguf"
PROMPT = (
    "Return plain Python code with no Markdown, comments, or docstring. "
    "Define odd_squares_sum(values) using a for loop: add the square of each odd "
    "integer to a running total, then return the total. End with these two tests, "
    "one per line: assert odd_squares_sum([-3, 4, 5]) == 34; "
    "assert odd_squares_sum([]) == 0."
)
C = 512
N = 128
BATCH = 16
THREADS = 3


def free_port():
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return listener.getsockname()[1]


def request(opener, url, payload=None, timeout=2):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with opener.open(req, timeout=timeout) as response:
        return json.load(response)


def parsed_python(answer):
    fence = re.search(r"```(?:python)?\s*\n(.*?)```", answer, re.S)
    try:
        return ast.parse(fence[1] if fence else answer)
    except SyntaxError:
        return None


def cpu_speed_limit():
    output = subprocess.run(
        ["/usr/bin/pmset", "-g", "therm"], capture_output=True, text=True, check=False,
    ).stdout
    match = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", output)
    return int(match[1]) if match else None


def run(engine, model, mode, gpu_layers, timeout, *, batch=BATCH, workload=None):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    device = "none" if mode == "cpu" else "MTL0"
    layers = "0" if mode == "cpu" else gpu_layers
    command = [
        str(engine), "-m", str(model), "--alias", "benchmark-model", "-dev", device,
        "-ngl", layers, "--fit", "off", "--load-mode", "mmap",
        "-c", str(C), "-b", str(batch), "-ub", str(batch), "-t", str(THREADS),
        "-tb", str(THREADS), "-np", "1", "--cache-ram", "0", "--no-warmup",
        "--no-mmproj", "--reasoning", "off", "--offline", "--no-webui",
        "--cors-origins", "localhost", "--host", "127.0.0.1", "--port", str(port),
        "-lv", "4",
    ]
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("TOSH_", "LLAMA_ARG_"))}
    environment["TOSH_FA_AMD"] = "1"
    starting_speed_limit = cpu_speed_limit()
    with tempfile.TemporaryFile() as log:
        start = time.monotonic()
        process = subprocess.Popen(
            command, cwd=engine.parent, env=environment,
            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
        )
        result = {
            "mode": mode, "device": device, "requested_gpu_layers": layers,
            "batch": batch, "ubatch": batch,
        }
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited with status {process.returncode}")
                if time.monotonic() - start >= timeout:
                    raise TimeoutError("server startup timed out")
                try:
                    healthy = request(opener, url + "/health").get("status") == "ok"
                except (OSError, urllib.error.URLError):
                    healthy = False
                if healthy:
                    break
                time.sleep(0.1)
            ready = time.monotonic()
            result["load_wall_s"] = round(ready - start, 3)
            if workload is not None:
                result.update(workload(opener, url, timeout))
                result["end_to_end_wall_s"] = round(time.monotonic() - start, 3)
            else:
                completion = request(
                    opener, url + "/v1/chat/completions",
                    {
                        "model": "benchmark-model", "temperature": 0, "seed": 42,
                        "max_tokens": N, "stream": False,
                        "messages": [{"role": "user", "content": PROMPT}],
                    }, timeout=timeout,
                )
                finished = time.monotonic()
                choice = completion["choices"][0]
                answer = choice["message"]["content"] or ""
                result.update({
                    "request_wall_s": round(finished - ready, 3),
                    "end_to_end_wall_s": round(finished - start, 3),
                    "answer": answer,
                    "python_syntax_valid": parsed_python(answer) is not None,
                    "finish_reason": choice["finish_reason"],
                    "usage": completion.get("usage"),
                    "timings": completion.get("timings"),
                })
                if not result["timings"]:
                    result["error"] = "server did not expose completion timing"
                if completion.get("usage", {}).get("completion_tokens", 0) < 32:
                    result["error"] = "completion contained fewer than 32 generated tokens"
        except (OSError, urllib.error.URLError, RuntimeError, KeyError, ValueError) as error:
            result["error"] = str(error)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            log.seek(0)
            lines = log.read().decode(errors="replace").splitlines()

    lines = [
        line.replace(str(engine), engine.name).replace(str(model), model.name).replace(str(Path.home()), "~")
        for line in lines
    ]
    result["cpu_speed_limit_before_pct"] = starting_speed_limit
    result["cpu_speed_limit_after_pct"] = cpu_speed_limit()
    result["allocation_log"] = [
        line for line in lines
        if re.search(r"offload(?:ing|ed)|model buffer size|kv buffer size|compute buffer size", line, re.I)
    ]
    if "error" in result:
        result["log_tail"] = lines[-16:]
    return result


def summarize(results):
    summary = {}
    for mode in ("cpu", "gpu"):
        rows = [row for row in results if row["mode"] == mode and "error" not in row]
        if not rows:
            continue
        summary[mode] = {
            "successful_trials": len(rows),
            "load_wall_s": round(median(row["load_wall_s"] for row in rows), 3),
            "request_wall_s": round(median(row["request_wall_s"] for row in rows), 3),
            "end_to_end_wall_s": round(median(row["end_to_end_wall_s"] for row in rows), 3),
            "prompt_tokens_per_s": round(median(row["timings"]["prompt_per_second"] for row in rows), 3),
            "generated_tokens_per_s": round(median(row["timings"]["predicted_per_second"] for row in rows), 3),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", default=DEFAULT_ENGINE, type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL, type=Path)
    parser.add_argument("--gpu-layers", default="all")
    parser.add_argument("--repeats", default=2, type=int, help="measured trials per mode")
    parser.add_argument("--timeout", default=90, type=int, help="startup and request timeout, individually")
    parser.add_argument("--note", help="record any relevant background workload")
    parser.add_argument("--output", default=PROJECT / "results/inference.json", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or args.timeout < 1:
        parser.error("--repeats and --timeout must be positive")
    engine, model = args.engine.resolve(), args.model.resolve()
    for path in (engine, model):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")

    trials = []
    for trial in range(args.repeats):
        order = ("cpu", "gpu") if trial % 2 == 0 else ("gpu", "cpu")
        for mode in order:
            print(f"Trial {trial + 1}, {mode}", flush=True)
            row = run(engine, model, mode, args.gpu_layers, args.timeout)
            trials.append({"trial": trial + 1, **row})
            print(row.get("error") or row["timings"], flush=True)
    successful = [row for row in trials if "error" not in row]
    answers = {row["answer"] for row in successful}
    trees = [
        parsed_python(row["answer"]) for row in successful
    ]
    report = {
        "engine": "ToshLLM llama-server",
        "model": model.name,
        "model_bytes": model.stat().st_size,
        "note": args.note,
        "command_settings": {
            "context": C, "batch": BATCH, "ubatch": BATCH, "threads": THREADS,
            "max_output_tokens": N, "temperature": 0, "seed": 42, "mmap": True, "warmup": False,
            "cache_ram_mib": 0, "tosh_fa_amd": 1, "gpu_layers": args.gpu_layers,
        },
        "prompt": PROMPT,
        "trials": trials,
        "summary_medians": summarize(trials),
        "all_answers_byte_identical": len(successful) == len(trials) and len(answers) == 1,
        "all_answers_python_ast_identical": (
            len(successful) == len(trials)
            and all(tree is not None for tree in trees)
            and len({ast.dump(tree) for tree in trees}) == 1
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary_medians"], indent=2), flush=True)
    return int(len(successful) != len(trials))


if __name__ == "__main__":
    raise SystemExit(main())
