#!/usr/bin/env python3
"""Compare Gemma throughput across two ToshLLM builds using 3 Radeon layers."""

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

from compare_inference import DEFAULT_ENGINE, LOCAL, PROJECT, free_port, request
from gemma_offload import DEFAULT_MODEL, EXPECTED, PROMPT

DEFAULT_AVX2_ENGINE = LOCAL / "toshllm-engine-0.88.0-macos13-x86_64-avx2/bin/llama-server"
DEFAULT_ENGINES = (("compatibility", DEFAULT_ENGINE), ("avx2", DEFAULT_AVX2_ENGINE))
CONTEXT = 1024
BATCH = 16
THREADS = 3
LAYERS = 3
SEED = 42
CODING_PROMPT = (
    "Evaluate this Python function: def f(xs): return sum(x*x for x in xs if x % 2). "
    "Return only a JSON array of four integers giving f([]), f([1,2,3]), "
    "f([-3,-2,1]), and f([2,4]), in that order."
)
CODING_EXPECTED = [0, 10, 10, 0]
CODING_MAX_TOKENS = 48
STREAM_PROMPT = "Continue with the requested repeated sentence exactly: The Moon orbits Earth."
STREAM_EXPECTED = " ".join([EXPECTED] * 160)
GRAMMAR = 'root ::= "The Moon orbits Earth." ' + " ".join(["tail"] * 159)
GRAMMAR += '\ntail ::= " The Moon orbits Earth."'
SETTINGS = {
    "context": CONTEXT, "batch": BATCH, "ubatch": BATCH, "cpu_threads": THREADS,
    "metal_device": "MTL0", "gpu_layers": LAYERS, "seed": SEED, "temperature": 0,
    "mmap": True, "warmup": False, "cache_ram_mib": 0, "tosh_fa_amd": 1,
}
ENVIRONMENT = {k: v for k, v in os.environ.items() if not k.startswith(("TOSH_", "LLAMA_ARG_"))}
ENVIRONMENT["TOSH_FA_AMD"] = "1"


def thermal():
    out = subprocess.check_output(["/usr/bin/pmset", "-g", "therm"], text=True, timeout=3)
    limits = {}
    for name in ("CPU_Speed_Limit", "CPU_Scheduler_Limit"):
        match = re.search(rf"{name}\s*=\s*(\d+)", out)
        limits[name.lower() + "_pct"] = int(match[1]) if match else None
    warning = re.search(r"Thermal Warning Level\s*[:=]\s*(\d+)", out, re.I)
    limits["thermal_warning"] = bool(warning and int(warning[1]) > 0)
    return limits


def wait_for_cool_cpu(max_wait):
    start = time.monotonic()
    limits = thermal()
    if limits["cpu_speed_limit_pct"] is not None and limits["cpu_speed_limit_pct"] < 90 and max_wait:
        print(f"Waiting up to {max_wait:g}s for CPU speed limit to recover to 90%", flush=True)
    while (
        limits["cpu_speed_limit_pct"] is not None
        and limits["cpu_speed_limit_pct"] < 90
        and time.monotonic() - start < max_wait
    ):
        time.sleep(min(5, max(0, max_wait - (time.monotonic() - start))))
        limits = thermal()
    return round(time.monotonic() - start, 3), limits


def chat(opener, url, prompt, max_tokens, timeout, *, alias="gemma-benchmark"):
    start = time.monotonic()
    response = request(opener, url + "/v1/chat/completions", {
        "model": alias, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": SEED, "max_tokens": max_tokens, "stream": False,
        "cache_prompt": False,
    }, timeout=timeout)
    choice = response["choices"][0]
    return {
        "answer": choice["message"]["content"],
        "finish_reason": choice["finish_reason"],
        "usage": response.get("usage"),
        "timings": response.get("timings"),
        "request_wall_s": round(time.monotonic() - start, 3),
        "thermal_after": thermal(),
    }


def coding_answer_is_correct(answer):
    text = (answer or "").strip()
    fence = re.fullmatch(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    try:
        values = json.loads(fence[1] if fence else text)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(values, list) and all(type(value) is int for value in values)
        and values == CODING_EXPECTED
    )


def binary_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while block := source.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def stream(opener, url, token_budget, decode_so_far, tokens_so_far, progress, timeout):
    payload = {
        "prompt": STREAM_PROMPT, "grammar": GRAMMAR, "n_predict": token_budget,
        "temperature": 0, "seed": SEED, "stream": True, "cache_prompt": False,
        "return_tokens": True, "timings_per_token": True,
    }
    req = urllib.request.Request(
        url + "/completion", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    answer = []
    token_ids = []
    event_count = 0
    first_text_s = None
    final = None
    start = time.monotonic()
    last_sample_time = start
    with opener.open(req, timeout=timeout) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            elapsed_s = time.monotonic() - start
            if event.get("stop"):
                final = event
                break
            content = event.get("content", "")
            if content:
                answer.append(content)
                event_count += 1
                if first_text_s is None:
                    first_text_s = elapsed_s
            token_ids.extend(event.get("tokens") or [])
            timings = event.get("timings") or {}
            if time.monotonic() - last_sample_time >= 15:
                limits = thermal()
                if limits["thermal_warning"]:
                    raise RuntimeError("macOS reported a thermal warning during generation")
                last_sample_time = time.monotonic()

            decoded = timings.get("predicted_ms")
            observed = timings.get("predicted_n", event.get("tokens_predicted"))
            if decoded is None and first_text_s is not None:
                decoded = max(0, 1000 * (elapsed_s - first_text_s))
            if observed is None:
                observed = len(token_ids) or event_count
            if decoded is not None:
                decoded_s = decode_so_far + decoded / 1000
                marker = (len(progress) + 1) * 60
                if decoded_s >= marker:
                    point = {
                        "decode_s": round(decoded_s, 3),
                        "generated_tokens": tokens_so_far + observed,
                        "basis": "engine timing" if timings.get("predicted_ms") is not None else
                            ("stream timings and token counts" if event.get("tokens_predicted") is not None or token_ids
                             else "stream chunks; approximate"),
                        **thermal(),
                    }
                    previous = progress[-1] if progress else {"decode_s": 0, "generated_tokens": 0}
                    point["interval_tokens_per_s"] = round(
                        (point["generated_tokens"] - previous["generated_tokens"])
                        / (point["decode_s"] - previous["decode_s"]), 3,
                    )
                    progress.append(point)
                    print(
                        f"Minute {len(progress)}: {point['interval_tokens_per_s']:.3f} tok/s, "
                        f"CPU limit {point['cpu_speed_limit_pct']}%, "
                        f"{point['generated_tokens']} total tokens ({point['basis']})",
                        flush=True,
                    )
    if final is None or not final.get("timings"):
        raise RuntimeError("stream ended without final engine timings")
    text = "".join(answer)
    if not text or not STREAM_EXPECTED.startswith(text):
        raise RuntimeError(f"generated content deviated from the exact continuation: {text[:120]!r}")
    if final["timings"]["predicted_n"] != token_budget:
        raise RuntimeError(
            f"engine generated {final['timings']['predicted_n']} tokens, expected {token_budget}"
        )
    return {
        "answer": text, "token_ids": token_ids, "exact_expected_prefix": True,
        "timings": final["timings"], "first_text_s": first_text_s,
        "stop_type": final.get("stop_type"), "prompt_truncated": final.get("truncated"),
        "request_wall_s": round(time.monotonic() - start, 3),
        "thermal_after": thermal(),
    }


def run_engine(row, engine, model, args, checkpoint):
    port = free_port()
    url = f"http://127.0.0.1:{port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command = [
        str(engine), "-m", str(model), "--alias", "gemma-benchmark", "-dev", "MTL0",
        "-ngl", str(LAYERS), "--fit", "off", "--load-mode", "mmap",
        "-c", str(CONTEXT), "-b", str(BATCH), "-ub", str(BATCH), "-t", str(THREADS),
        "-tb", str(THREADS), "-np", "1", "--cache-ram", "0", "--no-warmup",
        "--no-mmproj", "--reasoning", "off", "--offline", "--no-ui",
        "--cors-origins", "localhost", "--host", "127.0.0.1", "--port", str(port), "--verbose",
    ]
    name = row["build"]
    print(f"Starting {name}", flush=True)
    with tempfile.TemporaryFile() as log:
        start = time.monotonic()
        process = subprocess.Popen(
            command, cwd=engine.parent, env=ENVIRONMENT, stdin=subprocess.DEVNULL,
            stdout=log, stderr=subprocess.STDOUT,
        )
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup ({process.returncode})")
                if time.monotonic() - start > args.timeout:
                    raise TimeoutError("server startup timed out")
                try:
                    if request(opener, url + "/health").get("status") == "ok":
                        break
                except urllib.error.URLError:
                    pass
                time.sleep(0.25)
            row["load_wall_s"] = round(time.monotonic() - start, 3)
            row["copy_check"] = chat(opener, url, PROMPT, 10, args.timeout)
            answer = row["copy_check"]["answer"]
            row["copy_check"]["exact"] = answer == EXPECTED
            if not row["copy_check"]["exact"]:
                raise RuntimeError(f"normal chat copy check gave {answer!r}")
            print(f"{name}: exact chat copy passed", flush=True)
            checkpoint()
            row["coding_request"] = chat(opener, url, CODING_PROMPT, CODING_MAX_TOKENS, args.timeout)
            coding = row["coding_request"]
            coding["correct"] = coding_answer_is_correct(coding["answer"])
            print(f"{name}: coding check correct={coding['correct']}, {coding['request_wall_s']:.2f}s", flush=True)
            checkpoint()
            decoded_s = 0
            token_total = 0
            while decoded_s < args.seconds:
                result = stream(
                    opener, url, args.tokens, decoded_s, token_total,
                    row["minute_progress"], args.timeout,
                )
                row["requests"].append(result)
                decoded_s += result["timings"]["predicted_ms"] / 1000
                token_total += result["timings"]["predicted_n"]
                row["decode_s"] = round(decoded_s, 3)
                row["generated_tokens"] = token_total
                row["combined_tokens_per_s"] = round(token_total / decoded_s, 3)
                print(f"{name} request {len(row['requests'])}: {decoded_s:.1f} s total decode, "
                      f"{row['combined_tokens_per_s']:.3f} tok/s", flush=True)
                checkpoint()
            row["status"] = "complete"
            if row["minute_progress"]:
                row["last_full_minute_tokens_per_s"] = row["minute_progress"][-1]["interval_tokens_per_s"]
        except (OSError, RuntimeError, KeyError, ValueError) as error:
            row["status"] = "error"
            row["error"] = str(error)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(8)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            row["thermal_after"] = thermal()
            log.seek(0)
            lines = log.read().decode(errors="replace").splitlines()
    row["engine_log"] = [
        line.replace(str(Path.home()), "~")
        for line in lines if re.search(
            r"offloaded \d+/\d+ layers to GPU|MTL0_Private .+ buffer size|AVX2|F16C|FMA|ToshLLM engine",
            line,
        )
    ]
    if not any(f"offloaded {LAYERS}/" in line for line in row["engine_log"]):
        row.setdefault("error", "engine did not verify requested Metal layer count")
        row["status"] = "error"
    if row["status"] == "error":
        row["log_tail"] = [line.replace(str(Path.home()), "~") for line in lines[-12:]]


def reference_comparison(rows):
    complete = [row for row in rows if row["status"] == "complete"]
    if len(complete) < 2:
        return None
    first, second = (row["requests"][0] for row in complete[:2])
    return {
        "copy_answers_exact": all(row["copy_check"]["exact"] for row in complete[:2]),
        "coding_answers_correct": [row["coding_request"]["correct"] for row in complete[:2]],
        "coding_request_wall_s": [row["coding_request"]["request_wall_s"] for row in complete[:2]],
        "first_stream_text_identical": first["answer"] == second["answer"],
        "first_stream_token_ids_identical": (
            first["token_ids"] == second["token_ids"] if first["token_ids"] and second["token_ids"] else None
        ),
        "matched_first_request_wall_s": [first["request_wall_s"], second["request_wall_s"]],
        "matched_first_request_generation_tokens_per_s": [
            first["timings"]["predicted_per_second"], second["timings"]["predicted_per_second"],
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", nargs=2, action="append", metavar=("LABEL", "BINARY"),
                        help="engine to run, in the order given (default: installed compatibility, then AVX2)")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--seconds", type=float, default=300, help="minimum cumulative decode time per engine")
    parser.add_argument("--tokens", type=int, default=144, help="output budget per repeated request")
    parser.add_argument("--timeout", type=float, default=120, help="startup and per-read timeouts")
    parser.add_argument("--cooldown", type=float, default=30, help="seconds between engines")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/gemma-avx2.json")
    parser.add_argument("--note", help="measurement conditions")
    args = parser.parse_args()
    if args.seconds <= 0 or args.timeout <= 0 or args.cooldown < 0 or not 1 <= args.tokens <= 768:
        parser.error("seconds and timeout must be positive, cooldown nonnegative, and tokens between 1 and 768")
    try:
        model = args.model.resolve(strict=True)
        engines = [
            (name, Path(path).expanduser().resolve(strict=True))
            for name, path in (args.engine or DEFAULT_ENGINES)
        ]
    except OSError as error:
        parser.error(str(error))
    report = {
        "model": model.name, "model_bytes": model.stat().st_size,
        "settings": SETTINGS, "copy_prompt": PROMPT, "copy_expected": EXPECTED,
        "coding_prompt": CODING_PROMPT, "coding_expected": CODING_EXPECTED,
        "coding_max_output_tokens": CODING_MAX_TOKENS,
        "stream_prompt": STREAM_PROMPT, "stream_grammar": "exact sentence repeated 160 times",
        "tokens_per_request": args.tokens, "target_decode_s": args.seconds, "note": args.note,
        "builds": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint():
        report["comparison"] = reference_comparison(report["builds"])
        temp = args.output.with_suffix(args.output.suffix + ".tmp")
        temp.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        temp.replace(args.output)

    for index, (name, engine) in enumerate(engines):
        if index:
            time.sleep(args.cooldown)
        wait_s, before = wait_for_cool_cpu(max(0, 90 - (args.cooldown if index else 0)))
        row = {
            "build": name, "binary_sha256": binary_sha256(engine), "thermal_before": before,
            "thermal_wait_s": wait_s,
            "minute_progress": [], "requests": [], "status": "running",
        }
        report["builds"].append(row)
        run_engine(row, engine, model, args, checkpoint)
        checkpoint()
    return int(any(row["status"] != "complete" for row in report["builds"]))


if __name__ == "__main__":
    raise SystemExit(main())
