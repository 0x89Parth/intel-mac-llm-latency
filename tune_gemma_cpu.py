#!/usr/bin/env python3
"""Profile a stock Intel Gemma decoder, then screen CPU threads and Metal layer counts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile
import time
import urllib.request

from compare_avx2 import DEFAULT_AVX2_ENGINE, ENVIRONMENT, binary_sha256, thermal
from compare_inference import DEFAULT_ENGINE, PROJECT, free_port, request
from compare_repack import memory_state
from gemma_offload import DEFAULT_MODEL

# Screening only. The real-answer benchmark uses different prompts.
TASKS = (
    ("shipment", "A carton holds 6 mugs. A shop receives 8 full cartons and gives away 5 mugs. How many mugs remain? Reply with the integer only.", "43"),
    ("routing", "The directory lists Amber as printer support, Violet as payroll, and Cedar as facilities. Which team handles a missing payslip? Reply with the team name only.", "Violet"),
    ("sorting", "Sort 12, 3, 19, 7 from smallest to largest. Return just a JSON array of the four integers.", [3, 7, 12, 19]),
)
PROFILES = ((3, 3), (4, 2), (2, 3), (3, 4), (4, 3), (3, 2))
BODY = {"temperature": 0, "seed": 42, "n_predict": 24, "cache_prompt": False}
PROFILE_PROMPT = "Continue this list of fruits, separating each word with a comma: apple, pear, peach,"
BASE = ["--alias", "gemma-tune", "-dev", "MTL0", "--fit", "off", "--load-mode", "mmap",
        "-c", "1024", "-b", "16", "-ub", "16", "-np", "1", "--cache-ram", "0",
        "--no-warmup", "--no-mmproj", "--reasoning", "off", "--no-repack", "--offline",
        "--no-ui", "--host", "127.0.0.1", "-lv", "4"]


def correct(answer, expected):
    answer = answer.strip().strip("`").strip()
    if isinstance(expected, str):
        return answer.casefold().rstrip(".!") == expected.casefold()
    try:
        values = json.loads(answer)
    except ValueError:
        return False
    return type(values) is list and all(type(x) is int for x in values) and values == expected


def state():
    snapshot = {"thermal": thermal(), "memory": memory_state()}
    power = subprocess.check_output(["/usr/bin/pmset", "-g", "batt"], text=True)
    snapshot["ac_power"] = "'AC Power'" in power
    pressure = subprocess.run(["/usr/bin/memory_pressure", "-Q"], capture_output=True)
    snapshot["memory_pressure_status"] = pressure.returncode
    return snapshot


def sample_decode(client, url, proc, directory):
    # Arm sampling only once generated content has actually reached the client.
    output = Path(directory) / "stock.sample"
    template = request(client, url + "/apply-template", {"messages": [{"role":"user", "content":PROFILE_PROMPT}]})
    payload = {**BODY, "prompt": template["prompt"], "n_predict": 32, "stream": True, "ignore_eos": True}
    req = urllib.request.Request(url + "/completion", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"})
    sampler = None
    timings = {}
    with client.open(req, timeout=150) as response:
        for line in response:
            if not line.startswith(b"data:") or line[5:].strip() == b"[DONE]":
                continue
            event = json.loads(line[5:])
            if event.get("stop"):
                timings = event.get("timings", {})
                break
            if event.get("content") and sampler is None:
                sampler = subprocess.Popen(["/usr/bin/sample", str(proc.pid), "3", "5", "-file", str(output)],
                                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    if sampler is None:
        raise RuntimeError("stock probe produced no content")
    _, error = sampler.communicate(timeout=15)
    if sampler.returncode:
        raise RuntimeError("macOS sample failed: " + error.replace(str(Path.home()), "~")[-180:])
    report = output.read_text(errors="replace")
    functions = []
    marker = "Sort by top of stack"
    section = report.partition(marker)[2].split("Binary Images:", 1)[0]
    for line in section.splitlines():
        match = re.match(r"^\s*(.+?)(?:\s{2,})(\d+)\s*$", line)
        if match:
            function = match[1].strip().split("  (in ")[0].split(" (in ")[0]
            if not function.startswith("0x"):
                functions.append({"symbol": function, "leaf_samples": int(match[2])})
    signatures = sorted({s for s in re.findall(r"\b(?:ggml_(?:vec_dot|compute_forward|compute_forward_mul_mat)[a-zA-Z0-9_]*)", report)})
    return {"prompt": PROFILE_PROMPT, "method":"macOS sample, 3 s at 5 ms, started after first content event; all-process thread leaf counts include waiting threads",
            "decode":timings, "top_leaf_symbols":functions[:40], "cpu_kernel_symbols_on_stacks":signatures[:45]}


def block(engine, model, threads, layers, profiling=False):
    port = free_port()
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    url = f"http://127.0.0.1:{port}"
    row = {"threads":threads, "gpu_layers":layers, "before": state(), "trials":[]}
    if not row["before"]["ac_power"] or row["before"]["memory_pressure_status"] != 0 or row["before"]["thermal"]["thermal_warning"]:
        raise RuntimeError("AC power and nominal memory pressure with no thermal warning needed")
    setting = ["-ngl", str(layers), "-t", str(threads), "-tb", str(threads)]
    row["engine_flags"] = BASE + setting
    with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as log:
        begin = time.monotonic()
        proc = subprocess.Popen([str(engine), "-m", str(model), *BASE, *setting, "--port", str(port)], cwd=engine.parent,
                                env=ENVIRONMENT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None and time.monotonic()-begin < 130:
                try:
                    if request(client, url + "/health").get("status") == "ok":
                        break
                except OSError:
                    pass
                time.sleep(.3)
            else:
                raise RuntimeError(f"startup failed or timed out, exit {proc.poll()}")
            row["startup_seconds"] = time.monotonic()-begin
            if profiling:
                row["sample"] = sample_decode(client, url, proc, directory)
            else:
                for name, prompt, expected in (("warmup", "Reply with the word ready.", None), *TASKS):
                    if state()["thermal"]["thermal_warning"]:
                        raise RuntimeError("macOS reports thermal warning")
                    template = request(client, url + "/apply-template", {"messages":[{"role":"user","content":prompt}]})
                    start = time.monotonic()
                    answer = request(client, url + "/completion", {**BODY, "n_predict":3 if expected is None else BODY["n_predict"], "prompt":template["prompt"]}, timeout=125)
                    elapsed = time.monotonic()-start
                    if expected is not None:
                        row["trials"].append({"task":name, "request_seconds":elapsed, "answer":answer["content"],
                                              "correct":correct(answer["content"],expected), "timings":answer["timings"]})
            row["after"] = state()
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            row["error"] = str(exc)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=10)
        log.seek(0)
        lines = log.read().decode(errors="replace").splitlines()
    row["placement"] = [s.strip() for s in lines if "offloaded " in s and "layers to GPU" in s]
    row["metal_buffers"] = [s.strip() for s in lines if "MTL0_Private" in s and "buffer size" in s]
    if not any(f"offloaded {layers}/" in s for s in row["placement"]):
        row.setdefault("error", "requested GPU placement unconfirmed")
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--stock", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--candidate", type=Path, default=DEFAULT_AVX2_ENGINE)
    parser.add_argument("--output", type=Path, default=PROJECT/"results/gemma-cpu-tuning.json")
    opt = parser.parse_args()
    model = opt.model.expanduser().resolve(strict=True)
    stock, candidate = [p.expanduser().resolve(strict=True) for p in (opt.stock,opt.candidate)]
    out = {"started_utc":datetime.now(timezone.utc).isoformat(), "model":model.name, "model_bytes":model.stat().st_size,
           "engine_sha256":{"stock":binary_sha256(stock),"candidate":binary_sha256(candidate)}, "tasks":TASKS,"profiles":PROFILES,
           "sampling":BODY,"selection_rule":"Require all three correct. Minimize geometric mean of matched wall-time ratios against 3 threads and 3 layers. Single-pass exploratory tuning.", "blocks":[]}
    opt.output.parent.mkdir(exist_ok=True, parents=True)
    def save():
        stage = opt.output.with_suffix(".tmp")
        stage.write_text(json.dumps(out,indent=2)+"\n")
        stage.replace(opt.output)
    print("profiling stock 3/3",flush=True)
    out["stock_profile"] = block(stock,model,3,3,True)
    save()
    for threads,layers in PROFILES:
        print(f"screening {threads} threads / {layers} GPU layers",flush=True)
        row = block(candidate,model,threads,layers)
        out["blocks"].append(row)
        save()
        print([(t["task"], t["correct"], round(t["request_seconds"],2)) for t in row["trials"]], row.get("error",""),flush=True)
    reference = out["blocks"][0]
    eligible = []
    if len(reference["trials"]) == len(TASKS) and all(t["correct"] for t in reference["trials"]) and "error" not in reference:
        for row in out["blocks"]:
            if len(row["trials"]) == len(TASKS) and all(t["correct"] for t in row["trials"]) and "error" not in row:
                ratios = [trial["request_seconds"]/ref["request_seconds"] for trial,ref in zip(row["trials"],reference["trials"])]
                row["wall_ratio_vs_reference"] = math.exp(sum(map(math.log,ratios))/len(ratios))
                eligible.append(row)
    best = min(eligible,key=lambda r:r["wall_ratio_vs_reference"]) if eligible else None
    out["selected_profile"] = {k: best[k] for k in ("threads","gpu_layers","wall_ratio_vs_reference")} if best else None
    save()
    print("selected",out["selected_profile"],flush=True)


if __name__ == "__main__":
    main()
