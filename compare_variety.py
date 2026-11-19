#!/usr/bin/env python3
"""Time twelve distinct warm Gemma requests on the stock and rebuilt Intel engines."""

import argparse
from collections import defaultdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid

from compare_avx2 import DEFAULT_AVX2_ENGINE, ENVIRONMENT, binary_sha256
from compare_correct_answers import PROFILES, SOURCE_REV, atomic_write, engine_identity, now, settings, shell_text, state
from compare_inference import DEFAULT_ENGINE, PROJECT, free_port, request
from variety_workloads import WORKLOADS, grade


WARMUP = {"prompt": "Reply with the word ready.", "max_tokens": 4}
DEFAULT_GGUF, _, _, _, EXPECTED_GGUF = PROFILES["gemma"]


def blocks_for(workloads):
    by_category = defaultdict(list)
    for slug, item in workloads.items():
        by_category[item["category"]].append(slug)
    if len(by_category) != 4 or any(len(group) != 3 for group in by_category.values()):
        raise ValueError("Expected three distinct prompts in each of four categories")
    mixed_order = [slug for column in zip(*by_category.values()) for slug in column]
    return [mixed_order[i:i + 3] for i in range(0, len(mixed_order), 3)]


def streamed_sample(client, base, item, timeout):
    start = time.monotonic()
    rendered = request(client, base + "/apply-template",
                       {"messages": [{"role": "user", "content": item["prompt"]}]}, timeout=timeout)["prompt"]
    payload = {"prompt": rendered, "temperature": 0, "seed": 42, "n_predict": item["max_tokens"],
               "stream": True, "return_tokens": True, "cache_prompt": False}
    req = urllib.request.Request(base + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    pieces, ids = [], []
    first = final = None
    remaining = timeout - (time.monotonic() - start)
    if remaining <= 0:
        raise TimeoutError("Template formatting exceeded request deadline")
    with client.open(req, timeout=remaining) as stream:
        for line in stream:
            when = time.monotonic()
            if when - start > timeout:
                raise TimeoutError("Stream exceeded request deadline")
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            content = event.get("content") or ""
            if content:
                if first is None:
                    first = when
                pieces.append(content)
            ids.extend(event.get("tokens") or [])
            if event.get("stop"):
                final = event
                break
    finished = time.monotonic()
    if not final or not final.get("timings"):
        raise RuntimeError("Stream ended without engine timings")
    timings = final["timings"]
    if timings.get("cache_n", 0) != 0:
        raise RuntimeError("The server reused a prompt prefix despite cache_prompt=false")
    return {
        "answer": "".join(pieces), "token_ids": ids, "timings": timings,
        "ttft_s": None if first is None else first - start, "complete_s": finished - start,
        "prompt_tokens": timings["prompt_n"], "output_tokens": timings["predicted_n"],
        "decode_tps": timings["predicted_per_second"],
        "prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "stop_type": final.get("stop_type"), "prompt_truncated": bool(final.get("truncated")),
        "budget_reached": bool(final.get("stopped_limit") or final.get("stop_type") == "limit"
                               or timings["predicted_n"] >= item["max_tokens"]),
    }


def run_arm(engine, gguf, protocol, row, checkpoint, timeout):
    row["status"] = "cooldown"
    row["cooldown_started_utc"] = now()
    checkpoint()
    time.sleep(protocol["cooldown_s"])
    row["before"] = state()
    if not row["before"]["ac_power"]:
        raise RuntimeError("The machine is not on AC power")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    command = [str(engine), "-m", str(gguf), *protocol["server_args"], "--port", str(port)]
    with tempfile.TemporaryFile() as log:
        row.update(status="running", launch_utc=now())
        checkpoint()
        launch = time.monotonic()
        proc = subprocess.Popen(command, cwd=engine.parent, env=ENVIRONMENT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None and time.monotonic() - launch < timeout:
                try:
                    if request(client, base + "/health").get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(.1)
            else:
                raise RuntimeError(f"Server failed to become ready (exit={proc.poll()})")
            row["startup_s"] = time.monotonic() - launch
            row["warmup"] = streamed_sample(client, base, WARMUP, timeout)
            checkpoint()
            for slug in row["workload_order"]:
                row["last_started_workload"] = slug
                sample = streamed_sample(client, base, WORKLOADS[slug], timeout)
                sample.update(grade(slug, sample["answer"], sample["output_tokens"]))
                sample["after"] = state()
                row["samples"][slug] = sample
                checkpoint()
                print(f"  {slug}: {sample['complete_s']:.2f}s; strict={sample['strict_correct']}", flush=True)
                if not sample["after"]["ac_power"]:
                    raise RuntimeError("AC power lost during the arm")
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.seek(0)
            lines = log.read().decode(errors="replace").splitlines()
            row["allocation"] = [s.strip() for s in lines if re.search(
                r"offloaded \d+/\d+ layers to GPU|MTL0_Private model buffer size", s)]
            if len(row["samples"]) != len(row["workload_order"]):
                row["log_tail"] = [s.replace(str(Path.home()), "~") for s in lines[-8:]]
    if not re.search(r"offloaded 3/\d+ layers to GPU", "\n".join(row["allocation"])):
        raise RuntimeError("Three Radeon layers could not be verified in the server log")
    row.update(status="complete", ended_utc=now())
    checkpoint()


def benchmark(args, output):
    gguf = args.gguf.expanduser().resolve(strict=True)
    sha = binary_sha256(gguf)
    if sha != EXPECTED_GGUF:
        raise ValueError(f"Unexpected Gemma model checksum {sha}")
    engines = {"stock": args.stock.expanduser().resolve(strict=True),
               "rebuilt": args.rebuilt.expanduser().resolve(strict=True)}
    flags = settings(argparse.Namespace(model="gemma", layers="3", threads=3, no_repack=True))
    blocks = blocks_for(WORKLOADS)
    protocol = {"blocks": blocks, "workloads": WORKLOADS, "warmup": WARMUP, "cooldown_s": 20,
                "server_args": flags, "seed": 42, "temperature": 0, "cache_prompt": False,
                "TOSH_FA_AMD": "1", "timeout_s": args.timeout,
                "runner_sha256": binary_sha256(Path(__file__)),
                "fixtures_sha256": binary_sha256(PROJECT / "variety_workloads.py")}
    identity = {"schema_version": 1, "model": "gemma", "gguf": gguf.name, "model_sha256": sha,
                "engines": {label: engine_identity(path) for label, path in engines.items()}, "protocol": protocol}
    if output.exists():
        if not args.resume:
            raise ValueError(f"{output} exists. Use --resume or a fresh output file.")
        data = json.loads(output.read_text())
        if any(data.get(k) != v for k, v in identity.items()):
            raise ValueError("Checkpoint identity, frozen protocol or input files differ")
        if data.get("status") == "complete":
            if len(data["blocks"]) != len(blocks) or any(b["status"] != "complete" for b in data["blocks"]):
                raise ValueError("Completed checkpoint contains missing or incomplete blocks")
            print("All four paired blocks are already complete.", flush=True)
            return
        if data["blocks"] and data["blocks"][-1]["status"] != "complete":
            data.setdefault("abandoned_blocks", []).append(data["blocks"].pop())
    else:
        data = {**identity, "run_id": str(uuid.uuid4()), "created_utc": now(), "status": "running",
                "hardware": {"model": shell_text("/usr/sbin/sysctl", "-n", "hw.model"),
                             "cpu": shell_text("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
                             "ram_bytes": int(shell_text("/usr/sbin/sysctl", "-n", "hw.memsize")),
                             "macos": platform.mac_ver()[0]},
                "engine_source": {"toshllm": "v0.88.0", "llama_cpp_commit": SOURCE_REV}, "blocks": []}

    def save():
        atomic_write(output, data)

    save()
    for i in range(len(data["blocks"]), len(blocks)):
        arm_order = ["stock", "rebuilt"] if i % 2 == 0 else ["rebuilt", "stock"]
        block = {"block": i + 1, "order": arm_order, "prompts": blocks[i], "status": "running", "arms": {}}
        data["blocks"].append(block)
        for arm in arm_order:
            print(f"Block {i + 1}/{len(blocks)}, {arm}", flush=True)
            row = block["arms"][arm] = {"workload_order": blocks[i], "samples": {}}
            try:
                run_arm(engines[arm], gguf, protocol, row, save, args.timeout)
            except BaseException as exc:
                row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                           error=f"{type(exc).__name__}: {exc}".replace(str(Path.home()), "~"))
                block["status"] = "incomplete"
                save()
                raise
        matched = {}
        for slug in blocks[i]:
            s, r = (block["arms"][arm]["samples"][slug] for arm in ("stock", "rebuilt"))
            matched[slug] = {"prompt_hash_match": s["prompt_sha256"] == r["prompt_sha256"],
                             "prompt_truncated": s["prompt_truncated"] or r["prompt_truncated"],
                             "token_ids_match": bool(s["token_ids"]) and s["token_ids"] == r["token_ids"]}
        block.update(status="complete", pairing=matched)
        save()
    data.update(status="complete", ended_utc=now())
    save()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--stock", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--rebuilt", type=Path, default=DEFAULT_AVX2_ENGINE)
    parser.add_argument("--timeout", type=int, default=600, help="startup or individual request deadline, seconds")
    parser.add_argument("--output", type=Path, default=PROJECT / "results/gemma-prompt-variety.json")
    parser.add_argument("--resume", action="store_true", help="keep completed blocks and repeat any interrupted block")
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("timeout must be positive")
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(tempfile.gettempdir()) / f"llm-correct-answers-{os.getuid()}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another benchmark is active on this laptop")
        benchmark(args, output)


if __name__ == "__main__":
    main()
