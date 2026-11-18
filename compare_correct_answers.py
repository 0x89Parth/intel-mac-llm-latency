#!/usr/bin/env python3
"""Compare process-cold and warm time to a graded answer in independent paired runs."""

import argparse
import fcntl
import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

from compare_avx2 import DEFAULT_AVX2_ENGINE, ENVIRONMENT, binary_sha256, thermal
from compare_inference import DEFAULT_ENGINE, DEFAULT_MODEL as QWEN, PROJECT, free_port, request
from compare_repack import memory_state
from correct_answer_workloads import WORKLOADS, grade
from gemma_offload import DEFAULT_MODEL as GEMMA


PROFILES = {
    "gemma": (GEMMA, "3", 16, 1024, "4b4f9d4b448fca16230f08d50c18a95c81ecfaefede95d3bdfcf1880b4b620f5"),
    "qwen": (QWEN, "all", 32, 2048, "1d9614638d18024d0fbb36575a15f1302a3adf044df10345688ec4f6e1c4ff32"),
}
WARM = ("explanation", "long_prompt", "code_edit", "fixed_decode")
SOURCE_REV = "b1b770d90d94cad9e9ea5b993bdbc383a522fb28"
KNOWN_ENGINES = {
    "c3c287023914f8e7612c669431ccfa80717a4e0c6fe322aa462ceaa4de1e0c86":
        "ToshLLM v0.88.0 official macOS 12+ x86_64 binary, SSE4.2 compatibility",
    "690aac3a10b2ef64eef094a77d94e10d7559ef37074a16c5f418d120d1bfa806":
        "ToshLLM v0.88.0 local Ventura build, AVX2/FMA/F16C/BMI2, see build-avx2.sh",
}


def now():
    return datetime.now(timezone.utc).isoformat()


def shell_text(*cmd):
    return subprocess.check_output(cmd, text=True, timeout=12).strip()


def state():
    power = shell_text("/usr/bin/pmset", "-g", "batt").splitlines()[0]
    return {
        "utc": now(), "ac_power": "'AC Power'" in power,
        "vm_pressure_level": int(shell_text("/usr/sbin/sysctl", "-n", "kern.memorystatus_vm_pressure_level")),
        **memory_state(), **thermal(),
    }


def engine_identity(path):
    sha = binary_sha256(path)
    output = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                            env=ENVIRONMENT, timeout=20, check=True)
    version = [line.strip() for line in (output.stdout + output.stderr).splitlines()
               if re.search(r"version:|built with|build:|compiler:", line, re.I)]
    return {"binary": path.name, "sha256": sha, "build": KNOWN_ENGINES.get(sha, "custom binary"),
            "reported_version": version}


def settings(args, candidate=False):
    _, default_layers, batch, context, _ = PROFILES[args.model]
    layers = args.layers if args.layers is not None else default_layers
    threads = args.threads
    if candidate:
        layers = args.candidate_layers if args.candidate_layers is not None else layers
        threads = args.candidate_threads if args.candidate_threads is not None else threads
    return ["--alias", "paired-benchmark", "-dev", "MTL0", "-ngl", layers, "--fit", "off",
            "--load-mode", "mmap", "-c", str(context), "-b", str(batch), "-ub", str(batch),
            "-t", str(threads), "-tb", str(threads), "-np", "1", "--cache-ram", "0",
            "--no-warmup", "--no-mmproj", "--reasoning", "off", *(["--no-repack"] if args.no_repack else []), "--offline",
            "--no-webui", "--host", "127.0.0.1", "-lv", "4"]


def atomic_write(path, record):
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix="." + path.name + ".",
                                     delete=False, encoding="utf8") as fp:
        partial = Path(fp.name)
        try:
            json.dump(record, fp, indent=2, ensure_ascii=False)
            fp.write("\n")
            fp.flush()
            os.fsync(fp.fileno())
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
    try:
        partial.replace(path)
    finally:
        partial.unlink(missing_ok=True)


def measure(client, base, name, timeout, launch_time=None):
    item = WORKLOADS[name]
    start = time.monotonic()
    prompt = item["prompt"]
    if name != "fixed_decode":
        prompt = request(client, base + "/apply-template",
                         {"messages": [{"role": "user", "content": prompt}]}, timeout=timeout)["prompt"]
    payload = {"prompt": prompt, "temperature": 0, "seed": 42, "n_predict": item["max_tokens"],
               "stream": True, "return_tokens": True, "cache_prompt": False}
    if item.get("grammar"):
        payload.update(grammar=item["grammar"], ignore_eos=True)
    req = urllib.request.Request(base + "/completion", data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    pieces, token_ids = [], []
    final = first = None
    with client.open(req, timeout=timeout) as stream:
        for line in stream:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                break
            event = json.loads(raw)
            if "error" in event:
                raise RuntimeError(str(event["error"]))
            text = event.get("content") or ""
            if text:
                if first is None:
                    first = time.monotonic()
                pieces.append(text)
            token_ids.extend(event.get("tokens") or [])
            if event.get("stop"):
                final = event
                break
    end = time.monotonic()
    if final is None or not final.get("timings"):
        raise RuntimeError("Stream terminated without engine timings")
    timings = final["timings"]
    if timings.get("cache_n", 0) or final.get("truncated"):
        raise RuntimeError("Unexpected prompt-cache reuse or context truncation")
    content = "".join(pieces)
    count = timings["predicted_n"]
    passed, reason = grade(name, content, count)
    row = {"correct": bool(passed), "grade_reason": reason, "answer": content,
           "ttft_s": None if first is None else first - start, "complete_s": end - start,
           "decode_tps": timings["predicted_per_second"], "output_tokens": count,
           "prompt_tokens": timings["prompt_n"], "timings": timings, "token_ids": token_ids,
           "stop_type": final.get("stop_type"), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    if launch_time is not None:
        row.update(cold_to_complete_s=end-launch_time,
                   cold_to_first_content_s=None if first is None else first-launch_time)
    return row


def run_arm(row, engine, model, server_args, cooldown_s, checkpoint, timeout):
    row["status"] = "running"
    row["cooldown_started_utc"] = now()
    checkpoint()
    time.sleep(cooldown_s)
    row["before"] = state()
    if not row["before"]["ac_power"]:
        raise RuntimeError("Connect AC power before benchmarking")
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cmd = [str(engine), "-m", str(model), *server_args, "--port", str(port)]
    with tempfile.TemporaryFile() as log:
        row["launch_utc"] = now()
        launch = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=engine.parent, env=ENVIRONMENT, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            while proc.poll() is None and time.monotonic() - launch < timeout:
                try:
                    if request(client, base + "/health").get("status") == "ok":
                        break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(.10)
            else:
                raise RuntimeError(f"Benchmark server failed to become ready (exit={proc.poll()})")
            row["startup_s"] = time.monotonic() - launch
            for name in row["workload_order"]:
                row["last_started_workload"] = name
                sample = measure(client, base, name, timeout, launch if name == "cold" else None)
                sample["after"] = state()
                row["samples"][name] = sample
                checkpoint()
                if not sample["after"]["ac_power"]:
                    raise RuntimeError("AC power lost during arm")
                print(f"  {name} correct={sample['correct']} elapsed={sample['complete_s']:.3f}s", flush=True)
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            log.seek(0)
            lines = log.read().decode(errors="replace").splitlines()
            row["allocation"] = [s.strip() for s in lines if re.search(r"offloaded \d+/\d+ layers to GPU|MTL0_Private model buffer size", s)]
            if len(row["samples"]) != len(row["workload_order"]):
                row["log_tail"] = [s.replace(str(Path.home()), "~") for s in lines[-8:]]
    matched = re.search(r"offloaded (\d+)/(\d+) layers to GPU", "\n".join(row["allocation"]))
    layers = server_args[server_args.index("-ngl") + 1]
    if not matched or matched[1] != (matched[2] if layers == "all" else layers):
        raise RuntimeError("Requested layer offload was not verified by engine log")
    row["status"] = "complete"
    row["ended_utc"] = now()
    checkpoint()


def benchmark(args, output):
    default_model, _, _, _, expected_hash = PROFILES[args.model]
    model = (args.gguf or default_model).expanduser().resolve(strict=True)
    engines = {key: value.expanduser().resolve(strict=True)
               for key, value in (("stock", args.stock), ("rebuilt", args.rebuilt))}
    model_hash = binary_sha256(model)
    if model_hash != expected_hash:
        raise ValueError(f"Unexpected {args.model} model checksum: {model_hash}")
    identities = {key: engine_identity(value) for key, value in engines.items()}
    candidate = args.candidate_layers is not None or args.candidate_threads is not None
    if candidate:
        engines["tuned"] = engines["rebuilt"]
        identities["tuned"] = identities["rebuilt"]
    protocol = {
        "rounds": args.rounds, "cooldown_s": args.cooldown, "seed": 42, "temperature": 0,
        "cache_prompt": False, "TOSH_FA_AMD": "1", "server_args": settings(args),
        "repack": "disabled" if args.no_repack else "engine default for both arms",
        "candidate_server_args": settings(args, candidate=True) if candidate else None,
        "workloads": WORKLOADS, "runner_sha256": binary_sha256(Path(__file__)),
        "fixtures_sha256": binary_sha256(PROJECT / "correct_answer_workloads.py"),
        "cold_definition": "New server process, no model warmup; filesystem cache retained. GGUF prehashed.",
        "latency_definition": "Client wall time from native chat formatting through final stream event. TTFT uses the first nonempty text chunk. Cold latency also includes process start and readiness polling.",
    }
    initial = {"schema_version": 1, "model": args.model, "gguf": model.name, "model_sha256": model_hash,
               "engines": identities, "protocol": protocol}
    if output.exists():
        if not args.resume:
            raise ValueError(f"{output} already exists; use --resume or a new --output")
        data = json.loads(output.read_text())
        if any(data.get(k) != v for k, v in initial.items()):
            raise ValueError("Checkpoint metadata differs from this protocol or the input files")
        if data["pairs"] and data["pairs"][-1]["status"] != "complete":
            data.setdefault("abandoned_pairs", []).append(data["pairs"].pop())
    else:
        data = {**initial, "run_id": str(uuid.uuid4()), "created_utc": now(), "pairs": [],
                "hardware": {"model": shell_text("/usr/sbin/sysctl", "-n", "hw.model"),
                             "cpu": shell_text("/usr/sbin/sysctl", "-n", "machdep.cpu.brand_string"),
                             "ram_bytes": int(shell_text("/usr/sbin/sysctl", "-n", "hw.memsize")),
                             "macos": platform.mac_ver()[0]},
                "reference_source": {"toshllm": "v0.88.0", "llama_cpp_commit": SOURCE_REV}}
    def checkpoint():
        atomic_write(output, data)
    checkpoint()
    for index in range(len(data["pairs"]), args.rounds):
        order = ["stock", "rebuilt"] if index % 2 == 0 else ["rebuilt", "stock"]
        if candidate:
            order.insert((2, 0, 1)[index % 3], "tuned")
        shift = index % len(WARM)
        workloads = ["cold", *(WARM[shift:] + WARM[:shift])]
        pair = {"round": index + 1, "status": "running", "order": list(order), "arms": {}}
        data["pairs"].append(pair)
        for arm in order:
            row = pair["arms"][arm] = {"workload_order": workloads, "samples": {}}
            print(f"{args.model} pair {index + 1}/{args.rounds} {arm}", flush=True)
            try:
                server_args = protocol["candidate_server_args"] if arm == "tuned" else protocol["server_args"]
                run_arm(row, engines[arm], model, server_args, args.cooldown, checkpoint, args.timeout)
            except BaseException as exc:
                row.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "error",
                           error=f"{type(exc).__name__}: {exc}".replace(str(Path.home()), "~"))
                pair["status"] = "incomplete"
                checkpoint()
                raise
        pair["status"] = "complete"
        checkpoint()
    data["status"] = "complete"
    checkpoint()
    print(f"Completed {len(data['pairs'])} independent paired rounds: {output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=tuple(PROFILES), required=True)
    parser.add_argument("--gguf", type=Path, help="file with the same verified model checksum")
    parser.add_argument("--stock", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--rebuilt", type=Path, default=DEFAULT_AVX2_ENGINE)
    parser.add_argument("--threads", type=int, default=3)
    parser.add_argument("--layers", help="identical GPU offload on both builds; defaults to three for Gemma, all for Qwen")
    parser.add_argument("--candidate-threads", type=int, help="add a third, tuned rebuilt arm")
    parser.add_argument("--candidate-layers", help="GPU offload for an optional third, tuned rebuilt arm")
    parser.add_argument("--no-repack", action="store_true", help="disable CPU weight repacking in all arms")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--cooldown", type=float, default=20, help="idle seconds before every arm")
    parser.add_argument("--timeout", type=float, default=600, help="startup and socket read timeout in seconds")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume", action="store_true", help="retain completed pairs; restart an interrupted pair")
    args = parser.parse_args()
    if args.rounds < 1 or args.threads < 1 or args.cooldown < 0 or args.timeout <= 0:
        parser.error("invalid round, thread, cooldown or timeout setting")
    if args.candidate_threads is not None and args.candidate_threads < 1:
        parser.error("--candidate-threads must be positive")
    for layers in (args.layers, args.candidate_layers):
        if layers is not None and layers != "all" and not layers.isdecimal():
            parser.error("GPU layers must be a nonnegative integer or all")
    output = (args.output or PROJECT / "results" / f"correct-answers-{args.model}.json").resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    # A shared host lock prevents competing copies of the paired benchmark on the same laptop.
    lock_path = Path(tempfile.gettempdir()) / f"llm-correct-answers-{os.getuid()}.lock"
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("another paired benchmark is active on this Mac")
        benchmark(args, output)


if __name__ == "__main__":
    main()
