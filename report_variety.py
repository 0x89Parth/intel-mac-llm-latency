#!/usr/bin/env python3
"""Export blinded answers and report Gemma latency and pass rates on distinct prompts."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import median

from analyze_results import good_number
from blind_answers import answer_id
from write_answer_review import load_assessments


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "results/gemma-prompt-variety.json"
ASSESSMENTS = ROOT / "results/variety-blind-review.json"
ARMS = ("stock", "rebuilt")
GRADES = ("review_semantic", "review_strict", "auto_semantic", "auto_strict")
COHORTS = {
    "review_strict": "Reviewed, fully correct",
    "same_tokens": "Reviewed, fully correct and identical output tokens",
    "auto_strict": "Frozen automatic strict grade",
}


def blind_records(data):
    """Export all distinct non-warmup replies, including any abandoned attempts."""
    fixtures = data["protocol"]["workloads"]
    distinct = {}
    for block in data.get("abandoned_blocks", []) + data["blocks"]:
        for arm in block["arms"].values():
            for task, sample in arm.get("samples", {}).items():
                fixture = fixtures[task]
                key = answer_id(fixture["prompt"], sample["answer"])
                entry = {"id": key, "task": task, "category": fixture["category"],
                         "prompt": fixture["prompt"], "rubric": fixture["rubric"],
                         "format": fixture["format"], "answer": sample["answer"]}
                if key in distinct and distinct[key] != entry:
                    raise ValueError("Answer ID corresponds to conflicting prompts or rubrics")
                distinct[key] = entry
    return [distinct[key] for key in sorted(distinct)]


def validate(data, reviews):
    fixtures = data["protocol"]["workloads"]
    blocks = data["blocks"]
    if data["model"] != "gemma" or data["schema_version"] != 1 or data["status"] != "complete":
        raise ValueError("The final report requires a completed Gemma variety run")
    categories = Counter(item["category"] for item in fixtures.values())
    if len(categories) != 4 or set(categories.values()) != {3}:
        raise ValueError("Expected three prompts in each of four categories")
    if len(blocks) != 4 or [b["prompts"] for b in blocks] != data["protocol"]["blocks"]:
        raise ValueError("Observed blocks do not match the four frozen blocks")
    observed = []
    for index, block in enumerate(blocks):
        order = list(ARMS) if index % 2 == 0 else list(reversed(ARMS))
        tasks = block["prompts"]
        if (block["status"] != "complete" or block["block"] != index + 1
                or block["order"] != order or len(tasks) != 3 or set(block["arms"]) != set(ARMS)):
            raise ValueError(f"Unexpected assignments in block {index + 1}")
        for arm in block["arms"].values():
            if (arm["status"] != "complete" or arm["workload_order"] != tasks
                    or set(arm["samples"]) != set(tasks)):
                raise ValueError("Missing samples or unmatched task orders")
        observed.extend(tasks)
    if len(observed) != len(set(observed)) or set(observed) != set(fixtures):
        raise ValueError("Measured prompts differ from the twelve distinct frozen prompts")
    ids = {r["id"]: r["task"] for r in blind_records(data)}
    if ids != {key: review["task"] for key, review in reviews.items()}:
        raise ValueError("Assessments must cover exactly the distinct measured answers and task IDs")


def judged(sample, fixture, reviews):
    ident = answer_id(fixture["prompt"], sample["answer"])
    review = reviews[ident]
    if not good_number(sample["complete_s"]):
        raise ValueError("Missing positive wall time for a completed request")
    if any(type(sample[k]) is not bool for k in ("semantic_correct", "instruction_pass", "strict_correct")):
        raise ValueError("Missing frozen automatic grade")
    if sample["strict_correct"] != (sample["semantic_correct"] and sample["instruction_pass"]):
        raise ValueError("Inconsistent frozen strict grade")
    code_fix = fixture["category"] == "code_fix"
    test_gate = sample["semantic_correct"] if code_fix else None
    strict = review["semantic_correct"] and review["instruction_pass"] and test_gate is not False
    metrics = {k: sample[k] for k in ("ttft_s", "complete_s", "decode_tps", "output_tokens",
                                      "prompt_tokens", "stop_type", "budget_reached", "prompt_truncated")}
    return {**metrics, "answer_id": ident, "review_semantic": review["semantic_correct"],
            "review_format": review["instruction_pass"], "functional_gate": test_gate,
            "review_strict": strict, "review_reason": review["reason"], "auto_semantic": sample["semantic_correct"],
            "auto_format": sample["instruction_pass"], "auto_strict": sample["strict_correct"],
            "auto_reason": sample["reason"]}


def case_row(block, task, fixture, reviews):
    left, right = (block["arms"][arm]["samples"][task] for arm in ARMS)
    paired_input = bool(left["prompt_sha256"]) and left["prompt_sha256"] == right["prompt_sha256"]
    truncated = left["prompt_truncated"] or right["prompt_truncated"]
    same_tokens = bool(left["token_ids"]) and left["token_ids"] == right["token_ids"]
    cases = {arm: judged(s, fixture, reviews) for arm, s in zip(ARMS, (left, right))}
    included = {}
    for key in COHORTS:
        grade = "review_strict" if key == "same_tokens" else key
        included[key] = (paired_input and not truncated and all(c[grade] for c in cases.values())
                         and (key != "same_tokens" or same_tokens))
    return {"task": task, "category": fixture["category"], "block": block["block"], "first": block["order"][0],
            "prompt_hash_match": paired_input, "prompt_truncated": truncated, "same_output_token_ids": same_tokens,
            "same_output_text": left["answer"] == right["answer"], "arms": cases, "included": included,
            "observed_time_saved_pct": 100 * (left["complete_s"] - right["complete_s"]) / left["complete_s"]}


def cohort_stats(rows, key):
    matched = [r for r in rows if r["included"][key]]
    deltas = [r["observed_time_saved_pct"] for r in matched]
    return {"n": len(matched), "prompt_ids": [r["task"] for r in matched],
            "stock_median_s": median(r["arms"]["stock"]["complete_s"] for r in matched) if matched else None,
            "rebuilt_median_s": median(r["arms"]["rebuilt"]["complete_s"] for r in matched) if matched else None,
            "median_saved_pct": median(deltas) if matched else None, "min_saved_pct": min(deltas, default=None),
            "max_saved_pct": max(deltas, default=None), "same_output_ids": sum(r["same_output_token_ids"] for r in matched)}


def pass_counts(rows):
    return {arm: {grade: sum(r["arms"][arm][grade] for r in rows) for grade in GRADES} for arm in ARMS}


def summarize(data, reviews):
    validate(data, reviews)
    fixtures = data["protocol"]["workloads"]
    rows = [case_row(b, task, fixtures[task], reviews) for b in data["blocks"] for task in b["prompts"]]
    categories = {}
    for name in dict.fromkeys(r["category"] for r in rows):
        group = [r for r in rows if r["category"] == name]
        categories[name] = {"prompts": len(group), "passes": pass_counts(group), "primary": cohort_stats(group, "review_strict")}
    abandoned = data.get("abandoned_blocks", [])
    return {"schema_version": 1, "run_id": data["run_id"], "prompt_count": len(rows),
            "distinct_reviewed_outputs": len(reviews), "abandoned_blocks": len(abandoned),
            "abandoned_replies": sum(len(a.get("samples", {})) for b in abandoned for a in b["arms"].values()),
            "passes": pass_counts(rows), "cohorts": {key: cohort_stats(rows, key) for key in COHORTS},
            "categories": categories, "prompts": rows}


def num(value):
    return "—" if value is None else f"{value:.2f}"


def signed(value):
    return "—" if value is None else f"{value:+.1f}%"


def mark(flag):
    return "pass" if flag else "fail"


def both(row, field, fmt=num):
    return " / ".join(fmt(row["arms"][arm][field]) for arm in ARMS)


def render(summary, links):
    total = summary["prompt_count"]
    primary = summary["cohorts"]["review_strict"]
    n = primary["n"]
    lines = ["# Gemma on 12 new prompts", ""]
    if n:
        lines += [f"Both builds produced a fully correct answer on **{n} of {total} prompts** with matched input. "
                  f"On those prompts the stock median was **{num(primary['stock_median_s'])} s** and the rebuilt median "
                  f"was **{num(primary['rebuilt_median_s'])} s**. The median paired time saved was "
                  f"**{signed(primary['median_saved_pct'])}**, ranging from {signed(primary['min_saved_pct'])} "
                  f"to {signed(primary['max_saved_pct'])}. {primary['same_output_ids']} of the {n} "
                  f"pairs had identical output token IDs.", ""]
    else:
        lines += ["No prompt produced two fully correct answers on matched inputs. The primary latency estimate is unavailable.", ""]
    lines += ["The prompts cover explanations, code comprehension, document extraction and code fixes. "
              "Each build answered each distinct prompt once on one Intel MacBook Pro. Four blocks used eight fresh servers, "
              "with one unrelated warmup and three measured requests per server. The percentages describe this designed set "
              "of prompts. Requests from the same block share a server launch. A positive saving means a shorter wait. "
              "Percentages are calculated within each prompt before taking the median.", "",
              "## Accuracy and timing", "",
              "| Build | Reviewed semantic | Reviewed strict, including functional checks | Automatic semantic | Automatic strict |",
              "|---|---:|---:|---:|---:|"]
    for arm in ARMS:
        vals = summary["passes"][arm]
        lines.append("| " + " | ".join([arm.title()] + [f"{vals[k]}/{total}" for k in GRADES]) + " |")
    lines += ["", "The separate AI review used distinct answers in hash order with build, timing and frozen grades hidden. "
              "Reviewed strict requires both semantic and format passes. It also requires the frozen functional test to pass for code fixes. "
              "The observed timings below include failed answers. The paired comparisons require identical, untruncated model input.", "",
              "| Eligible prompts | Pairs | Stock median (s) | Rebuilt median (s) | Median paired saving | Min to max | Identical output IDs |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for key, label in COHORTS.items():
        c = summary["cohorts"][key]
        lines.append(f"| {label} | {c['n']}/{total} | {num(c['stock_median_s'])} | {num(c['rebuilt_median_s'])} | "
                     f"{signed(c['median_saved_pct'])} | {signed(c['min_saved_pct'])} to {signed(c['max_saved_pct'])} | "
                     f"{c['same_output_ids']}/{c['n']} |")
    lines += ["", "## By category", "", "| Category | Stock strict | Rebuilt strict | Correct pairs | Median paired saving |",
              "|---|---:|---:|---:|---:|"]
    for name, group in summary["categories"].items():
        a, b = (group["passes"][arm]["review_strict"] for arm in ARMS)
        c = group["primary"]
        lines.append(f"| {name.replace('_', ' ')} | {a}/{group['prompts']} | {b}/{group['prompts']} | "
                     f"{c['n']}/{group['prompts']} | {signed(c['median_saved_pct'])} |")
    lines += ["", "## All measured replies", "", "Paired cells read **stock / rebuilt**. Time to first text and full request "
              "time include native chat formatting. Decode speed comes from the engine. Output tokens and timings include every "
              "recorded reply in the four retained blocks.", "",
              "| Block, first build | Prompt | Reviewed semantic | Reviewed strict | Auto strict | First text (s) | Complete (s) | "
              "Decode (tokens/s) | Output tokens | Identical IDs | Primary saving |",
              "|---|---|---|---|---|---:|---:|---:|---:|---|---:|"]
    for r in summary["prompts"]:
        eligible = r["included"]["review_strict"]
        lines.append(f"| {r['block']}, {r['first']} | `{r['task']}` | {both(r,'review_semantic',mark)} | "
                     f"{both(r,'review_strict',mark)} | {both(r,'auto_strict',mark)} | {both(r,'ttft_s')} | "
                     f"{both(r,'complete_s')} | {both(r,'decode_tps')} | {both(r,'output_tokens',str)} | "
                     f"{'yes' if r['same_output_token_ids'] else 'no'} | "
                     f"{signed(r['observed_time_saved_pct']) if eligible else '—'} |")
    lines += ["", f"The study retained {total} prompts per build. It recorded {summary['abandoned_blocks']} abandoned block attempts "
              f"containing {summary['abandoned_replies']} additional replies. Those attempts are stored in the raw data "
              "and excluded from the twelve-prompt comparisons.", "",
              f"[Frozen protocol](VARIETY.md) · [Raw measured requests]({links['raw']}) · "
              f"[Blind review decisions]({links['assessments']}) · [Per-prompt results](results/variety-summary.json)", ""]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    cmds = p.add_subparsers(dest="operation", required=True)
    export = cmds.add_parser("export", help="extract distinct replies even from a partial run")
    export.add_argument("--raw", type=Path, default=RAW)
    export.add_argument("--out", type=Path, help="defaults to standard output")
    report = cmds.add_parser("report", help="join a complete run with independently graded answers")
    report.add_argument("--raw", type=Path, default=RAW)
    report.add_argument("--assessments", type=Path, default=ASSESSMENTS)
    report.add_argument("--root", type=Path, default=ROOT, help="output root for markdown and results/")
    args = p.parse_args()
    try:
        data = json.loads(args.raw.read_text(encoding="utf-8"))
        if args.operation == "export":
            encoded = json.dumps(blind_records(data), ensure_ascii=False, indent=2)
            if args.out:
                args.out.write_text(encoded + "\n", encoding="utf-8")
            else:
                print(encoded)
            return
        summary = summarize(data, load_assessments(args.assessments))
        links = {key: os.path.relpath(getattr(args, key).resolve(), args.root.resolve())
                 for key in ("raw", "assessments")}
        body = render(summary, links)
        (args.root / "results").mkdir(parents=True, exist_ok=True)
        (args.root / "results/variety-summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.root / "VARIETY-RESULTS.md").write_text(body, encoding="utf-8")
    except (OSError, KeyError, ValueError, TypeError) as err:
        p.exit(1, str(err) + "\n")


if __name__ == "__main__":
    main()
