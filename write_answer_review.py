#!/usr/bin/env python3
"""Join blinded assessments to the measured answers and check sensitivity to the grading rule."""

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

from analyze_results import ARM_NAMES, MODELS, TASKS, comparison, spread
from blind_answers import answer_id


REVIEW_TASKS = tuple(key for key in TASKS if key != "fixed_decode")


def load_assessments(path):
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("assessments must be a JSON array of blinded decisions")
    assessments = {}
    required = ("id", "task", "semantic_correct", "instruction_pass", "reason")
    for record in records:
        if not isinstance(record, dict) or not all(key in record for key in required):
            raise ValueError("each assessment needs " + ", ".join(required))
        ident = record["id"]
        if not isinstance(ident, str) or len(ident) != 64 or any(x not in "0123456789abcdef" for x in ident):
            raise ValueError("invalid SHA-256 answer ID")
        if ident in assessments:
            raise ValueError(f"duplicate assessment for {ident}")
        if any(type(record[k]) is not bool for k in ("semantic_correct", "instruction_pass")):
            raise ValueError(f"assessment {ident} must use JSON booleans")
        if not isinstance(record["reason"], str) or not record["reason"].strip():
            raise ValueError(f"missing review reason for {ident}")
        assessments[ident] = {k: record[k] for k in required}
    return assessments


def review_model(data, assessments, seen):
    n = data["protocol"]["rounds"]
    pairs = data["pairs"]
    if (data.get("status") != "complete" or sorted(p["round"] for p in pairs) != list(range(1, n + 1))
            or any(p.get("status") != "complete" for p in pairs)):
        raise ValueError(f"{data['model']} has incomplete rounds; cannot finalize the review")
    arms = ["stock", "rebuilt"] + (["tuned"] if data["protocol"].get("candidate_server_args") else [])
    for pair in pairs:
        for arm in arms:
            run = pair.get("arms", {}).get(arm, {})
            if run.get("status") != "complete" or not set(REVIEW_TASKS).issubset(run.get("samples", {})):
                raise ValueError(f"{data['model']} round {pair['round']} lacks a complete {arm} arm")
    semantic, strict = deepcopy(pairs), deepcopy(pairs)
    abandoned = data.get("abandoned_pairs", [])
    counts = {task: {arm: Counter() for arm in arms} for task in REVIEW_TASKS}
    for index, pair in enumerate(pairs + abandoned):
        for arm, run in pair["arms"].items():
            if arm not in arms:
                raise ValueError(f"unrecognized arm {arm!r}")
            for task, sample in run.get("samples", {}).items():
                if task == "fixed_decode":
                    continue
                prompt = data["protocol"]["workloads"][task]["prompt"]
                ident = answer_id(prompt, sample["answer"])
                review = assessments.get(ident)
                if not review or review["task"] != task:
                    raise ValueError(f"missing or mismatched assessment for {task} {ident}")
                seen.add(ident)
                if type(sample.get("correct")) is not bool:
                    raise ValueError(f"missing frozen grade for {ident}")
                correct = review["semantic_correct"]
                compliant = correct and review["instruction_pass"]
                row = counts[task][arm]
                row.update(graded=1, frozen=sample["correct"], semantic=correct, strict=compliant)
                row["semantic_disagreement"] += correct != sample["correct"]
                if index < len(pairs):
                    semantic[index]["arms"][arm]["samples"][task]["correct"] = correct
                    strict[index]["arms"][arm]["samples"][task]["correct"] = compliant
    eligibility = []
    for task in REVIEW_TASKS:
        key = "cold_to_complete_s" if task == "cold" else "complete_s"
        for left, right in zip(arms, arms[1:]):
            summaries = [comparison(p, task, left, right) for p in (pairs, semantic, strict)]
            ids = [s["both_correct_rounds"] for s in summaries]
            if not (ids[0] == ids[1] == ids[2]):
                estimates = []
                for s in summaries:
                    saving = s["metrics"][key]["improvement_pct"]
                    estimates.append(f"{saving['n']}/{n}, " + (spread(saving) if saving["n"] else "no estimate"))
                eligibility.append((task, f"{ARM_NAMES[left]} → {ARM_NAMES[right]}", *estimates))
    return counts, eligibility


def render(reports, assessments):
    lines = ["# Independent review of generated answers", "",
             "A separate AI reviewer assessed each distinct measured answer with the prompt visible and the model, "
             "build, timing and automatic grade hidden. Semantic correctness checks the substance. Full compliance "
             "requires both a correct answer and adherence to the requested instructions. The frozen automatic checks "
             "accept a single surrounding Markdown fence for JSON and Python. A consumer requiring bare JSON or Python "
             "can reject those responses. The fixed 144-token decode control has its own budget and prefix checks.",
             "", "[Blinded export](blind_answers.py) · [Review decisions](results/blind-review.json) · "
             "[Predeclared primary analysis](REPORT.md)", ""]
    seen = set()
    for slug, data in reports.items():
        counts, changes = review_model(data, assessments, seen)
        lines += ["## " + MODELS[slug], "",
                  "| Task | Build | Graded | Frozen correct | Semantically correct | Fully compliant |",
                  "|---|---|---:|---:|---:|---:|"]
        for task, arms in counts.items():
            for arm, row in arms.items():
                lines.append(f"| {TASKS[task]} | {ARM_NAMES[arm]} | {row['graded']} | {row['frozen']} | "
                             f"{row['semantic']} | {row['strict']} |")
        different = sum(row["semantic_disagreement"] for by_arm in counts.values() for row in by_arm.values())
        lines += ["", f"Semantic and frozen grades disagreed on **{different}** measured replies. "
                  "The counts include every returned answer, including any abandoned attempts.", ""]
        if changes:
            lines += ["**Sensitivity to the grading rule.** Each cell lists the number of matched correct pairs out of "
                      "planned rounds and the median [minimum, maximum] percentage of elapsed time saved. A positive "
                      "number means a shorter wait. The table covers tasks where a different rule selected different "
                      "pairs. Cold time starts at process launch. Warm time starts at the client request.", "",
                      "| Task | Comparison | Frozen automatic | Semantic only | Full compliance |",
                      "|---|---|---|---|---|"]
            for task, *cols in changes:
                lines.append("| " + " | ".join([TASKS[task], *cols]) + " |")
            lines.append("")
        else:
            lines += ["All three rules selected the same eligible paired rounds for these tasks.", ""]
    extra = set(assessments) - seen
    if extra:
        raise ValueError(f"{len(extra)} assessment(s) have no matching measured answer")
    lines += ["## Reasons for individual answers", "",
              f"The review covers all **{len(seen)}** distinct prompt-and-answer combinations for the listed models. "
              "The identifiers below are prefixes of the SHA-256 values in the public review JSON. Identical replies "
              "to the same prompt share one assessment.", "",
              "| Task | Answer ID | Semantic | Instructions | Reviewer’s reason |", "|---|---|---|---|---|"]
    for r in sorted(assessments.values(), key=lambda v: (REVIEW_TASKS.index(v["task"]), v["id"])):
        reason = " ".join(r["reason"].splitlines()).replace("|", r"\|")
        lines.append(f"| {TASKS[r['task']]} | `{r['id'][:12]}` | {'Pass' if r['semantic_correct'] else 'Fail'} | "
                     f"{'Pass' if r['instruction_pass'] else 'Fail'} | {reason} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default=Path(__file__).resolve().parent, type=Path)
    p.add_argument("--assessments", required=True, type=Path, help="blinded decision array with id, task, semantic_correct, instruction_pass, reason")
    p.add_argument("--models", choices=MODELS.keys(), nargs="+", default=list(MODELS), help="require complete reports for each selected model")
    args = p.parse_args()
    try:
        assessments = load_assessments(args.assessments)
        reports = {}
        for slug in args.models:
            path = args.root / "results" / f"correct-answers-{slug}.json"
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("model") != slug or data.get("schema_version") != 1:
                raise ValueError(f"unexpected model or schema in {path}")
            if slug in reports:
                raise ValueError(f"model listed twice: {slug}")
            reports[slug] = data
        markdown = render(reports, assessments)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        p.error(str(exc))
    decisions = [assessments[key] for key in sorted(assessments)]
    (args.root / "ANSWER-REVIEW.md").write_text(markdown, encoding="utf-8")
    (args.root / "results/blind-review.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Published {len(decisions)} review decisions in {args.root}")


if __name__ == "__main__":
    main()
