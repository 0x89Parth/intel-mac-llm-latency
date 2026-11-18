#!/usr/bin/env python3
"""Summarize paired ToshLLM answer latency. Plotting is optional."""

import argparse
import json
import math
from pathlib import Path
from statistics import median


MODELS = {"gemma": "Gemma 4 26B-A4B", "qwen": "Qwen2.5-Coder 0.5B"}
TASKS = {
    "cold": "First answer after launch", "explanation": "Fresh explanation",
    "long_prompt": "Release handover", "code_edit": "Python code edit",
    "fixed_decode": "144-token decode control",
}
WARM_ANSWERS = ("explanation", "long_prompt", "code_edit")
ARM_NAMES = {"stock": "Stock", "rebuilt": "Rebuilt", "tuned": "Tuned"}
WARM_METRICS = ("ttft_s", "complete_s", "decode_tps")
COLD_METRICS = ("cold_to_first_content_s", "cold_to_complete_s", "decode_tps")
METRIC_NAMES = {
    "ttft_s": "First text (s)", "complete_s": "Full request (s)",
    "cold_to_first_content_s": "Launch to first text (s)",
    "cold_to_complete_s": "Launch to full answer (s)", "decode_tps": "Decode (tokens/s)",
}


def distribution(values):
    if not values:
        return {"n": 0, "median": None, "min": None, "max": None}
    return {"n": len(values), "median": median(values), "min": min(values), "max": max(values)}


def good_number(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def observed(pairs, arm, task):
    samples = [pair.get("arms", {}).get(arm, {}).get("samples", {}).get(task) for pair in pairs]
    grades = [sample["correct"] for sample in samples if sample and type(sample.get("correct")) is bool]
    return {"correct": sum(grades), "graded": len(grades)}


def unfiltered(pairs, arms):
    """Describe each actual server launch and completed reply, regardless of its grade."""
    launches = {}
    replies = {}
    for arm in arms:
        runs = [pair["arms"][arm] for pair in pairs if arm in pair.get("arms", {})]
        launches[arm] = distribution([run["startup_s"] for run in runs if good_number(run.get("startup_s"))])
        replies[arm] = {}
        for task in TASKS:
            samples = [run["samples"][task] for run in runs if task in run.get("samples", {})]
            samples = [s for s in samples if type(s.get("correct")) is bool]
            keys = COLD_METRICS if task == "cold" else WARM_METRICS
            replies[arm][task] = {
                "graded": len(samples), "correct": sum(s["correct"] for s in samples),
                "metrics": {key: distribution([s[key] for s in samples if good_number(s.get(key))]) for key in keys},
            }
    return {"startup_s": launches, "replies": replies}


def comparison(pairs, task, left, right):
    keys = COLD_METRICS if task == "cold" else WARM_METRICS
    output = {"left": left, "right": right, "paired_rounds": [], "excluded_rounds": {},
              "both_correct_rounds": [], "metrics": {}}
    matched = []
    for pair in pairs:
        number = pair["round"]
        arms = [pair.get("arms", {}).get(name, {}) for name in (left, right)]
        samples = [arm.get("samples", {}).get(task) for arm in arms]
        if pair.get("status") != "complete" or not all(arm.get("status") == "complete" for arm in arms):
            reason = "incomplete pair"
        elif not all(s and type(s.get("correct")) is bool for s in samples):
            reason = "missing graded sample"
        elif not samples[0].get("prompt_sha256") or samples[0]["prompt_sha256"] != samples[1].get("prompt_sha256"):
            reason = "missing or mismatched formatted prompt"
        else:
            output["paired_rounds"].append(number)
            if all(s["correct"] for s in samples):
                matched.append((number, *samples))
                output["both_correct_rounds"].append(number)
                continue
            reason = "incorrect " + ", ".join(name for name, sample in zip((left, right), samples) if not sample["correct"])
        output["excluded_rounds"][str(number)] = reason

    for key in keys:
        values = []
        for num, a, b in matched:
            first, second = a.get(key), b.get(key)
            if not (good_number(first) and good_number(second)):
                continue
            benefit = second - first if key == "decode_tps" else first - second
            values.append({"round": num, left: first, right: second,
                           "improvement": benefit, "improvement_pct": 100 * benefit / first})
        output["metrics"][key] = {
            field: distribution([item[field] for item in values]) for field in (left, right, "improvement", "improvement_pct")
        }
        output["metrics"][key]["by_round"] = values

    with_counts = [(a, b) for _, a, b in matched if all(isinstance(x.get("output_tokens"), int) for x in (a, b))]
    with_ids = [(a, b) for _, a, b in matched if a.get("token_ids") and b.get("token_ids")]
    output["output_tokens"] = {
        left: distribution([a["output_tokens"] for a, b in with_counts]),
        right: distribution([b["output_tokens"] for a, b in with_counts]),
        "same_count": sum(a["output_tokens"] == b["output_tokens"] for a, b in with_counts),
        "count_compared": len(with_counts),
        "same_ids": sum(a["token_ids"] == b["token_ids"] for a, b in with_ids), "ids_compared": len(with_ids),
        "same_text": sum(a["answer"] == b["answer"] for _, a, b in matched),
    }
    return output


def summarize(data):
    pairs = data["pairs"]
    if len({pair["round"] for pair in pairs}) != len(pairs):
        raise ValueError(f"Duplicate retained round number in {data['run_id']}")
    abandoned = data.get("abandoned_pairs", [])
    rounds = data["protocol"]["rounds"]
    comparisons = [("stock", "rebuilt")]
    if data["protocol"].get("candidate_server_args"):
        comparisons.append(("rebuilt", "tuned"))
    complete = sum(pair.get("status") == "complete" for pair in pairs)
    model = {"run_id": data["run_id"], "source_status": data.get("status", "incomplete"),
             "planned_rounds": rounds, "completed_rounds": complete, "abandoned_attempts": len(abandoned),
             "complete": data.get("status") == "complete" and complete == rounds, "tasks": {}}
    arms = list(dict.fromkeys(a for pair in comparisons for a in pair))
    model["all_observed"] = unfiltered(pairs + abandoned, arms)
    for task in TASKS:
        grades = {}
        for arm in arms:
            retained = observed(pairs, arm, task)
            additional = observed(abandoned, arm, task)
            all_trials = {key: retained[key] + additional[key] for key in retained}
            grades[arm] = {"retained": retained, "abandoned": additional, "all": all_trials,
                           "missing_retained": max(0, rounds - retained["graded"])}
        model["tasks"][task] = {
            "correctness": grades, "comparisons": {
                f"{a}_vs_{b}": comparison(pairs, task, a, b) for a, b in comparisons},
        }
    return model


def dash(value, signed=False):
    if value is None:
        return "—"
    return f"{value:+.2f}" if signed else f"{value:.2f}"


def spread(dist):
    if not dist["n"]:
        return "—"
    return f"{dash(dist['median'], True)}% [{dash(dist['min'], True)}, {dash(dist['max'], True)}]"


def token_range(dist):
    if not dist["n"]:
        return "—"
    return str(dist["min"]) if dist["min"] == dist["max"] else f"{dist['min']}–{dist['max']}"


def write_report(summary, chart_present):
    lines = ["# Time to a correct local answer", ""]
    if chart_present:
        lines += ["![Median warm response times for matched correct answers on each MacBook workload](results/warm-latency.png)", ""]
    lines += [
        "The primary comparison uses the released Intel ToshLLM engine and the local rebuild on the same laptop. "
        "The paired latency estimates use independent rounds in which both engines passed the frozen correctness checks "
        "and received the same formatted prompt. A dash means the estimate is unavailable.",
        "",
        "A positive saving means a shorter wait. For each qualifying round, the analysis divides the time saved by "
        "that round’s baseline time, then reports the median and the full range of those percentages. "
        "The two displayed median times can give a different percentage if divided directly.", "",
    ]
    for slug, display in MODELS.items():
        info = summary["models"].get(slug)
        lines += [f"## {display}", ""]
        if info is None:
            lines += ["No measurement file is available for this model.", ""]
            continue
        status = "Complete" if info["complete"] else "**Incomplete checkpoint**"
        lines += [f"{status}. {info['completed_rounds']}/{info['planned_rounds']} planned rounds completed. "
                  f"The raw record contains {info['abandoned_attempts']} abandoned round attempts. "
                  f"[Per-request JSON](results/correct-answers-{slug}.json)", ""]
        tasks = info["tasks"]
        for comp in tasks["cold"]["comparisons"]:
            left, right = comp.split("_vs_")
            a, b = ARM_NAMES[left], ARM_NAMES[right]
            lines += [f"### {a} and {b.lower()}", "",
                      f"| Task | {a} correct/graded | {b} correct/graded | Correct pairs | {a} median (s) | "
                      f"{b} median (s) | Paired time saved, median [min, max] |",
                      "|---|---:|---:|---:|---:|---:|---:|"]
            for task, desc in TASKS.items():
                record = tasks[task]
                cmp = record["comparisons"][comp]
                grades = [record["correctness"][key]["all"] for key in (left, right)]
                key = "cold_to_complete_s" if task == "cold" else "complete_s"
                m = cmp["metrics"][key]
                lines.append(f"| {desc} | {grades[0]['correct']}/{grades[0]['graded']} | "
                             f"{grades[1]['correct']}/{grades[1]['graded']} | {len(cmp['both_correct_rounds'])}/{info['planned_rounds']} | "
                             f"{dash(m[left]['median'])} | {dash(m[right]['median'])} | {spread(m['improvement_pct'])} |")
            lines += ["", "Correct/graded includes all returned answers across retained and abandoned attempts. "
                      "Correct pairs counts complete matched rounds only. Missing requests and failed grades remain visible "
                      "in the raw data and summary JSON. A task with zero correct pairs has no time-to-correct estimate.", "",
                      f"| Task | First text (s), {a} / {b} | Decode (tokens/s), {a} / {b} | "
                      f"Output token range, {a} / {b} | Same length | Same token IDs |", "|---|---:|---:|---:|---:|---:|"]
            for task, desc in TASKS.items():
                cmp = tasks[task]["comparisons"][comp]
                first = cmp["metrics"]["cold_to_first_content_s" if task == "cold" else "ttft_s"]
                decode = cmp["metrics"]["decode_tps"]
                tok = cmp["output_tokens"]
                lines.append(f"| {desc} | {dash(first[left]['median'])} / {dash(first[right]['median'])} | "
                             f"{dash(decode[left]['median'])} / {dash(decode[right]['median'])} | "
                             f"{token_range(tok[left])} / {token_range(tok[right])} | "
                             f"{tok['same_count']}/{tok['count_compared']} | {tok['same_ids']}/{tok['ids_compared']} |")
            lines += [""]
        observed_data = info["all_observed"]
        lines += ["### Observed startup and reply times, including incorrect answers", "",
                  "Startup runs from process launch until the server reports ready.", "",
                  "| Build | Ready launches | Median startup (s) |", "|---|---:|---:|"]
        for arm, startup in observed_data["startup_s"].items():
            lines.append(f"| {ARM_NAMES[arm]} | {startup['n']} | {dash(startup['median'])} |")
        lines += ["", "The medians below use every completed reply from each build, including replies from unfinished "
                  "or abandoned rounds. Correct/graded gives the number of those replies that passed. For the first reply, "
                  "both times start at process launch. For warm workloads, both start at the client request.", "",
                  "| Task | Build | Correct/graded | Median first text (s) | Median full reply (s) | Median decode (tokens/s) |",
                  "|---|---|---:|---:|---:|---:|"]
        for task, desc in TASKS.items():
            first, full, decode = COLD_METRICS if task == "cold" else WARM_METRICS
            for arm, observations in observed_data["replies"].items():
                obs = observations[task]
                m = obs["metrics"]
                lines.append(f"| {desc} | {ARM_NAMES[arm]} | {obs['correct']}/{obs['graded']} | "
                             f"{dash(m[first]['median'])} | {dash(m[full]['median'])} | {dash(m[decode]['median'])} |")
        lines += [""]
    lines += ["## Timing and interpretation", "",
              "For the first answer after launch, the clock starts when the benchmark launches a new server process. "
              "The operating system keeps its file cache. Warm times start at the client request, before formatting the model prompt. "
              "First-text time uses the first nonempty streamed text. Full request time includes prompt processing and generation. "
              "Decode speed uses the engine’s own output-token clock.", "",
              "Output length can change even when both answers pass. The fixed-length control checks identical token budgets "
              "and the required sentence pattern. Each ordinary workload has one held-out prompt per model. Five repetitions of the same "
              "prompt measure run-to-run variability on this machine. They give no population-wide accuracy estimate or evidence "
              "isolating a single CPU instruction flag.", "", "[Paired medians, spread, and per-round changes](results/paired-summary.json)", ""]
    return "\n".join(lines)


def plot(summary, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), squeeze=False)
    fig.subplots_adjust(left=.10, right=.95, top=.70, bottom=.18, wspace=.62)
    colors = ("#41566D", "#177860")
    for ax, (slug, label) in zip(axes[0], MODELS.items()):
        model = summary["models"].get(slug)
        max_value = 0
        for i, task in enumerate(WARM_ANSWERS):
            if model:
                m = model["tasks"][task]["comparisons"]["stock_vs_rebuilt"]["metrics"]["complete_s"]
                vals = [m[name]["median"] for name in ("stock", "rebuilt")]
            else:
                vals = [None, None]
            if vals[0] is None:
                ax.text(.02, i, "No paired data" if not model else "No matched correct pairs",
                        transform=ax.get_yaxis_transform(), va="center", fontsize=9, color="#4E5960")
                continue
            max_value = max(max_value, *vals)
            for j, val in enumerate(vals):
                y = i + (-.16 if j == 0 else .16)
                ax.barh(y, val, height=.24, color=colors[j], zorder=2)
                ax.annotate(f"{val:.2f}", xy=(val, y), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=9)
        ax.set_xlim(0, (max_value or 1) * 1.3)
        ax.set_ylim(len(WARM_ANSWERS) - .45, -.55)
        ax.set_yticks(range(len(WARM_ANSWERS)), [TASKS[t] for t in WARM_ANSWERS])
        ax.set_title(label, loc="left", weight="bold", fontsize=13, pad=15)
        ax.set_xlabel("Median seconds per request")
        ax.tick_params(axis="both", length=0, pad=5, labelsize=9)
        ax.grid(axis="x", color="#D7DEE0", linewidth=.65)
        ax.spines[["right", "left", "top"]].set_visible(False)
        ax.spines["bottom"].set_color("#9AA8AB")
    fig.text(.035, .93, "How long did a correct answer take?", fontsize=21, weight="bold")
    fig.text(.035, .85, "Warm requests on the 2019 Intel MacBook Pro. Each panel uses its own time scale.", fontsize=11)
    fig.legend(handles=[Patch(facecolor=c, label=n) for c, n in zip(colors, ("Stock ToshLLM", "Local rebuild"))],
               loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(.5, .01))
    fig.savefig(path, dpi=155, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="directory containing results/; writes REPORT.md here")
    parser.add_argument("--plot", choices=("auto", "required", "off"), default="auto",
                        help="auto writes the image when matplotlib is installed")
    args = parser.parse_args()
    folder = args.root / "results"
    summary = {"schema_version": 1, "models": {}}
    for model in MODELS:
        source = folder / f"correct-answers-{model}.json"
        if source.is_file():
            data = json.loads(source.read_text(encoding="utf-8"))
            if data.get("model") != model or data.get("schema_version") != 1:
                raise ValueError(f"Unexpected schema or model in {source}")
            summary["models"][model] = summarize(data)
    if not summary["models"]:
        parser.error("no recorded models in the results directory")
    chart = False
    if args.plot != "off":
        try:
            plot(summary, folder / "warm-latency.png")
            chart = True
        except ModuleNotFoundError as exc:
            if exc.name != "matplotlib" or args.plot == "required":
                raise
            print("matplotlib unavailable; generated numerical report without a chart")
    (folder / "paired-summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (args.root / "REPORT.md").write_text(write_report(summary, chart), encoding="utf-8")
    print(f"Report written to {args.root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
