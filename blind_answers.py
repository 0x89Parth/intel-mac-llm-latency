#!/usr/bin/env python3
"""Export distinct real answers, hiding run, build, model, timing and grader labels."""

import argparse
import hashlib
import json
from pathlib import Path


def answer_id(prompt, answer):
    return hashlib.sha256((prompt + "\0" + answer).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    args = parser.parse_args()
    records = {}
    for path in args.reports:
        data = json.loads(path.read_text())
        for pair in data["pairs"] + data.get("abandoned_pairs", []):
            for arm in pair["arms"].values():
                for name, row in arm["samples"].items():
                    if name == "fixed_decode":
                        continue
                    prompt = data["protocol"]["workloads"][name]["prompt"]
                    answer = row["answer"]
                    digest = answer_id(prompt, answer)
                    records[digest] = {"id": digest, "task": name, "prompt": prompt, "answer": answer}
    print(json.dumps([records[key] for key in sorted(records)], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
