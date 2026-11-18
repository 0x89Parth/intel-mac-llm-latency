# Independent review of generated answers

A separate AI reviewer assessed each distinct measured answer with the prompt visible and the model, build, timing and automatic grade hidden. Semantic correctness checks the substance. Full compliance requires both a correct answer and adherence to the requested instructions. The frozen automatic checks accept a single surrounding Markdown fence for JSON and Python. A consumer requiring bare JSON or Python can reject those responses. The fixed 144-token decode control has its own budget and prefix checks.

[Blinded export](blind_answers.py) · [Review decisions](results/blind-review.json) · [Predeclared primary analysis](REPORT.md)

## Gemma 4 26B-A4B

| Task | Build | Graded | Frozen correct | Semantically correct | Fully compliant |
|---|---|---:|---:|---:|---:|
| First answer after launch | Stock | 5 | 5 | 5 | 5 |
| First answer after launch | Rebuilt | 5 | 5 | 5 | 5 |
| First answer after launch | Tuned | 5 | 5 | 5 | 5 |
| Fresh explanation | Stock | 5 | 5 | 5 | 5 |
| Fresh explanation | Rebuilt | 5 | 5 | 5 | 5 |
| Fresh explanation | Tuned | 5 | 5 | 5 | 5 |
| Release handover | Stock | 5 | 5 | 5 | 0 |
| Release handover | Rebuilt | 5 | 5 | 5 | 0 |
| Release handover | Tuned | 5 | 5 | 5 | 0 |
| Python code edit | Stock | 5 | 5 | 5 | 0 |
| Python code edit | Rebuilt | 5 | 5 | 5 | 0 |
| Python code edit | Tuned | 5 | 5 | 5 | 0 |

Semantic and frozen grades disagreed on **0** measured replies. The counts include every returned answer, including any abandoned attempts.

**Sensitivity to the grading rule.** Each cell lists the number of matched correct pairs out of planned rounds and the median [minimum, maximum] percentage of elapsed time saved. A positive number means a shorter wait. The table covers tasks where a different rule selected different pairs. Cold time starts at process launch. Warm time starts at the client request.

| Task | Comparison | Frozen automatic | Semantic only | Full compliance |
|---|---|---|---|---|
| Release handover | Stock → Rebuilt | 5/5, +68.53% [+66.92, +70.78] | 5/5, +68.53% [+66.92, +70.78] | 0/5, no estimate |
| Release handover | Rebuilt → Tuned | 5/5, +4.88% [+0.27, +6.17] | 5/5, +4.88% [+0.27, +6.17] | 0/5, no estimate |
| Python code edit | Stock → Rebuilt | 5/5, +57.33% [+52.59, +62.95] | 5/5, +57.33% [+52.59, +62.95] | 0/5, no estimate |
| Python code edit | Rebuilt → Tuned | 5/5, -1.31% [-11.66, +9.56] | 5/5, -1.31% [-11.66, +9.56] | 0/5, no estimate |

## Qwen2.5-Coder 0.5B

| Task | Build | Graded | Frozen correct | Semantically correct | Fully compliant |
|---|---|---:|---:|---:|---:|
| First answer after launch | Stock | 5 | 0 | 0 | 0 |
| First answer after launch | Rebuilt | 5 | 0 | 0 | 0 |
| Fresh explanation | Stock | 5 | 0 | 0 | 0 |
| Fresh explanation | Rebuilt | 5 | 0 | 0 | 0 |
| Release handover | Stock | 5 | 5 | 5 | 0 |
| Release handover | Rebuilt | 5 | 5 | 5 | 0 |
| Python code edit | Stock | 5 | 5 | 5 | 0 |
| Python code edit | Rebuilt | 5 | 5 | 5 | 0 |

Semantic and frozen grades disagreed on **0** measured replies. The counts include every returned answer, including any abandoned attempts.

**Sensitivity to the grading rule.** Each cell lists the number of matched correct pairs out of planned rounds and the median [minimum, maximum] percentage of elapsed time saved. A positive number means a shorter wait. The table covers tasks where a different rule selected different pairs. Cold time starts at process launch. Warm time starts at the client request.

| Task | Comparison | Frozen automatic | Semantic only | Full compliance |
|---|---|---|---|---|
| Release handover | Stock → Rebuilt | 5/5, +0.04% [-1.02, +1.87] | 5/5, +0.04% [-1.02, +1.87] | 0/5, no estimate |
| Python code edit | Stock → Rebuilt | 5/5, -0.41% [-3.81, +0.82] | 5/5, -0.41% [-3.81, +0.82] | 0/5, no estimate |

## Reasons for individual answers

The review covers all **7** distinct prompt-and-answer combinations for the listed models. The identifiers below are prefixes of the SHA-256 values in the public review JSON. Identical replies to the same prompt share one assessment.

| Task | Answer ID | Semantic | Instructions | Reviewer’s reason |
|---|---|---|---|---|
| First answer after launch | `59314300dfe1` | Fail | Pass | The reply contains just an integer, but eight minus three is five. |
| First answer after launch | `f3998e039415` | Pass | Pass | Five slots are free, and the response is the single integer 5. |
| Fresh explanation | `9a472ca6b2a7` | Fail | Fail | It gives 7 as the default but omits both the absence of insertion and the subsequent KeyError. It exceeds two sentences and ends mid-sentence. |
| Fresh explanation | `acf7f336bd02` | Pass | Pass | The two sentences give the default 7, explain that the dictionary stays empty, and identify KeyError on lookup. |
| Release handover | `f6f164c14255` | Pass | Fail | All three keys and values match the final Atlas record. The Markdown fence prevents the whole reply from being a JSON object. |
| Python code edit | `0d3aab0daee9` | Pass | Fail | The signature supports positional and keyword calls, and the conditional clamps the difference at zero. The Markdown fence adds text outside the requested function. |
| Python code edit | `bff16ba1a076` | Pass | Fail | The signature is intact and both return paths give the clamped difference. Markdown fencing adds text outside the complete function. |
