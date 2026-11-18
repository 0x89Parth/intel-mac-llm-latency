# Independent review of generated answers

The timed study uses the frozen correctness checks published before inference began. As a second check, a separate AI reviewer received each distinct answer and the corresponding prompt with build, model, timings, and automatic grade hidden. Answers were deduplicated and sorted by the SHA-256 of the prompt and output text. The reviewer graded the meaning of the response and strict adherence to formatting separately. The 144-token control has a fixed-prefix and token-budget check in the harness. [Blinded export tool](blind_answers.py) · [Review decisions keyed by answer hash](results/blind-review.json)

## Qwen2.5-Coder 0.5B

Across both builds and all five pairs there were four distinct responses to the four ordinary prompts. Each build consistently gave the same answer as the other. The review confirmed the substantive verdicts of the frozen grader.

| Prompt | Review finding |
|---|---|
| Cold arithmetic | Returned 2. Eight slots minus three occupied leaves 5. |
| Dictionary explanation | Mentioned the fallback value 7. Omitted the absence of insertion and the subsequent `KeyError`. The answer ended mid-sentence. |
| Release handover | Returned the correct owner, time, and rollback ticket. |
| Python edit | Preserved the callable interface and correctly clamped negative billed minutes to zero. |

The handover and code answers were wrapped in Markdown fences. The frozen graders accept a single surrounding fence. A consumer that requires bare JSON or Python would reject both builds on those tasks, leaving no time-to-correct comparison for that stricter consumer.
