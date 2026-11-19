# Gemma on 12 new prompts

This follow-up asks whether the earlier latency result extends beyond one repeated Python explanation. The sample comprises twelve distinct prompts, with three each for explanations, code comprehension or debugging, document extraction, and code fixes. The prompts and rubrics are frozen in Git before measurement. The fixed model and server settings come from the original primary comparison. This phase changes the workload and evaluates the two complete engines.

## Protocol

- Use the original Q4_K Gemma GGUF, stock ToshLLM v0.88.0 and the same local rebuild on the same i9 MacBook Pro. Hold context, batch, three CPU threads, three Radeon layers, seed and each prompt's token budget fixed across builds. Disable prompt lookup. Use AC power, an equal cooldown and an unrelated warmup before the measured prompts on each server.
- Test each prompt once per build. Use four blocks of three prompts. Each block starts a fresh server for each build, with stock first in blocks one and three and the rebuild first in blocks two and four. Use the same prompts in the same order within a block. This gives 24 timed requests across eight fresh server launches. The twelve prompts are a designed sample of explanation, coding and information-extraction tasks.
- Record client time to first visible text and to the complete reply, engine decode speed, prompt and output token counts, output token IDs, full answers, machine state and input hashes. Use warm request time for the headline comparison.
- Report substantive passes and strict passes out of all twelve prompts per build. Strict means the response gives the correct answer and obeys the requested output format. Publish the frozen automatic grades alongside a separate AI review of the distinct answers with build labels, timings and automatic decisions hidden.
- For the primary reviewed latency comparison, include a prompt only when both builds received identical formatted input and both replies satisfy the frozen content and format requirements in the blind review. For code fixes require a pass on the frozen functional checks as well. Report the number of qualifying pairs, both median times, the median and full range of the per-prompt percentage savings, and the number with identical output tokens. Give the same-token subset and automatic-grade analysis separately where they differ. Show unsuccessful prompts and their timings too.
- Compute each saving as `100 × (stock time − rebuilt time) / stock time`. Do not count repeated prompts from the earlier study in this sample. The sampling unit here is the distinct prompt, with one observation per build; requests sharing a server launch share its state. This study does not isolate individual instruction flags.

The prompts span Python, SQL and JavaScript behavior, short technical explanations, extraction from records, and tested Python expressions. The benchmark sends the prompt text to the model. Expected answers, rubrics and functional cases stay in the scoring code. [Frozen prompts and checks](variety_workloads.py) · [Measurement script](compare_variety.py)

With the same models and engines installed as in the [setup instructions](README.md#reproduce), this command writes a separate run file under `runs/`.

```sh
caffeinate -i python3 -Bu compare_variety.py --output runs/gemma-prompt-variety.json
```
