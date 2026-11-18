# Time to a correct local LLM answer on an Intel Mac

How much sooner does a correct answer arrive after rebuilding the inference engine for a 2019 Intel MacBook Pro? This project compares the released ToshLLM v0.88.0 Intel binary with a local rebuild on the same Core i9-9880H and Radeon Pro 560X. Gemma 4 26B-A4B and Qwen2.5-Coder 0.5B are evaluated separately with their respective GGUF files.

The earlier fixed-sentence test measured Gemma decoding at 1.72 versus 4.14 output tokens/s. This study measures answer latency and correctness on additional workloads. [Earlier throughput experiment](https://github.com/0x89Parth/intel-mac-llm-bench/blob/944e7c7083cc854425ca4aaece1f2d15c9e975d0/README.md)

## Measurement plan

- Use five paired rounds per model. Each arm starts a fresh server. Alternate which build runs first and give each arm the same scheduled cooldown. Measure requests sequentially on AC power and record the reported CPU speed limit and memory conditions.
- Fix the GGUF, prompt, model context, output budget, seed, threads and GPU layers within each stock/rebuild comparison. The primary comparison uses three CPU threads for both models, three Radeon layers for Gemma and full GPU offload for Qwen. The comparison estimates the effect of switching between the two complete builds. The rebuild enables AVX2, FMA, F16C and BMI2 together.
- Measure launch to the first completed answer with a common cold probe. Cold refers to starting a new process and loading a model. macOS manages the filesystem cache throughout the experiment.
- On each loaded server, time a fresh explanation, a longer document question, a code edit checked against tests, and a fixed-length decode control. Record the time to first visible text, time to completion, output token count, engine-reported decode throughput and the actual answer.
- For each model and workload, report correct answers out of all trials and the median and range of paired time savings. For each pair where both answers pass, the time saved is `100 × (stock_seconds − rebuilt_seconds) / stock_seconds`. Keep incorrect answers and missing pairs visible. A task with no mutually correct pair has no time-to-correct speed estimate.
- Inspect output lengths and grade the explanations against prewritten content requirements. On the fixed-length task, check conformity to the requested text and token budget. Multiple requests within a server launch count as one round for each workload.
- Profile a disposable Gemma process and screen nearby thread and GPU-layer settings on separate tuning prompts. A selected configuration must face different prompts before claiming a tuning gain. Keep prompt lookup disabled for the primary rebuild comparison.

## Machine

2019 MacBook Pro, Intel Core i9-9880H, 16 GB RAM, Radeon Pro 560X with 4 GB VRAM, macOS Ventura 13.6.5. One machine limits conclusions about other systems.
