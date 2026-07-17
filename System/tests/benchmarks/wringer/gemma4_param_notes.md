# Gemma-4 Parameters & Tuning Notes

This document summarizes findings on optimal parameters for the **Gemma-4** model family to maximize accuracy in coding, reasoning, and instruction-following, while minimizing the probability of response skips or generation failures.

---

## 1. Parameter Presets for Coding & Reasoning

Gemma-4 is a highly sensitive reasoning model. The table below lists the recommended baseline parameters vs. standard general-purpose defaults:

| Parameter | Coding / Reasoning Preset | General Default | Rationale |
|---|---|---|---|
| **Temperature** | `0.3` to `0.5` | `1.0` | High temperature causes token divergence, resulting in broken syntax or early loop termination in code. |
| **Repeat Penalty** | `1.0` (disabled) or `1.05` | `1.3` | Code naturally contains structural repetition (e.g. `import`, `for`, `def`, brackets). A high repeat penalty penalizes these valid repetitions, causing syntax corruption or failure to complete blocks. |
| **Min P** | `0.05` to `0.10` | `0.05` | Filters out noise tokens while allowing the model to trace logical alternative branches. |
| **Top P** | `0.90` to `0.95` | `0.95` | Focuses attention on high-confidence tokens. |
| **Top K** | `50` to `64` | `64` | Traditional top-k bounds to balance sample diversity. |

---

## 2. Preventing Response Skipping (Empty Responses)

Gemma-4 models occasionally produce empty outputs or fail to respond when sampling parameters are too aggressive. To resolve this:
1. **Reduce Temperature**: Do not run reasoning models at temperatures above `0.8`.
2. **Add Strict Stop Sequences**: Ensure `<turn|>` and `<|end_of_turn|>` are explicitly appended to the `stop` list in inference requests.
3. **KV Cache & Flash Attention**: Gemma models require Flash Attention to be enabled when KV cache quantization (Q4_0 / Q8_0) is active. Failing to use Flash Attention with quantized KV cache causes SWA (Sliding Window Attention) padding logic to crash context initialization, leading to silent generation failure.
