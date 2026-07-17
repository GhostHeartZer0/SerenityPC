---
trigger: model_decision
description: only for files in Live folder, NOT ALL in SerenityPC.
---

# 🧠 Project: Serenity Live

## Hardware Guardrails

- **VRAM Cap:** 6GB total. Max allocation for AI: 2.5GB during gaming sessions.
- **RAM Pocket:** 18GB usable. Reserved for Encoder offloading and 4b "Deep Cooks."
- **Thermal Rule:** If GPU hits 82°C, throttle inference speed.

## Core Priority Logic

1. **Cormal (1b-1b):** - *Role:* Real-time response, screen-seeing, gaming advice.
   - *Hardware:* GPU Encoder (CUDA) / CPU Decoder (RAM).
2. **INTELLI-CORE (4b-4b):** - *Role:* "Deep Cook" accuracy, long-term thinking.
   - *Hardware:* CPU-Heavy. Encoder on RAM; Decoder on GPU (bnb-nf4).
3. **TROUBLESHOOTER (2b-CodeGemma):** - *Role:* Self-improvement and code-editing.
   - *Hardware:* GPU-Exclusive

## Code Style

- Use `T5Gemma2ForConditionalGeneration` for 1b and 4b.
- Use `AutoModelForCausalLM` for CodeGemma.
- Ensure all models use `bnb_config` with `nf4` and `double_quant`.
