import torch
import os
import sys
from transformers import AutoConfig, AutoModelForSeq2SeqLM

LIVE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(LIVE_ROOT, "t5gemma-2-1b-1b")

print(f"Loading config from {MODEL_PATH}")
config = AutoConfig.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("Loading model (CPU)...")
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=False)

print("\n--- Model Modules ---")
for name, _ in model.named_modules():
    if "." not in name or name.count(".") < 2:
        print(name)

print("\n--- Meta Check ---")
meta_params = [n for n, p in model.named_parameters() if p.device.type == "meta"]
if meta_params:
    print(f"FOUND {len(meta_params)} META PARAMETERS:")
    for p in meta_params[:10]:
        print(f"  - {p}")
else:
    print("No meta parameters found on CPU load.")

print("\n--- Device ---")
print(f"Model device: {model.device}")
