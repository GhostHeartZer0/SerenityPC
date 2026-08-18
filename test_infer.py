import os
import sys
import re

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from llama_cpp import Llama
from serenity_resources import PERSONA_PROMPTS

mpath = os.path.join(script_dir, "Models", "26B-A4B", "gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf")

model = Llama(
    model_path=mpath,
    n_gpu_layers=7,
    n_ctx=2048,
    n_threads=6,
    flash_attn=True,
    verbose=False
)

sys_prompt = PERSONA_PROMPTS[7]

messages = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "What is the capital of France?"}
]

res = model.create_chat_completion(
    messages=messages,
    temperature=0.7,
    top_p=0.95,
    min_p=0.05,
    repeat_penalty=1.0,
    max_tokens=250
)

raw_answer = res["choices"][0]["message"]["content"]

# Separate thought and answer
closers = [
    r'<channel\|>', r'<\/channel\|>', r'<\/think>', r'<\|channel>text', r'<\|channel>assistant', r'\[\/DRAFT\]',
    r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
]
all_splits = []
for tag_pattern in closers:
    for m in re.finditer(tag_pattern, raw_answer, re.IGNORECASE):
        all_splits.append(m.end())

all_splits.sort()
best_split = -1
if all_splits:
    for split in all_splits:
        remaining = raw_answer[split:].strip()
        if re.search(r'<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>', remaining, re.IGNORECASE):
            continue
        best_split = split
        break
    if best_split == -1:
        best_split = all_splits[-1]

if best_split != -1:
    think_log = raw_answer[:best_split].strip()
    final_answer = raw_answer[best_split:].strip()
else:
    think_log = ""
    final_answer = raw_answer

tag_clean_pattern = r'(?i)<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\/think>|<\/thought>|\[\/DRAFT\]|<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|<\|tool_call>|<tool_call\|>|<\|tool_response>|<tool_response\|>|<\|tool>|<tool\|>|<ctrl42>|<\/ctrl42>|<\|?turn\|?>'
think_log = re.sub(tag_clean_pattern, '', think_log).strip()
final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()

print("==================================================")
print("             THOUGHT LOG (UI THOUGHT TAB)         ")
print("==================================================")
print(think_log)

print("\n==================================================")
print("             FINAL ANSWER (UI CHAT TAB)           ")
print("==================================================")
print(final_answer if final_answer else "[Awaiting additional tokens / synthesis]")
