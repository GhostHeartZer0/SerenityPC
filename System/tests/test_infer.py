import json
from llama_cpp import Llama

model_path = "S:/SerenityPC/Models/gemma-4-E4B/gemma-4-E4B-it-UD-Q4_K_XL.gguf"
print(f"Loading {model_path}...")
model = Llama(
    model_path=model_path,
    n_gpu_layers=10,
    n_ctx=4096,
    verbose=False
)

sys_msg = "You are Serenity Lvl 3. You focus on projects, collaboration, and debating when essential. Be intelligent as to when to help and how. Get details right. Better memory and flexible response length."
sys_msg += "\n[CRITICAL RESTRICTION]: You will begin in the thought channel. When your internal reasoning is complete, you MUST explicitly output '<channel|>' to close the thought process and begin your final response."

prompt_str = f"<|turn>system\n<|think|>\n{sys_msg}<turn|>\n<|turn>user\nHelp me devise a plan to make money nowadays.<turn|>\n<|turn>model\n<|channel>thought\n"

print("Generating...")
stream = model(prompt_str, stream=True, echo=False, max_tokens=1024, stop=["<turn|>", "<|turn>", "<|file_separator|>", "<eos>"])

output = ""
for chunk in stream:
    c = chunk['choices'][0]['text']
    print(c, end='', flush=True)
    output += c

print("\n\n--- TEST COMPLETE ---")
print(f"Contains <channel|>? {'<channel|>' in output}")
