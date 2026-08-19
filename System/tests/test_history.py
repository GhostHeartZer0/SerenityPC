import sys
from llama_cpp import Llama

model_path = "S:/SerenityPC/Models/gemma-4-26B MOE/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf"
model = Llama(
    model_path=model_path,
    n_gpu_layers=10,
    n_ctx=4096,
    verbose=False
)

sys_msg = "You are Serenity Lvl 3."
sys_msg += "\n[CRITICAL RESTRICTION]: You will begin in the thought channel. You must complete ALL planning, tone adjustments, and notes INSIDE the thought channel. When you are 100% ready to speak to the user, you MUST explicitly output '<channel|>' to close the thought process, followed immediately by your polished final response."

fake_history = ""
for i in range(10):
    fake_history += f"<|turn>user\nUser message {i}<turn|>\n<|turn>model\n<|channel>thought\nSome thoughts.\n<channel|>\nModel response {i}<turn|>\n"

prompt_str = f"<|turn>system\n<|think|>\n{sys_msg}<turn|>\n{fake_history}<|turn>user\nhelp me sleep...<turn|>\n<|turn>model\n<|channel>thought\n"

stream = model(prompt_str, stream=True, echo=False, max_tokens=1024, stop=["<turn|>", "<|turn>", "<|file_separator|>", "<eos>"])

output = ""
for chunk in stream:
    c = chunk['choices'][0]['text']
    output += c

print(f"Output length: {len(output)}")
print(f"Output: {repr(output)}")
