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
prompt_str = f"<|turn>system\n<|think|>\n{sys_msg}<turn|>\n<|turn>user\nhelp me sleep...<turn|>\n<|turn>model\n<|channel>thought\n"

stream = model(prompt_str, stream=True, echo=False, max_tokens=1024, stop=["<turn|>", "<|turn>", "<|file_separator|>", "<eos>", "\n\n\n"])

output = ""
for chunk in stream:
    c = chunk['choices'][0]['text']
    output += c

print(f"Output length: {len(output)}")
print(f"Output: {repr(output)}")
