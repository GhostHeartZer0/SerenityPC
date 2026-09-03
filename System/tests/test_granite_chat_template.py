"""
Test: Granite 4.2 Chat Template Rendering for llama.cpp / Jinja2
Covers thinking, non-thinking, low-effort modes, and history truncation.
"""
import re
import sys
from jinja2.sandbox import ImmutableSandboxedEnvironment as SandboxEnvironment

GRANITE_4_2_TEMPLATE = (
    "{%- if messages[0]['role'] == 'system' -%}"
        "{%- set system_message = messages[0]['content'] -%}"
        "{%- set loop_messages = messages[1:] -%}"
    "{%- else -%}"
        "{%- set system_message = '' -%}"
        "{%- set loop_messages = messages -%}"
    "{%- endif -%}"
    "{%- if system_message -%}"
        "{{ '<|im_start|>system\\n' + system_message + '<|im_end|>\\n' }}"
    "{%- endif -%}"
    "{%- for message in loop_messages -%}"
        "{%- set content = message['content'] -%}"
        "{%- if message['role'] == 'user' -%}"
            "{%- if loop.last and low_effort -%}"
                "{%- set content = content + ' {reasoning effort: low}' -%}"
            "{%- endif -%}"
            "{{ '<|im_start|>user\\n' + content + '<|im_end|>\\n' }}"
        "{%- elif message['role'] == 'assistant' -%}"
            "{%- if (truncate_history_thinking is not defined or truncate_history_thinking) and not loop.last -%}"
                "{%- set content = strip_thinking(content) -%}"
            "{%- endif -%}"
            "{{ '<|im_start|>assistant\\n' + content + '<|im_end|>\\n' }}"
        "{%- endif -%}"
    "{%- endfor -%}"
    "{%- if add_generation_prompt -%}"
        "{{ '<|im_start|>assistant\\n' }}"
        "{%- if enable_thinking is not defined or enable_thinking -%}"
            "{{ '<think>\\n' }}"
        "{%- else -%}"
            "{{ '<think></think>' }}"
        "{%- endif -%}"
    "{%- endif -%}"
)

def strip_thinking(text: str) -> str:
    return re.sub(r"<think>[\s\S]*?</think>\s*", "", text)

def render_granite(messages, enable_thinking=True, low_effort=False, truncate_history_thinking=True, add_generation_prompt=True):
    env = SandboxEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals["strip_thinking"] = strip_thinking
    tpl = env.from_string(GRANITE_4_2_TEMPLATE)
    return tpl.render(
        messages=messages,
        enable_thinking=enable_thinking,
        low_effort=low_effort,
        truncate_history_thinking=truncate_history_thinking,
        add_generation_prompt=add_generation_prompt
    )

def test_thinking_mode():
    msgs = [{"role": "user", "content": "Hello"}]
    out = render_granite(msgs, enable_thinking=True)
    assert out.endswith("<|im_start|>assistant\n<think>\n"), f"Failed thinking prompt: {out}"
    print("[PASS] Thinking mode generation prompt")

def test_non_thinking_mode():
    msgs = [{"role": "user", "content": "Hello"}]
    out = render_granite(msgs, enable_thinking=False)
    assert out.endswith("<|im_start|>assistant\n<think></think>"), f"Failed non-thinking prompt: {out}"
    print("[PASS] Non-thinking mode generation prompt")

def test_low_effort_mode():
    msgs = [{"role": "user", "content": "Quick math 2+2"}]
    out = render_granite(msgs, enable_thinking=True, low_effort=True)
    assert "Quick math 2+2 {reasoning effort: low}" in out
    assert out.endswith("<|im_start|>assistant\n<think>\n")
    print("[PASS] Low-effort mode parameter injection")

def test_history_truncation():
    msgs = [
        {"role": "user", "content": "Step 1"},
        {"role": "assistant", "content": "<think>\nLong internal trace\n</think>\nStep 1 result."},
        {"role": "user", "content": "Step 2"}
    ]
    out = render_granite(msgs, truncate_history_thinking=True)
    assert "Long internal trace" not in out, f"Thinking leaked in history: {out}"
    assert "Step 1 result." in out
    print("[PASS] History thinking truncation")

if __name__ == "__main__":
    test_thinking_mode()
    test_non_thinking_mode()
    test_low_effort_mode()
    test_history_truncation()
    print("\nALL GRANITE 4.2 TESTS PASSED")
