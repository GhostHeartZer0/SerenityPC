"""
Test: Chat Template Rendering and override rules for Gemma-4 Chat Handlers
"""
import sys
import os

# Simulate the exact Jinja rendering environment used by llama-cpp-python / Llava15ChatHandler
class MockLlava15ChatHandler:
    def __init__(self):
        # The exact custom CHAT_FORMAT we set on the handler for Gemma-4 models
        self.CHAT_FORMAT = (
            "{% for message in messages %}"
            "{% if message.role == 'system' %}"
            "<|turn>system\n{{ message.content }}<turn|>\n"
            "{% endif %}"
            "{% if message.role == 'user' %}"
            "<|turn>user\n"
            "{% if message.content is string %}"
            "{% if 'data:image' in message.content %}"
            "{{ message.content }}"
            "{% else %}"
            "{{ message.content }}"
            "{% endif %}"
            "{% endif %}"
            "{% if message.content is iterable %}"
            "{% for content in message.content %}"
            "{% if content.type == 'image_url' and content.image_url is string %}"
            "{{ content.image_url }}"
            "{% endif %}"
            "{% if content.type == 'image_url' and content.image_url is mapping %}"
            "{{ content.image_url.url }}"
            "{% endif %}"
            "{% endfor %}"
            "{% for content in message.content %}"
            "{% if content.type == 'text' %}"
            "{{ content.text }}"
            "{% endif %}"
            "{% endfor %}"
            "{% endif %}"
            "<turn|>\n"
            "{% endif %}"
            "{% if message.role == 'assistant' and message.content is not none %}"
            "<|turn>model\n{{ message.content }}<turn|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "<|turn>model\n"
            "{% endif %}"
        )

    def render(self, messages, add_generation_prompt=True):
        from jinja2.sandbox import ImmutableSandboxedEnvironment as SandboxEnvironment
        template = SandboxEnvironment(
            trim_blocks=True,
            lstrip_blocks=True,
        ).from_string(self.CHAT_FORMAT)
        return template.render(
            messages=messages,
            add_generation_prompt=add_generation_prompt
        )

def test_gemma4_custom_chat_format_rendering():
    print("=" * 60)
    print("TEST: Gemma-4 Custom Chat Format Rendering")
    print("=" * 60)
    
    # Test messages list with system prompt and user multimodal query
    messages = [
        {"role": "system", "content": "<|think|>\nYou are Serenity."},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,abcdef"}},
            {"type": "text", "text": "Analyze these two screenshots."}
        ]}
    ]
    
    handler = MockLlava15ChatHandler()
    rendered = handler.render(messages, add_generation_prompt=True)
    
    print("Rendered Output:")
    print("-" * 40)
    print(rendered)
    print("-" * 40)
    
    # Assertions to ensure standard Gemma-4 turns and formatting rules are satisfied
    assert "<|turn>system\n<|think|>\nYou are Serenity.<turn|>\n" in rendered, "System turn format is incorrect"
    assert "<|turn>user\ndata:image/jpeg;base64,abcdefAnalyze these two screenshots.<turn|>\n" in rendered, "User turn format is incorrect"
    assert rendered.endswith("<|turn>model\n"), "Generation prompt format is incorrect"
    print("[PASS] Chat format rendered correctly with custom templates")
    return True

if __name__ == "__main__":
    if test_gemma4_custom_chat_format_rendering():
        print("\nALL TESTS PASSED")
        sys.exit(0)
    else:
        print("\nSOME TESTS FAILED")
        sys.exit(1)
