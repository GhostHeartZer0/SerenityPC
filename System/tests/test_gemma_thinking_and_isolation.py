# Unit test verifying Gemma thinking enablement and cross-model compatibility.
import llama_cpp
from llama_cpp import llama_chat_format

def test_gemma_native_thinking_enabled():
    """Verify that enable_thinking=True on Gemma-4 allows native reasoning without closing empty thought block."""
    model_path = "S:/LLM/gemma/gemma-4-unc/26B/SC117/gemma-4-26B-A4B-it-qat-heretic-UD-Q4_K_XL.gguf"
    m = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
    m.chat_format = "chat_template.default"

    prompts = []
    orig_tokenize = m.tokenize
    def mock_tokenize(text, **kwargs):
        prompts.append(text.decode("utf-8") if isinstance(text, bytes) else text)
        return orig_tokenize(text, **kwargs)
    m.tokenize = mock_tokenize

    try:
        m.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are Serenity."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            chat_template_kwargs={"enable_thinking": True}
        )
    except Exception:
        pass

    assert len(prompts) > 0
    p = prompts[0]
    # Verify <|think|> is at the top of system turn
    assert "<|think|>" in p
    # Verify the empty thought channel is NOT closed before generation
    assert "<|channel>thought\n<channel|>" not in p
    assert p.endswith("<|turn>model\n")

def test_gemma_native_thinking_disabled():
    """Verify that enable_thinking=False on Gemma-4 properly suppresses reasoning channel."""
    model_path = "S:/LLM/gemma/gemma-4-unc/26B/SC117/gemma-4-26B-A4B-it-qat-heretic-UD-Q4_K_XL.gguf"
    m = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
    m.chat_format = "chat_template.default"

    prompts = []
    orig_tokenize = m.tokenize
    def mock_tokenize(text, **kwargs):
        prompts.append(text.decode("utf-8") if isinstance(text, bytes) else text)
        return orig_tokenize(text, **kwargs)
    m.tokenize = mock_tokenize

    try:
        m.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are Serenity."},
                {"role": "user", "content": "What is 2+2?"}
            ],
            chat_template_kwargs={"enable_thinking": False}
        )
    except Exception:
        pass

    assert len(prompts) > 0
    p = prompts[0]
    # When disabled, template closes thought channel immediately
    assert "<|channel>thought\n<channel|>" in p

def test_non_gemma_model_compatibility():
    """Verify other model families (e.g. Olmo-3-Think) ignore enable_thinking and keep their native thinking tokens."""
    model_path = "S:/LLM/Olmo-3-7B-Think-UD-Q4_K_XL.gguf"
    m = llama_cpp.Llama(model_path=model_path, vocab_only=True, verbose=False)
    m.chat_format = "chat_template.default"

    prompts = []
    orig_tokenize = m.tokenize
    def mock_tokenize(text, **kwargs):
        prompts.append(text.decode("utf-8") if isinstance(text, bytes) else text)
        return orig_tokenize(text, **kwargs)
    m.tokenize = mock_tokenize

    try:
        m.create_chat_completion(
            messages=[
                {"role": "system", "content": "You are Serenity."},
                {"role": "user", "content": "Hello"}
            ],
            chat_template_kwargs={"enable_thinking": True}
        )
    except Exception:
        pass

    assert len(prompts) > 0
    p = prompts[0]
    # Qwen still emits <think>
    assert "<think>" in p
