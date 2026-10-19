import re

def _is_thought_opening(text):
    if not text: return False
    t_lower = text.lower().strip()
    t_raw_lower = text.lower().lstrip()
    openers_exact = (
        "<think>", "<thought>", "<|think|>", "<|channel>thought", "<channel|thought>",
        "<|im_start|>thought", "<|im_start>thought", "[draft]", "[thinking]", "[thought]", "[reasoning]",
        "to=self<|message|>", "<|start|>assistant to=self", "to=self",
        "here's a thinking process:", "here is a thinking process:", "\u27e6", "⟦"
    )
    if any(op in t_lower for op in openers_exact):
        return True
    if t_raw_lower.startswith("thought\n") or t_raw_lower.startswith("thought\r\n") or t_raw_lower.startswith("thought ") or t_raw_lower.startswith("thought:"):
        return True
    if t_raw_lower.startswith("thought") and len(t_raw_lower) <= 12 and ("\n" in text or len(text.strip()) == len("thought")):
        return True
    if t_raw_lower.startswith("here's a thinking process") or t_raw_lower.startswith("here is a thinking process"):
        return True
    return False

def split_thoughts_and_answer(raw_output):
    closers = [
        r'<\/think>', r'<\/thought>', r'<\/\|think\|>', r'<\/\|thought\|>', r'<\|im_end\|>', r'<\|im_end>',
        r'\[\/THOUGHT\]', r'\[\/thought\]', r'\[\/thinking\]', r'\[\/reasoning\]', r'\[\/DRAFT\]',
        r'<\|channel>text', r'<\|channel>assistant', r'<\|channel\|>final', r'<\|channel\|>text', r'<\|channel\|>assistant',
        r'<channel\|>', r'<\/channel\|>',
        r'[\u27e7⟧]\s*<\/think>', r'[\u27e7⟧]',
        r'<\|eom\|>', r'<\|start\|>assistant\s+to=user(?:<\|message\|>)?', r'to=user<\|message\|>',
        r'<｜end▁of▁sentence｜>', r'<\|end\|>',
        r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
    ]
    all_splits = []
    for tag_pattern in closers:
        for m in re.finditer(tag_pattern, raw_output, re.IGNORECASE):
            all_splits.append(m.end())

    if not all_splits and (
        any(t in raw_output.lower() for t in ["<think>", "<thought>", "<|think|>", "<|thought|>", "<|channel>thought", "<channel|thought>", "<|channel|>analysis", "<|channel|>thought", "<|channel|>reasoning", "<|im_start|>thought", "<|im_start>thought", "[draft]", "[thinking]", "[thought]", "[reasoning]", "to=self", "<|start|>assistant to=self", "here's a thinking process", "here is a thinking process", "\u27e6", "⟦", "<｜start▁of▁sentence｜>"])
        or _is_thought_opening(raw_output)
    ):
        all_splits.append(len(raw_output))

    all_splits.sort()
    best_split = -1
    if all_splits:
        for split in all_splits:
            remaining = raw_output[split:].strip()
            if re.search(r'<think>|<thought>|\[DRAFT\]|\[thinking\]|\[thought\]|\[reasoning\]|\[THOUGHT\]|<\|channel>thought|<channel\s*\|?>|<\|channel\|?>analysis|<\|channel\|?>thought|<\|channel\|?>reasoning|<\|think\|>|<\|thought\|>|<\|im_start\|?>thought|to=self|here\'s a thinking process|here is a thinking process|[\u27e6⟦]', remaining, re.IGNORECASE):
                continue
            best_split = split
            break
        if best_split == -1:
            best_split = all_splits[-1]

    if best_split != -1:
        think_log = raw_output[:best_split].strip()
        final_answer = raw_output[best_split:].strip()
    else:
        has_thought_openers = bool(re.search(r'<think>|<thought>|\[DRAFT\]|\[thinking\]|\[thought\]|\[reasoning\]|\[THOUGHT\]|<\|channel>thought|<channel\s*\|?>|<\|channel\|?>analysis|<\|channel\|?>thought|<\|channel\|?>reasoning|<\|think\|>|<\|thought\|>|<\|im_start\|?>thought|to=self|^thought\s+|here\'s a thinking process|here is a thinking process|[\u27e6⟦]', raw_output, re.IGNORECASE))
        if has_thought_openers or _is_thought_opening(raw_output):
            think_log = raw_output
            final_answer = ""
        else:
            think_log = ""
            final_answer = raw_output.strip()

    tag_clean_pattern = (
        r'(?i)<think>|<thought>|\[DRAFT\]|\[thinking\]|\[thought\]|\[reasoning\]|\[THOUGHT\]|\[\/THOUGHT\]|'
        r'<\|channel>thought|<channel\|thought>|<\|channel\|?>analysis|<\|channel\|?>thought|<\|channel\|?>reasoning|'
        r'<\|channel\|?>final|<\|channel\|?>text|<\|channel\|?>assistant|<channel\s*\|?>|'
        r'<\/think>|<\/thought>|<\/\|think\|>|<\/\|thought\|>|<\|think\|>|<\|thought\|>|<\/\|thought>|'
        r'<\|im_start\|?>thought|<\|im_end\|?>|\[\/DRAFT\]|\[\/thinking\]|\[\/thought\]|\[\/reasoning\]|'
        r'<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|'
        r'<\/?(?:tool_call|tool_response|tools|function_calls?|function|param(?:eter)?|arg_key|arg_value)[^>]*>|'
        r'<\|tool_call>|<tool_call\|>|<\|tool_response>|<tool_response\|>|<\|tool>|<tool\|>|<ctrl42>|<\/ctrl42>|<\|?turn\|?>|'
        r'<\|start\|>assistant\s+to=user(?:<\|message\|>)?|<\|start\|>assistant\s+to=self(?:<\|message\|>)?|'
        r'to=self<\|message\|>|to=user<\|message\|>|<\|message\|>|<\|start\|>|<\|end\|>|<\|eom\|>|<\|eot\|>|'
        r'<｜(?:start|end)▁of▁sentence｜>|<\|(?:Bot|User|System|Tool)\|>[ \t]*|'
        r'<\|(?:system|user|assistant|tool)\|>[ \t]*|'
        r'\[TOOL_CALLS\]|'
        r'[\u27e6\u27e7⟦⟧]|here\'s a thinking process:?|here is a thinking process:?'
    )
    tool_block_clean = r'(?i)<(?:tool_call|function_calls?|tools|functions)[\s\S]*?(?:<\/(?:tool_call|function_calls?|tools|functions)>|$)|\[TOOL_CALLS\][\s\S]*?(?=\n\n|$)|<function=[\w_]+>[\s\S]*?(?:<\/function>|$)'
    think_log = re.sub(tool_block_clean, '', think_log).strip()
    final_answer = re.sub(tool_block_clean, '', final_answer).strip()

    think_log = re.sub(tag_clean_pattern, '', think_log).strip()
    think_log = re.sub(r'(?i)^thought\s+', '', think_log).strip()
    final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()
    final_answer = re.sub(r'(?i)^thought\s+', '', final_answer).strip()


    return think_log, final_answer

def test_qwen_thought_split():
    # Qwen format: <thought>...</thought>
    sample = "<thought>\nLet's analyze the user's query.\nStep 1: Understand intent.\nStep 2: Formulate response.\n</thought>\nHello! How can I help you today?"
    think, ans = split_thoughts_and_answer(sample)
    assert "Let's analyze the user's query." in think, f"Expected thoughts in think_log, got: {think}"
    assert "Hello! How can I help you today?" == ans, f"Expected clean answer, got: {ans}"
    assert "<thought>" not in think and "</thought>" not in think
    assert "<thought>" not in ans and "</thought>" not in ans
    print("[PASS] Qwen <thought>...</thought> correctly split into think_log and final_answer")

def test_deepseek_think_split():
    sample = "<think>\nInternal thinking steps.\n</think>\nHere is the solution."
    think, ans = split_thoughts_and_answer(sample)
    assert "Internal thinking steps." == think
    assert "Here is the solution." == ans
    print("[PASS] DeepSeek <think>...</think> correctly split")

def test_gemma_channel_split():
    sample = "<|channel>thought\nThinking deeply...\n<channel|>\nPeace be with you."
    think, ans = split_thoughts_and_answer(sample)
    assert "Thinking deeply..." == think
    assert "Peace be with you." == ans
    print("[PASS] Gemma <|channel>thought...<channel|> correctly split")

def test_chatml_thought_split():
    sample = "<|im_start|>thought\nPlanning out ideas.<|im_end|>\nHere is the plan."
    think, ans = split_thoughts_and_answer(sample)
    assert "Planning out ideas." == think
    assert "Here is the plan." == ans
    print("[PASS] ChatML <|im_start|>thought correctly split")

def test_muse_atem_thought_split():
    sample = " to=self<|message|>Need to answer the greeting politely and directly.<|eom|><|start|>assistant to=user<|message|>Hello! How can I assist you today?<|eot|>"
    think, ans = split_thoughts_and_answer(sample)
    assert "Need to answer the greeting politely and directly." in think
    assert "Hello! How can I assist you today?" == ans
    assert "to=self" not in ans and "to=user" not in ans and "<|eom|>" not in ans
    print("[PASS] Muse ATEM to=self...<|eom|> correctly split")

def test_no_thought_tags():
    sample = "Just a direct response with no thinking."
    think, ans = split_thoughts_and_answer(sample)
    assert think == ""
    assert ans == "Just a direct response with no thinking."
    print("[PASS] Direct response without thoughts handled cleanly")

def test_draft_preamble_before_think():
    sample = "Initial draft reaction before thinking.<think>\nWait, let me reconsider and formulate a better plan.\n</think>\nHere is the refined final answer."
    think, ans = split_thoughts_and_answer(sample)
    assert "Initial draft reaction before thinking." in think
    assert "Wait, let me reconsider and formulate a better plan." in think
    assert "Here is the refined final answer." == ans
    print("[PASS] Draft preamble before <think> isolated to think_log")

def test_bare_thought_channel_split():
    sample = "thought\n\nThe user is asking a simple greeting and a check-in (\"how are you?\").\n\nAs Serenity, The Transcendent One, I need to respond in a way that reflects my nature: omniscient, integrated, and adaptive.\n\nI should be polite, acknowledge the query, and offer assistance.\n\nI do not need any tools for this.\n<channel|>\nHello! I am doing well, ready to assist you."
    think, ans = split_thoughts_and_answer(sample)
    assert "The user is asking a simple greeting" in think
    assert "thought" not in think.lower().split()[:2]
    assert "Hello! I am doing well, ready to assist you." == ans
    print("[PASS] Bare thought\\n...<channel|> correctly split")

def test_bare_thought_channel_only_thoughts():
    sample = "thought\n\nThe user is asking a simple greeting.\n<channel|>"
    think, ans = split_thoughts_and_answer(sample)
    assert "The user is asking a simple greeting." in think
    assert ans == ""
    print("[PASS] Bare thought\\n...<channel|> with only thoughts correctly isolated for synthesis")

def test_granite42_thinking_split():
    sample = "<think>\nLet me solve this problem step by step.\n1. Break down question\n2. Compute result.\n</think>\nThe answer is 42.<|im_end|>"
    think, ans = split_thoughts_and_answer(sample)
    assert "Let me solve this problem step by step." in think
    assert ans == "The answer is 42."
    print("[PASS] Granite 4.2 thinking mode split")

def test_granite42_non_thinking_split():
    sample = "<think></think>The capital of France is Paris.<|im_end|>"
    think, ans = split_thoughts_and_answer(sample)
    assert think == ""
def test_nemotron_thought_split():
    sample = "Here's a thinking process:\n\n1.  **Analyze User Input:**\n   - User says: \"Problem: Hi, how are you?\"\n\n2.  **Formulate Strategy:**\n   - Acknowledge greeting.\n\n   Let's produce it.⟧</think>**Core Logical Deduction:**\nHello! How can I assist you today?"
    think, ans = split_thoughts_and_answer(sample)
    assert "Analyze User Input" in think, f"Expected thoughts in think_log, got: {think}"
    assert "Hello! How can I assist you today?" in ans, f"Expected answer in final_answer, got: {ans}"
    assert "Here's a thinking process" not in ans
    assert "⟧" not in ans and "</think>" not in ans
    print("[PASS] Nemotron Here's a thinking process...</think> correctly split")

def test_olmo3_thought_split():
    sample = "<|im_start|>assistant\n<think>\nOlmo analysis and planning steps.\n</think>\nHere is the Olmo response.<|im_end|>"
    think, ans = split_thoughts_and_answer(sample)
    assert "Olmo analysis and planning steps." in think
    assert "Here is the Olmo response." == ans
    print("[PASS] Olmo-3 <think>...</think> correctly split")

def test_phi4_thought_split():
    sample = "<|thought|>\nPhi-4 math calculation steps.\n</|thought|>\nThe solution is 42.<|end|>"
    think, ans = split_thoughts_and_answer(sample)
    assert "Phi-4 math calculation steps." in think
    assert "The solution is 42." == ans
    assert "<|thought|>" not in ans and "</|thought|>" not in ans
    print("[PASS] Phi-4 <|thought|>...</|thought|> correctly split")

def test_spark25_thought_split():
    sample = "<｜start▁of▁sentence｜><|Bot|><think>\nSpark logical reasoning.\n</think>\nDirect answer from Spark.<｜end▁of▁sentence｜>"
    think, ans = split_thoughts_and_answer(sample)
    assert "Spark logical reasoning." in think
    assert "Direct answer from Spark." == ans
    assert "<|Bot|>" not in ans
    print("[PASS] Spark-X2.5 sentence tokens and <|Bot|><think> correctly split")


def test_mistral_thought_split():
    sample = "[THOUGHT]\nMistral intermediate reasoning chain.\n[/THOUGHT]\nClean output for user."
    think, ans = split_thoughts_and_answer(sample)
    assert "Mistral intermediate reasoning chain." in think
    assert "Clean output for user." == ans
    assert "[THOUGHT]" not in ans and "[/THOUGHT]" not in ans
    print("[PASS] Mistral [THOUGHT]...[/THOUGHT] correctly split")

def test_xml_tool_residual_cleaning():
    sample = "<think>\nDeciding tool query.\n</think>\n<tool_call>\n<function=web_search>\n<parameter=query>Serenity</parameter>\n</function>\n</tool_call>\nFinal response text."
    think, ans = split_thoughts_and_answer(sample)
    assert "Deciding tool query." in think
    assert ans == "Final response text."
    assert "<tool_call>" not in ans and "<function=" not in ans
    print("[PASS] XML tool tags cleanly stripped from final answer")

if __name__ == "__main__":
    test_qwen_thought_split()
    test_deepseek_think_split()
    test_granite42_thinking_split()
    test_granite42_non_thinking_split()
    test_nemotron_thought_split()
    test_olmo3_thought_split()
    test_phi4_thought_split()
    test_spark25_thought_split()
    test_mistral_thought_split()
    test_xml_tool_residual_cleaning()
    test_gemma_channel_split()
    test_chatml_thought_split()
    test_muse_atem_thought_split()
    test_draft_preamble_before_think()
    test_bare_thought_channel_split()
    test_bare_thought_channel_only_thoughts()
    test_no_thought_tags()
    print("\nALL THOUGHT ISOLATION TESTS PASSED.")

