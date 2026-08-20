import re

def split_thoughts_and_answer(raw_output):
    closers = [
        r'<\/think>', r'<\/thought>', r'<\/\|think\|>', r'<\|im_end\|>', r'<\|im_end>',
        r'<\|channel>text', r'<\|channel>assistant', r'<channel\|>', r'<\/channel\|>', r'\[\/DRAFT\]',
        r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
    ]
    all_splits = []
    for tag_pattern in closers:
        for m in re.finditer(tag_pattern, raw_output, re.IGNORECASE):
            all_splits.append(m.end())

    if not all_splits and any(t in raw_output.lower() for t in ["<think>", "<thought>", "<|think|>", "<|channel>thought", "<|im_start|>thought", "<|im_start>thought", "[draft]"]):
        all_splits.append(len(raw_output))

    all_splits.sort()
    best_split = -1
    if all_splits:
        for split in all_splits:
            remaining = raw_output[split:].strip()
            if re.search(r'<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\|think\|>|<\|im_start\|?>thought', remaining, re.IGNORECASE):
                continue
            best_split = split
            break
        if best_split == -1:
            best_split = all_splits[-1]

    if best_split != -1:
        think_log = raw_output[:best_split].strip()
        final_answer = raw_output[best_split:].strip()
    else:
        think_log = ""
        final_answer = raw_output.strip()

    tag_clean_pattern = r'(?i)<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\/think>|<\/thought>|<\/\|think\|>|<\|think\|>|<\|im_start\|?>thought|<\|im_end\|?>|\[\/DRAFT\]|<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|<\|tool_call>|<tool_call\|>|<\|tool_response>|<tool_response\|>|<\|tool>|<tool\|>|<ctrl42>|<\/ctrl42>|<\|?turn\|?>'
    think_log = re.sub(tag_clean_pattern, '', think_log).strip()
    final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()

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

if __name__ == "__main__":
    test_qwen_thought_split()
    test_deepseek_think_split()
    test_gemma_channel_split()
    test_chatml_thought_split()
    test_draft_preamble_before_think()
    test_no_thought_tags()
    print("\nALL THOUGHT ISOLATION TESTS PASSED.")
