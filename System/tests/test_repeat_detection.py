import re
import sys

class DummyApp:
    def __init__(self, repeat_mode="lazy"):
        self.config = {"repeat_detection_mode": repeat_mode}

    def _detect_repetition(self, text, mode=None):
        """
        Detects repetitive generation loops according to the configured mode:
        - 'off': Loop detection disabled (never interrupts).
        - 'lazy': Relaxed detection for coding and repeated tool calling (ignores code blocks/tool syntax, high repetition threshold).
        - 'hyper': Aggressive detection with tight substring matching and stall phrase checks.
        """
        if not text:
            return False
            
        if mode is None:
            mode = self.config.get("repeat_detection_mode", "lazy").lower()
        else:
            mode = str(mode).lower()

        if mode == "off":
            return False

        if mode == "hyper":
            recent = text[-400:]
            n = len(recent)
            
            stall_phrases = ["re-read", "reread", "look again", "let me look", "actually the prompt", "wait, the input"]
            stall_count = sum(recent.lower().count(phrase) for phrase in stall_phrases)
            if stall_count >= 3:
                return True

            min_len = 35
            max_repeats = 3
            if n < min_len * max_repeats:
                return False
            
            seen = set()
            for i in range(n - min_len + 1):
                sub = recent[i:i+min_len]
                if sub in seen:
                    continue
                seen.add(sub)
                if not any(c.isalnum() for c in sub):
                    continue
                if recent.count(sub) >= max_repeats:
                    return True
            return False

        # Default / "lazy" mode:
        # 1. Filter out code blocks and tool calls so repetitive code / tool queries don't trigger false positives
        cleaned = re.sub(r'```[\s\S]*?```', ' [CODE_BLOCK] ', text)
        cleaned = re.sub(r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:|<(?:channel\|)?(?:execute_tool|executetool)>)[\s\S]*?(?:<\/(?:execute_tool|executetool)>|\}|\n|$)', ' [TOOL_CALL] ', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(?:web_search|read_file|get_system_stats|control_rgb|generate_image)\s*\([\s\S]*?\)', ' [TOOL_CALL] ', cleaned, flags=re.IGNORECASE)

        recent = cleaned[-800:]
        n = len(recent)

        # Check for consecutive repeating identical lines (e.g. spamming the exact same sentence 5+ times)
        lines = [line.strip() for line in recent.split('\n') if len(line.strip()) >= 15]
        if len(lines) >= 5:
            consecutive_count = 1
            for j in range(1, len(lines)):
                if lines[j] == lines[j - 1] and lines[j] not in ["[CODE_BLOCK]", "[TOOL_CALL]"]:
                    consecutive_count += 1
                    if consecutive_count >= 5:
                        return True
                else:
                    consecutive_count = 1

        min_len = 80
        max_repeats = 4
        if n < min_len * max_repeats:
            return False

        seen = set()
        for i in range(n - min_len + 1):
            sub = recent[i:i+min_len]
            if sub in seen:
                continue
            seen.add(sub)
            if not any(c.isalnum() for c in sub):
                continue
            if recent.count(sub) >= max_repeats:
                return True
        return False


def test_repeat_mode_off():
    app = DummyApp(repeat_mode="off")
    looping_text = "This is a degenerate loop text that repeats over and over again! " * 20
    assert not app._detect_repetition(looping_text)
    assert not app._detect_repetition(looping_text, mode="off")


def test_repeat_mode_lazy_allows_repeated_tool_calls():
    app = DummyApp(repeat_mode="lazy")
    # Repeated search calls should NOT trigger loop detection in lazy mode
    tool_text = """
I will now search for information across multiple queries:
action: web_search{"query": "search hi"}
action: web_search{"query": "search low"}
action: web_search{"query": "search windows"}
action: web_search{"query": "search system files"}
"""
    assert not app._detect_repetition(tool_text)

    # Python stub syntax tool calls
    py_tool_text = """
```python
web_search("search hi")
web_search("search low")
web_search("search windows")
web_search("search files")
```
"""
    assert not app._detect_repetition(py_tool_text)


def test_repeat_mode_lazy_allows_code_blocks():
    app = DummyApp(repeat_mode="lazy")
    code_text = """
Here is the implementation:
```python
def test_case_1():
    assert calculate_sum(1, 2) == 3
def test_case_2():
    assert calculate_sum(1, 2) == 3
def test_case_3():
    assert calculate_sum(1, 2) == 3
def test_case_4():
    assert calculate_sum(1, 2) == 3
```
"""
    assert not app._detect_repetition(code_text)


def test_repeat_mode_lazy_catches_true_runaways():
    app = DummyApp(repeat_mode="lazy")
    # 85-char string repeating 5 times
    runaway_phrase = "The quick brown fox jumps over the lazy dog and runs across the green open sunny meadow. "
    runaway_text = runaway_phrase * 5
    assert app._detect_repetition(runaway_text)


def test_repeat_mode_hyper_catches_tight_loops():
    app = DummyApp(repeat_mode="hyper")
    # 40-char string repeating 3 times
    tight_loop = "Calculating the recursive algorithm step. " * 4
    assert app._detect_repetition(tight_loop)

    # Stall phrases
    stall_text = "Let me look again at the input. Let me look again. Actually the prompt says to reread. Wait, the input..."
    assert app._detect_repetition(stall_text)


def test_empty_or_short_input():
    app = DummyApp(repeat_mode="lazy")
    assert not app._detect_repetition("")
    assert not app._detect_repetition(None)
    assert not app._detect_repetition("Short text")

if __name__ == "__main__":
    test_repeat_mode_off()
    test_repeat_mode_lazy_allows_code_blocks()
    test_repeat_mode_lazy_catches_true_runaways()
    test_repeat_mode_hyper_catches_tight_loops()
    test_empty_or_short_input()
    print("[PASS] test_repeat_detection passed successfully.")
