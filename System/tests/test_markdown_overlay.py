# System/tests/test_markdown_overlay.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import tkinter as tk
from System.markdown_engine import MarkdownEngine

def run_tests():
    print("=== Testing MarkdownEngine Overlay & Math Preservation ===")

    # 1. Verify 3*3*5*5 is NEVER captured as italic
    math_expr = "Calculate 3*3*5*5 for the dimensions."
    intervals = MarkdownEngine.parse_overlay_intervals(math_expr)
    print(f"[TEST 1] Raw: {math_expr}")
    print(f"[TEST 1] Intervals found: {intervals}")
    # There should be 0 intervals because 3*3*5*5 is pure math, not markdown
    assert len(intervals) == 0, f"Expected 0 intervals for math expression, got {intervals}"

    # 2. Verify standard markdown inline formatting with math
    mixed_text = "Here is **bold text** and *italic word* plus 3*3*5*5 and `inline_code()`."
    intervals = MarkdownEngine.parse_overlay_intervals(mixed_text)
    print(f"\n[TEST 2] Mixed: {mixed_text}")
    tags_found = [t for _, _, t in intervals]
    print(f"[TEST 2] Tags found: {tags_found}")
    assert "md_bold" in tags_found
    assert "md_italic" in tags_found
    assert "md_code" in tags_found
    assert "md_syntax" in tags_found

    # Verify extracted chunks matching tags
    for s, e, t in intervals:
        chunk = mixed_text[s:e]
        if t == "md_bold":
            assert chunk == "bold text"
        elif t == "md_italic":
            assert chunk == "italic word"
        elif t == "md_code":
            assert chunk == "inline_code()"
        # Confirm 3*3*5*5 is never part of any tagged span
        assert "3*3*5*5" not in chunk

    # 3. Verify Code Blocks & Headers
    block_text = "# Main Title\n\n```python\nresult = 3 * 3 * 5 * 5\n```\n"
    b_intervals = MarkdownEngine.parse_overlay_intervals(block_text)
    print(f"\n[TEST 3] Block intervals count: {len(b_intervals)}")
    b_tags = [t for _, _, t in b_intervals]
    assert "md_header_1" in b_tags
    assert "md_code" in b_tags
    assert "md_syntax" in b_tags

    for s, e, t in b_intervals:
        chunk = block_text[s:e]
        if t == "md_header_1":
            assert chunk == "Main Title"
        elif t == "md_code":
            assert "result = 3 * 3 * 5 * 5" in chunk

    # 4. Headless Tkinter Text Widget Non-Destructive Overlay Test
    root = tk.Tk()
    root.withdraw()
    try:
        txt = tk.Text(root)
        txt.tag_config("md_syntax", elide=True)
        txt.tag_config("md_bold", font=("TkDefaultFont", 10, "bold"))
        txt.tag_config("md_italic", font=("TkDefaultFont", 10, "italic"))

        sample_source = "User asked: What is 3*3*5*5? Cecilia said: **225**."
        txt.insert("1.0", sample_source)

        # Apply overlay intervals exactly as main._apply_markdown does
        raw = txt.get("1.0", "end-1c")
        ov_intervals = MarkdownEngine.parse_overlay_intervals(raw)
        for s_off, e_off, tag in ov_intervals:
            tag_start = txt.index(f"1.0 + {s_off} chars")
            tag_end = txt.index(f"1.0 + {e_off} chars")
            txt.tag_add(tag, tag_start, tag_end)

        # CRITICAL ASSERTION: The widget text must be bit-for-bit identical to source!
        retrieved_text = txt.get("1.0", "end-1c")
        print(f"\n[TEST 4] Source:    {repr(sample_source)}")
        print(f"[TEST 4] Retrieved: {repr(retrieved_text)}")
        assert retrieved_text == sample_source, f"Source was mutated! Expected {sample_source!r}, got {retrieved_text!r}"
        assert "3*3*5*5" in retrieved_text
        assert "**225**" in retrieved_text

        # Verify tags applied correctly
        # The bold text '225' should have 'md_bold' tag
        bold_pos = sample_source.index("225")
        idx_in_widget = txt.index(f"1.0 + {bold_pos} chars")
        applied_tags = txt.tag_names(idx_in_widget)
        print(f"[TEST 4] Tags on '225': {applied_tags}")
        assert "md_bold" in applied_tags

        # The '**' syntax delimiters should have 'md_syntax' tag
        delim_pos = sample_source.index("**")
        idx_delim = txt.index(f"1.0 + {delim_pos} chars")
        assert "md_syntax" in txt.tag_names(idx_delim)
    finally:
        root.destroy()

    print("\n=== ALL MARKDOWN OVERLAY TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()
