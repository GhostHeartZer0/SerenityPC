# System/tests/test_markdown.py
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.markdown_engine import MarkdownEngine

def run_tests():
    print("=== Testing Interval-Based MarkdownEngine ===")

    # 1. LaTeX Math Conversion Test
    math_latex = r"\frac{a + b}{c \cdot d} + \sqrt[3]{x^2 + y^2} \ge \alpha \times \beta \pm \infty"
    converted_math = MarkdownEngine.convert_latex_to_unicode(math_latex)
    print(f"[MATH TEST] Input: {math_latex}")
    print(f"[MATH TEST] Output: {converted_math}")
    assert "≥" in converted_math or ">=" in converted_math
    assert "√" in converted_math
    assert "×" in converted_math
    assert "±" in converted_math
    assert "∞" in converted_math
    assert "α" in converted_math
    assert "β" in converted_math

    # 2. GFM Table Formatting Test
    gfm_table = """
| Column A | Column B | Column C |
| :--- | :---: | ---: |
| Value 1 | Center Val | $100.00 |
| Short | Longer Table Value | 42 |
"""
    table_rendered = MarkdownEngine.format_gfm_table(gfm_table)
    print("\n[TABLE TEST] Formatted Table:\n" + table_rendered)
    assert "┌" in table_rendered and "┬" in table_rendered and "┐" in table_rendered
    assert "│ Column A" in table_rendered
    assert "├" in table_rendered and "┼" in table_rendered and "┤" in table_rendered
    assert "└" in table_rendered and "┴" in table_rendered and "┘" in table_rendered

    # 3. Code Block & Inline Code Test (Verify ZERO null bytes, ZERO placeholder leakage)
    code_markdown = """
Prerequisites
You will need to install these libraries:
```bash
pip install PyOpenGL glfw numpy
```

The Code:
```python
# Header Comment inside code
def calculate_metrics(a, b, c):
    my_var_name = a * b * c
    return my_var_name
```

Technical Breakdown:
1. The Geometry (4D SDF): Instead of polygons, the `hypercube_sdf` function defines the shape.
2. The Rendering (Ray Marching): The fragment shader marches rays from the camera.
"""
    spans = MarkdownEngine.parse_to_spans(code_markdown)
    print(f"\n[CODE SPANS TEST] Total Spans: {len(spans)}")
    for txt, tags in spans:
        print(f"  Span: {repr(txt)} -> {tags}")
        # Assert NO NULL BYTES exist anywhere in spans
        assert "\x00" not in txt, f"Null byte leaked in span: {repr(txt)}"
        assert "CODE_" not in txt or "hypercube_sdf" in txt, f"Placeholder leaked: {repr(txt)}"
        if "md_code" in tags:
            if "pip install" in txt:
                assert "pip install PyOpenGL glfw numpy" in txt
            elif "calculate_metrics" in txt:
                assert "# Header Comment inside code" in txt
                assert "a * b * c" in txt
            elif "hypercube_sdf" in txt:
                assert txt == "hypercube_sdf"

    # 4. Disambiguation Test
    mixed_text = "The variable `user_input_id` and price $100.00 vs equation $x^2 + y^2 = r^2$."
    mixed_spans = MarkdownEngine.parse_to_spans(mixed_text)
    for txt, tags in mixed_spans:
        assert "\x00" not in txt

    # 5. Arithmetic Multiplication & Power Preservation Test (e.g. 3*3*5*5 -> 3*3*5*5, NEVER 3355)
    arithmetic_prompt = "Calculate 3*3*5*5 and a*b*c with power 3**2."
    spans = MarkdownEngine.parse_to_spans(arithmetic_prompt)
    reconstructed = "".join(txt for txt, tags in spans)
    print(f"\n[ARITHMETIC TEST] Input: {repr(arithmetic_prompt)}")
    print(f"[ARITHMETIC TEST] Output: {repr(reconstructed)}")
    assert "3*3*5*5" in reconstructed, f"3*3*5*5 was corrupted into: {reconstructed}"
    assert "a*b*c" in reconstructed, f"a*b*c was corrupted into: {reconstructed}"
    assert "3**2" in reconstructed, f"3**2 was corrupted into: {reconstructed}"
    assert not any("md_italic" in tags for txt, tags in spans if "3" in txt)

    # 6. Non-Destructive Overlay Intervals Test
    overlay_text = "Here is **bold** text and *italic* and `code` with 3*3*5*5 math."
    tag_ranges, replacements = MarkdownEngine.get_overlay_intervals(overlay_text)
    print(f"\n[OVERLAY TEST] Tag ranges count: {len(tag_ranges)}, replacements: {len(replacements)}")
    # Replacements must be empty since there are no tables
    assert len(replacements) == 0
    # Must have md_hidden and styling tags
    tags_used = {t for _, _, t in tag_ranges}
    assert "md_hidden" in tags_used
    assert "md_bold" in tags_used
    assert "md_italic" in tags_used
    assert "md_code" in tags_used
    # Verify exact spans
    for s, e, tag in tag_ranges:
        slice_txt = overlay_text[s:e]
        if tag == "md_bold":
            assert slice_txt == "bold"
        elif tag == "md_italic":
            assert slice_txt == "italic"
        elif tag == "md_code":
            assert slice_txt == "code"

    # 7. Advanced LaTeX & Math Formatting Test (TODO Example + Currency Disambiguation)
    todo_math = r'$$S_n = \sum_{i=0}^{n} 5^i = \frac{5^{n+1} - 1}{5 - 1} = \frac{5^{n+1} - 1}{4}$$, the sum of $5^0$ through $5^{21}$ is $596,046,447,753,906.'
    spans_todo = MarkdownEngine.parse_to_spans(todo_math)
    math_blocks = [txt for txt, tags in spans_todo if "md_math_block" in tags]
    math_inlines = [txt for txt, tags in spans_todo if "md_math_inline" in tags]
    raw_texts = "".join(txt for txt, tags in spans_todo if "md_math_block" not in tags and "md_math_inline" not in tags)
    print(f"\n[TODO MATH TEST] Math Blocks: {math_blocks}")
    print(f"[TODO MATH TEST] Math Inlines: {math_inlines}")
    print(f"[TODO MATH TEST] Preserved Raw: {raw_texts}")
    assert len(math_blocks) == 1
    assert "Sₙ = ∑₍ᵢ₌₀₎ⁿ 5ⁱ = (5ⁿ⁺¹ - 1)/(5 - 1) = (5ⁿ⁺¹ - 1)/(4)" in math_blocks[0]
    assert "5⁰" in math_inlines
    assert "5²¹" in math_inlines
    assert "$596,046,447,753,906." in raw_texts

    # 8. Forward-Slash Commands Test (/frac, /sum, /times, /mathbf, /left, /right)
    forward_math = r'/sum_{i=0}^{n} /frac{a}{b} /times /mathbf{x} /left( x /right)'
    converted_forward = MarkdownEngine.convert_latex_to_unicode(forward_math)
    print(f"\n[FORWARD SLASH TEST] Output: {converted_forward}")
    assert "∑₍ᵢ₌₀₎ⁿ" in converted_forward
    assert "a/b" in converted_forward
    assert "×" in converted_forward
    assert "x ( x )" in converted_forward

    # 9. Quad Dollar Syntax ($$$$...$$$$) Test
    quad_math = r'$$$$E = m \cdot c^2$$$$'
    spans_quad = MarkdownEngine.parse_to_spans(quad_math)
    assert any("md_math_block" in tags and "m · c²" in txt for txt, tags in spans_quad)

    # 10. Single-letter variables and complex inline LaTeX
    single_var = r'Let $U$ be internal energy, $x$ position, and **High Specific Heat Capacity** ($\approx 4.18\text{ kJ/kg}\cdot^\circ\text{C}$).'
    spans_single = MarkdownEngine.parse_to_spans(single_var)
    assert any("md_math_inline" in tags and txt == "U" for txt, tags in spans_single)
    assert any("md_math_inline" in tags and txt == "x" for txt, tags in spans_single)
    assert any("md_math_inline" in tags and "4.18 kJ/kg·°C" in txt for txt, tags in spans_single)

    # 11. Standalone slash symbols in text outside math ($)
    slash_text = r'Change is /Delta and approx is /approx and /circ'
    spans_slash = MarkdownEngine.parse_to_spans(slash_text)
    assert any("md_math_inline" in tags and txt == "Δ" for txt, tags in spans_slash)
    assert any("md_math_inline" in tags and txt == "≈" for txt, tags in spans_slash)
    assert any("md_math_inline" in tags and txt == "°" for txt, tags in spans_slash)

    print("\n=== ALL INTERVAL MARKDOWN TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()
