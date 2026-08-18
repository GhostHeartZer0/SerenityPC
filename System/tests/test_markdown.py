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

    print("\n=== ALL INTERVAL MARKDOWN TESTS PASSED ===")

if __name__ == "__main__":
    run_tests()
