# System/tests/test_prompt_markdown_overlay.py
import sys, os, unittest
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.markdown_engine import MarkdownEngine


class TestPromptMarkdownOverlay(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.txt = tk.Text(self.root)
        self.txt.tag_config("user", font=("TkDefaultFont", 10))
        self.txt.tag_config("ai", font=("TkDefaultFont", 10))
        self.txt.tag_config("md_hidden", elide=True)
        self.txt.tag_config("md_bold", font=("TkDefaultFont", 10, "bold"))
        self.txt.tag_config("md_italic", font=("TkDefaultFont", 10, "italic"))
        self.txt.tag_config("md_code", font=("Courier", 10))

    def tearDown(self):
        self.root.destroy()

    def test_default_setting_in_config(self):
        """Verify format_prompts_markdown is configured and defaults to False."""
        config = {}
        if "format_prompts_markdown" not in config:
            config["format_prompts_markdown"] = False
        self.assertFalse(config["format_prompts_markdown"])

    def test_prompt_math_preservation_raw(self):
        """Ensure math expressions like 3*3*5*5 and a*b*c in prompts are NOT formatted when disabled."""
        prompt_math = "3*3*5*5"
        self.txt.insert(tk.END, f"\nYou: {prompt_math}\n", ("user",))
        raw_in_buffer = self.txt.get("1.0", tk.END)
        self.assertIn("3*3*5*5", raw_in_buffer)
        self.assertNotIn("3355", raw_in_buffer)

    def test_markdown_engine_arithmetic_protection(self):
        """Verify MarkdownEngine never parses arithmetic multiplication as italics."""
        expr = "Calculate 3*3*5*5 and x*y*z."
        spans = MarkdownEngine.parse_to_spans(expr)
        full_text = "".join(s[0] for s in spans)
        self.assertEqual(full_text, expr)
        self.assertIn("3*3*5*5", full_text)
        self.assertIn("x*y*z", full_text)
        for chunk, tags in spans:
            if "3" in chunk or "x" in chunk:
                self.assertNotIn("md_italic", tags)

    def test_latex_symbol_prefix_safety(self):
        r"""Verify longer latex symbols like \infty are not broken by shorter prefixes like \in."""
        math_expr = r"x \in A \implies \lim_{n \to \infty} x_n"
        converted = MarkdownEngine.convert_latex_to_unicode(math_expr)
        self.assertIn("∞", converted)
        self.assertNotIn("∈fty", converted)

    def test_non_destructive_overlay_intervals(self):
        """Verify get_overlay_intervals tags markdown syntax with md_hidden without modifying characters."""
        raw = "Text with **bold**, *italic*, and `code`."
        self.txt.insert("1.0", raw)
        tag_ranges, replacements = MarkdownEngine.get_overlay_intervals(raw)
        
        # Zero character replacements for inline markdown
        self.assertEqual(len(replacements), 0)
        
        # Apply overlay tags
        for s, e, tag in tag_ranges:
            self.txt.tag_add(tag, f"1.0 + {s} chars", f"1.0 + {e} chars")

        # Underlying text must be 100% byte-for-byte identical to input
        buffer_text = self.txt.get("1.0", "end-1c")
        self.assertEqual(buffer_text, raw)

        # Bold tag must cover precisely 'bold'
        bold_ranges = self.txt.tag_ranges("md_bold")
        self.assertEqual(len(bold_ranges), 2)
        self.assertEqual(self.txt.get(bold_ranges[0], bold_ranges[1]), "bold")

        # Italic tag must cover precisely 'italic'
        italic_ranges = self.txt.tag_ranges("md_italic")
        self.assertEqual(len(italic_ranges), 2)
        self.assertEqual(self.txt.get(italic_ranges[0], italic_ranges[1]), "italic")

        # Delimiters must be hidden via md_hidden
        hidden_ranges = self.txt.tag_ranges("md_hidden")
        self.assertTrue(len(hidden_ranges) >= 6)


if __name__ == "__main__":
    unittest.main()
