# System/markdown_engine.py
# High-Performance Interval-Based Markdown, Math, Table, and Code Renderer for SerenityPC.

import re
from typing import List, Tuple, Dict, Any, Optional

class MarkdownEngine:
    """
    High-performance, memory-safe markdown engine for Tkinter Text widgets.
    Parses Markdown into tagged spans using direct interval scanning without placeholders or null bytes.
    """

    SUPERSCRIPT_MAP = {
        '0': '⁰', '1': '¹', '2': '²', '3': '³', '4': '⁴',
        '5': '⁵', '6': '⁶', '7': '⁷', '8': '⁸', '9': '⁹',
        '+': '⁺', '-': '⁻', '=': '⁼', '(': '⁽', ')': '⁾',
        'n': 'ⁿ', 'i': 'ⁱ', 'x': 'ˣ', 'y': 'ʸ'
    }

    SUBSCRIPT_MAP = {
        '0': '₀', '1': '₁', '2': '₂', '3': '₃', '4': '₄',
        '5': '₅', '6': '₆', '7': '₇', '8': '₈', '9': '₉',
        '+': '₊', '-': '₋', '=': '₌', '(': '₍', ')': '₎',
        'a': 'ₐ', 'e': 'ₑ', 'i': 'ᵢ', 'j': 'ⱼ', 'k': 'ₖ',
        'm': 'ₘ', 'n': 'ₙ', 'o': 'ₒ', 'p': 'ₚ', 'r': 'ᵣ',
        's': 'ₛ', 't': 'ₜ', 'u': 'ᵤ', 'v': 'ᵥ', 'x': 'ₓ'
    }

    LATEX_SYMBOLS = {
        r'\rightarrow': '→', r'\to': '→', r'\Rightarrow': '⇒', r'\implies': '⇒',
        r'\leftarrow': '←', r'\Leftarrow': '⇐', r'\leftrightarrow': '↔', r'\Leftrightarrow': '⇔', r'\iff': '⇔',
        r'\checkmark': '✓', r'\neg': '¬', r'\neq': '≠', r'\ne': '≠',
        r'\times': '×', r'\cdot': '·', r'\div': '÷', r'\pm': '±', r'\mp': '∓',
        r'\approx': '≈', r'\sim': '~', r'\equiv': '≡', r'\cong': '≅',
        r'\le': '≤', r'\leq': '≤', r'\ge': '≥', r'\geq': '≥',
        r'\ll': '≪', r'\gg': '≫',
        r'\sum': '∑', r'\prod': '∏', r'\int': '∫', r'\iint': '∬', r'\iiint': '∭', r'\oint': '∮',
        r'\partial': '∂', r'\nabla': '∇',
        r'\in': '∈', r'\notin': '∉', r'\subset': '⊂', r'\subseteq': '⊆', r'\supset': '⊃', r'\supseteq': '⊇',
        r'\cap': '∩', r'\cup': '∪', r'\setminus': '∖',
        r'\forall': '∀', r'\exists': '∃', r'\nexists': '∄',
        r'\infty': '∞', r'\degree': '°', r'^\circ': '°',
        r'\alpha': 'α', r'\beta': 'β', r'\gamma': 'γ', r'\delta': 'δ', r'\epsilon': 'ε', r'\varepsilon': 'ε',
        r'\zeta': 'ζ', r'\eta': 'η', r'\theta': 'θ', r'\vartheta': 'ϑ', r'\iota': 'ι', r'\kappa': 'κ',
        r'\lambda': 'λ', r'\mu': 'μ', r'\nu': 'ν', r'\xi': 'ξ', r'\pi': 'π', r'\varpi': 'ϖ',
        r'\rho': 'ρ', r'\varrho': 'ϱ', r'\sigma': 'σ', r'\varsigma': 'ς', r'\tau': 'τ', r'\upsilon': 'υ',
        r'\phi': 'φ', r'\varphi': 'ϕ', r'\chi': 'χ', r'\psi': 'ψ', r'\omega': 'ω',
        r'\Gamma': 'Γ', r'\Delta': 'Δ', r'\Theta': 'Θ', r'\Lambda': 'Λ', r'\Xi': 'Ξ',
        r'\Pi': 'Π', r'\Sigma': 'Σ', r'\Upsilon': 'Υ', r'\Phi': 'Φ', r'\Psi': 'Ψ', r'\Omega': 'Ω',
        r'\dots': '…', r'\ldots': '…', r'\cdots': '⋯', r'\vdots': '⋮', r'\ddots': '⋱',
        r'\hbar': 'ℏ', r'\ell': 'ℓ', r'\Re': 'ℜ', r'\Im': 'ℑ', r'\aleph': 'ℵ',
        r'\emptyset': '∅', r'\angle': '∠', r'\triangle': '△', r'\perp': '⊥', r'\parallel': '∥'
    }

    @classmethod
    def convert_latex_to_unicode(cls, text: str) -> str:
        """Converts LaTeX math expressions to highly readable Unicode representations."""
        if not text:
            return ""
        s = text

        # 1. Structural fractions \frac{a}{b} or \dfrac{a}{b} -> (a)/(b)
        def replace_frac(m):
            num, den = m.group(1).strip(), m.group(2).strip()
            if len(num) <= 3 and len(den) <= 3 and '/' not in num and '/' not in den:
                return f"{num}/{den}"
            return f"({num})/({den})"
        s = re.sub(r'\\(?:d|t)?frac\{([^{}]*)\}\{([^{}]*)\}', replace_frac, s)

        # 2. Roots \sqrt[n]{x} -> ⁿ√(x), \sqrt{x} -> √(x)
        def replace_root(m):
            deg = m.group(1).strip()
            val = m.group(2).strip()
            sup_deg = "".join(cls.SUPERSCRIPT_MAP.get(c, c) for c in deg)
            return f"{sup_deg}√({val})"
        s = re.sub(r'\\sqrt\[([^\]]+)\]\{([^{}]*)\}', replace_root, s)
        s = re.sub(r'\\sqrt\{([^{}]*)\}', r'√(\1)', s)

        # 3. Formatting macros \mathbf, \text, \mathrm, etc.
        tags = ["mathbf", "mathrm", "mathit", "text", "textbf", "textit", "underline", "mathbb", "mathcal", "boldsymbol"]
        pattern = r'\\(?:' + '|'.join(tags) + r')\{([^}]*)\}'
        s = re.sub(pattern, r'\1', s)

        # 4. Bracket modifiers \left, \right
        s = s.replace(r'\left(', '(').replace(r'\right)', ')')
        s = s.replace(r'\left[', '[').replace(r'\right]', ']')
        s = s.replace(r'\left\{', '{').replace(r'\right\}', '}')
        s = s.replace(r'\left|', '|').replace(r'\right|', '|')

        # 5. Spacing commands
        s = re.sub(r'\\(?:quad|qquad|,|;|!|\s)', ' ', s)

        # 6. Replace symbols
        for lat, uni in cls.LATEX_SYMBOLS.items():
            s = s.replace(lat, uni)

        # 7. Convert simple superscripts: x^2 -> x², x^{10} -> x¹⁰
        def replace_sup(m):
            base, sup = m.group(1), m.group(2)
            sup_clean = "".join(cls.SUPERSCRIPT_MAP.get(c, c) for c in sup)
            return f"{base}{sup_clean}"
        s = re.sub(r'([a-zA-Z0-9\)])\^\{([a-zA-Z0-9\+\-]+)\}', replace_sup, s)
        s = re.sub(r'([a-zA-Z0-9\)])\^([0-9nixy\+\-])', replace_sup, s)

        # 8. Convert simple subscripts: x_1 -> x₁, x_{10} -> x₁₀
        def replace_sub(m):
            base, sub = m.group(1), m.group(2)
            sub_clean = "".join(cls.SUBSCRIPT_MAP.get(c, c) for c in sub)
            return f"{base}{sub_clean}"
        s = re.sub(r'([a-zA-Z0-9\)])_\{([0-9aeijkmnoprstuvx\+\-]+)\}', replace_sub, s)
        s = re.sub(r'([a-zA-Z0-9\)])_([0-9aeijkmnoprstuvx])', replace_sub, s)

        # 9. Clean up residual backslashes on basic math
        s = re.sub(r'\\([a-zA-Z]+)', r'\1', s)
        return s.strip()

    @classmethod
    def format_gfm_table(cls, table_str: str) -> str:
        """Formats a Markdown GFM table into a clean aligned Unicode box-drawing table."""
        lines = [line.strip() for line in table_str.strip().split('\n') if line.strip()]
        if len(lines) < 2:
            return table_str

        raw_rows = []
        for line in lines:
            parts = [p.strip().replace(r'\|', '|') for p in re.split(r'(?<!\\)\|', line)]
            if line.startswith('|') and parts:
                parts = parts[1:]
            if line.endswith('|') and parts:
                parts = parts[:-1]
            raw_rows.append(parts)

        if len(raw_rows) < 2:
            return table_str

        sep_row = raw_rows[1]
        is_sep = all(re.match(r'^:?-+:?$', cell.strip()) for cell in sep_row if cell.strip())

        alignments = []
        if is_sep:
            for cell in sep_row:
                c = cell.strip()
                if c.startswith(':') and c.endswith(':'):
                    alignments.append('center')
                elif c.endswith(':'):
                    alignments.append('right')
                else:
                    alignments.append('left')
            raw_rows.pop(1)
        else:
            alignments = ['left'] * max(len(r) for r in raw_rows)

        if not raw_rows:
            return table_str

        num_cols = max(len(r) for r in raw_rows)
        for r in raw_rows:
            while len(r) < num_cols:
                r.append('')
        while len(alignments) < num_cols:
            alignments.append('left')

        col_widths = [0] * num_cols
        for r in raw_rows:
            for idx, cell in enumerate(r):
                col_widths[idx] = max(col_widths[idx], len(cell))
        col_widths = [max(w, 3) for w in col_widths]

        formatted_lines = []
        top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
        formatted_lines.append(top_border)

        header_cells = []
        for idx, cell in enumerate(raw_rows[0]):
            w, align = col_widths[idx], alignments[idx]
            cell_str = cell.center(w) if align == 'center' else cell.rjust(w) if align == 'right' else cell.ljust(w)
            header_cells.append(f" {cell_str} ")
        formatted_lines.append("│" + "│".join(header_cells) + "│")

        mid_border = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
        formatted_lines.append(mid_border)

        for r in raw_rows[1:]:
            row_cells = []
            for idx, cell in enumerate(r):
                w, align = col_widths[idx], alignments[idx]
                cell_str = cell.center(w) if align == 'center' else cell.rjust(w) if align == 'right' else cell.ljust(w)
                row_cells.append(f" {cell_str} ")
            formatted_lines.append("│" + "│".join(row_cells) + "│")

        bot_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"
        formatted_lines.append(bot_border)

        return "\n".join(formatted_lines)

    @classmethod
    def _parse_inline_spans(cls, text: str, base_tags: tuple) -> List[Tuple[str, tuple]]:
        """
        Direct interval scanner for inline elements:
        1. Inline code: `...`
        2. Inline math: $...$, \\(...\\)
        3. Bold-italic: ***...***, ___...___
        4. Bold: **...**, __...__
        5. Strike: ~~...~~
        6. Italic: *...*, word-boundary safe _..._
        Uses strictly non-overlapping intervals without placeholders.
        """
        if not text:
            return []

        # Find all match candidates with (start, end, rendered_text, tags, priority)
        intervals = []

        # 1. Inline code (Highest priority)
        for m in re.finditer(r'`([^`\n]+?)`', text):
            intervals.append((m.start(), m.end(), m.group(1), base_tags + ("md_code",), 1))

        # 2. Math display inside text: $$...$$, \[...\]
        for m in re.finditer(r'(?s)\$\$(.+?)\$\$|\\\[(.+?)\\\]', text):
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            converted = cls.convert_latex_to_unicode(inner)
            intervals.append((m.start(), m.end(), f" {converted} ", base_tags + ("md_math_block",), 2))

        # 3. Inline math: $...$, \(...\)
        for m in re.finditer(r'(?<!\\)\$([^\$\n\s](?:[^\$\n]*?[^\$\n\s])?)\$|\\\((.+?)\\\)', text):
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            if inner and not re.match(r'^\d+(?:\.\d+)?(?:\s*,\s*\d+)?$', inner.strip()):
                converted = cls.convert_latex_to_unicode(inner)
                intervals.append((m.start(), m.end(), converted, base_tags + ("md_math_inline",), 3))

        # 4. Bold-italic
        for m in re.finditer(r'\*\*\*(.+?)\*\*\*|___(.+?)___', text):
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            intervals.append((m.start(), m.end(), inner, base_tags + ("md_bold_italic",), 4))

        # 5. Bold
        for m in re.finditer(r'\*\*(.+?)\*\*|__(.+?)__', text):
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            intervals.append((m.start(), m.end(), inner, base_tags + ("md_bold",), 5))

        # 6. Strike
        for m in re.finditer(r'~~(.+?)~~', text):
            intervals.append((m.start(), m.end(), m.group(1), base_tags + ("md_strike",), 6))

        # 7. Italic
        for m in re.finditer(r'\*([^\*\n\s](?:[^\*\n]*?[^\*\n\s])?)\*', text):
            intervals.append((m.start(), m.end(), m.group(1), base_tags + ("md_italic",), 7))

        for m in re.finditer(r'(?<=\s)_([^_ \n](?:[^_\n]*?[^_ \n])?)_(?=\s|[.,;:!?\)]|$)', text):
            intervals.append((m.start(), m.end(), m.group(1), base_tags + ("md_italic",), 7))

        # Resolve overlapping intervals by priority (lowest priority number wins)
        intervals.sort(key=lambda x: (x[0], x[4], -(x[1] - x[0])))
        non_overlapping = []
        last_end = 0

        for start, end, inner_txt, tags, prio in intervals:
            if start >= last_end:
                non_overlapping.append((start, end, inner_txt, tags))
                last_end = end

        # Build final spans
        spans: List[Tuple[str, tuple]] = []
        curr = 0
        for start, end, inner_txt, tags in non_overlapping:
            if start > curr:
                spans.append((text[curr:start], base_tags))
            spans.append((inner_txt, tags))
            curr = end

        if curr < len(text):
            spans.append((text[curr:], base_tags))

        return spans

    @classmethod
    def parse_to_spans(cls, text: str, base_tags: tuple = ("ai",), is_thought: bool = False) -> List[Tuple[str, tuple]]:
        """
        Parses raw Markdown text into tagged spans.
        Guarantees code blocks, tables, and math equations are isolated and accurately styled.
        """
        if not text:
            return []

        # Thoughts handling
        if is_thought:
            thought_tags = base_tags if "md_thought" in base_tags else base_tags + ("md_thought",)
            parts = re.split(r'(```[\w]*\n?[\s\S]*?```|`[^`\n]+?`)', text)
            spans: List[Tuple[str, tuple]] = []
            for part in parts:
                if not part:
                    continue
                if part.startswith('```') and part.endswith('```'):
                    inner = re.sub(r'^```[\w]*\n?', '', part)
                    inner = re.sub(r'\n?```$', '', inner)
                    spans.append((f"\n{inner}\n", base_tags + ("md_code",)))
                elif part.startswith('`') and part.endswith('`') and len(part) >= 2:
                    spans.append((part[1:-1], base_tags + ("md_code",)))
                else:
                    spans.extend(cls._parse_inline_spans(part, thought_tags))
            return spans

        # Phase 1: Identify all Top-Level Block Intervals (Fenced Code, Math Display, Tables)
        # Each block item: (start_idx, end_idx, block_type, data)
        blocks = []

        # 1. Fenced Code Blocks: ```lang\n...```
        for m in re.finditer(r'```(\w*)\r?\n([\s\S]*?)```', text):
            lang = m.group(1).strip() if m.group(1) else ""
            code_content = m.group(2)
            header_str = f"[{lang.upper()}]\n" if lang else ""
            formatted_code = f"\n{header_str}{code_content.rstrip()}\n\n"
            blocks.append((m.start(), m.end(), "code", formatted_code))

        # 2. Display Math Blocks: $$...$$ or \[...\]
        for m in re.finditer(r'\$\$([\s\S]+?)\$\$|\\\[([\s\S]+?)\\\]', text):
            inner = m.group(1) if m.group(1) is not None else m.group(2)
            converted = cls.convert_latex_to_unicode(inner.strip())
            formatted_math = f"\n  {converted}\n\n"
            blocks.append((m.start(), m.end(), "math_block", formatted_math))

        # Sort blocks by start index and filter overlaps
        blocks.sort(key=lambda x: x[0])
        resolved_blocks = []
        last_block_end = 0
        for b_start, b_end, b_type, b_data in blocks:
            if b_start >= last_block_end:
                resolved_blocks.append((b_start, b_end, b_type, b_data))
                last_block_end = b_end

        # Phase 2: Split text by top-level blocks and parse the intermediate text line-by-line
        final_spans: List[Tuple[str, tuple]] = []
        pos = 0

        for b_start, b_end, b_type, b_data in resolved_blocks:
            # Parse text between previous block and this block
            if b_start > pos:
                inter_text = text[pos:b_start]
                cls._parse_text_lines(inter_text, base_tags, final_spans)

            # Emit the block span directly
            if b_type == "code":
                final_spans.append((b_data, base_tags + ("md_code",)))
            elif b_type == "math_block":
                final_spans.append((b_data, base_tags + ("md_math_block",)))

            pos = b_end

        # Parse remaining text after last block
        if pos < len(text):
            inter_text = text[pos:]
            cls._parse_text_lines(inter_text, base_tags, final_spans)

        return final_spans

    @classmethod
    def _parse_text_lines(cls, text_chunk: str, base_tags: tuple, out_spans: List[Tuple[str, tuple]]) -> None:
        """Parses a text chunk (not containing fenced code blocks or display math) line by line."""
        if not text_chunk:
            return

        lines = text_chunk.split('\n')
        table_buf = []
        in_table = False

        def flush_table():
            nonlocal table_buf, in_table
            if not table_buf:
                return
            if len(table_buf) >= 2 and any(re.search(r'\|\s*:?-+-*:?\s*\|', l) or re.search(r'^:?-+-*:?$', l.replace('|', '').strip()) for l in table_buf):
                formatted = cls.format_gfm_table('\n'.join(table_buf))
                out_spans.append((f"\n{formatted}\n\n", base_tags + ("md_table",)))
            else:
                for tbl_l in table_buf:
                    cls._parse_single_line(tbl_l, base_tags, out_spans)
                    out_spans.append(("\n", base_tags))
            table_buf = []
            in_table = False

        for l_idx, line in enumerate(lines):
            stripped = line.strip()

            # Table line detection
            if '|' in line:
                if not in_table:
                    in_table = True
                table_buf.append(line)
                continue
            else:
                if in_table:
                    flush_table()

            # Empty line
            if not stripped:
                out_spans.append(("\n", base_tags))
                continue

            # Headers: #, ##, ###
            header_match = re.match(r'^(#{1,6})\s+(.*)$', line)
            if header_match:
                level = len(header_match.group(1))
                h_text = header_match.group(2)
                h_tag = "md_header_1" if level == 1 else "md_header_2" if level == 2 else "md_header_3"
                out_spans.append((f"\n{h_text}\n", base_tags + (h_tag,)))
                continue

            # Horizontal rules
            if re.match(r'^(?:---|\*\*\*|___)\s*$', stripped):
                out_spans.append(("\n────────────────────────────────────────\n", base_tags + ("md_table",)))
                continue

            # Blockquotes: > quote
            quote_match = re.match(r'^\s*>\s*(.*)$', line)
            if quote_match:
                q_text = quote_match.group(1)
                out_spans.append(("  ▎ ", base_tags + ("md_quote",)))
                out_spans.extend(cls._parse_inline_spans(q_text, base_tags + ("md_quote",)))
                out_spans.append(("\n", base_tags))
                continue

            # Unordered lists: * item, - item, + item
            ul_match = re.match(r'^(\s*)[\*\-\+]\s+(.*)$', line)
            if ul_match:
                indent = len(ul_match.group(1)) // 2
                bullet_prefix = "  " * indent + " • "
                item_text = ul_match.group(2)
                out_spans.append((bullet_prefix, base_tags + ("md_list",)))
                out_spans.extend(cls._parse_inline_spans(item_text, base_tags + ("md_list",)))
                out_spans.append(("\n", base_tags))
                continue

            # Ordered lists: 1. item
            ol_match = re.match(r'^(\s*)(\d+\.)\s+(.*)$', line)
            if ol_match:
                indent = len(ol_match.group(1)) // 2
                num_prefix = "  " * indent + f" {ol_match.group(2)} "
                item_text = ol_match.group(3)
                out_spans.append((num_prefix, base_tags + ("md_list",)))
                out_spans.extend(cls._parse_inline_spans(item_text, base_tags + ("md_list",)))
                out_spans.append(("\n", base_tags))
                continue

            # Standard paragraph line
            cls._parse_single_line(line, base_tags, out_spans)
            if l_idx < len(lines) - 1:
                out_spans.append(("\n", base_tags))

        if in_table:
            flush_table()

    @classmethod
    def _parse_single_line(cls, line: str, base_tags: tuple, out_spans: List[Tuple[str, tuple]]) -> None:
        """Parses inline formatting for a single line."""
        spans = cls._parse_inline_spans(line, base_tags)
        out_spans.extend(spans)
