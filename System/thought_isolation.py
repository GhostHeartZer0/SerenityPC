# System/thought_isolation.py
"""Thought Channel Isolation & Stage Separation for Cecilia.

Enforces zero-leakage boundaries between internal reasoning scratchpads
and external persona responses, adhering strictly to thought_channel_delicacy rules.
"""

import re
from typing import Tuple


REASONING_TAGS = {
    'DECONSTRUCTION', 'MOTIVE', 'SUBTEXT', 'STRATEGIC IMPLICATIONS',
    'COGNITIVE PIVOT', 'THE CONTRAST HEURISTIC', 'FUNCTIONAL GOAL',
    'INPUT DECONSTRUCTION', 'SYNTACTIC STRUCTURE', 'LINGUISTIC PROFILE',
    'CONTEXTUAL VOID', 'REASONING ENGINE', 'ANALYSIS_START', 'ANALYSIS_END',
    'INTERNAL_THOUGHT', 'THOUGHT_START', 'THOUGHT_END',
    'INTENT GUARDRAIL TRIGGERED', 'SYSTEM INSTRUCTION', 'SYSTEM DIRECTIVE',
    'DMN ANALYTICAL CONCEPTS', 'GROUNDED WORLDVIEW', 'PARAM'
}

RAW_RESPONSE_TAGS = {
    'TACTICAL_ANALYSIS_COMPLETE', 'STATUS: UNFILTERED', 'STATUS',
    'TACTICAL_ANALYSIS', 'TACTICAL ANALYSIS', 'UNFILTERED',
    'RAW_RESPONSE', 'RAW RESPONSE', 'PRELIMINARY_ANALYSIS',
    'DUAL_RESPONSE', 'UNFILTERED_OUTPUT', 'UNFILTERED_RESPONSE',
    'TACTICAL EVALUATION', 'TECHNICAL ANALYSIS', 'STRATEGIC ANALYSIS',
    'STRATEGIC', 'SYSTEM ANALYSIS', 'SYSTEM_ANALYSIS',
    'ARCHITECTURAL STATE', 'ARCHITECTURAL', 'STRATEGIC ROADMAP',
    'ROADMAP', 'TECHNICAL SUMMARY', 'STRATEGIC GAP ANALYSIS',
    'GAP ANALYSIS', 'STATE DECONSTRUCTION', 'SYSTEM',
    'STAGE 2', 'STAGE 2: RAW UNFILTERED RESPONSE', 'STAGE_2', 'STAGE 2:'
}


class ThoughtIsolation:
    """Enforces strict isolation between private reasoning scratchpads and speech output."""

    @staticmethod
    def strip_control_tags(text: str) -> str:
        """Removes XML and token wrappers from raw generation text."""
        if not text:
            return ""
        patterns = [
            r"</?\|thought\|>",
            r"</?\|channel>thought", r"</?channel\|>",
            r"</?\|channel\|>thought", r"</?\|channel\|>",
            r"</?thought>",
            r"</?think>",
            r"<start_of_turn>", r"<end_of_turn>",
            r"<\|turn>[a-z_]*\n?", r"<turn\|>", r"</turn>",
            r"<bos>", r"<eos>",
            r"\[/DRAFT\]", r"\[/thinking\]", r"\[/thought\]", r"\[/reasoning\]",
            r"\[DRAFT\]", r"\[thinking\]", r"\[thought\]", r"\[reasoning\]",
            r"[\u27e6\u27e7⟦⟧]"
        ]
        res = text
        for pat in patterns:
            res = re.sub(pat, "", res, flags=re.IGNORECASE | re.DOTALL)
        return res.strip()

    def isolate_stages(self, thought_stream: str, raw_response: str) -> Tuple[str, str, str]:
        """
        Splits generation output into:
        1. thought_stream: Private internal reasoning monologue.
        2. raw_response: Unfiltered tactical/technical analysis (Stage 2).
        3. final_response: Authentic refined Cecilia dialogue (Stage 3).
        """
        t_cleaned = self.strip_control_tags(thought_stream)
        r_cleaned = raw_response.strip()

        # Extract any embedded thought blocks in raw_response
        thought_match = re.search(r'(?s)<(?:think|thought|\|thought\|)>(.*?)</(?:think|thought|\|thought\|)>', r_cleaned, re.IGNORECASE)
        if thought_match:
            embedded_thought = self.strip_control_tags(thought_match.group(1).strip())
            if not t_cleaned:
                t_cleaned = embedded_thought
            else:
                t_cleaned = f"{t_cleaned}\n{embedded_thought}".strip()
            r_cleaned = (r_cleaned[:thought_match.start()] + " " + r_cleaned[thought_match.end():]).strip()

        # Check for explicit Stage markers
        stage3_match = re.search(
            r'(?:(?:\*\*|\*|__)?\[\s*(?:STAGE\s*3(?::\s*REFINED\s*RESPONSE)?|PERSONA|SPEECH|REFINED_RESPONSE|AUTHENTIC_VOICE|CECILIA)\s*\](?:\*\*|\*|__)?|(?:\*\*|\*|__)\s*(?:STAGE\s*3(?::\s*REFINED\s*RESPONSE)?|PERSONA|SPEECH|REFINED_RESPONSE|AUTHENTIC_VOICE|CECILIA)\s*(?:\*\*|\*|__)|⚡?\s*Cecilia(?:\s*\([^)]*\))?:)',
            r_cleaned,
            re.IGNORECASE
        )

        stage2_match = re.search(
            r'(?:(?:\*\*|\*|__)?\[\s*(?:STAGE\s*2(?::\s*RAW\s*UNFILTERED\s*RESPONSE)?|RAW_RESPONSE|UNFILTERED)\s*\](?:\*\*|\*|__)?|(?:\*\*|\*|__)\s*(?:STAGE\s*2(?::\s*RAW\s*UNFILTERED\s*RESPONSE)?|RAW_RESPONSE|UNFILTERED)\s*(?:\*\*|\*|__))',
            r_cleaned,
            re.IGNORECASE
        )

        if stage3_match:
            split_idx = stage3_match.end()
            speech = r_cleaned[split_idx:].strip()
            raw_tactical = r_cleaned[:stage3_match.start()].strip()
            if stage2_match and stage2_match.start() < stage3_match.start():
                raw_tactical = r_cleaned[stage2_match.end():stage3_match.start()].strip()
            # Clean persona prefix if duplicated
            speech = re.sub(r'^(?:⚡?\s*Cecilia(?:\s*\([^)]*\))?:\s*)', '', speech, flags=re.IGNORECASE).strip()
            return t_cleaned, raw_tactical, self.strip_control_tags(speech)

        # Fallback: check if entire raw response is wrapped in thought/deconstruction headers
        lines = r_cleaned.splitlines()
        speech_lines = []
        is_speech = False

        for line in lines:
            s = line.strip()
            if self._is_speech_header(s):
                is_speech = True
                continue
            if is_speech:
                speech_lines.append(line)

        if speech_lines:
            speech = "\n".join(speech_lines).strip()
            speech = re.sub(r'^(?:⚡?\s*Cecilia(?:\s*\([^)]*\))?:\s*)', '', speech, flags=re.IGNORECASE).strip()
            return t_cleaned, "", self.strip_control_tags(speech)

        clean_resp = re.sub(r'^(?:⚡?\s*Cecilia(?:\s*\([^)]*\))?:\s*)', '', r_cleaned, flags=re.IGNORECASE).strip()
        return t_cleaned, "", self.strip_control_tags(clean_resp)

    def _is_speech_header(self, line: str) -> bool:
        s = line.strip()
        if re.match(r'^⚡?\s*Cecilia(\s*\([^)]*\))?:', s, re.IGNORECASE):
            return True
        if re.match(r'^(?:\*\*\s*)?\[\s*(STAGE\s*3(?::\s*REFINED\s*RESPONSE)?|PERSONA|SPEECH|REFINED_RESPONSE|AUTHENTIC_VOICE|CECILIA)\s*\](?:\s*\*\*)?', s, re.IGNORECASE):
            return True
        if re.match(r'^\*\*\s*(STAGE\s*3(?::\s*REFINED\s*RESPONSE)?|PERSONA|SPEECH|REFINED_RESPONSE|AUTHENTIC_VOICE|CECILIA)\s*\*\*', s, re.IGNORECASE):
            return True
        return False

    def sanitize_speech(self, raw_text: str) -> str:
        """Sanitizes final text ensuring no leaked thinking tags or internal headers remain."""
        _, _, speech = self.isolate_stages("", raw_text)
        return speech
