import math
from typing import List, Dict, Any

class TriAttentionScorer:
    """
    Simulates the Trigonometric KV Compression logic (TriAttention 2026).
    Since we cannot natively hook into the C++ KV pre-RoPE activations,
    this Python equivalent uses a trigonometric sparsity sampler to score
    conversation turns (or token chunks) based on their relative positional
    importance to maintain "logical momentum" across massive contexts.
    """
    def __init__(self, budget_ratio: float = 0.5):
        # The fraction of context we want to retain when pruning kicks in
        self.budget_ratio = max(0.1, min(0.9, budget_ratio))

    def _trig_score(self, index: int, total: int) -> float:
        """
        Generates a score based on distance from the endpoints using 
        a simulated Tri-Attention inverse trigonometric decay curve.
        Keys near the beginning (system) and end (recent context) score highly.
        """
        if total <= 1:
            return 1.0
        
        # Normalize position to [0, 1]
        pos = index / (total - 1)
        
        # Simulated trigonometric attention concentration:
        # High at pos=0 (System/Root), High at pos=1 (Recent)
        # Trough in the middle, but with high-frequency "ripples" to catch structural data
        base_curve = math.cos(math.pi * pos) ** 2
        ripple = 0.1 * math.sin(10 * math.pi * pos) ** 2
        
        return base_curve + ripple
        
    def score_messages(self, messages: List[Dict[str, str]]) -> List[float]:
        """
        Assigns a retention score to each message in the context.
        Higher score means the message is more critical to retain.
        """
        scores = []
        total = len(messages)
        for i, msg in enumerate(messages):
            # System prompt is heavily protected
            if msg.get("role") == "system":
                scores.append(100.0)
                continue
                
            # Most recent 2 turns are heavily protected
            if i >= total - 2:
                scores.append(10.0 + self._trig_score(i, total))
                continue
                
            # Score based on position
            score = self._trig_score(i, total)
            
            # Semantic weight: Penalize 'hollow' short AI acknowledgments
            content = msg.get("content", "")
            if msg.get("role") == "assistant" and len(content) < 50:
                score *= 0.5
                
            scores.append(score)
            
        return scores

    def select_top_k_indices(self, scores: List[float], budget_count: int) -> List[int]:
        """
        Returns the indices of the messages that should be retained to fit the budget.
        Re-sorts indices chronologically to preserve sequence.
        """
        if budget_count >= len(scores):
            return list(range(len(scores)))
            
        # Get indices sorted by score descending
        sorted_by_score = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        # Keep the top 'budget_count'
        kept_indices = sorted_by_score[:budget_count]
        # Re-sort chronologically
        return sorted(kept_indices)
