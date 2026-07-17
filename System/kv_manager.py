from typing import List, Dict, Any, Optional
from System.tri_attention_core import TriAttentionScorer
import logging

class KVManager:
    """
    Manages the context window by pruning messages when they exceed
    the maximum context threshold. It uses TriAttention simulated
    scoring to determine which messages to keep.
    """
    def __init__(self, max_context_tokens: int = 120000, prune_ratio: float = 0.5):
        # Default target of 120k for massive context, but configurable
        self.max_context_tokens = max_context_tokens
        self.prune_ratio = prune_ratio
        self.scorer = TriAttentionScorer(budget_ratio=prune_ratio)
        self.logger = logging.getLogger("KVManager")

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (typically 4 chars = 1 token for LLMs)"""
        return max(1, len(text) // 4)

    def _get_messages_token_count(self, messages: List[Dict[str, str]]) -> int:
        return sum(self._estimate_tokens(msg.get("content", "")) for msg in messages)

    def enforce_kv_budget(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Evaluates the message list. If it exceeds the maximum context threshold,
        it prunes the sequence down to `prune_ratio` * max_context via Tri-Attention.
        """
        current_tokens = self._get_messages_token_count(messages)
        
        # If we have headroom, we don't prune. We use a 90% threshold for triggering
        trigger_threshold = int(self.max_context_tokens * 0.9)
        
        if current_tokens < trigger_threshold:
            return messages
            
        self.logger.info(f"[TriAttention] KV Cache overflow detected ({current_tokens} > {trigger_threshold}). Pruning...")
        
        # Calculate how many messages we can afford based on average tokens per message
        # Let's target the exact budget fraction of the total token count
        target_tokens = int(self.max_context_tokens * self.prune_ratio)
        
        # Score the messages
        scores = self.scorer.score_messages(messages)
        
        # Sort indices by score descending, keep adding until budget is full
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        kept_indices = []
        tokens_accumulated = 0
        
        # Always keep System and the last User message if possible
        for idx in sorted_indices:
            msg_tokens = self._estimate_tokens(messages[idx].get("content", ""))
            
            # System and last request are mandatory
            is_mandatory = (messages[idx].get("role") == "system" or idx == len(messages) - 1)
            
            if tokens_accumulated + msg_tokens <= target_tokens or is_mandatory:
                kept_indices.append(idx)
                tokens_accumulated += msg_tokens
                
        # Chronologically sort the allowed indices
        kept_indices = sorted(list(set(kept_indices)))
        
        pruned_messages = [messages[i] for i in kept_indices]
        new_tokens = self._get_messages_token_count(pruned_messages)
        
        self.logger.info(f"[TriAttention] KV Pruning Complete. Kept {len(kept_indices)}/{len(messages)} turns. Tokens: {current_tokens} -> {new_tokens}")
        
        return pruned_messages
        
    def enforce_string_kv_budget(self, text: str) -> str:
        """
        Prunes a single massive string (like deep cook draft history) using chunking.
        """
        current_tokens = self._estimate_tokens(text)
        trigger_threshold = int(self.max_context_tokens * 0.9)
        
        if current_tokens < trigger_threshold:
            return text
            
        self.logger.info(f"[TriAttention] String overflow detected ({current_tokens} > {trigger_threshold}). Chunk Pruning...")
        
        # Chunk text loosely by paragraphs
        chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
        if len(chunks) < 3: return text # Too few chunks to safely prune
        
        target_tokens = int(self.max_context_tokens * self.prune_ratio)
        dummy_messages = [{"role": "user", "content": c} for c in chunks]
        
        scores = self.scorer.score_messages(dummy_messages)
        sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        kept_indices = []
        tokens_accumulated = 0
        
        for idx in sorted_indices:
            chunk_tokens = self._estimate_tokens(chunks[idx])
            # Keep the very final chunk (most recent reasoning) always
            if tokens_accumulated + chunk_tokens <= target_tokens or idx == len(chunks) - 1:
                kept_indices.append(idx)
                tokens_accumulated += chunk_tokens
                
        kept_indices = sorted(list(set(kept_indices)))
        pruned_text = "\n\n[...] (Sparsified by TriAttention) [...]\n\n".join([chunks[i] for i in kept_indices])
        return pruned_text
