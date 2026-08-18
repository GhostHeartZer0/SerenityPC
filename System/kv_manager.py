from typing import List, Dict, Any, Optional
from System.tri_attention_core import TriAttentionScorer
import logging
import numpy as np
import os

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


class TurboVecIndex:
    def __init__(self, history_dir, mode: str = "on"):
        import os
        import threading
        self.history_dir = history_dir
        self.mode = mode.lower() if isinstance(mode, str) else "on"
        self._ingested_files = set()
        self.metadata = []
        self.lock = threading.RLock()
        self.collection = None
        self.embedder = None

        if self.mode == "off":
            print("[TURBOVEC] Mode: OFF (indexing and search disabled).")
            return

        if self.mode == "fallback":
            print("[TURBOVEC] Mode: FALLBACK (keyword index active, heavy vector embeddings bypassed).")
            return

        # Mode == "on": Set quiet environment to eliminate HF Hub warnings & progress bar spam
        os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        try:
            import turbovec
            self.collection = turbovec.TurboQuantIndex(384)
        except Exception as e:
            print(f"[TURBOVEC] Failed to initialize TurboQuantIndex: {e}")

        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("[TURBOVEC] Mode: ON (quantized vector index & MiniLM ready on CPU).")
        except Exception as e:
            print(f"[TURBOVEC] Failed to initialize SentenceTransformer: {e}")
            print("[TURBOVEC] Gracefully falling back to keyword indexing.")
            self.mode = "fallback"

    def ingest_needed_files(self, active_model_path: Optional[str], active_level: Optional[int], lookup_mode: str):
        if self.mode == "off":
            return

        import glob
        import os
        import re
        import json
        import zlib

        with self.lock:
            active_model_name = None
            if active_model_path:
                active_model_name = os.path.splitext(os.path.basename(active_model_path))[0].lower()

            files = glob.glob(os.path.join(self.history_dir, "*.history.jsonz"))
            files_to_load = []

            for f in files:
                basename = os.path.basename(f)
                match = re.search(r"^(.*)_lvl(\d+)\.history\.jsonz$", basename)
                if not match:
                    continue
                
                f_model = match.group(1).lower()
                f_level = int(match.group(2))

                should_load = False
                if lookup_mode == "targeted":
                    if active_model_name and f_model == active_model_name and active_level is not None and f_level == active_level:
                        should_load = True
                elif lookup_mode == "model":
                    if active_model_name and f_model == active_model_name:
                        should_load = True
                elif lookup_mode == "level":
                    if active_level is not None and f_level == active_level:
                        should_load = True
                elif lookup_mode == "all":
                    should_load = True

                if should_load:
                    files_to_load.append((f, f_model, f_level))

            newly_ingested = 0
            for f, f_model, f_level in files_to_load:
                if f in self._ingested_files:
                    continue
                try:
                    with open(f, 'rb') as fp:
                        history = json.loads(zlib.decompress(fp.read()).decode('utf-8'))
                    
                    texts = []
                    temp_metadata = []
                    for msg in history:
                        content = msg.get("content", "")
                        if len(content) > 20:
                            texts.append(content)
                            temp_metadata.append({
                                "content": content,
                                "role": msg.get("role"),
                                "file": f,
                                "model": f_model,
                                "level": f_level
                            })
                    
                    if texts and self.mode == "on" and self.embedder is not None and self.collection is not None:
                        vecs = self.embedder.encode(texts, convert_to_numpy=True)
                        self.collection.add(vecs.astype(np.float32))
                        self.metadata.extend(temp_metadata)
                        newly_ingested += len(texts)
                    elif texts:
                        self.metadata.extend(temp_metadata)
                        newly_ingested += len(texts)

                    self._ingested_files.add(f)
                except Exception as e:
                    print(f"[TURBOVEC] Failed to parse history {f}: {e}")

            if newly_ingested > 0:
                print(f"[TURBOVEC] Ingested {newly_ingested} new chunks. Total indexed: {len(self.metadata)}")

    def search(self, query: str, top_k: int = 3, active_model_path: Optional[str] = None, active_level: Optional[int] = None, lookup_mode: str = "targeted"):
        if self.mode == "off":
            return []

        import numpy as np
        
        with self.lock:
            self.ingest_needed_files(active_model_path, active_level, lookup_mode)
            
            if not self.metadata:
                return []

            active_model_name = None
            if active_model_path:
                active_model_name = os.path.splitext(os.path.basename(active_model_path))[0].lower()

            mask = np.zeros(len(self.metadata), dtype=bool)
            for i, meta in enumerate(self.metadata):
                match = False
                if lookup_mode == "targeted":
                    if active_model_name and meta["model"] == active_model_name and active_level is not None and meta["level"] == active_level:
                        match = True
                elif lookup_mode == "model":
                    if active_model_name and meta["model"] == active_model_name:
                        match = True
                elif lookup_mode == "level":
                    if active_level is not None and meta["level"] == active_level:
                        match = True
                elif lookup_mode == "all":
                    match = True
                
                mask[i] = match

            if not np.any(mask):
                return []

            # Vector similarity search if mode is ON
            if self.mode == "on" and self.embedder is not None and self.collection is not None:
                try:
                    query_vec = self.embedder.encode([query], convert_to_numpy=True).astype(np.float32)
                    distances, indices = self.collection.search(query_vec, k=top_k, mask=mask)
                    
                    results = []
                    if len(indices) > 0:
                        for idx in indices[0]:
                            if 0 <= idx < len(self.metadata):
                                results.append(self.metadata[idx]["content"])
                    return results
                except Exception as e:
                    print(f"[TURBOVEC] Vector search failed ({e}), falling back to keyword search.")

            # Fallback keyword search
            query_terms = [t for t in query.lower().split() if len(t) > 3]
            scored_res = []
            for i, meta in enumerate(self.metadata):
                if mask[i]:
                    c_lower = meta["content"].lower()
                    score = sum(c_lower.count(t) for t in query_terms)
                    if score > 0:
                        scored_res.append((score, meta["content"]))
            
            scored_res.sort(key=lambda x: x[0], reverse=True)
            return [text for _, text in scored_res[:top_k]]
