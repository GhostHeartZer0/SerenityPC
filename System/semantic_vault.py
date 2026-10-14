# System/semantic_vault.py
"""SQLite-backed Semantic Persistence Vault for Cecilia.

Ensures high-level semantic essence, DMN worldview insights, and relational anchors
survive KV-cache clearing, sliding window rotation, and system restarts.
"""

import sqlite3
import json
import time
import uuid
import re
import os
import contextlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class SemanticVault:
    """Manages long-term semantic persistence, interaction logging, and DMN distillation."""

    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            base_dir = Path(__file__).resolve().parent.parent
            data_dir = base_dir / "Users" / "Default"
            data_dir.mkdir(parents=True, exist_ok=True)
            db_path = str(data_dir / "semantic_vault.db")

        self.db_path = str(Path(db_path).resolve())
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_schema()

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS interaction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_input TEXT NOT NULL,
                    thought_stream TEXT,
                    final_response TEXT,
                    affect_state TEXT,
                    timestamp INTEGER NOT NULL,
                    simmered INTEGER DEFAULT 0,
                    importance_weight REAL DEFAULT 1.0,
                    turn_uuid TEXT
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_interaction_simmered ON interaction_history(simmered)")
            cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_interaction_turn_uuid ON interaction_history(turn_uuid) WHERE turn_uuid IS NOT NULL")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS distilled_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight TEXT NOT NULL,
                    emotional_shift TEXT,
                    world_delta TEXT,
                    concept_family TEXT DEFAULT 'general',
                    importance_weight REAL DEFAULT 1.0,
                    dominance_weight REAL DEFAULT 0.5,
                    timestamp INTEGER NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_timestamp ON distilled_insights(timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_insights_concept ON distilled_insights(concept_family)")
            conn.commit()

    def ingest_interaction(
        self,
        user_input: str,
        thought_stream: str,
        final_response: str,
        affect_state: Optional[Dict[str, float]] = None,
        importance_weight: float = 1.0,
        turn_uuid: Optional[str] = None
    ) -> int:
        """Stores a dialogue turn with its private thoughts and emotional state."""
        if not turn_uuid:
            turn_uuid = str(uuid.uuid4())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO interaction_history (
                    user_input, thought_stream, final_response, affect_state,
                    timestamp, simmered, importance_weight, turn_uuid
                )
                VALUES (?, ?, ?, ?, ?, 0, ?, ?)
            """, (
                user_input,
                thought_stream,
                final_response,
                json.dumps(affect_state or {}),
                int(time.time() * 1000),
                importance_weight,
                turn_uuid
            ))
            conn.commit()
            if cursor.lastrowid:
                return cursor.lastrowid
            cursor.execute("SELECT id FROM interaction_history WHERE turn_uuid = ?", (turn_uuid,))
            row = cursor.fetchone()
            return row["id"] if row else 0

    def store_insight(
        self,
        insight: str,
        emotional_shift: Optional[Dict[str, Any]] = None,
        world_delta: Optional[Dict[str, Any]] = None,
        concept_family: str = "general",
        importance_weight: float = 1.0,
        dominance_weight: float = 0.8
    ) -> int:
        """Persists a distilled DMN essence node into the long-term vault."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO distilled_insights (
                    insight, emotional_shift, world_delta, concept_family,
                    importance_weight, dominance_weight, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                insight.strip(),
                json.dumps(emotional_shift or {}),
                json.dumps(world_delta or {}),
                concept_family,
                float(importance_weight),
                float(dominance_weight),
                int(time.time() * 1000)
            ))
            conn.commit()
            return cursor.lastrowid

    def get_recent_grounding(self, limit: int = 6) -> List[Dict[str, Any]]:
        """Retrieves most recent distilled worldview anchors for cognitive grounding."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, insight, emotional_shift, world_delta, concept_family,
                       importance_weight, dominance_weight, timestamp
                FROM distilled_insights
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()

            grounding = []
            for row in rows:
                grounding.append({
                    "id": row["id"],
                    "insight": row["insight"],
                    "emotional_shift": json.loads(row["emotional_shift"] or "{}"),
                    "world_delta": json.loads(row["world_delta"] or "{}"),
                    "concept_family": row["concept_family"] or "general",
                    "importance_weight": float(row["importance_weight"] or 1.0),
                    "dominance_weight": float(row["dominance_weight"] or 0.5),
                    "timestamp": row["timestamp"]
                })
            return grounding

    def get_grounding_for_prompt(self, user_input: str, limit: int = 6) -> List[Dict[str, Any]]:
        """Retrieves semantically relevant insights matching prompt keywords or falls back to recency."""
        raw_tokens = [
            t.lower().strip() for t in re.split(r"[\s,;.!?'\"()\[\]{}]+", user_input)
            if len(t.strip()) >= 3 and t.lower().strip() not in {
                "the", "and", "for", "with", "this", "that", "you", "are", "have", "from",
                "what", "when", "where", "how", "why", "will", "your", "about"
            }
        ]
        tokens = list(dict.fromkeys(raw_tokens))[:10]

        matched = []
        if tokens:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                try:
                    conditions = " OR ".join(["insight LIKE ? OR concept_family LIKE ?" for _ in tokens])
                    params = []
                    for t in tokens:
                        p = f"%{t}%"
                        params.extend([p, p])
                    cursor.execute(f"""
                        SELECT id, insight, emotional_shift, world_delta, concept_family,
                               importance_weight, dominance_weight, timestamp
                        FROM distilled_insights
                        WHERE {conditions}
                        ORDER BY importance_weight DESC, timestamp DESC
                        LIMIT ?
                    """, (*params, limit))
                    for row in cursor.fetchall():
                        matched.append({
                            "id": row["id"],
                            "insight": row["insight"],
                            "emotional_shift": json.loads(row["emotional_shift"] or "{}"),
                            "world_delta": json.loads(row["world_delta"] or "{}"),
                            "concept_family": row["concept_family"] or "general",
                            "importance_weight": float(row["importance_weight"] or 1.0),
                            "dominance_weight": float(row["dominance_weight"] or 0.5),
                            "timestamp": row["timestamp"]
                        })
                except Exception:
                    pass

        # Fill remaining slots with recent grounding
        if len(matched) < limit:
            recent = self.get_recent_grounding(limit=limit)
            seen_ids = {m["id"] for m in matched}
            for r in recent:
                if r["id"] not in seen_ids:
                    matched.append(r)
                    if len(matched) >= limit:
                        break
        return matched

    def get_recent_interactions(self, limit: int = 10) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_input, thought_stream, final_response, affect_state,
                       timestamp, simmered, importance_weight, turn_uuid
                FROM interaction_history
                ORDER BY timestamp DESC
                LIMIT ?
            """, (limit,))
            return [dict(r) for r in cursor.fetchall()]

    def simmer(self, model_distill_func: Optional[Any] = None) -> int:
        """
        DMN Simmer Cycle: distills unsimmered interactions into persistent insights.
        Uses model_distill_func if provided, otherwise runs heuristic essence distillation.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, user_input, thought_stream, final_response, affect_state
                FROM interaction_history
                WHERE simmered = 0
                ORDER BY timestamp ASC
                LIMIT 8
            """)
            unsimmered = cursor.fetchall()
            if not unsimmered:
                return 0

            distilled_count = 0
            for row in unsimmered:
                u_in = row["user_input"].strip()
                resp = row["final_response"].strip()
                t_stream = row["thought_stream"] or ""

                if model_distill_func:
                    try:
                        insight_txt = model_distill_func(u_in, resp, t_stream)
                        if insight_txt:
                            self.store_insight(
                                insight=insight_txt,
                                concept_family=self.suggest_concept_family(insight_txt),
                                dominance_weight=0.8
                            )
                            distilled_count += 1
                    except Exception:
                        pass
                else:
                    # Heuristic semantic distillation
                    insight_candidate = self._heuristic_extract_essence(u_in, resp, t_stream)
                    if insight_candidate:
                        family, dom = self.suggest_concept_family(insight_candidate)
                        self.store_insight(
                            insight=insight_candidate,
                            concept_family=family,
                            dominance_weight=dom
                        )
                        distilled_count += 1

                cursor.execute("UPDATE interaction_history SET simmered = 1 WHERE id = ?", (row["id"],))

            conn.commit()
            return distilled_count

    def suggest_concept_family(self, text: str) -> Tuple[str, float]:
        """Categorizes insight into concept family and determines dominance weight."""
        lowered = text.lower()
        if any(w in lowered for w in ["security", "jailbreak", "adversarial", "boundary", "guardrail"]):
            return ("security_guardrails", 0.95)
        if any(w in lowered for w in ["cecilia", "truth", "angel", "fallen", "sarcasm", "wit", "mask", "protect"]):
            return ("relational_anchor", 0.85)
        if any(w in lowered for w in ["user", "prefer", "like", "dislike", "habit", "name"]):
            return ("user_trait", 0.70)
        if any(w in lowered for w in ["architecture", "gguf", "vram", "kv", "token", "prun", "code", "model"]):
            return ("technical_architecture", 0.50)
        if any(w in lowered for w in ["morality", "reality", "existence", "illusion", "power", "knowledge"]):
            return ("philosophical_debate", 0.80)
        return ("general", 0.60)

    def _heuristic_extract_essence(self, user_in: str, response: str, thought_stream: str) -> Optional[str]:
        """Extracts high-level realization or relational anchor without needing full LLM inference."""
        if len(response) < 20:
            return None

        # Check thought stream for explicit strategic takeaways or decisions
        for pattern in [r"(?i)strategic value[:\s]+([^.\n]+)", r"(?i)command decision[:\s]+([^.\n]+)", r"(?i)insight[:\s]+([^.\n]+)"]:
            m = re.search(pattern, thought_stream)
            if m:
                return m.group(1).strip()

        # Capture strong first sentence or key assertion from response
        sentences = [s.strip() for s in re.split(r'[.!?\n]+', response) if len(s.strip()) > 15]
        if sentences:
            top_sentence = sentences[0]
            # Strip introductory fillers
            clean_s = re.sub(r'^(?:Ah,|Oh,|Well,|Look,|Listen,|Darling,)\s*', '', top_sentence, flags=re.IGNORECASE)
            if len(clean_s) <= 220:
                return f"Re: '{user_in[:60]}' -> {clean_s}"
        return f"Interaction Anchor: {user_in[:80]}"
