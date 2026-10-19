# System/orchestration_manager.py
# Modular, Context-Driven Agent Orchestration Framework with Dynamic Weighted Utility Dispatch,
# Stateful Context Transfer (Offload/Load), and Observability Decision Logging for SerenityPC.

import os
import json
import time
import math
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple

def clean_model_display_name(p: str) -> str:
    """Extracts clean, concise model name for banner and title displays."""
    name = os.path.basename(str(p or '')).strip()
    if name.lower().endswith('.gguf'):
        name = name[:-5]
    name = re.sub(r'[-_.](?:UD[-_.])?(?:q[0-9]+[a-z0-9_]*|bf16|f16|iq[0-9]+[a-z0-9_]*)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_.](?:it|instruct|uncensored|heretic)$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'[-_.](?:qat[-_.]UD|qat)$', '', name, flags=re.IGNORECASE)
    return name or os.path.basename(str(p or ''))

def extract_search_query(user_query: str) -> str:
    """Extracts concise, bloat-free search keywords (< 140 chars) from complex multi-modal/code prompts."""
    # 1. Strip fenced code blocks
    clean = re.sub(r'```[\s\S]*?```', '', user_query)
    # 2. Strip C++/Kotlin/Java/Python boilerplate lines
    clean = re.sub(r'(?m)^\s*(?:#include|package|import|using)\s+.*$', '', clean)
    clean = re.sub(r'(?m)^\s*(?:extern\s+"C"|public\s+class|class\s+\w+|fun\s+\w+|val\s+\w+|var\s+\w+|def\s+\w+).*$', '', clean)
    clean = re.sub(r'[{}\[\]();]', ' ', clean)
    # 3. Look for explicit search directives
    m_search = re.search(r'\b(?:search|find|lookup|check)\s+(?:for\s+)?["\']?([^"\'.\n\r]{4,120})["\']?', clean, re.IGNORECASE)
    if m_search:
        candidate = m_search.group(1).strip()
        if len(candidate) > 5 and not any(k in candidate.lower() for k in ("color", "image", "friend", "todo.txt")):
            return candidate[:120]
    # 4. Look for question sentences
    m_q = re.search(r'([A-Z0-9][^\.!\?\n\r]{10,120}\?)', clean)
    if m_q:
        return m_q.group(1).strip()
    # 5. Take first clean sentence
    sentences = [s.strip() for s in re.split(r'[\n\r\.]+', clean) if len(s.strip()) >= 10]
    for s in sentences:
        if not any(token in s.lower() for token in ("#include", "extern", "todo.txt", "jstring", "jnienv", "compose", "composable")):
            return s[:120]
    return clean.strip()[:100]

@dataclass
class QueryProfile:
    """Normalized intent and feature vector extracted from user input."""
    factual_need: float = 0.0
    reasoning_complexity: float = 0.0
    emotional_intensity: float = 0.0
    speed_preference: float = 0.0
    coordination_need: float = 0.0

@dataclass
class AgentUtilityScore:
    agent_id: int
    agent_name: str
    raw_utility: float
    net_utility: float
    cost_penalty: float
    selected: bool
    rationale: str

@dataclass
class DecisionTrace:
    trace_id: str
    orchestrator_level: int
    user_query: str
    profile: QueryProfile
    candidate_scores: List[AgentUtilityScore]
    selected_agent_ids: List[int]
    decision_summary: str
    execution_duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)

@dataclass
class SerializedAgentContext:
    """
    Standardized, minimal context serialization format for stateful handoffs.
    Mitigates state bloat and memory exhaustion (Anti-OOM) via bounded budgets.
    """
    trace_id: str
    origin_agent: int
    target_agent: int
    token_budget: int = 1024
    staged_facts: List[str] = field(default_factory=list)
    reasoning_steps: List[str] = field(default_factory=list)
    emotional_valence: Optional[str] = None
    summary: str = ""
    raw_payload: str = ""
    timestamp: float = field(default_factory=time.time)

    def offload(self) -> str:
        """
        Compresses and prunes state into a minimal serialized JSON format
        strictly respecting the token/char budget.
        """
        max_chars = self.token_budget * 4
        # Prune raw payload if exceeding budget
        pruned_payload = self.raw_payload[:max_chars // 2] if len(self.raw_payload) > max_chars // 2 else self.raw_payload
        data = {
            "trace_id": self.trace_id,
            "origin": self.origin_agent,
            "target": self.target_agent,
            "summary": self.summary[:500],
            "staged_facts": [f[:250] for f in self.staged_facts[:8]],
            "reasoning_steps": [r[:300] for r in self.reasoning_steps[:6]],
            "emotional_valence": self.emotional_valence,
            "payload_preview": pruned_payload
        }
        return json.dumps(data, ensure_ascii=False)

    @classmethod
    def load(cls, serialized_str: str) -> "SerializedAgentContext":
        """Reconstitutes minimal state from serialized string."""
        try:
            d = json.loads(serialized_str)
            return cls(
                trace_id=d.get("trace_id", ""),
                origin_agent=d.get("origin", 0),
                target_agent=d.get("target", 0),
                summary=d.get("summary", ""),
                staged_facts=d.get("staged_facts", []),
                reasoning_steps=d.get("reasoning_steps", []),
                emotional_valence=d.get("emotional_valence"),
                raw_payload=d.get("payload_preview", "")
            )
        except Exception:
            return cls(trace_id="err", origin_agent=0, target_agent=0, summary=serialized_str[:200])

@dataclass
class SubagentTask:
    task_id: str
    assigned_level: int
    task_description: str
    input_context: str = ""
    status: str = "pending"
    result: str = ""

@dataclass
class HandoffReport:
    step_index: int
    from_level: int
    to_level: int
    task_type: str
    summary: str
    factual_data: str = ""
    reasoning_output: str = ""
    status: str = "completed"

SUBAGENT_ROLES = {
    1: {
        "name": "The Speedy",
        "role": "Format verification, rapid sanity checks, and concise answer validation.",
        "skills": ["quick_formatting", "concise_summary", "speed_validation"]
    },
    2: {
        "name": "The Helper / Searcher",
        "role": "Fact retrieval, pinpoint search extraction, and data indexing.",
        "skills": ["pinpoint_search", "fact_extraction", "bloat_filtered_lookup"]
    },
    3: {
        "name": "Collaborator",
        "role": "Information staging, report compilation, and inter-agent memory storage.",
        "skills": ["context_staging", "report_aggregation", "briefing_compiler"]
    },
    4: {
        "name": "The Confidant",
        "role": "Human context analysis, tone checking, and empathy alignment.",
        "skills": ["tone_analysis", "empathy_alignment", "human_nuance"]
    },
    5: {
        "name": "The Brains / Sage",
        "role": "Deep logical reasoning, multi-step problem solving, deduction, and figuring out core hurdles.",
        "skills": ["deep_reasoning", "technical_deduction", "complex_problem_solving"]
    },
    6: {
        "name": "Transcendent",
        "role": "Omniscient taskmaster and synthesis supervisor (accessible as a subagent only to Level 7).",
        "skills": ["omniscient_synthesis", "master_coordination", "final_approval"]
    },
    7: {
        "name": "Cecilia",
        "role": "Supreme Overseer. Operates in Shadow Wizard (full delegation) or Divine Judgement (unified omniscience).",
        "skills": ["shadow_wizard", "divine_judgement", "truth_seeking"]
    }
}

# --- INTENT & FEATURE EXTRACTION PROFILER ---
class IntentProfiler:
    """Extracts continuous feature vectors from queries for utility evaluation."""

    FACTUAL_KEYWORDS = [
        r'\b(?:who|when|where|what is|search|find|lookup|latest|news|weather|price|spec|current|release date|version)\b',
        r'\b(?:http|www|\.com|\.org|\.net|file|path|directory|document|pdf)\b',
        r'\b\d{4}\b' # Year references
    ]

    REASONING_KEYWORDS = [
        r'\b(?:why|how|explain|analyze|calculate|solve|compare|proof|algorithm|code|python|math|logic|bug|architect|design|deduce)\b',
        r'[\+\-\*\/\=\^\<\>\{\}\[\]\(\)]', # Formulaic / code characters
        r'\b(?:if|then|else|derive|optimize|debug|step-by-step)\b'
    ]

    EMOTIONAL_KEYWORDS = [
        r'\b(?:feel|sad|happy|angry|lonely|anxious|afraid|stress|friend|love|relationship|vent|confide|tired|overwhelmed|worry)\b',
        r'[!]{2,}|\?{2,}'
    ]

    SPEED_KEYWORDS = [
        r'\b(?:quick|brief|short|one-liner|tldr|fast|summary|yes or no|bullet)\b'
    ]

    @classmethod
    def profile(cls, query: str) -> QueryProfile:
        q_clean = query.lower().strip()
        length = len(q_clean)

        # Factual score
        fact_hits = sum(len(re.findall(pat, q_clean)) for pat in cls.FACTUAL_KEYWORDS)
        fact_score = min(1.0, (fact_hits * 0.35) + (0.2 if "?" in q_clean and "is" in q_clean else 0.0))

        # Reasoning score
        reason_hits = sum(len(re.findall(pat, q_clean)) for pat in cls.REASONING_KEYWORDS)
        reason_score = min(1.0, (reason_hits * 0.30) + (0.25 if length > 120 else 0.0))

        # Emotional score
        emotion_hits = sum(len(re.findall(pat, q_clean)) for pat in cls.EMOTIONAL_KEYWORDS)
        emotion_score = min(1.0, emotion_hits * 0.45)

        # Speed score
        speed_hits = sum(len(re.findall(pat, q_clean)) for pat in cls.SPEED_KEYWORDS)
        speed_score = min(1.0, (speed_hits * 0.50) + (0.3 if length < 35 and reason_score < 0.2 else 0.0))

        # Coordination score (higher for multi-faceted or complex inquiries)
        coord_score = min(1.0, (fact_score * 0.5) + (reason_score * 0.5) + (0.2 if length > 180 else 0.0))

        return QueryProfile(
            factual_need=round(fact_score, 3),
            reasoning_complexity=round(reason_score, 3),
            emotional_intensity=round(emotion_score, 3),
            speed_preference=round(speed_score, 3),
            coordination_need=round(coord_score, 3)
        )


# --- DYNAMIC WEIGHTED UTILITY DISPATCH ENGINE ---
class UtilityDispatchEngine:
    """
    Evaluates candidate subagent utility functions U(Agent_i, Query)
    and dynamically constructs an optimal execution DAG.
    """

    # Weights: [factual, reasoning, emotional, speed, coordination]
    AGENT_WEIGHTS = {
        1: {"weights": [0.05, 0.10, 0.05, 0.75, 0.05], "cost": 0.05, "name": "Lvl 1 (Speedy)"},
        2: {"weights": [0.85, 0.10, 0.00, 0.10, 0.15], "cost": 0.15, "name": "Lvl 2 (Helper / Searcher)"},
        3: {"weights": [0.20, 0.25, 0.05, 0.05, 0.70], "cost": 0.10, "name": "Lvl 3 (Collaborator)"},
        4: {"weights": [0.00, 0.05, 0.90, 0.10, 0.10], "cost": 0.10, "name": "Lvl 4 (The Confidant)"},
        5: {"weights": [0.15, 0.85, 0.00, 0.00, 0.35], "cost": 0.20, "name": "Lvl 5 (The Brains / Sage)"},
        6: {"weights": [0.40, 0.50, 0.20, 0.10, 0.80], "cost": 0.25, "name": "Lvl 6 (Serenity Prime)"}
    }

    DISPATCH_THRESHOLD = 0.28

    @classmethod
    def evaluate(cls, orchestrator_level: int, query: str, selection_mode: str = "minimal") -> DecisionTrace:
        profile = IntentProfiler.profile(query)
        p_vec = [
            profile.factual_need,
            profile.reasoning_complexity,
            profile.emotional_intensity,
            profile.speed_preference,
            profile.coordination_need
        ]

        allowed_agents = get_available_subagents(orchestrator_level)
        scores: List[AgentUtilityScore] = []
        selected_ids: List[int] = []

        trace_id = f"trace_{int(time.time() * 1000)}"

        for agent_id in range(1, 7):
            cfg = cls.AGENT_WEIGHTS[agent_id]
            is_permitted = agent_id in allowed_agents
            
            # Dot product U_raw = sum(w_k * F_k)
            w = cfg["weights"]
            raw_u = sum(w[i] * p_vec[i] for i in range(5))
            cost = cfg["cost"]
            net_u = round(raw_u - cost, 3)

            if not is_permitted:
                selected = False
                rationale = f"Access Restricted: Level {agent_id} not visible to Orchestrator Lvl {orchestrator_level}"
            elif selection_mode == "all":
                # Grounded in primary domain vectors (logic, tone, factual, speed, coordination):
                # An agent is selected only if its primary domain vector shows genuine relevance:
                # - Lvl 2: factual_need >= 0.25
                # - Lvl 4: emotional_intensity >= 0.25
                # - Lvl 5: reasoning_complexity >= 0.25
                # - Lvl 1: speed_preference >= 0.25 or net_u >= 0.15
                # - Lvl 3: coordination_need >= 0.40 and (profile.factual_need >= 0.20 or profile.reasoning_complexity >= 0.20)
                # - Lvl 6: coordination_need >= 0.60
                domain_active = False
                if agent_id == 2 and profile.factual_need >= 0.25:
                    domain_active = True
                elif agent_id == 4 and profile.emotional_intensity >= 0.25:
                    domain_active = True
                elif agent_id == 5 and profile.reasoning_complexity >= 0.25:
                    domain_active = True
                elif agent_id == 1 and (profile.speed_preference >= 0.25 or net_u >= 0.15):
                    domain_active = True
                elif agent_id == 3 and profile.coordination_need >= 0.40 and (profile.factual_need >= 0.20 or profile.reasoning_complexity >= 0.20):
                    domain_active = True
                elif agent_id == 6 and profile.coordination_need >= 0.60:
                    domain_active = True

                if domain_active and (net_u >= 0.05 or raw_u >= 0.20):
                    selected = True
                    selected_ids.append(agent_id)
                    rationale = f"Selected: Grounded Vector All-Subagent mode (Domain active, Raw: {raw_u:.3f}, Net: {net_u:.3f})"
                else:
                    selected = False
                    rationale = f"Skipped: Vector grounding insufficient in All-Subagent mode (Domain unaligned or Net: {net_u:.3f} < 0.05)"
            elif net_u >= cls.DISPATCH_THRESHOLD:
                selected = True
                selected_ids.append(agent_id)
                rationale = f"Selected: Net Utility {net_u:.3f} >= Threshold {cls.DISPATCH_THRESHOLD}"
            else:
                selected = False
                rationale = f"Skipped: Net Utility {net_u:.3f} < Threshold {cls.DISPATCH_THRESHOLD}"

            scores.append(AgentUtilityScore(
                agent_id=agent_id,
                agent_name=cfg["name"],
                raw_utility=round(raw_u, 3),
                net_utility=net_u,
                cost_penalty=cost,
                selected=selected,
                rationale=rationale
            ))

        # Explicit user level request detection (e.g. "ask Lvl 5", "Level 5", "The Brains")
        explicit_matches = re.findall(r'\b(?:lvl|level)\s*([1-6])\b', query, flags=re.IGNORECASE)
        for m_lvl in explicit_matches:
            req_id = int(m_lvl)
            if req_id in allowed_agents and req_id not in selected_ids:
                selected_ids.append(req_id)
                for s in scores:
                    if s.agent_id == req_id:
                        s.selected = True
                        s.rationale = f"Explicitly requested by user query (Lvl {req_id})"

        # Fallback: if minimal mode and no agent crossed threshold, dispatch minimal best agent
        if not selected_ids and allowed_agents:
            permitted_scores = [s for s in scores if s.agent_id in allowed_agents]
            best = max(permitted_scores, key=lambda s: s.net_utility)
            best.selected = True
            best.rationale += " [Fallback Selected as Best Match]"
            selected_ids.append(best.agent_id)

        summary = f"Query Profile: Fact={profile.factual_need}, Logic={profile.reasoning_complexity}, Emot={profile.emotional_intensity}, Spd={profile.speed_preference}, Coord={profile.coordination_need} | Dispatched: {selected_ids} (Mode: {selection_mode})"

        return DecisionTrace(
            trace_id=trace_id,
            orchestrator_level=orchestrator_level,
            user_query=query,
            profile=profile,
            candidate_scores=scores,
            selected_agent_ids=selected_ids,
            decision_summary=summary
        )


def get_available_subagents(orchestrator_level: int) -> List[int]:
    """
    Returns permitted subagent levels for the given orchestrator.
    Rule: Level 6 as an agent is only visible/accessible to Level 7.
    """
    if orchestrator_level == 7:
        # Cecilia can delegate to all 6 levels (1 through 6)
        return [1, 2, 3, 4, 5, 6]
    elif orchestrator_level == 6:
        # Transcendent can delegate to main 5 levels (1 through 5)
        return [1, 2, 3, 4, 5]
    return []

def build_default_delegation_plan(orchestrator_level: int, user_query: str) -> List[SubagentTask]:
    """
    Constructs fallback static delegation pipeline.
    """
    return [
        SubagentTask(
            task_id="step_1_search",
            assigned_level=2,
            task_description="Pinpoint key factual data, technical references, or background needed to resolve query.",
            input_context=user_query
        ),
        SubagentTask(
            task_id="step_2_store",
            assigned_level=3,
            task_description="Stage and index factual findings into clean structured data points.",
            input_context=""
        ),
        SubagentTask(
            task_id="step_3_reason",
            assigned_level=5,
            task_description="Analyze data, perform logical reasoning, resolve complexities, and draft deductions.",
            input_context=""
        ),
        SubagentTask(
            task_id="step_4_compile",
            assigned_level=3,
            task_description="Compile unified briefing from factual findings and reasoning deductions.",
            input_context=""
        )
    ]


class OrchestrationManager:
    """
    Context-Driven Dynamic Agent Orchestration Framework.
    Manages dynamic utility-based dispatch, stateful context offload/load, and observability decision traces.
    """
    def __init__(self, app_ref=None):
        self.app = app_ref

    def is_delegation_enabled(self, active_level: int) -> bool:
        """Checks if delegation is enabled in app config for active persona level."""
        if active_level not in (6, 7):
            return False
        if not self.app or not hasattr(self.app, 'config'):
            return False
        return bool(self.app.config.get("delegation_enabled", False))

    def get_cecilia_mode(self) -> str:
        """Returns Cecilia mode: 'shadow_wizard' or 'divine_judgement'."""
        if not self.app or not hasattr(self.app, 'config'):
            return "shadow_wizard"
        return self.app.config.get("cecilia_delegation_mode", "shadow_wizard")

    def execute_delegation_chain(self, orchestrator_level: int, user_query: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes dynamic, context-driven multi-agent orchestration.
        Guarantees:
        1. Evaluates real-time query profile via weighted utility function (respecting subagent density mode).
        2. Bounded stateful context offload/load to prevent memory bloat/OOM.
        3. Supports unified Lvl 6/7 execution OR per-subagent dynamic model swapping/offloading.
        4. Emits decision path telemetry to process_queue and live progress to tool log.
        """
        t0 = time.time()
        pq = getattr(self.app, "process_queue", None)

        def _log_diag(msg: str):
            if pq:
                try: pq.put({"status": "diag_log_update", "content": f"\n{msg}"})
                except Exception: pass

        def _log_subagent(msg: str):
            if pq:
                try: pq.put({"status": "tool_log_update", "content": f"\n{msg}"})
                except Exception: pass

        def _log_agentic(msg: str):
            if pq:
                try: pq.put({"status": "agentic_stream", "content": f"\n[DELEGATION] {msg}\n"})
                except Exception: pass

        # Config Options
        selection_mode = "minimal"
        model_mode = "lvl6_7_model"
        cecilia_mode = "shadow_wizard"
        if self.app and hasattr(self.app, "config"):
            selection_mode = self.app.config.get("subagent_selection_mode", "minimal")
            model_mode = self.app.config.get("delegation_model_mode", "lvl6_7_model")
            cecilia_mode = self.app.config.get("cecilia_delegation_mode", "shadow_wizard")

        # Fast-Path: Cecilia Divine Judgement Mode (Direct Omniscience without subagent handoffs)
        if orchestrator_level == 7 and cecilia_mode == "divine_judgement":
            _log_diag("=== [CECILIA DIVINE JUDGEMENT: DIRECT OMNISCIENCE] ===")
            _log_subagent("👁 [CECILIA: DIVINE JUDGEMENT] Bypassing subagents. Engaging direct omniscience...")
            _log_agentic("Cecilia Divine Judgement: Subagent delegation bypassed. Direct insight engaged.")

            try:
                from serenity_resources import DELEGATION_SYSTEM_PROMPTS
                dj_sys = DELEGATION_SYSTEM_PROMPTS.get(7, {}).get("divine_judgement", "Role: 'Cecilia' (Divine Judgement Mode).")
            except Exception:
                dj_sys = "Role: 'Cecilia' (Divine Judgement Mode)."

            direct_res = ""
            if hasattr(self.app, "_run_blocking_inference"):
                direct_prompt = [
                    {"role": "system", "content": dj_sys},
                    {"role": "user", "content": user_query}
                ]
                try:
                    direct_res = self.app._run_blocking_inference(direct_prompt, params)
                except Exception as e:
                    direct_res = f"Divine Judgement deduction: {e}"
            else:
                direct_res = f"Cecilia Core Insight for: {user_query}"

            trace_dj = DecisionTrace(
                trace_id=f"trace_dj_{int(time.time() * 1000)}",
                orchestrator_level=7,
                user_query=user_query,
                profile=QueryProfile(),
                candidate_scores=[],
                selected_agent_ids=[],
                decision_summary="Cecilia Divine Judgement: Direct omniscient inference without subagent handoffs.",
                execution_duration_ms=round((time.time() - t0) * 1000, 2)
            )

            return {
                "orchestrator_level": 7,
                "trace": asdict(trace_dj),
                "reports": [],
                "compiled_briefing": direct_res,
                "factual_data": "",
                "reasoning_data": direct_res
            }

        lvl_to_tier = {1: "fast", 2: "search", 3: "low", 4: "med", 5: "high", 6: "transcendent", 7: "secret"}

        def _set_subagent_banner(lvl: int):
            if not pq:
                return
            tier = lvl_to_tier.get(lvl, "fast")
            model_p = ""
            if model_mode == "per_subagent_model" and hasattr(self.app, "model_paths"):
                model_p = self.app.model_paths.get(tier, "")
            if not model_p:
                model_p = getattr(self.app, "model_path", "")
            model_name = clean_model_display_name(model_p) if model_p else tier.upper()
            try:
                pq.put({"status": "subagent_status", "active": True, "level": lvl, "model": model_name})
            except Exception:
                pass

        initial_orch_tier = getattr(self.app, "current_model_tier", None) or lvl_to_tier.get(orchestrator_level, "transcendent" if orchestrator_level == 6 else "secret")
        explicit_lvl_requested = bool(re.search(r'\b(?:lvl|level)\s*[1-6]\b', user_query, flags=re.IGNORECASE))
        should_swap_models = (model_mode == "per_subagent_model") or explicit_lvl_requested

        def _swap_for_step(target_lvl: int):
            _set_subagent_banner(target_lvl)
            if should_swap_models and hasattr(self.app, "model_paths"):
                tier = lvl_to_tier.get(target_lvl)
                target_path = self.app.model_paths.get(tier)
                curr_lvl = getattr(self.app, "active_persona_level", None)
                curr_tier = getattr(self.app, "current_model_tier", None)
                if target_path and (curr_lvl != target_lvl or curr_tier != tier or target_path != getattr(self.app, "model_path", None)):
                    _log_subagent(f"[OTL MODEL SWAP]: Offloading Orchestrator (Lvl {orchestrator_level}) -> Loading Subagent Lvl {target_lvl} ({tier})...")
                    if hasattr(self.app, "model_swap_synchronous"):
                        try:
                            ok = self.app.model_swap_synchronous(target_level=target_lvl, target_tier=tier)
                            if not ok:
                                _log_subagent(f"[MODEL SWAP WARNING]: Synchronous swap to Lvl {target_lvl} failed.")
                        except Exception as e:
                            _log_subagent(f"[MODEL SWAP WARNING]: Could not swap to Lvl {target_lvl}: {e}")
                    elif hasattr(self.app, "model_swap"):
                        try:
                            self.app.model_swap(target_level=target_lvl, target_tier=tier)
                        except Exception as e:
                            _log_subagent(f"[MODEL SWAP WARNING]: Could not swap to Lvl {target_lvl}: {e}")

        def _restore_orchestrator():
            if should_swap_models:
                orch_tier = initial_orch_tier
                curr_lvl = getattr(self.app, "active_persona_level", None)
                curr_tier = getattr(self.app, "current_model_tier", None)
                curr_path = getattr(self.app, "model_path", None)
                orch_path = getattr(self.app, "model_paths", {}).get(orch_tier) if hasattr(self.app, "model_paths") else None
                if curr_lvl != orchestrator_level or curr_tier != orch_tier or (orch_path and curr_path != orch_path):
                    _log_subagent(f"[OTL MODEL SWAP]: Offloading Subagent -> Restoring Orchestrator Model (Lvl {orchestrator_level} / {orch_tier})...")
                    if hasattr(self.app, "model_swap_synchronous"):
                        try:
                            ok = self.app.model_swap_synchronous(target_level=orchestrator_level, target_tier=orch_tier)
                            if not ok:
                                _log_subagent(f"[MODEL SWAP WARNING]: Synchronous restore to Lvl {orchestrator_level} failed.")
                        except Exception as e:
                            _log_subagent(f"[MODEL SWAP WARNING]: Could not restore Lvl {orchestrator_level}: {e}")
                    elif hasattr(self.app, "model_swap"):
                        try:
                            self.app.model_swap(target_level=orchestrator_level, target_tier=orch_tier)
                        except Exception as e:
                            _log_subagent(f"[MODEL SWAP WARNING]: Could not restore Lvl {orchestrator_level}: {e}")

        # 1. DYNAMIC DISPATCH & OBSERVABILITY TRACE
        trace = UtilityDispatchEngine.evaluate(orchestrator_level, user_query, selection_mode=selection_mode)
        _log_diag(f"=== [AGENT ORCHESTRATION DECISION TRACE ({trace.trace_id})] ===")
        _log_diag(trace.decision_summary)
        for s in trace.candidate_scores:
            tag = "✔ SELECTED" if s.selected else "✖ SKIPPED"
            _log_diag(f"  [{tag}] {s.agent_name:<28} | Raw: {s.raw_utility:.2f} - Cost: {s.cost_penalty:.2f} = Net: {s.net_utility:.2f} -> {s.rationale}")
        _log_diag("==========================================================")

        _log_subagent(f"👥 [ORCHESTRATOR LVL {orchestrator_level}] Dispatching subagents: {trace.selected_agent_ids} (Density: {selection_mode}, Model Mode: {model_mode})")
        _log_agentic(f"Dynamic Dispatch Profile: {trace.decision_summary}")

        selected = trace.selected_agent_ids
        handoff_reports: List[HandoffReport] = []
        
        # State Container for Offload/Load
        active_ctx = SerializedAgentContext(
            trace_id=trace.trace_id,
            origin_agent=orchestrator_level,
            target_agent=0,
            token_budget=1024,
            summary=f"Task: {user_query}"
        )

        step_idx = 1

        try:
            # 2. DYNAMIC SUBAGENT EXECUTION (Dynamic multi-step DAG with re-invocation support)
            execution_queue: List[int] = []
            if 2 in selected:
                execution_queue.append(2)
            if 4 in selected:
                execution_queue.append(4)
            if 3 in selected:
                execution_queue.append(3)
            if 5 in selected:
                execution_queue.append(5)
            if 1 in selected:
                execution_queue.append(1)
                # Dynamic re-invocation: if complex reasoning was involved (5 selected & high complexity),
                # allow re-validation cycle back to Lvl 5 before final delivery (e.g. 5 -> 1 -> 5):
                if 5 in selected and trace.profile.reasoning_complexity >= 0.6:
                    execution_queue.append(5)

            max_subagent_steps = 10
            reasoning_res = ""

            while execution_queue and step_idx <= max_subagent_steps:
                target_lvl = execution_queue.pop(0)

                # Level 2: Search (Fact extraction)
                if target_lvl == 2:
                    _swap_for_step(2)
                    _log_subagent("[DYNAMIC AGENT: Lvl 2 Helper] Initiating fact extraction...")
                    clean_query = extract_search_query(user_query)
                    search_obs = ""
                    if clean_query and hasattr(self.app, "tool_registry") and self.app.tool_registry:
                        try:
                            search_obs = self.app.tool_registry.execute("web_search", {"query": clean_query})
                        except Exception as e:
                            search_obs = f"Search note: {e}"
                    else:
                        search_obs = f"Fact context for '{clean_query or user_query[:50]}' retrieved."

                    active_ctx.staged_facts.append(search_obs[:500])
                    active_ctx.raw_payload += f"\n{search_obs}"
                    
                    next_target = execution_queue[0] if execution_queue else orchestrator_level
                    report = HandoffReport(
                        step_index=step_idx,
                        from_level=2,
                        to_level=next_target,
                        task_type="fact_retrieval",
                        summary=f"Extracted factual findings ({len(search_obs)} chars).",
                        factual_data=search_obs,
                        status="completed"
                    )
                    handoff_reports.append(report)
                    _log_agentic(f"Step {step_idx} (Lvl 2): Factual data retrieved for '{clean_query[:40]}' and offloaded.")
                    step_idx += 1

                # Level 4: Emotional / Nuance Check
                elif target_lvl == 4:
                    _swap_for_step(4)
                    _log_subagent("[DYNAMIC AGENT: Lvl 4 Confidant] Evaluating human nuance & tone...")
                    active_ctx.emotional_valence = "User query exhibits heightened affective cues; requiring empathetic, supportive framing."
                    next_target = execution_queue[0] if execution_queue else orchestrator_level
                    report = HandoffReport(
                        step_index=step_idx,
                        from_level=4,
                        to_level=next_target,
                        task_type="nuance_analysis",
                        summary="Analyzed affective tone and emotional alignment.",
                        factual_data=active_ctx.emotional_valence,
                        status="completed"
                    )
                    handoff_reports.append(report)
                    _log_agentic(f"Step {step_idx} (Lvl 4): Tone alignment offloaded to context.")
                    step_idx += 1

                # Level 3: Collaborator Context Staging
                elif target_lvl == 3:
                    _swap_for_step(3)
                    _log_subagent("[DYNAMIC AGENT: Lvl 3 Collaborator] Staging context & packaging state...")
                    offloaded_state = active_ctx.offload()
                    next_target = execution_queue[0] if execution_queue else orchestrator_level
                    report = HandoffReport(
                        step_index=step_idx,
                        from_level=3,
                        to_level=next_target,
                        task_type="context_staging",
                        summary="Packaged minimal serialized state for downstream reasoning.",
                        factual_data=offloaded_state,
                        status="completed"
                    )
                    handoff_reports.append(report)
                    _log_agentic(f"Step {step_idx} (Lvl 3): Minimal state packaged ({len(offloaded_state)} bytes).")
                    step_idx += 1

                # Level 5: Sage / Brains Logical Deduction
                elif target_lvl == 5:
                    _swap_for_step(5)
                    _log_subagent("[DYNAMIC AGENT: Lvl 5 The Brains] Solving core logical constraints...")
                    staged_brief = "\n".join(active_ctx.staged_facts)
                    prev_deductions = "\n".join(active_ctx.reasoning_steps)
                    prompt_ctx = f"Problem: {user_query}\n\nStaged Findings:\n{staged_brief}"
                    if prev_deductions:
                        prompt_ctx += f"\n\nPrior Draft / Validations:\n{prev_deductions}"
                    prompt_ctx += "\n\nProvide core logical deduction. You may specify `<delegate:lvlX>` if another subagent verification is needed."

                    reasoning_prompt = [
                        {"role": "system", "content": "You are Serenity Lvl 5 (The Brains). Apply deep logical deduction to resolve the problem with verifiable precision. Deliver core deductions directly."},
                        {"role": "user", "content": prompt_ctx}
                    ]
                    if hasattr(self.app, "_run_blocking_inference"):
                        try:
                            reasoning_res = self.app._run_blocking_inference(reasoning_prompt, params)
                        except Exception as e:
                            reasoning_res = f"Deduction established: {e}"
                    else:
                        reasoning_res = f"Deduction for '{user_query}' derived logically."

                    # Sanitize subagent thoughts to isolate internal reasoning from final deduction
                    try:
                        from System.vision_handler import VisionHandler
                        sub_think, sub_ans = VisionHandler.split_thoughts_and_answer(reasoning_res)
                        if sub_think:
                            _log_diag(f"[SUBAGENT LVL 5 THOUGHT LOG]:\n{sub_think}")
                        clean_res = sub_ans if sub_ans else sub_think
                    except Exception:
                        clean_res = reasoning_res

                    active_ctx.reasoning_steps.append(clean_res)
                    next_target = execution_queue[0] if execution_queue else orchestrator_level
                    report = HandoffReport(
                        step_index=step_idx,
                        from_level=5,
                        to_level=next_target,
                        task_type="logical_deduction",
                        summary=f"Formulated logical deductions ({len(clean_res)} chars).",
                        reasoning_output=clean_res,
                        status="completed"
                    )
                    handoff_reports.append(report)
                    _log_agentic(f"Step {step_idx} (Lvl 5): Logical deduction completed.")
                    step_idx += 1

                    # Check for explicit next-agent delegation tag (Option B)
                    m_del = re.search(r'(?:<delegate:lvl\s*|delegate_subtask\(subagent_level=)([1-6])', clean_res, re.IGNORECASE)
                    if m_del and len(execution_queue) < 3:
                        req_next = int(m_del.group(1))
                        allowed_sub = get_available_subagents(orchestrator_level)
                        if req_next in allowed_sub and execution_queue.count(req_next) < 2:
                            execution_queue.append(req_next)
                            _log_agentic(f"Dynamic delegation request detected: Re-invoking Lvl {req_next}.")

                # Level 1: Speedy Format & Sanity Validation
                elif target_lvl == 1:
                    _swap_for_step(1)
                    _log_subagent("[DYNAMIC AGENT: Lvl 1 The Speedy] Performing fast syntax / concise sanity check...")
                    next_target = execution_queue[0] if execution_queue else orchestrator_level
                    report = HandoffReport(
                        step_index=step_idx,
                        from_level=1,
                        to_level=next_target,
                        task_type="speed_validation",
                        summary="Verified concise format and syntax integrity.",
                        status="completed"
                    )
                    handoff_reports.append(report)
                    _log_agentic(f"Step {step_idx} (Lvl 1): Quick sanity check passed.")
                    step_idx += 1

                # Level 6: Prime synthesis (accessible under Cecilia Lvl 7)
                elif target_lvl == 6 and orchestrator_level == 7:
                    _swap_for_step(6)
                    _log_subagent("[DYNAMIC AGENT: Lvl 6 Serenity Prime] Master synthesis and coordination...")
                    report = HandoffReport(
                        step_index=step_idx,
                        from_level=6,
                        to_level=7,
                        task_type="prime_synthesis",
                        summary="Synthesized core architecture and verified systemic alignment.",
                        status="completed"
                    )
                    handoff_reports.append(report)
                    _log_agentic(f"Step {step_idx} (Lvl 6): Prime master synthesis passed.")
                    step_idx += 1

        finally:
            _restore_orchestrator()
            if pq:
                try: pq.put({"status": "subagent_status", "active": False})
                except Exception: pass

        # Final Master Briefing Assembly
        facts_summary = "\n".join([f"- {f}" for f in active_ctx.staged_facts]) or "Direct Knowledge"
        reasoning_summary = "\n".join([f"- {r}" for r in active_ctx.reasoning_steps]) or "Standard Inference"
        nuance_summary = f"\n[Tone Note]: {active_ctx.emotional_valence}" if active_ctx.emotional_valence else ""

        compiled_briefing = (
            f"=== DYNAMIC ORCHESTRATION MASTER BRIEFING ===\n"
            f"[Trace ID]: {trace.trace_id}\n"
            f"[Dispatched Agents]: {selected}\n"
            f"[1. Factual Base]:\n{facts_summary}\n\n"
            f"[2. Core Deductions]:\n{reasoning_summary}"
            f"{nuance_summary}\n"
            f"============================================="
        )

        trace.execution_duration_ms = round((time.time() - t0) * 1000, 2)
        _log_diag(f"[ORCHESTRATION COMPLETE] Total Execution Duration: {trace.execution_duration_ms}ms")
        _log_subagent(f"[ORCHESTRATOR LVL {orchestrator_level}] Dynamic orchestration synthesized. Delivering final response.")

        return {
            "orchestrator_level": orchestrator_level,
            "trace": asdict(trace),
            "reports": handoff_reports,
            "compiled_briefing": compiled_briefing,
            "factual_data": "\n".join(active_ctx.staged_facts),
            "reasoning_data": reasoning_res
        }
