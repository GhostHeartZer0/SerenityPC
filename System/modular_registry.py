# System/modular_registry.py
# Reusable Modular Registry Framework & Dynamic Parameter Auto-Adjustment Engine for Serenity AI.

import re
from typing import Dict, Any, Callable, List, Optional, Tuple

class ModularRegistry:
    """
    Generic, thread-safe, extensible registry pattern replacing monolithic if-elif chains.
    Supports decorator-based registration, metadata attachments, and safe execution dispatch.
    """
    def __init__(self, name: str = "ModularRegistry"):
        self.name = name
        self._handlers: Dict[str, Callable] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def register(self, key: str, handler: Optional[Callable] = None, **metadata):
        """Register a handler function with an identifier key and optional metadata."""
        if handler is None:
            def decorator(fn: Callable):
                self._handlers[key] = fn
                self._metadata[key] = metadata
                return fn
            return decorator
        self._handlers[key] = handler
        self._metadata[key] = metadata
        return handler

    def get(self, key: str, default: Any = None) -> Optional[Callable]:
        """Retrieve a handler by key."""
        return self._handlers.get(key, default)

    def get_metadata(self, key: str) -> Dict[str, Any]:
        """Retrieve metadata associated with a key."""
        return self._metadata.get(key, {})

    def has(self, key: str) -> bool:
        """Check if a key is registered."""
        return key in self._handlers

    def execute(self, key: str, *args, **kwargs) -> Any:
        """Execute a registered handler by key."""
        handler = self._handlers.get(key)
        if not handler:
            raise KeyError(f"[{self.name}] No handler registered for key: '{key}'")
        return handler(*args, **kwargs)

    def list_keys(self) -> List[str]:
        """Return all registered keys."""
        return list(self._handlers.keys())


class DynamicParamRegistry(ModularRegistry):
    """
    Intelligent dynamic parameter adjustment registry.
    Evaluates prompt intent and applies domain-specific sampling adjustments in-memory
    without modifying user configuration files on disk.
    """
    def __init__(self):
        super().__init__(name="DynamicParamRegistry")
        self._register_default_rules()

    def _register_default_rules(self):
        # 1. Coding / Programming / Software Engineering
        @self.register("coding", priority=100)
        def coding_rule(text: str, base_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            code_pattern = r'\b(def |class |import |function|const |var |let |fn |async |struct |enum |interface |sql|query|regex|json|yaml|html|css|javascript|typescript|python|c\+\+|rust|golang|bash|powershell|refactor|debug|unittest|pytest|syntax|compile|algorithm|binary search|recursion|nullpointer|traceback|exception)\b'
            if re.search(code_pattern, text, re.IGNORECASE) or "```" in text:
                adjusted = dict(base_params)
                base_temp = float(base_params.get("temperature", 0.8))
                adjusted["temperature"] = max(0.2, min(0.35, base_temp * 0.45))
                adjusted["min_p"] = max(float(base_params.get("min_p", 0.05)), 0.10)
                adjusted["top_p"] = min(float(base_params.get("top_p", 0.95)), 0.88)
                adjusted["repeat_penalty"] = max(float(base_params.get("repeat_penalty", 1.0)), 1.05)
                return {"domain": "Coding / Technical", "params": adjusted}
            return None

        # 2. Math & Logic / Exact Calculus / Physics
        @self.register("math_logic", priority=90)
        def math_rule(text: str, base_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            math_pattern = r'\b(solve|calculate|equation|integral|derivative|matrix|vector|proof|theorem|polynomial|eigenvalue|probability|statistics|formula|arithmetic|prime numbers|gcd|lcm)\b|[\$\\\^]'
            if re.search(math_pattern, text, re.IGNORECASE):
                adjusted = dict(base_params)
                base_temp = float(base_params.get("temperature", 0.8))
                adjusted["temperature"] = max(0.2, min(0.3, base_temp * 0.4))
                adjusted["min_p"] = max(float(base_params.get("min_p", 0.05)), 0.10)
                adjusted["top_p"] = min(float(base_params.get("top_p", 0.95)), 0.85)
                return {"domain": "Math & Logic", "params": adjusted}
            return None

        # 3. Creative / Storytelling / Roleplay / Brainstorming
        @self.register("creative", priority=80)
        def creative_rule(text: str, base_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            creative_pattern = r'\b(story|poem|poetry|fiction|novel|creative|roleplay|narrative|dialogue|character|lore|fantasy|sci-fi|brainstorm|lyrics|metaphor|creative writing)\b'
            if re.search(creative_pattern, text, re.IGNORECASE):
                adjusted = dict(base_params)
                base_temp = float(base_params.get("temperature", 0.8))
                adjusted["temperature"] = min(1.0, max(0.88, base_temp * 1.15))
                adjusted["top_p"] = min(0.98, max(float(base_params.get("top_p", 0.95)), 0.96))
                adjusted["min_p"] = min(0.04, float(base_params.get("min_p", 0.05)))
                return {"domain": "Creative / Narrative", "params": adjusted}
            return None

        # 4. Factual / Quick Extraction / Definition
        @self.register("factual", priority=70)
        def factual_rule(text: str, base_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            factual_pattern = r'^(who is|what is|when did|where is|define|list the|summarize|extract|translate)\b'
            if re.search(factual_pattern, text.strip(), re.IGNORECASE):
                adjusted = dict(base_params)
                base_temp = float(base_params.get("temperature", 0.8))
                adjusted["temperature"] = max(0.4, min(0.65, base_temp * 0.75))
                adjusted["top_p"] = min(float(base_params.get("top_p", 0.95)), 0.90)
                return {"domain": "Factual / Extraction", "params": adjusted}
            return None

    def adjust_params(self, prompt: str, base_params: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
        """
        Walks registered domain rules to compute non-destructive dynamic parameter adjustments.
        Returns: (final_params_dict, domain_label_or_None)
        """
        if not prompt:
            return dict(base_params), None

        # Sort registered rules by priority descending
        sorted_keys = sorted(
            self._handlers.keys(),
            key=lambda k: self._metadata.get(k, {}).get("priority", 0),
            reverse=True
        )

        for key in sorted_keys:
            rule_fn = self._handlers[key]
            try:
                result = rule_fn(prompt, base_params)
                if result and isinstance(result, dict) and "params" in result:
                    return result["params"], result.get("domain", key)
            except Exception as e:
                print(f"[DYNAMIC PARAMS ERROR] Rule '{key}' failed: {e}")

        return dict(base_params), None
