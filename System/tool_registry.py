# System/tool_registry.py
# Modular Tool Registry for Programmatic & Template Tool Calling in SerenityPC.
# Includes V2.3.0 Level 2 P2P Messaging, Encrypted Mail (.smp), 72-Hour Trust Gate,
# AST Multi-Line PTC Parser, and SerenityDev MCP Frontend Delegation.

import os
import re
import ast
import json
import threading
import subprocess
try:
    import psutil
except ImportError:
    psutil = None
from typing import List, Dict, Any, Optional, Tuple
try:
    from System.modular_registry import ModularRegistry
except ImportError:
    try:
        from modular_registry import ModularRegistry
    except ImportError:
        class ModularRegistry:
            def __init__(self, name: str = "DefaultRegistry"):
                self.name = name
                self._registry = {}
            def register(self, key, component=None, metadata=None, overwrite=True):
                if component is not None:
                    self._registry[key] = component
                    return component
                def decorator(fn):
                    self._registry[key] = fn
                    return fn
                return decorator
            def has(self, key): return key in self._registry
            def execute(self, key, *args, **kwargs): return self._registry[key](*args, **kwargs)
            def get(self, key, default=None): return self._registry.get(key, default)
            def get_all(self): return dict(self._registry)
            def list_keys(self): return list(self._registry.keys())

try:
    from System.serenity_mail_courier import (
        SerenityEnvResolver,
        Level2CourierAgent,
        PacketType,
        ExchangeMode,
        CrossStateP2PTransport,
        SerenityMailCrypto,
    )
except ImportError:
    from serenity_mail_courier import (
        SerenityEnvResolver,
        Level2CourierAgent,
        PacketType,
        ExchangeMode,
        CrossStateP2PTransport,
        SerenityMailCrypto,
    )


class GemmaToolRegistry:
    """Handles tool definitions, schemas, AST PTC parsing, and modular execution for Serenity models."""

    def __init__(self, chatbot_app):
        self.app = chatbot_app
        self.registry = ModularRegistry(name="ToolRegistry")
        self.courier = Level2CourierAgent(app=chatbot_app)
        if chatbot_app is not None:
            try:
                setattr(chatbot_app, "tool_registry", self)
            except Exception:
                pass
            try:
                try:
                    from System.serenity_p2p_ui_wiring import wire_serenity_app
                except ImportError:
                    from serenity_p2p_ui_wiring import wire_serenity_app
                wire_serenity_app(chatbot_app)
            except Exception:
                pass
        self.tools = [
            {
                "function": {
                    "name": "get_system_stats",
                    "description": "Returns current CPU, RAM, and GPU utilization for hardware health monitoring.",
                    "parameters": {"type": "object", "properties": {}, "required": []}
                }
            },
            {
                "function": {
                    "name": "read_file",
                    "description": "Reads the first 5000 characters of a local text file for analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute or workspace-relative path to the file."}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "function": {
                    "name": "read_file_range",
                    "description": "Reads an inclusive range of 1-based lines from a local text file for analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute or workspace-relative path to the file."},
                            "start": {"type": "integer", "description": "First line number to read (1-based, inclusive)."},
                            "end": {"type": "integer", "description": "Last line number to read (1-based, inclusive)."}
                        },
                        "required": ["path", "start", "end"]
                    }
                }
            },
            {
                "function": {
                    "name": "web_search",
                    "description": "Searches the live web for real-time data, current events, weather, news, and specialized technical info not present in your training data.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "The search query."}
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "function": {
                    "name": "control_rgb",
                    "description": "Adjusts the system RGB lighting color or style.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "color": {"type": "array", "items": {"type": "integer"}, "description": "[R, G, B] values (0-255)."},
                            "style": {"type": "string", "description": "Hardware style: 'Steady', 'Breathing', 'Rainbow', 'Flash', etc."}
                        },
                        "required": []
                    }
                }
            },
            {
                "function": {
                    "name": "generate_image",
                    "description": "Generates an image or diagram. Use markdown formatting or Mermaid logic if drawing a technical diagram.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Description of image or raw Mermaid/SVG code."},
                            "type": {"type": "string", "description": "Type: 'image' or 'diagram'"}
                        },
                        "required": ["prompt", "type"]
                    }
                }
            },
            # --- V2.3.0 LEVEL 2 P2P MESSAGING, ENCRYPTED MAIL & SERENITYDEV MCP TOOLS ---
            {
                "function": {
                    "name": "send_peer_message",
                    "description": "Sends an encrypted P2P chat or progress ping to a paired friend's Serenity instance (Main or Legacy).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "peer_id": {"type": "string", "description": "Friend's peer ID or display name (e.g. 'Jesse')."},
                            "message": {"type": "string", "description": "The message or progress update to send."},
                            "target_level": {"type": "integer", "description": "Recipient persona level (default 2 for The Helper; 7 for Cecilia)."},
                            "mode": {"type": "string", "description": "Exchange mode: 'agent_courier', 'direct_transfer', or 'agent_discussion'."}
                        },
                        "required": ["peer_id", "message"]
                    }
                }
            },
            {
                "function": {
                    "name": "send_peer_mail",
                    "description": "Sends an end-to-end encrypted store-and-forward mail packet (.smp) with optional file or report attachment to a peer across states.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "peer_id": {"type": "string", "description": "Friend's peer ID or display name."},
                            "subject": {"type": "string", "description": "Mail subject line."},
                            "body": {"type": "string", "description": "Full mail content or technical notes."},
                            "attachment_path": {"type": "string", "description": "Optional workspace path to attach (e.g. Wringer report or snippet)."},
                            "target_level": {"type": "integer", "description": "Recipient persona level (default 2)."}
                        },
                        "required": ["peer_id", "subject", "body"]
                    }
                }
            },
            {
                "function": {
                    "name": "check_serenity_mailbox",
                    "description": "Checks the Serenity P2P Mailbox (inbox, sent, outbox, or quarantine) and returns messages.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "folder": {"type": "string", "description": "Folder to inspect: 'inbox', 'outbox', 'sent', or 'quarantine'."},
                            "unread_only": {"type": "boolean", "description": "If true, returns only unread messages."},
                            "limit": {"type": "integer", "description": "Maximum number of messages to retrieve (default 10)."}
                        },
                        "required": []
                    }
                }
            },
            {
                "function": {
                    "name": "manage_quarantine_mail",
                    "description": "Admits or rejects an inbound peer message held in the Contextual Quarantine Sandbox.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "quarantine_id": {"type": "integer", "description": "ID of the quarantined item."},
                            "action": {"type": "string", "description": "Moderation action: 'admit' or 'reject'."},
                            "reason": {"type": "string", "description": "Optional note explaining the moderation decision."}
                        },
                        "required": ["quarantine_id", "action"]
                    }
                }
            },
            {
                "function": {
                    "name": "share_chat_with_peer",
                    "description": "Packages recent conversation turns (or a compressed OTL context snapshot) into an encrypted mail packet and shares it with a friend.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "peer_id": {"type": "string", "description": "Friend's peer ID or display name."},
                            "turn_count": {"type": "integer", "description": "Number of recent chat turns to include (default 5)."},
                            "include_thoughts": {"type": "boolean", "description": "Whether to include isolated <thought> reasoning streams (default false)."},
                            "note": {"type": "string", "description": "Introductory note from Serenity accompanying the shared chat."}
                        },
                        "required": ["peer_id"]
                    }
                }
            },
            {
                "function": {
                    "name": "pair_serenity_peer",
                    "description": "Generates a signed cross-state 'serenity-peer://' invite card (with STUN WAN discovery) or pairs a friend from their invite code.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "'generate_invite', 'accept_invite', or 'list_peers'."},
                            "invite_code": {"type": "string", "description": "The 'serenity-peer://...' invite string when action is 'accept_invite'."},
                            "endpoint_url": {"type": "string", "description": "Optional custom HTTPS/relay endpoint URL."},
                            "dead_drop_path": {"type": "string", "description": "Optional shared folder path for offline .smp dead-drop sync."}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "function": {
                    "name": "delegate_to_serenitydev",
                    "description": "Delegates a coding, codebase exploration, or IDE file edit task to SerenityDev over MCP (requires 72h profile age for '/execute').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "SerenityDev slash mode: 'explore', 'plan', 'execute', or 'agent'."},
                            "prompt": {"type": "string", "description": "Detailed coding or refactoring objective."},
                            "workspace_path": {"type": "string", "description": "Optional target repository path."}
                        },
                        "required": ["command", "prompt"]
                    }
                }
            }
        ]
        self._register_handlers()

    def _register_handlers(self):
        """Registers modular handlers using ModularRegistry."""

        def clean_search_bloat(text: str) -> str:
            """Filters out web boilerplate, cookie notices, navigation junk, and duplicate noise."""
            if not text:
                return ""
            noise_patterns = [
                r'(?i)^\s*(?:accept all cookies|cookie settings|privacy policy|terms of use|terms & conditions|all rights reserved|copyright \d{4}).*$',
                r'(?i)^\s*(?:sign in|sign up|log in|register|subscribe now|subscribe|join now|download app).*$',
                r'(?i)^\s*(?:skip to (?:content|main|navigation)|menu|search query|toggle navigation).*$',
                r'(?i)^\s*(?:advertisement|sponsored|ad choices|share this article|related articles).*$',
                r'(?i)^\s*(?:enable javascript|please enable javascript to view).*$'
            ]
            lines = []
            seen = set()
            for line in text.splitlines():
                l_str = line.strip()
                if not l_str or len(l_str) < 4:
                    continue
                if any(re.match(pat, l_str) for pat in noise_patterns):
                    continue
                l_norm = l_str.lower()
                if l_norm in seen:
                    continue
                seen.add(l_norm)
                lines.append(l_str)
            cleaned = "\n".join(lines)
            cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
            return cleaned.strip()

        @self.registry.register("web_search")
        def handle_web_search(args: Dict[str, Any]) -> str:
            query = args.get("query", "").strip()
            if not query:
                return "Notice: Search query was empty. Proceeding to answer using baseline knowledge."

            # Keyword-actuated lookup extraction to filter bloat (V2.9.0)
            if len(query) > 90 or "\n" in query or "```" in query:
                try:
                    from System.orchestration_manager import extract_search_query
                    extracted = extract_search_query(query)
                    if extracted and len(extracted) >= 4:
                        query = extracted
                except Exception:
                    pass
            if len(query) > 160:
                query = query[:160].rsplit(' ', 1)[0].strip()

            from System.network_guard import is_offline_mode
            if is_offline_mode() or (self.app and getattr(self.app, 'config', {}).get("offline_mode", False)):
                msg = f"[OFFLINE MODE] Live web search blocked by offline policy for query: '{query}'. Please answer using internal offline knowledge."
                pq = getattr(self.app, "process_queue", None)
                if pq:
                    try:
                        pq.put({"status": "tool_log_update", "content": f"\n{msg}"})
                    except Exception:
                        pass
                return msg

            import requests
            import urllib.parse
            from bs4 import BeautifulSoup

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'DNT': '1'
            }

            try:
                brave_url = "https://search.brave.com/search?q=" + urllib.parse.quote(query)
                resp = requests.get(brave_url, headers=headers, timeout=8)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    results = []
                    for res in soup.select('div.snippet, div.result, .search-result, .snippet'):
                        title = res.select_one('h2, .title, .search-snippet-title')
                        snippet = res.select_one('p, .content, .snippet-description, .snippet-content')
                        if title and snippet:
                            c_title = clean_search_bloat(title.get_text(strip=True))
                            c_snip = clean_search_bloat(snippet.get_text(strip=True))
                            if c_title and c_snip:
                                results.append(f"[{c_title}]\n{c_snip}")

                    if len(results) >= 1:
                        proof_msg = f"[SEARCH PROOF] Provider: Brave | Status: {resp.status_code} | Cleaned Extracts: {len(results)}"
                        if self.app and hasattr(self.app, "process_queue"):
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                        return f"Brave Search Results for '{query}':\n\n" + "\n\n".join(results[:5])
            except Exception as e:
                print(f"[SEARCH DEBUG] Brave failed: {e}")

            try:
                bing_url = f"https://www.bing.com/search?q={urllib.parse.quote(query)}"
                resp = requests.get(bing_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    results = []
                    for res in soup.select('li.b_algo'):
                        title = res.select_one('h2')
                        snippet = res.select_one('.b_caption p, .b_snippet')
                        if title and snippet:
                            c_title = clean_search_bloat(title.get_text(strip=True))
                            c_snip = clean_search_bloat(snippet.get_text(strip=True))
                            if c_title and c_snip:
                                results.append(f"[{c_title}]\n{c_snip}")

                    if len(results) >= 2:
                        proof_msg = f"[SEARCH PROOF] Provider: Bing | Status: {resp.status_code} | Cleaned Extracts: {len(results)}"
                        if self.app and hasattr(self.app, "process_queue"):
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                        return f"Bing Search Context for '{query}':\n\n" + "\n\n".join(results[:5])
            except Exception:
                pass

            try:
                ddg_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
                resp = requests.get(ddg_url, headers=headers, timeout=5)
                if resp.status_code == 200:
                    soup = BeautifulSoup(resp.text, 'html.parser')
                    results = []
                    for res in soup.select('.result__body'):
                        title = res.select_one('.result__title')
                        snippet = res.select_one('.result__snippet')
                        if title and snippet:
                            c_title = clean_search_bloat(title.get_text(strip=True))
                            c_snip = clean_search_bloat(snippet.get_text(strip=True))
                            if c_title and c_snip:
                                results.append(f"[{c_title}]\n{c_snip}")
                    if len(results) >= 2:
                        proof_msg = f"[SEARCH PROOF] Provider: DuckDuckGo | Status: {resp.status_code} | Cleaned Extracts: {len(results)}"
                        if self.app and hasattr(self.app, "process_queue"):
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                        return f"DuckDuckGo Context for '{query}':\n\n" + "\n\n".join(results[:5])
            except Exception:
                pass

            return f"Notice: Live search returned 0 retrievable hits for '{query[:60]}'. Proceeding with internal baseline knowledge."

        @self.registry.register("delegate_subtask")
        def handle_delegate_subtask(args: Dict[str, Any]) -> str:
            subagent_lvl = args.get("subagent_level", 2)
            task_desc = args.get("task_description", "")
            return f"[SUBAGENT LVL {subagent_lvl} DISPATCH]: Task '{task_desc}' received and queued for handoff."

        @self.registry.register("generate_image")
        def handle_generate_image(args: Any) -> str:
            import sys
            if isinstance(args, str):
                try:
                    parsed = ast.literal_eval(args)
                    if isinstance(parsed, dict):
                        args = parsed
                    else:
                        args = {"prompt": args}
                except Exception:
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {"prompt": args}
            elif not isinstance(args, dict):
                args = {"prompt": str(args)}

            for k in ("action_input", "parameters", "arguments", "input"):
                if k in args:
                    nested = args[k]
                    if isinstance(nested, str):
                        try:
                            nested_parsed = ast.literal_eval(nested)
                            if isinstance(nested_parsed, dict):
                                nested = nested_parsed
                        except Exception:
                            try:
                                nested_parsed = json.loads(nested)
                                if isinstance(nested_parsed, dict):
                                    nested = nested_parsed
                            except Exception:
                                pass
                    if isinstance(nested, dict):
                        for nk, nv in nested.items():
                            args.setdefault(nk, nv)
                    elif isinstance(nested, str) and not args.get("prompt"):
                        args["prompt"] = nested

            prompt = args.get("prompt") or args.get("description") or args.get("code") or args.get("query") or args.get("text") or ""
            req_type = args.get("type", "image")
            if isinstance(prompt, dict):
                req_type = prompt.get("type", req_type)
                prompt = prompt.get("prompt") or prompt.get("description") or str(prompt)

            clean_prompt = str(prompt).strip()
            clean_prompt = re.sub(r'<\|"?|\\"?\|?>?|<\|+|\|>', '', clean_prompt).strip(' "<|>\\')

            def spawn_viewer():
                try:
                    base_dir = SerenityEnvResolver.get_writable_root(self.app)
                    bundle_dir = SerenityEnvResolver.get_bundle_root(self.app)
                    viewer_script = os.path.join(base_dir, "System", "hud_viewer.py")
                    if not os.path.isfile(viewer_script):
                        viewer_script = os.path.join(bundle_dir, "System", "hud_viewer.py")
                    tmp_dir = os.path.join(base_dir, ".tmp")
                    os.makedirs(tmp_dir, exist_ok=True)
                    payload_path = os.path.join(tmp_dir, f"hud_payload_{int(time.time() * 1000)}.json")
                    with open(payload_path, "w", encoding="utf-8") as fp:
                        json.dump({"req_type": req_type, "content": clean_prompt}, fp, ensure_ascii=False)
                    py_bin = SerenityEnvResolver.find_python_interpreter(self.app) or sys.executable
                    subprocess.Popen([py_bin, viewer_script, payload_path])
                except Exception as e:
                    print(f"[TOOL] Image viewer error: {e}")

            threading.Thread(target=spawn_viewer, daemon=True).start()
            if self.app and hasattr(self.app, "process_queue"):
                try:
                    self.app.process_queue.put({"status": "tool_log_update", "content": f"\n[IMAGE / VISUAL HUD] Rendered {req_type}: {clean_prompt[:80]}...\n"})
                except Exception:
                    pass
            return f"Successfully generated and displayed {req_type} visual prompt in HUD overlay: '{clean_prompt[:60]}...'"

        @self.registry.register("get_system_stats")
        def handle_get_system_stats(args: Dict[str, Any]) -> str:
            stats = {}
            try:
                stats["cpu"] = f"{psutil.cpu_percent()}%"
                stats["ram"] = f"{psutil.virtual_memory().percent}%"
            except Exception:
                stats["cpu"] = "Normal"
                stats["ram"] = "Normal"

            nvidia_ml = getattr(self.app, 'nvidia_ml', None)
            if not nvidia_ml:
                try:
                    import pynvml as nvml
                    nvml.nvmlInit()
                    nvidia_ml = nvml
                except Exception:
                    pass
            if nvidia_ml:
                try:
                    handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(handle)
                    stats["vram"] = f"{mem.used/1024**2:.0f} / {mem.total/1024**2:.0f} MB"
                except Exception:
                    pass
            return json.dumps(stats)

        def _is_sandboxed_path(target_p: str) -> bool:
            """
            Enforces strict tool file sandboxing with the 72-Hour Profile Age Gate:
            - Default (unauthenticated, locked, or profile < 72h old): strictly restricted to Users/, Logs/, and .tmp/.
            - Password-protected, unlocked profile >= 72 hours old: allowed read access across the repository root.
            - Sensitive cryptographic/credential files (.env, vault_state.json, *.encz, vault.db, *.key, *.pem) are always blocked.
            """
            w_root = os.path.realpath(SerenityEnvResolver.get_writable_root(self.app))
            b_root = os.path.realpath(SerenityEnvResolver.get_bundle_root(self.app))
            real_p = os.path.realpath(target_p)
            fname_lower = os.path.basename(real_p).lower()
            if (
                fname_lower in {".env", "vault_state.json", "vault.db", "serenity_mailbox.db"}
                or fname_lower.endswith((".encz", ".key", ".pem"))
            ):
                return False

            vm = getattr(self.app, "vault_manager", None) if self.app else None
            full_repo_allowed = False
            if self.app is None:
                full_repo_allowed = True
            elif vm is not None and hasattr(vm, "has_full_repo_read_access"):
                try:
                    full_repo_allowed = bool(vm.has_full_repo_read_access(min_age_hours=72.0))
                except Exception:
                    full_repo_allowed = False

            try:
                if full_repo_allowed:
                    roots = [w_root] if w_root == b_root else [w_root, b_root]
                    return any(os.path.commonpath([r, real_p]) == r for r in roots)
                allowed_dirs = [
                    os.path.realpath(os.path.join(w_root, "Users")),
                    os.path.realpath(os.path.join(w_root, "Logs")),
                    os.path.realpath(os.path.join(w_root, ".tmp")),
                    os.path.realpath(os.path.join(w_root, "tmp")),
                    os.path.realpath(os.path.join(w_root, "History")),
                ]
                return any(os.path.commonpath([d, real_p]) == d for d in allowed_dirs)
            except Exception:
                return False

        self._is_sandboxed_path = _is_sandboxed_path

        @self.registry.register("read_file")
        def handle_read_file(args: Dict[str, Any]) -> str:
            path = args.get("path")
            if not path:
                return "Notice: No file path provided."

            target_path = path
            if not os.path.isabs(target_path):
                base_dir = SerenityEnvResolver.get_writable_root(self.app)
                target_path = os.path.join(base_dir, path)

            if not _is_sandboxed_path(target_path):
                return f"[SECURITY RESTRICTION] File operations outside Serenity workspace directories are blocked: '{path}'."

            if not os.path.exists(target_path):
                return f"Notice: File '{path}' was not found. Please proceed to answer based on available context and inform user that the path was not found."

            try:
                with open(target_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read(5000)
            except Exception as e:
                return f"Notice: Error reading file '{path}': {str(e)}"

        @self.registry.register("read_file_range")
        def handle_read_file_range(args: Dict[str, Any]) -> str:
            path = args.get("path")
            if not path:
                return "Notice: No file path provided."

            start_value = args.get("start")
            end_value = args.get("end")
            if start_value is None or end_value is None:
                return "Notice: Start and end must be integer line numbers."
            try:
                start = int(start_value)
                end = int(end_value)
            except (TypeError, ValueError):
                return "Notice: Start and end must be integer line numbers."
            if start < 1 or end < start:
                return "Notice: Start and end must define a valid 1-based inclusive line range."

            target_path = path
            if not os.path.isabs(target_path):
                base_dir = SerenityEnvResolver.get_writable_root(self.app)
                target_path = os.path.join(base_dir, path)

            if not _is_sandboxed_path(target_path):
                return f"[SECURITY RESTRICTION] File operations outside Serenity workspace directories are blocked: '{path}'."

            if not os.path.exists(target_path):
                return f"Notice: File '{path}' was not found."

            try:
                with open(target_path, "r", encoding="utf-8", errors="ignore") as f:
                    selected_lines = [
                        line for line_number, line in enumerate(f, start=1)
                        if start <= line_number <= end
                    ]
                return "".join(selected_lines)
            except Exception as e:
                return f"Notice: Error reading file '{path}': {str(e)}"

        @self.registry.register("control_rgb")
        def handle_control_rgb(args: Dict[str, Any]) -> str:
            base_dir = SerenityEnvResolver.get_writable_root(self.app)
            state_path = os.path.join(base_dir, "System", "rgb_state.json")
            try:
                state = {}
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        state = json.load(f)

                if "color" in args:
                    state["manual_color"] = args["color"]
                if "style" in args:
                    state["manual_style"] = args["style"]
                state["mode"] = "manual"

                with open(state_path, 'w') as f:
                    json.dump(state, f, indent=4)
                return f"RGB adjusted: Mode=Manual, Color={args.get('color')}, Style={args.get('style')}"
            except Exception as e:
                return f"Notice: RGB controller not fully accessible ({str(e)}). Simulated state applied."

        # --- V2.3.0 HANDLERS: P2P MESSAGING, MAIL, QUARANTINE & SERENITYDEV MCP ---

        @self.registry.register("send_peer_message")
        def handle_send_peer_message(args: Dict[str, Any]) -> str:
            peer_id = str(args.get("peer_id", "")).strip()
            message = str(args.get("message", "")).strip()
            if not peer_id or not message:
                return "Notice: Both 'peer_id' and 'message' are required."
            target_lvl = int(args.get("target_level") or 2)
            mode_str = str(args.get("mode") or "agent_courier")
            try:
                ex_mode = ExchangeMode(mode_str)
            except ValueError:
                ex_mode = ExchangeMode.AGENT_COURIER

            res = self.courier.dispatch_peer_packet(
                peer_query=peer_id,
                packet_type=PacketType.PEER_CHAT,
                subject="P2P Message",
                body=message,
                target_level=target_lvl,
                exchange_mode=ex_mode,
            )
            return json.dumps(res)

        @self.registry.register("send_peer_mail")
        def handle_send_peer_mail(args: Dict[str, Any]) -> str:
            peer_id = str(args.get("peer_id", "")).strip()
            subject = str(args.get("subject", "Serenity Mail")).strip()
            body = str(args.get("body", "")).strip()
            if not peer_id or not body:
                return "Notice: 'peer_id' and 'body' are required to send mail."

            att_path = str(args.get("attachment_path") or "").strip()
            att_type = "none"
            att_data: Dict[str, Any] = {}
            if att_path:
                target_p = att_path
                if not os.path.isabs(target_p):
                    base_dir = SerenityEnvResolver.get_writable_root(self.app)
                    target_p = os.path.join(base_dir, att_path)
                if not _is_sandboxed_path(target_p):
                    return f"[SECURITY RESTRICTION] Cannot attach protected or out-of-sandbox file: '{att_path}'."
                if os.path.isfile(target_p):
                    with open(target_p, "r", encoding="utf-8", errors="ignore") as fp:
                        att_data = {
                            "filename": os.path.basename(target_p),
                            "content": fp.read(25_000),
                        }
                    att_type = "file_snippet"

            target_lvl = int(args.get("target_level") or 2)
            res = self.courier.dispatch_peer_packet(
                peer_query=peer_id,
                packet_type=PacketType.PEER_MAIL,
                subject=subject,
                body=body,
                attachment_type=att_type,
                attachment_data=att_data,
                target_level=target_lvl,
                exchange_mode=ExchangeMode.AGENT_COURIER,
            )
            return json.dumps(res)

        @self.registry.register("check_serenity_mailbox")
        def handle_check_serenity_mailbox(args: Dict[str, Any]) -> str:
            self.courier.sync_environment_and_user()
            if bool(args.get("sync_deaddrop", True)):
                self.courier.poll_deaddrop_and_outbox()
            folder = str(args.get("folder") or "inbox").strip().lower()
            unread_only = bool(args.get("unread_only", False))
            limit = max(1, min(50, int(args.get("limit") or 10)))

            with self.courier.mailbox._get_connection() as conn:
                cursor = conn.cursor()
                if folder == "quarantine":
                    cursor.execute("""
                        SELECT id, packet_id, sender_id, sender_name, msg_type, subject,
                               content, intent_label, risk_score, hold_reason, status, quarantined_at
                        FROM quarantine_sandbox
                        WHERE status = 'quarantined'
                        ORDER BY quarantined_at DESC
                        LIMIT ?
                    """, (limit,))
                    rows = [dict(r) for r in cursor.fetchall()]
                    return json.dumps({"folder": "quarantine", "count": len(rows), "items": rows})

                if folder == "outbox":
                    cursor.execute("""
                        SELECT id, packet_id, peer_id, smp_file_path, retry_count, status, created_at
                        FROM outbox_queue
                        WHERE status = 'pending'
                        ORDER BY created_at DESC
                        LIMIT ?
                    """, (limit,))
                    rows = [dict(r) for r in cursor.fetchall()]
                    return json.dumps({"folder": "outbox", "count": len(rows), "items": rows})

                q = """
                    SELECT id, packet_id, folder, msg_type, sender_id, sender_name,
                           recipient_id, subject, body, attachment_type, target_level,
                           influence_weight, is_read, timestamp
                    FROM mailbox_messages
                    WHERE folder = ?
                """
                params: List[Any] = [folder]
                if unread_only:
                    q += " AND is_read = 0"
                q += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                cursor.execute(q, params)
                rows = [dict(r) for r in cursor.fetchall()]

            return json.dumps({
                "folder": folder,
                "count": len(rows),
                "trust_status": self.courier.get_trust_status(),
                "messages": rows,
            })

        @self.registry.register("manage_quarantine_mail")
        def handle_manage_quarantine_mail(args: Dict[str, Any]) -> str:
            q_id = int(args.get("quarantine_id") or 0)
            action = str(args.get("action") or "admit").strip().lower()
            reason = str(args.get("reason") or "").strip()
            if q_id <= 0 or action not in ("admit", "reject"):
                return "Notice: Valid 'quarantine_id' and action ('admit' or 'reject') required."
            res = self.courier.resolve_quarantine_item(q_id, action=action, reason=reason)
            return json.dumps(res)

        @self.registry.register("share_chat_with_peer")
        def handle_share_chat_with_peer(args: Dict[str, Any]) -> str:
            peer_id = str(args.get("peer_id", "")).strip()
            if not peer_id:
                return "Notice: 'peer_id' is required."
            turn_count = max(1, min(25, int(args.get("turn_count") or 5)))
            include_thoughts = bool(args.get("include_thoughts", False))
            note = str(args.get("note") or "Shared conversation context from Serenity.").strip()

            turns_data = []
            if self.app and hasattr(self.app, "messages") and isinstance(self.app.messages, list):
                raw_msgs = self.app.messages[-turn_count * 2 :]
                for m in raw_msgs:
                    content = str(m.get("content", ""))
                    if not include_thoughts:
                        content = re.sub(r"<(?:thought|think|reasoning)>.*?</(?:thought|think|reasoning)>", "", content, flags=re.DOTALL).strip()
                    turns_data.append({"role": m.get("role", "user"), "content": content[:1500]})

            res = self.courier.dispatch_peer_packet(
                peer_query=peer_id,
                packet_type=PacketType.SHARED_CONTEXT,
                subject=f"Shared Chat ({len(turns_data)} messages)",
                body=note,
                attachment_type="chat_turns",
                attachment_data={"turns": turns_data, "thoughts_included": include_thoughts},
                target_level=2,
                exchange_mode=ExchangeMode.AGENT_COURIER,
            )
            return json.dumps(res)

        @self.registry.register("pair_serenity_peer")
        def handle_pair_serenity_peer(args: Dict[str, Any]) -> str:
            self.courier.sync_environment_and_user()
            action = str(args.get("action") or "list_peers").strip().lower()
            if action == "generate_invite":
                ident = self.courier.mailbox.identity
                endpoint = str(args.get("endpoint_url") or "").strip()
                if not endpoint:
                    wan = CrossStateP2PTransport.discover_wan_endpoint()
                    if wan:
                        endpoint = f"https://{wan}"
                invite = CrossStateP2PTransport.generate_invite_code(
                    peer_id=ident["peer_id"],
                    display_name=ident["display_name"],
                    identity_keys=ident,
                    endpoint_url=endpoint,
                    app_edition=ident.get("app_edition", "SerenityPC-Main"),
                    dead_drop_hint=str(args.get("dead_drop_path") or ""),
                )
                return json.dumps({
                    "status": "invite_generated",
                    "peer_id": ident["peer_id"],
                    "fingerprint": ident["fingerprint"],
                    "detected_wan_endpoint": endpoint,
                    "invite_code": invite,
                })
            elif action == "accept_invite":
                code = str(args.get("invite_code") or "").strip()
                if not code:
                    return "Notice: 'invite_code' is required for accept_invite."
                peer = self.courier.mailbox.register_peer_from_invite(code)
                return json.dumps({"status": "peer_paired", "peer": peer})
            else:
                peers = self.courier.mailbox.list_peers()
                return json.dumps({"status": "ok", "count": len(peers), "peers": peers})

        @self.registry.register("delegate_to_serenitydev")
        def handle_delegate_to_serenitydev(args: Dict[str, Any]) -> str:
            command = str(args.get("command") or "plan").strip()
            prompt = str(args.get("prompt") or "").strip()
            workspace_path = str(args.get("workspace_path") or "").strip()
            if not prompt:
                return "Notice: 'prompt' is required for delegate_to_serenitydev."
            res = self.courier.delegate_to_serenitydev(
                command=command,
                prompt=prompt,
                workspace_path=workspace_path,
            )
            return json.dumps(res)

    # --- AST MULTI-LINE PTC PARSER (PORTED FROM SERENITYDEV v1.6.0) ---

    def parse_ptc_calls(self, model_output: str) -> List[Tuple[str, Dict[str, Any]]]:
        """
        Extracts Programmatic Tool Calling (PTC) Python function invocations from model output
        Parses tool invocations across multiple model formats:
        - Python PTC (AST parsed)
        - Nemotron XML (<function=name><parameter=k>v</parameter></function>)
        - MiniCPM XML (<function name="..."><param name="...">v</param></function>)
        - Spark-X2.5 XML (<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>)
        - Olmo-3 (<function_calls>[...]</function_calls>)
        - Mistral ([TOOL_CALLS] [...])
        - Tag/JSON formats (<ctrl42>call:name{...}, {"name": "...", "arguments": ...})
        """
        if not model_output:
            return []

        calls: List[Tuple[str, Dict[str, Any]]] = []
        valid_names = set(self.registry.list_keys())
        alias_map = {
            "readfile": "read_file",
            "search": "web_search",
            "google_search": "web_search",
            "read_file_content": "read_file",
        }

        def _resolve_name(raw_name: str) -> Optional[str]:
            if not raw_name:
                return None
            n = raw_name.strip().lower()
            if n in valid_names:
                return n
            if n in alias_map and alias_map[n] in valid_names:
                return alias_map[n]
            return None

        # Build schema parameter mapping for positional args
        schema_props_map: Dict[str, List[str]] = {}
        for t in self.tools:
            fn = t.get("function", {})
            fname = fn.get("name")
            if fname:
                schema_props_map[fname] = list(fn.get("parameters", {}).get("properties", {}).keys())

        # 1. AST parser for Python-style code blocks or direct invocations
        candidates = re.findall(r"```(?:python|py|tool_call)?\s*(.*?)\s*```", model_output, flags=re.DOTALL)
        if not candidates:
            candidates = [model_output]

        for block in candidates:
            try:
                tree = ast.parse(block.strip(), mode="exec")
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func_name = None
                    if isinstance(node.func, ast.Name):
                        func_name = node.func.id
                    elif isinstance(node.func, ast.Attribute):
                        func_name = node.func.attr
                    resolved = _resolve_name(func_name)
                    if not resolved:
                        continue

                    parsed_args: Dict[str, Any] = {}
                    schema_props = schema_props_map.get(resolved, [])
                    for idx, pos_arg in enumerate(node.args):
                        try:
                            val = ast.literal_eval(pos_arg)
                            key = schema_props[idx] if idx < len(schema_props) else f"arg_{idx}"
                            parsed_args[key] = val
                        except Exception:
                            pass
                    for kw in node.keywords:
                        if kw.arg:
                            try:
                                parsed_args[kw.arg] = ast.literal_eval(kw.value)
                            except Exception:
                                pass
                    calls.append((resolved, parsed_args))

        if calls:
            return calls

        # 2. Nemotron XML format: <function=func_name><parameter=key>value</parameter></function>
        nemotron_matches = re.finditer(
            r'<function=([\w_]+)>(.*?)(?:<\/function>|(?=<function=)|$)',
            model_output,
            re.DOTALL | re.IGNORECASE
        )
        for m in nemotron_matches:
            raw_fn = m.group(1).strip()
            resolved = _resolve_name(raw_fn)
            if not resolved:
                continue
            body = m.group(2)
            parsed_args: Dict[str, Any] = {}
            param_matches = re.finditer(
                r'<parameter=([\w_]+)>(.*?)(?:<\/parameter>|(?=<parameter=)|$)',
                body,
                re.DOTALL | re.IGNORECASE
            )
            for pm in param_matches:
                k = pm.group(1).strip()
                v = pm.group(2).strip()
                if v.startswith("{") or v.startswith("["):
                    try: v = json.loads(v)
                    except Exception: pass
                parsed_args[k] = v
            calls.append((resolved, parsed_args))

        if calls:
            return calls

        # 3. MiniCPM XML format: <function name="func_name"><param name="key">value</param></function>
        minicpm_matches = re.finditer(
            r'<function\s+name=["\']?([\w_]+)["\']?\s*>(.*?)(?:<\/function>|$)',
            model_output,
            re.DOTALL | re.IGNORECASE
        )
        for m in minicpm_matches:
            raw_fn = m.group(1).strip()
            resolved = _resolve_name(raw_fn)
            if not resolved:
                continue
            body = m.group(2)
            parsed_args: Dict[str, Any] = {}
            param_matches = re.finditer(
                r'<param\s+name=["\']?([\w_]+)["\']?\s*>(.*?)(?:<\/param>|$)',
                body,
                re.DOTALL | re.IGNORECASE
            )
            for pm in param_matches:
                k = pm.group(1).strip()
                v = pm.group(2).strip()
                cdata = re.search(r'<!\[CDATA\[(.*?)\]\]>', v, re.DOTALL)
                if cdata:
                    v = cdata.group(1)
                elif v.startswith("{") or v.startswith("["):
                    try: v = json.loads(v)
                    except Exception: pass
                parsed_args[k] = v
            calls.append((resolved, parsed_args))

        if calls:
            return calls

        # 4. Spark-X2.5 XML format: <tool_call>func_name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>
        spark_matches = re.finditer(
            r'<tool_call>\s*([a-zA-Z_]\w*)([\s\S]*?)(?:<\/tool_call>|$)',
            model_output,
            re.IGNORECASE
        )
        for m in spark_matches:
            raw_fn = m.group(1).strip()
            resolved = _resolve_name(raw_fn)
            if not resolved:
                continue
            body = m.group(2)
            parsed_args: Dict[str, Any] = {}
            kv_matches = re.findall(
                r'<arg_key>\s*(.*?)\s*<\/arg_key>\s*<arg_value>\s*(.*?)\s*<\/arg_value>',
                body,
                re.DOTALL | re.IGNORECASE
            )
            for k, v in kv_matches:
                v = v.strip()
                if v.startswith("{") or v.startswith("["):
                    try: v = json.loads(v)
                    except Exception: pass
                parsed_args[k] = v
            calls.append((resolved, parsed_args))

        if calls:
            return calls

        # 5. Mistral / Olmo / JSON array / dict within [TOOL_CALLS] or <tool_call> or <function_calls> or text
        json_containers = []
        mistral_m = re.search(r'\[TOOL_CALLS\]\s*(\[[\s\S]*?\]|\{[\s\S]*?\})', model_output)
        if mistral_m:
            json_containers.append(mistral_m.group(1))
        for tc_m in re.finditer(r'<tool_call>\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*<\/tool_call>', model_output, re.IGNORECASE):
            json_containers.append(tc_m.group(1))
        for fc_m in re.finditer(r'<function_calls?>\s*([\s\S]*?)\s*<\/function_calls?>', model_output, re.IGNORECASE):
            raw = fc_m.group(1).strip()
            if raw.startswith("[") or raw.startswith("{"):
                json_containers.append(raw)
            else:
                for sub_fc in re.finditer(r'<function_call>\s*(\{[\s\S]*?\})\s*<\/function_call>', raw, re.IGNORECASE):
                    json_containers.append(sub_fc.group(1))

        # Check tag calls: <ctrl42>call:name{...}
        for call_m in re.finditer(
            r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:|<(?:channel\|)?(?:execute_tool|executetool)>)\s*([\w_]+)\s*\{(.*?)\}(?:<\/(?:execute_tool|executetool)>)?',
            model_output,
            re.DOTALL | re.IGNORECASE
        ):
            raw_fn = call_m.group(1).strip()
            resolved = _resolve_name(raw_fn)
            if not resolved:
                continue
            args_str = "{" + call_m.group(2).strip() + "}"
            try:
                parsed_args = json.loads(args_str)
            except Exception:
                try:
                    parsed_args = ast.literal_eval(args_str)
                except Exception:
                    parsed_args = {}
            calls.append((resolved, parsed_args if isinstance(parsed_args, dict) else {}))

        if calls:
            return calls

        if not json_containers:
            for jm in re.finditer(r'\{\s*["\'](?:action|tool|name|function)["\']\s*:\s*["\'][\w_]+["\'][\s\S]*?\}', model_output):
                json_containers.append(jm.group(0))

        for jc in json_containers:
            parsed = None
            try:
                parsed = json.loads(jc.replace(r"\'", "'"))
            except Exception:
                try:
                    parsed = ast.literal_eval(jc)
                except Exception:
                    pass
            if isinstance(parsed, list):
                items = parsed
            elif isinstance(parsed, dict):
                items = [parsed]
            else:
                items = []

            for item in items:
                if not isinstance(item, dict):
                    continue
                raw_name = (
                    item.get("name")
                    or item.get("action")
                    or item.get("tool")
                    or item.get("function")
                )
                if isinstance(raw_name, dict):
                    raw_name = raw_name.get("name")
                resolved = _resolve_name(str(raw_name) if raw_name else "")
                if not resolved:
                    continue
                args_data = (
                    item.get("arguments")
                    or item.get("parameters")
                    or item.get("action_input")
                    or item.get("input")
                    or {}
                )
                if isinstance(args_data, str):
                    try:
                        args_data = json.loads(args_data)
                    except Exception:
                        try:
                            args_data = ast.literal_eval(args_data)
                        except Exception:
                            args_data = {"raw": args_data}
                if not isinstance(args_data, dict):
                    args_data = {"raw": args_data}
                calls.append((resolved, args_data))

        return calls

    parse_tool_calls = parse_ptc_calls


    def execute(self, call_name: str, args: Dict[str, Any]) -> str:
        """Executes a tool call via the modular registry with graceful fallbacks."""
        print(f"[TOOL] Executing: {call_name} with args: {args}")
        try:
            if self.registry.has(call_name):
                return self.registry.execute(call_name, args)
            return f"Notice: Tool '{call_name}' is not recognized in the registry. Please answer directly using your baseline knowledge."
        except Exception as e:
            return f"Notice: Tool '{call_name}' execution encountered an issue ({str(e)}). Please answer using available knowledge."

    def get_definitions(self, level: int = 1) -> List[Dict[str, Any]]:
        """Returns tool definitions permitted for the current persona level and offline state."""
        if level < 2:
            return []

        from System.network_guard import is_offline_mode
        is_offline = is_offline_mode() or (self.app and getattr(self.app, 'config', {}).get("offline_mode", False))

        base_tools = self.tools
        if is_offline:
            base_tools = [t for t in base_tools if t["function"]["name"] not in ("web_search",)]

        allowed_names = {
            "get_system_stats",
            "control_rgb",
            "generate_image",
            "read_file",
            "read_file_range",
            "send_peer_message",
            "send_peer_mail",
            "check_serenity_mailbox",
            "manage_quarantine_mail",
            "share_chat_with_peer",
            "pair_serenity_peer",
            "delegate_to_serenitydev",
        }
        if not is_offline:
            allowed_names.add("web_search")
        return [t for t in base_tools if t["function"]["name"] in allowed_names]

    def get_python_stubs(self, level: int = 1) -> str:
        """Generates typed Python function stubs for Programmatic Tool Calling (PTC - arXiv:2608.06370v1)."""
        tools = self.get_definitions(level)
        if not tools:
            return ""
        stubs = ["# Available Tools (Programmatic Tool Calling - execute via Python function call):"]
        for t in tools:
            f = t["function"]
            name = f["name"]
            desc = f["description"]
            params = f.get("parameters", {}).get("properties", {})
            req = f.get("parameters", {}).get("required", [])

            args_list = []
            for p_name, p_info in params.items():
                p_type = p_info.get("type", "str")
                py_type = "str"
                if p_type in ("integer", "int"):
                    py_type = "int"
                elif p_type in ("array", "list"):
                    py_type = "list"
                elif p_type in ("object", "dict"):
                    py_type = "dict"
                elif p_type in ("boolean", "bool"):
                    py_type = "bool"

                if p_name in req:
                    args_list.append(f"{p_name}: {py_type}")
                else:
                    args_list.append(f"{p_name}: {py_type} = None")

            args_str = ", ".join(args_list)
            stubs.append(f'def {name}({args_str}):\n    """{desc}"""\n    ...')
        return "\n\n".join(stubs)

    def get_gemma_declarations(self, level: int) -> str:
        """Generates Gemma-4 official template-aligned tool declarations string."""
        def official_q(s):
            return f'<|"|>{s}<|"|>'
        tool_defs = ""
        tools = self.get_definitions(level)
        for t in tools:
            f = t["function"]
            t_params = f.get("parameters", {})
            t_props = t_params.get("properties", {})
            t_prop_str = ",".join([
                f"{k}:{{description:{official_q(v.get('description',''))},type:{official_q(v.get('type','STRING'))}}}"
                for k, v in t_props.items()
            ])
            t_req_str = ",".join([official_q(r) for r in t_params.get("required", [])])

            tool_defs += f"<|tool>declaration:{f['name']}{{description:{official_q(f['description'])},parameters:{{properties:{{{t_prop_str}}},required:[{t_req_str}]}}}}<tool|>\n"
        if tools:
            tool_defs += "\n"
        return tool_defs
