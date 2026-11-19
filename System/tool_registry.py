# System/tool_registry.py
# Modular Tool Registry for Programmatic & Template Tool Calling in SerenityPC.
# Includes V2.3.0 Level 2 P2P Messaging, Encrypted Mail (.smp), 72-Hour Trust Gate,
# AST Multi-Line PTC Parser, and SerenityDev MCP Frontend Delegation.

import os
import sys
import time
import re
import ast
import json
import shutil
import tempfile
import threading
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
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
    try:
        from serenity_mail_courier import (
            SerenityEnvResolver,
            Level2CourierAgent,
            PacketType,
            ExchangeMode,
            CrossStateP2PTransport,
            SerenityMailCrypto,
        )
    except ImportError:
        class SerenityEnvResolver:
            @staticmethod
            def get_writable_root(app=None):
                return getattr(app, "script_dir", os.getcwd()) if app else os.getcwd()
            @staticmethod
            def get_bundle_root(app=None):
                return getattr(app, "script_dir", os.getcwd()) if app else os.getcwd()
            @staticmethod
            def find_python_interpreter(app=None):
                return sys.executable
        class Level2CourierAgent:
            def __init__(self, app=None): self.app = app
        PacketType = ExchangeMode = CrossStateP2PTransport = SerenityMailCrypto = None


class GemmaToolRegistry:
    """Handles tool definitions, schemas, AST PTC parsing, and modular execution for Serenity models."""

    TOOL_ALIAS_MAP = {
        "readfile": "read_file",
        "writefile": "write_file",
        "search": "web_search",
        "google_search": "web_search",
        "read_file_content": "read_file",
        "list_dir": "list_directory",
        "ls": "list_directory",
        "mkdir": "organize_files",
        "move_file": "organize_files",
        "serenitydev": "control_serenitydev",
        "terminal": "run_terminal_command",
        "run_command": "run_terminal_command",
        "generate_gif": "generate_video",
        "create_gif": "generate_video",
        "create_video": "generate_video",
        "make_gif": "generate_video",
    }

    DEFAULT_TOOLS = [
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
                            "path": {"type": "string", "description": "Absolute or workspace-relative path to the file. Always pass the exact full path; never abbreviate with '...'."}
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
                            "path": {"type": "string", "description": "Absolute or workspace-relative path to the file. Always pass the exact full path; never abbreviate with '...'."},
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
                            "type": {"type": "string", "description": "Type: 'image', 'diagram', 'gif', or 'video'"},
                            "resolution": {"type": "string", "description": "Optional resolution e.g. '768x768' or '512x512'."}
                        },
                        "required": ["prompt", "type"]
                    }
                }
            },
            {
                "function": {
                    "name": "generate_video",
                    "description": "Generates an animated GIF or short video clip from a text prompt.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Visual description of the motion scene to synthesize."},
                            "format": {"type": "string", "description": "Output format: 'gif' or 'video' (mp4). Default is 'gif'."},
                            "frames": {"type": "integer", "description": "Number of frames to generate (default 16)."},
                            "fps": {"type": "integer", "description": "Frames per second (default 8)."}
                        },
                        "required": ["prompt"]
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
            },
            {
                "function": {
                    "name": "write_file",
                    "description": "Writes or appends text content to a local file (auto-creates parent folders for websites/projects within permitted sandbox or expanded access roots).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Absolute or workspace-relative path to write."},
                            "content": {"type": "string", "description": "Text content to write to the file."},
                            "mode": {"type": "string", "description": "'overwrite' (default) or 'append'."}
                        },
                        "required": ["path", "content"]
                    }
                }
            },
            {
                "function": {
                    "name": "list_directory",
                    "description": "Lists files and folders in a local directory within permitted sandbox or expanded access roots.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Directory path to list (defaults to workspace root). Always pass the exact full path; never abbreviate with '...'."}
                        },
                        "required": []
                    }
                }
            },
            {
                "function": {
                    "name": "organize_files",
                    "description": "Creates directories or moves/renames/copies files and folders within permitted sandbox or expanded access roots.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "'mkdir', 'move', 'rename', or 'copy'."},
                            "source_path": {"type": "string", "description": "Target directory for mkdir, or source file/folder path."},
                            "destination_path": {"type": "string", "description": "Destination path for move, rename, or copy."}
                        },
                        "required": ["action", "source_path"]
                    }
                }
            },
            {
                "function": {
                    "name": "control_serenitydev",
                    "description": "Controls SerenityDev server lifecycle and requests ('status', 'start', 'stop', 'restart', 'preload', 'ask').",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "description": "'status', 'start', 'stop', 'restart', 'preload', or 'ask'."},
                            "model": {"type": "string", "description": "Optional target model name for preload or restart."},
                            "prompt": {"type": "string", "description": "Prompt text when action is 'ask'."}
                        },
                        "required": ["action"]
                    }
                }
            },
            {
                "function": {
                    "name": "run_terminal_command",
                    "description": "Runs a shell command in SerenityPC's integrated terminal drawer and returns stdout/stderr.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "Shell command to execute (e.g., 'dir /s /b'). Pass target folder in 'cwd' instead of embedding paths here."},
                            "cwd": {"type": "string", "description": "Optional absolute or workspace-relative working directory path."}
                        },
                        "required": ["command"]
                    }
                }
            }
    ]

    def __init__(self, chatbot_app):
        self.app = chatbot_app
        self.registry = ModularRegistry(name="ToolRegistry")
        self.courier = Level2CourierAgent(app=chatbot_app)
        self._last_listed_dirs: List[str] = []
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
        self.tools = list(self.DEFAULT_TOOLS)
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

            cfg = getattr(self.app, "config", {}) if self.app else {}
            kiwix_enabled = cfg.get("kiwix_search_enabled", True)
            zim_dir = cfg.get("kiwix_zim_dir", "")

            def _emit_proof(msg: str) -> None:
                pq = getattr(self.app, "process_queue", None) if self.app else None
                if pq:
                    try:
                        pq.put({"status": "tool_log_update", "content": f"\n{msg}"})
                    except Exception:
                        pass

            kiwix_results = []
            if kiwix_enabled and zim_dir:
                try:
                    from System.kiwix_search import search_kiwix_zims
                    for kh in search_kiwix_zims(query, zim_dir, max_results=4):
                        c_title = clean_search_bloat(kh["title"])
                        c_snip = clean_search_bloat(kh["snippet"])
                        if c_title and c_snip:
                            kiwix_results.append(f"[Kiwix: {kh['source']} | {c_title}]\n{c_snip}")
                except Exception as ke:
                    print(f"[SEARCH DEBUG] Kiwix search error: {ke}")

            from System.network_guard import is_offline_mode
            if is_offline_mode() or cfg.get("offline_mode", False):
                if kiwix_results:
                    _emit_proof(f"[SEARCH PROOF] Provider: Kiwix Offline | Extracts: {len(kiwix_results)}")
                    return f"Kiwix Offline Results for '{query}':\n\n" + "\n\n".join(kiwix_results)
                msg = f"[OFFLINE MODE] Live web search blocked by offline policy for query: '{query}'. Please answer using internal offline knowledge."
                _emit_proof(msg)
                return msg

            import requests
            import urllib.parse
            from bs4 import BeautifulSoup

            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.7',
            }

            web_results = []
            web_provider = ""

            # 1. DDGS (Yahoo / auto backend)
            try:
                from ddgs import DDGS
                for backend in ("yahoo", "auto"):
                    try:
                        hits = DDGS().text(query, backend=backend, max_results=5)
                        for h in hits or []:
                            c_title = clean_search_bloat(h.get("title", ""))
                            c_snip = clean_search_bloat(h.get("body", ""))
                            if c_title and c_snip:
                                web_results.append(f"[Web: {c_title}]\n{c_snip}")
                        if web_results:
                            web_provider = f"DDGS({backend})"
                            break
                    except Exception:
                        continue
            except Exception as e:
                print(f"[SEARCH DEBUG] DDGS failed: {e}")

            # 2. Direct Yahoo HTML fallback
            if not web_results:
                try:
                    resp = requests.get(
                        "https://search.yahoo.com/search",
                        params={"p": query},
                        headers=headers,
                        timeout=8,
                    )
                    if resp.status_code == 200:
                        soup = BeautifulSoup(resp.text, 'html.parser')
                        for res in soup.select('div.dd.algo, div.algo-sr'):
                            title = res.select_one('h3, .title')
                            snippet = res.select_one('.compText, p')
                            if title and snippet:
                                c_title = clean_search_bloat(title.get_text(" ", strip=True))
                                c_snip = clean_search_bloat(snippet.get_text(" ", strip=True))
                                if c_title and c_snip:
                                    web_results.append(f"[Web: {c_title}]\n{c_snip}")
                        if web_results:
                            web_provider = f"Yahoo({resp.status_code})"
                except Exception as e:
                    print(f"[SEARCH DEBUG] Yahoo failed: {e}")

            # 3. Wikipedia API fallback for factual queries
            if not web_results:
                try:
                    resp = requests.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={"action": "query", "list": "search", "srsearch": query, "srlimit": 4, "format": "json"},
                        headers={"User-Agent": "SerenityPC/1.0"},
                        timeout=6,
                    )
                    if resp.status_code == 200:
                        for item in resp.json().get("query", {}).get("search", []):
                            c_title = clean_search_bloat(item.get("title", ""))
                            c_snip = clean_search_bloat(BeautifulSoup(item.get("snippet", ""), "html.parser").get_text(" ", strip=True))
                            if c_title and c_snip:
                                web_results.append(f"[Wikipedia: {c_title}]\n{c_snip}")
                        if web_results:
                            web_provider = f"WikipediaAPI({resp.status_code})"
                except Exception:
                    pass

            combined = kiwix_results + web_results[:5]
            if combined:
                prov = " + ".join(p for p in ([f"Kiwix({len(kiwix_results)})"] if kiwix_results else []) + ([f"{web_provider}({len(web_results[:5])})"] if web_results else []))
                _emit_proof(f"[SEARCH PROOF] Provider: {prov} | Total Extracts: {len(combined)}")
                return f"Search Results for '{query}':\n\n" + "\n\n".join(combined)

            return f"Notice: Live & Kiwix search returned 0 retrievable hits for '{query[:60]}'. Proceeding with internal baseline knowledge."

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
            req_type = str(args.get("type") or args.get("format") or "image").lower().strip()
            if isinstance(prompt, dict):
                req_type = str(prompt.get("type", req_type)).lower().strip()
                prompt = prompt.get("prompt") or prompt.get("description") or str(prompt)

            clean_prompt = str(prompt).strip()
            clean_prompt = re.sub(r'<\|"?|\"?\|?>?|<\|+\|\>', '', clean_prompt).strip(' "<|>\\')

            cfg = getattr(self.app, "config", {}) if self.app else {}
            wait_flag = bool(args.get("wait", cfg.get("wait_on_tool_calls", False)))
            synth_meta: Dict[str, Any] = {}
            try:
                from System.media_generator import MediaGenerator
                gen_engine = MediaGenerator(self.app)
                media_mode = req_type if req_type in ("image", "diagram", "gif", "video") else "image"
                synth_meta = gen_engine.generate(
                    prompt=clean_prompt,
                    media_type=media_mode,
                    resolution=args.get("resolution"),
                    steps=args.get("steps"),
                    frames=args.get("frames"),
                    fps=args.get("fps"),
                    wait=wait_flag
                )
            except Exception as gen_err:
                print(f"[TOOL] MediaGenerator notice: {gen_err}")

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
                    payload_data = {
                        "req_type": req_type,
                        "content": clean_prompt,
                        "file_path": synth_meta.get("file_path", ""),
                        "fps": synth_meta.get("fps", 8),
                    }
                    with open(payload_path, "w", encoding="utf-8") as fp:
                        json.dump(payload_data, fp, ensure_ascii=False)
                    py_bin = SerenityEnvResolver.find_python_interpreter(self.app) or sys.executable
                    proc = subprocess.Popen([py_bin, viewer_script, payload_path])
                    if wait_flag:
                        timeout_s = float(cfg.get("tool_call_timeout_sec", 30.0))
                        try:
                            proc.wait(timeout=timeout_s)
                        except Exception:
                            pass
                except Exception as e:
                    print(f"[TOOL] Image viewer error: {e}")

            if wait_flag:
                spawn_viewer()
            else:
                threading.Thread(target=spawn_viewer, daemon=True).start()

            file_note = f" (Saved: {synth_meta.get('file_path')})" if synth_meta.get("file_path") else ""
            if self.app and hasattr(self.app, "process_queue"):
                try:
                    self.app.process_queue.put({"status": "tool_log_update", "content": f"\n[IMAGE / VISUAL HUD] Rendered {req_type}: {clean_prompt[:80]}...{file_note}\n"})
                except Exception:
                    pass
            return f"Successfully generated and displayed {req_type} visual prompt in HUD overlay: '{clean_prompt[:60]}...'{file_note}"

        @self.registry.register("generate_video")
        def handle_generate_video(args: Any) -> str:
            if isinstance(args, str):
                args = {"prompt": args, "type": "gif"}
            elif isinstance(args, dict):
                args = dict(args)
                fmt = str(args.get("format") or args.get("type") or "gif").lower().strip()
                args["type"] = "video" if fmt in ("mp4", "video") else "gif"
            else:
                args = {"prompt": str(args), "type": "gif"}
            return handle_generate_image(args)

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
            Enforces strict tool file sandboxing with the 72-Hour Profile Age Gate
            plus optional opt-in expanded access:
            - Sensitive credential/crypto/config files and OS/secret dirs are ALWAYS blocked.
            - Expanded access (`expanded_file_access` True, opt-in) permits user folders, home, non-OS drives, and configured roots.
            - Default (unauthenticated, locked, or profile < 72h old): strictly restricted to Users/, Logs/, .tmp/, History/.
            - Password-protected, unlocked profile >= 72 hours old: allowed read/write access across the repository root.
            """
            w_root = os.path.realpath(SerenityEnvResolver.get_writable_root(self.app))
            b_root = os.path.realpath(SerenityEnvResolver.get_bundle_root(self.app))
            real_p = os.path.realpath(target_p)
            path_obj = Path(real_p)
            fname_lower = path_obj.name.lower()
            parts_lower = {part.lower() for part in path_obj.parts}

            if (
                fname_lower == ".env"
                or fname_lower.startswith(".env.")
                or fname_lower in {
                    "vault_state.json", "vault.db", "semantic_vault.db", "serenity_mailbox.db",
                    "rlhf_logs.json", "id_rsa", "id_ed25519", "credentials",
                    ".git-credentials", ".netrc", ".npmrc",
                }
                or fname_lower.endswith((".encz", ".key", ".pem", ".pfx", ".p12", ".sqlite", ".sqlite3"))
                or bool(parts_lower & {".ssh", ".aws", ".gnupg", ".kube"})
            ):
                return False

            if fname_lower == "config.json":
                parent_low = path_obj.parent.name.lower()
                grandparent_low = path_obj.parent.parent.name.lower()
                if parent_low == "system" or grandparent_low == "users":
                    return False

            blocked_sys_roots = []
            if os.name == "nt":
                sys_drive = os.environ.get("SystemDrive", "C:") + os.sep
                for sub in ("Windows", "Program Files", "Program Files (x86)", "ProgramData", "System Volume Information", "$Recycle.Bin"):
                    blocked_sys_roots.append(os.path.realpath(os.path.join(sys_drive, sub)))
            else:
                blocked_sys_roots.extend(["/etc", "/usr", "/bin", "/sbin", "/boot", "/proc", "/sys", "/dev"])
            for s_root in blocked_sys_roots:
                try:
                    if os.path.commonpath([s_root, real_p]) == s_root:
                        return False
                except Exception:
                    pass

            cfg = getattr(self.app, "config", {}) if self.app else {}
            if not isinstance(cfg, dict):
                cfg = {}
            expanded_on = bool(cfg.get("expanded_file_access", False))

            if expanded_on:
                allowed_expanded = [w_root, b_root, os.path.realpath(str(Path.home())), os.path.realpath(tempfile.gettempdir())]
                raw_roots = str(cfg.get("expanded_file_roots", "") or "")
                for r in re.split(r"[;,]", raw_roots):
                    if r.strip():
                        allowed_expanded.append(os.path.realpath(r.strip()))
                # Allow non-OS drives (e.g. D:/..Z:/), home, and temp when user opts in
                drive_head = os.path.splitdrive(real_p)[0].upper()
                if os.name == "nt" and drive_head and drive_head != os.environ.get("SystemDrive", "C:").upper():
                    return True
                for root in allowed_expanded:
                    try:
                        if os.path.commonpath([root, real_p]) == root:
                            return True
                    except Exception:
                        pass

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

        def _resolve_target_path(raw_path: str) -> str:
            p = str(raw_path or "").strip()
            if not p:
                return SerenityEnvResolver.get_writable_root(self.app)
            if not os.path.isabs(p):
                return os.path.join(SerenityEnvResolver.get_writable_root(self.app), p)
            return p

        def _resolve_file_path(raw_path: str) -> Tuple[str, Optional[str]]:
            """Resolves paths verbatim or recovers '...' ellipsis paths against recently listed dirs."""
            raw_p = str(raw_path or "").strip().strip("\"'")
            is_abbrev = (
                raw_p == "..."
                or raw_p.startswith((".../", "...\\"))
                or "/.../" in raw_p
                or "\\...\\" in raw_p
            )
            if is_abbrev:
                m_suf = re.split(r'\.\.\.[\\/]+', raw_p)
                suffix = m_suf[-1].lstrip("\\/") if len(m_suf) > 1 else ""
                if suffix:
                    for base_d in self._last_listed_dirs:
                        for cand in (os.path.join(base_d, suffix), os.path.join(base_d, os.path.basename(suffix))):
                            if os.path.exists(cand):
                                return cand, None
                hint = f" (recent dir: '{self._last_listed_dirs[0]}')" if self._last_listed_dirs else ""
                return raw_p, (
                    f"Notice: Path '{raw_path}' uses '...' abbreviation and could not be resolved. "
                    f"Pass the full verbatim path{hint}."
                )

            target_p = _resolve_target_path(raw_p)
            if raw_p and not os.path.isabs(raw_p) and not os.path.exists(target_p):
                for base_d in self._last_listed_dirs:
                    cand = os.path.join(base_d, raw_p)
                    if os.path.exists(cand):
                        return cand, None
            return target_p, None

        @self.registry.register("read_file")
        def handle_read_file(args: Dict[str, Any]) -> str:
            path = args.get("path")
            if not path:
                return "Notice: No file path provided."

            target_path, resolve_err = _resolve_file_path(path)
            if resolve_err:
                return resolve_err

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

            target_path, resolve_err = _resolve_file_path(path)
            if resolve_err:
                return resolve_err

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

        @self.registry.register("write_file")
        def handle_write_file(args: Dict[str, Any]) -> str:
            path = args.get("path")
            content = args.get("content")
            mode = str(args.get("mode") or "overwrite").strip().lower()
            if not path or content is None:
                return "Notice: Both 'path' and 'content' are required for write_file."

            target_path = _resolve_target_path(path)
            if not _is_sandboxed_path(target_path):
                return f"[SECURITY RESTRICTION] Write access blocked outside permitted workspace/expanded roots: '{path}'."

            try:
                parent = os.path.dirname(target_path)
                if parent:
                    os.makedirs(parent, exist_ok=True)
                open_mode = "a" if mode == "append" else "w"
                with open(target_path, open_mode, encoding="utf-8") as f:
                    f.write(str(content))
                return f"Success: Wrote {len(str(content))} chars to '{target_path}' (mode={mode})."
            except Exception as e:
                return f"Notice: Error writing file '{path}': {str(e)}"

        @self.registry.register("list_directory")
        def handle_list_directory(args: Dict[str, Any]) -> str:
            path = str(args.get("path") or ".").strip()
            if path in ("", "."):
                target_path = _resolve_target_path(path)
            else:
                target_path, resolve_err = _resolve_file_path(path)
                if resolve_err:
                    return resolve_err
            if not _is_sandboxed_path(target_path):
                return f"[SECURITY RESTRICTION] Directory listing blocked outside permitted roots: '{path}'."
            if not os.path.exists(target_path) or not os.path.isdir(target_path):
                return f"Notice: Directory '{path}' not found."
            try:
                real_dir = os.path.realpath(target_path)
                self._last_listed_dirs = [real_dir] + [d for d in self._last_listed_dirs if d != real_dir][:9]
                items = sorted(os.listdir(target_path))
                lines = []
                for name in items[:200]:
                    full = os.path.join(target_path, name)
                    if os.path.isdir(full):
                        lines.append(f"[DIR]  {name}/")
                    else:
                        sz = os.path.getsize(full)
                        lines.append(f"[FILE] {name} ({sz} bytes)")
                if len(items) > 200:
                    lines.append(f"... ({len(items) - 200} more items)")
                listing = "\n".join(lines) if lines else "(empty directory)"
                return f"Directory: {target_path}\n{listing}"
            except Exception as e:
                return f"Notice: Error listing directory '{path}': {str(e)}"

        @self.registry.register("organize_files")
        def handle_organize_files(args: Dict[str, Any]) -> str:
            action = str(args.get("action") or "").strip().lower()
            src_raw = str(args.get("source_path") or args.get("path") or "").strip()
            dst_raw = str(args.get("destination_path") or args.get("dest") or "").strip()
            if not action or not src_raw:
                return "Notice: 'action' and 'source_path' are required for organize_files."

            src_p = _resolve_target_path(src_raw)
            if not _is_sandboxed_path(src_p):
                return f"[SECURITY RESTRICTION] Source path blocked outside permitted roots: '{src_raw}'."

            if action == "mkdir":
                try:
                    os.makedirs(src_p, exist_ok=True)
                    return f"Success: Directory created at '{src_p}'."
                except Exception as e:
                    return f"Notice: mkdir failed for '{src_raw}': {str(e)}"

            if not dst_raw:
                return f"Notice: 'destination_path' is required for action '{action}'."
            dst_p = _resolve_target_path(dst_raw)
            if not _is_sandboxed_path(dst_p):
                return f"[SECURITY RESTRICTION] Destination path blocked outside permitted roots: '{dst_raw}'."
            if not os.path.exists(src_p):
                return f"Notice: Source path '{src_raw}' does not exist."
            if os.path.isdir(src_p):
                for root_d, dir_names, file_names in os.walk(src_p):
                    for dn in dir_names:
                        if not _is_sandboxed_path(os.path.join(root_d, dn)):
                            return f"[SECURITY RESTRICTION] Directory '{src_raw}' contains protected path '{dn}'."
                    for fn in file_names:
                        if not _is_sandboxed_path(os.path.join(root_d, fn)):
                            return f"[SECURITY RESTRICTION] Directory '{src_raw}' contains protected file '{fn}'."

            try:
                dst_parent = os.path.dirname(dst_p)
                if dst_parent:
                    os.makedirs(dst_parent, exist_ok=True)
                if action in ("move", "rename"):
                    shutil.move(src_p, dst_p)
                    return f"Success: Moved '{src_p}' -> '{dst_p}'."
                elif action == "copy":
                    if os.path.isdir(src_p):
                        shutil.copytree(src_p, dst_p, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src_p, dst_p)
                    return f"Success: Copied '{src_p}' -> '{dst_p}'."
                return f"Notice: Unsupported organize_files action '{action}'."
            except Exception as e:
                return f"Notice: organize_files '{action}' failed: {str(e)}"

        @self.registry.register("control_serenitydev")
        def handle_control_serenitydev(args: Dict[str, Any]) -> str:
            import urllib.request
            action = str(args.get("action") or "status").strip().lower()
            model = str(args.get("model") or "").strip()
            prompt = str(args.get("prompt") or "").strip()

            cfg = getattr(self.app, "config", {}) if self.app else {}
            mcp_url = str(cfg.get("serenitydev_mcp_url", "http://127.0.0.1:8002/mcp") if isinstance(cfg, dict) else "http://127.0.0.1:8002/mcp")
            base_url = mcp_url.rsplit("/mcp", 1)[0] if "/mcp" in mcp_url else mcp_url.rstrip("/")

            parsed_url = urllib.parse.urlparse(base_url)
            if parsed_url.scheme not in ("http", "https") or (parsed_url.hostname or "").lower() not in ("127.0.0.1", "localhost", "::1"):
                return "[SECURITY RESTRICTION] SerenityDev URL must use http(s) on loopback (127.0.0.1/localhost/::1)."

            if action in ("start", "stop", "shutdown", "restart", "preload"):
                trust = self.courier.get_trust_status()
                if not trust.get("veteran_unlocked"):
                    return json.dumps({
                        "status": "blocked_72h_gate",
                        "action": action,
                        "trust_status": trust.get("label", ""),
                        "message": f"[72-HOUR TRUST GATE] SerenityDev '{action}' requires an unlocked password-protected profile >= 72 hours old.",
                    })

            auth_headers = {}
            mcp_token = os.environ.get("SERENITY_MCP_TOKEN", "").strip() or (
                str(cfg.get("serenitydev_mcp_token", "")).strip() if isinstance(cfg, dict) else ""
            )
            if mcp_token:
                auth_headers["Authorization"] = f"Bearer {mcp_token}"

            if action == "start":
                try:
                    req_st = urllib.request.Request(f"{base_url}/api/status", headers=auth_headers, method="GET")
                    with urllib.request.urlopen(req_st, timeout=2.0) as resp:
                        return json.dumps({"status": "already_running", "details": json.loads(resp.read().decode("utf-8", errors="replace"))})
                except Exception:
                    pass
                w_root = SerenityEnvResolver.get_writable_root(self.app)
                candidates = [
                    os.path.join(os.path.dirname(w_root), "SerenityDev", "serenitydevserver.py"),
                    os.path.join(w_root, "SerenityDev", "serenitydevserver.py"),
                ]
                server_py = next((c for c in candidates if os.path.isfile(c)), None)
                if not server_py:
                    return "Notice: SerenityDev serenitydevserver.py not found on disk."
                try:
                    py_exe = SerenityEnvResolver.find_python_interpreter(self.app)
                    cflags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
                    proc = subprocess.Popen([py_exe, server_py], cwd=os.path.dirname(server_py), creationflags=cflags)
                    return json.dumps({"status": "starting", "pid": proc.pid, "server": server_py})
                except Exception as e:
                    return f"Notice: Failed to launch SerenityDev server: {e}"

            try:
                json_headers = {"Content-Type": "application/json", **auth_headers}
                if action == "status":
                    req = urllib.request.Request(f"{base_url}/api/status", headers=auth_headers, method="GET")
                    with urllib.request.urlopen(req, timeout=4.0) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                elif action in ("stop", "shutdown"):
                    req = urllib.request.Request(f"{base_url}/shutdown", data=b"{}", headers=json_headers, method="POST")
                    with urllib.request.urlopen(req, timeout=4.0) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                elif action == "restart":
                    payload = json.dumps({"model": model} if model else {}).encode("utf-8")
                    req = urllib.request.Request(f"{base_url}/api/restart", data=payload, headers=json_headers, method="POST")
                    with urllib.request.urlopen(req, timeout=8.0) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                elif action == "preload":
                    if not model:
                        return "Notice: 'model' is required for preload action."
                    encoded_model = urllib.parse.quote(model, safe="")
                    req = urllib.request.Request(f"{base_url}/api/preload/{encoded_model}", data=b"{}", headers=json_headers, method="POST")
                    with urllib.request.urlopen(req, timeout=15.0) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                elif action == "ask":
                    if not prompt:
                        return "Notice: 'prompt' is required for ask action."
                    payload = json.dumps({"prompt": prompt, "session_id": "serenitypc"}).encode("utf-8")
                    req = urllib.request.Request(f"{base_url}/ask", data=payload, headers=json_headers, method="POST")
                    with urllib.request.urlopen(req, timeout=60.0) as resp:
                        return resp.read().decode("utf-8", errors="replace")
                return f"Notice: Unknown SerenityDev action '{action}'."
            except Exception as e:
                return f"Notice: SerenityDev request '{action}' failed ({e})."

        @self.registry.register("run_terminal_command")
        def handle_run_terminal_command(args: Dict[str, Any]) -> str:
            cmd = str(args.get("command") or "").strip()
            cwd_raw = str(args.get("cwd") or "").strip()
            if not cmd:
                return "Notice: 'command' is required for run_terminal_command."
            target_cwd = _resolve_target_path(cwd_raw) if cwd_raw else SerenityEnvResolver.get_writable_root(self.app)
            if not _is_sandboxed_path(target_cwd):
                return f"[SECURITY RESTRICTION] Terminal cwd blocked outside permitted roots: '{cwd_raw}'."

            sensitive_cmd_pat = re.compile(
                r'(?i)(?:\.env\b|vault_state\.json|serenity_mailbox\.db|semantic_vault\.db|rlhf_logs\.json|'
                r'SERENITY_ENCRYPTED_STORAGE_KEY|SERENITY_BMB_PASSPHRASE|SERENITY_MCP_TOKEN|'
                r'id_rsa|id_ed25519|\.(?:encz|pem|key|pfx|p12|sqlite3?)\b)'
            )
            if sensitive_cmd_pat.search(cmd):
                return "[SECURITY RESTRICTION] Terminal commands referencing protected credentials, keys, or vault files are blocked."

            cfg = getattr(self.app, "config", {}) if self.app else {}
            if not isinstance(cfg, dict):
                cfg = {}
            from System.network_guard import is_offline_mode
            if is_offline_mode() or cfg.get("offline_mode", False):
                net_cli_pat = re.compile(
                    r'(?i)\b(?:curl|wget|invoke-webrequest|invoke-restmethod|iwr|irm|bitsadmin|certutil|'
                    r'nc|ncat|ssh|scp|sftp|ftp|nslookup|ping|tracert|telnet|python\s+-m\s+http)\b'
                )
                if net_cli_pat.search(cmd):
                    return "[OFFLINE MODE] Network utility commands are blocked while Offline Mode is enabled."

            if cmd.lower() == "cd" or cmd.lower().startswith("cd "):
                dest = cmd[2:].strip().strip('"').strip("'")
                if dest:
                    cand_dir = os.path.abspath(os.path.join(target_cwd, dest))
                    if not _is_sandboxed_path(cand_dir):
                        return f"[SECURITY RESTRICTION] Terminal cd blocked outside permitted roots: '{dest}'."

            confirm_fn = getattr(self.app, "_confirm_terminal_command", None) if self.app else None
            if not (callable(confirm_fn) and confirm_fn(cmd, target_cwd) is True):
                return (
                    f"[USER DENIED] User declined executing terminal command '{cmd}'. "
                    "Do not retry this command without asking the user; answer using permitted tools or knowledge."
                )

            if self.app and hasattr(self.app, "run_terminal_drawer_command"):
                return str(self.app.run_terminal_drawer_command(cmd, cwd=target_cwd))
            try:
                proc = subprocess.run(cmd, shell=True, cwd=target_cwd, capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace")
                out = (proc.stdout or "") + (f"\n[STDERR]\n{proc.stderr}" if proc.stderr else "")
                return out.strip() or f"(Command exited with code {proc.returncode})"
            except Exception as e:
                return f"Notice: Terminal command failed: {e}"

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

    @classmethod
    def parse_tool_calls_static(cls, model_output: str) -> List[Tuple[str, Dict[str, Any]]]:
        """Stateless tool-call parser backed by class-level DEFAULT_TOOLS schemas."""
        schema_props_map: Dict[str, List[str]] = {}
        valid_names = set()
        for t in cls.DEFAULT_TOOLS:
            fn = t.get("function", {})
            fname = fn.get("name")
            if fname:
                valid_names.add(fname)
                schema_props_map[fname] = list(fn.get("parameters", {}).get("properties", {}).keys())
        return cls._parse_ptc_calls_with_schema(model_output, valid_names, schema_props_map)

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
        valid_names = set(self.registry.list_keys())
        schema_props_map: Dict[str, List[str]] = {}
        for t in self.tools:
            fn = t.get("function", {})
            fname = fn.get("name")
            if fname:
                schema_props_map[fname] = list(fn.get("parameters", {}).get("properties", {}).keys())
        return self._parse_ptc_calls_with_schema(model_output, valid_names, schema_props_map)

    @classmethod
    def _parse_ptc_calls_with_schema(
        cls,
        model_output: str,
        valid_names: set,
        schema_props_map: Dict[str, List[str]],
    ) -> List[Tuple[str, Dict[str, Any]]]:
        if not model_output:
            return []

        calls: List[Tuple[str, Dict[str, Any]]] = []
        alias_map = cls.TOOL_ALIAS_MAP

        def _resolve_name(raw_name: str) -> Optional[str]:
            if not raw_name:
                return None
            n = raw_name.strip().lower()
            if n in valid_names:
                return n
            if n in alias_map and alias_map[n] in valid_names:
                return alias_map[n]
            return None

        def _repair_json_backslashes(text: str) -> str:
            text = re.sub(
                r'([A-Za-z]:(?:(?!\\")\\[^"\\]*)+)',
                lambda m: re.sub(r'\\\\|\\', lambda _: r'\\', m.group(1)),
                text,
            )
            return re.sub(
                r'\\(\\|["\\/bfnrt]|u[0-9a-fA-F]{4})|\\',
                lambda m: m.group(0) if m.group(1) else r'\\',
                text,
            )

        # 1. AST parser for Python-style code blocks or direct invocations
        candidates = re.findall(r"```(?:python|py|tool_call)?\s*(.*?)\s*```", model_output, flags=re.DOTALL)
        if not candidates:
            candidates = [model_output]

        for block in candidates:
            block_src = block.strip()
            try:
                tree = ast.parse(block_src, mode="exec")
            except SyntaxError:
                try:
                    tree = ast.parse(re.sub(r'\\(?![\\\'"abfnrtvoxuU0-7])', r'\\\\', block_src), mode="exec")
                except SyntaxError:
                    try:
                        tree = ast.parse(re.sub(r'\\(?![\\\'"])', r'\\\\', block_src), mode="exec")
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

        # Check tag calls: <ctrl42>call:name{...} / <|tool_call>call:name{path:"C:\..."}
        for call_m in re.finditer(
            r'(?:<ctrl42>call:|<\|tool_call>call:|<\|tool_call\|>call:|<\|tool>call:|call:|action:|<(?:channel\|)?(?:execute_tool|executetool)>)\s*([\w_]+)\s*\{(.*?)\}(?:<\/(?:execute_tool|executetool)>)?',
            model_output,
            re.DOTALL | re.IGNORECASE
        ):
            raw_fn = call_m.group(1).strip()
            resolved = _resolve_name(raw_fn)
            if not resolved:
                continue
            raw_orig = call_m.group(2).strip()
            if '<|"|>' in raw_orig:
                raw_orig = re.sub(
                    r'<\|"\|>(.*?)<\|"\|>',
                    lambda m: json.dumps(m.group(1)),
                    raw_orig,
                    flags=re.DOTALL,
                )
            raw_inner = raw_orig.replace('<|"', '"').replace('"|>', '"')
            args_str = "{" + raw_inner + "}"
            parsed_args: Any = {}
            try:
                parsed_args = json.loads(args_str)
            except Exception:
                try:
                    parsed_args = ast.literal_eval(args_str)
                except Exception:
                    # Handle Gemma-4 unquoted keys ({path:"..."}) and raw Windows backslashes ("C:\Users\...")
                    # without ever treating Windows drive letters (e.g. C:\) as separate keys.
                    try:
                        fixed = re.sub(
                            r'("(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\')|((?:^|[{,])\s*)([A-Za-z_]\w*)\s*:(?!\s*[\\/])',
                            lambda m: m.group(1) if m.group(1) is not None else f'{m.group(2)}"{m.group(3)}":',
                            raw_inner,
                        )
                        fixed = re.sub(r',\s*$', '', fixed.strip())
                        fixed = _repair_json_backslashes(fixed)
                        parsed_args = json.loads("{" + fixed + "}")
                    except Exception:
                        parsed_args = {}
                        schema_props = schema_props_map.get(resolved, [])
                        key_matches = [
                            km for km in re.finditer(
                                r'(?:^|[,\n])\s*["\']?([A-Za-z_]\w*)["\']?\s*:(?!\s*[\\/])',
                                raw_inner,
                            )
                            if not (len(km.group(1)) == 1 and re.match(r'\s*[\\/]', raw_inner[km.end():]))
                        ]
                        if schema_props:
                            schema_matches = [km for km in key_matches if km.group(1) in schema_props]
                            if schema_matches:
                                key_matches = schema_matches
                        for idx, km in enumerate(key_matches):
                            k = km.group(1).strip()
                            v_start = km.end()
                            v_end = key_matches[idx + 1].start() if idx + 1 < len(key_matches) else len(raw_inner)
                            raw_v = raw_inner[v_start:v_end].strip().rstrip(',').strip()
                            if len(raw_v) >= 2 and raw_v[0] == raw_v[-1] and raw_v[0] in ('"', "'"):
                                q_ch = raw_v[0]
                                v: Any = raw_v[1:-1].replace(f'\\{q_ch}', q_ch)
                            else:
                                try:
                                    v = json.loads(raw_v)
                                except Exception:
                                    v = raw_v
                            parsed_args[k] = v
            calls.append((resolved, parsed_args if isinstance(parsed_args, dict) else {}))

        if calls:
            return calls

        if not json_containers:
            for jm in re.finditer(r'\{\s*["\'](?:action|tool|name|function)["\']\s*:\s*["\'][\w_]+["\']', model_output):
                start_idx = jm.start()
                brace_count = 0
                end_idx = -1
                in_str = False
                str_char = ''
                escape = False
                for i in range(start_idx, len(model_output)):
                    ch = model_output[i]
                    if escape:
                        escape = False
                        continue
                    if ch == '\\':
                        escape = True
                        continue
                    if in_str:
                        if ch == str_char:
                            in_str = False
                    elif ch in ('"', "'"):
                        in_str = True
                        str_char = ch
                    elif ch == '{':
                        brace_count += 1
                    elif ch == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_idx = i + 1
                            break
                if end_idx != -1:
                    json_containers.append(model_output[start_idx:end_idx])
                else:
                    json_containers.append(model_output[start_idx:])

        for jc in json_containers:
            parsed = None
            try:
                parsed = json.loads(jc.replace(r"\'", "'"))
            except Exception:
                try:
                    parsed = ast.literal_eval(jc)
                except Exception:
                    try:
                        parsed = json.loads(_repair_json_backslashes(jc.replace(r"\'", "'")))
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

        if not calls and valid_names:
            # Inline prose-embedded PTC fallback (e.g. "Here is info: send_peer_message()")
            names_pat = "|".join(sorted((re.escape(n) for n in valid_names | set(alias_map.keys())), key=len, reverse=True))
            for pm in re.finditer(rf'\b({names_pat})\s*\(([^()\n]*)\)', model_output, re.DOTALL | re.IGNORECASE):
                resolved = _resolve_name(pm.group(1))
                if not resolved:
                    continue
                raw_args = pm.group(2).strip()
                parsed_inline: Dict[str, Any] = {}
                if raw_args:
                    try:
                        expr = ast.parse(f"{resolved}({raw_args})", mode="eval")
                        if isinstance(expr.body, ast.Call):
                            schema_props = schema_props_map.get(resolved, [])
                            for idx, pos_arg in enumerate(expr.body.args):
                                key = schema_props[idx] if idx < len(schema_props) else f"arg_{idx}"
                                parsed_inline[key] = ast.literal_eval(pos_arg)
                            for kw in expr.body.keywords:
                                if kw.arg:
                                    parsed_inline[kw.arg] = ast.literal_eval(kw.value)
                    except Exception:
                        pass
                calls.append((resolved, parsed_inline))

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
            "generate_video",
            "read_file",
            "read_file_range",
            "write_file",
            "list_directory",
            "organize_files",
            "control_serenitydev",
            "run_terminal_command",
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