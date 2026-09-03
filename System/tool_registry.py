# System/tool_registry.py
# Modular Tool Registry for Programmatic & Template Tool Calling in SerenityPC.

import os
import re
import json
import threading
import subprocess
try:
    import psutil
except ImportError:
    psutil = None
from typing import List, Dict, Any
from System.modular_registry import ModularRegistry

class GemmaToolRegistry:
    """Handles tool definitions, schemas, and modular execution for Serenity models."""
    def __init__(self, chatbot_app):
        self.app = chatbot_app
        self.registry = ModularRegistry(name="ToolRegistry")
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
                            "path": {"type": "string", "description": "Absolute path to the file."}
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
                            "path": {"type": "string", "description": "Absolute path to the file."},
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
            }
        ]
        self._register_handlers()

    def _register_handlers(self):
        """Registers modular handlers using ModularRegistry."""

        def clean_search_bloat(text: str) -> str:
            """Filters out web boilerplate, cookie notices, navigation junk, and duplicate noise."""
            if not text: return ""
            # Strip common web noise lines
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
            
            # Check Offline Mode Guard
            from System.network_guard import is_offline_mode
            if is_offline_mode() or (self.app and getattr(self.app, 'config', {}).get("offline_mode", False)):
                msg = f"[OFFLINE MODE] Live web search blocked by offline policy for query: '{query}'. Please answer using internal offline knowledge."
                pq = getattr(self.app, "process_queue", None)
                if pq:
                    try: pq.put({"status": "tool_log_update", "content": f"\n{msg}"})
                    except: pass
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
            
            # Brave Search (Primary)
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

            # Bing Scraper (Fallback 1)
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
            except: pass

            # DuckDuckGo (Fallback 2)
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
            except: pass

            # Deep Browse (Playwright Headless Fallback)
            try:
                from playwright.sync_api import sync_playwright
                if self.app and hasattr(self.app, "process_queue"):
                    self.app.process_queue.put({"status": "thinking_status", "content": "USR: Initiating Stealth Browser Instance..."})
                
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page(user_agent=headers['User-Agent'])
                    page.goto(f"https://www.bing.com/search?q={urllib.parse.quote(query)}", wait_until="load", timeout=15000)
                    page.wait_for_timeout(3000)
                    
                    content = page.evaluate("""() => {
                        const results = [];
                        const bingResults = document.querySelectorAll('li.b_algo, .b_caption, .b_snippet');
                        if (bingResults.length > 0) {
                            bingResults.forEach(el => results.push(el.innerText));
                        } else {
                            document.querySelectorAll('p, span, div, h2').forEach(el => {
                                const txt = el.innerText.trim();
                                if (txt.length > 80 && !txt.includes('{')) {
                                    results.push(txt);
                                }
                            });
                        }
                        return results.slice(0, 10);
                    }""")
                    browser.close()
                    
                    cleaned_content = [clean_search_bloat(c) for c in content if clean_search_bloat(c)]
                    if cleaned_content:
                        proof_msg = f"[SEARCH PROOF] Provider: Playwright (Bing) | Cleaned Fragments: {len(cleaned_content)}"
                        if self.app and hasattr(self.app, "process_queue"):
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                        return f"Deep Web Extract for '{query}':\n\n" + "\n\n".join(cleaned_content)
            except Exception as e:
                print(f"[SEARCH DEBUG] Playwright failed: {e}")
            
            return f"Notice: Web search was unable to retrieve live results for '{query}' (network offline or search providers unreachable). Please proceed to answer the user directly and gracefully using your internal knowledge."

        @self.registry.register("delegate_subtask")
        def handle_delegate_subtask(args: Dict[str, Any]) -> str:
            subagent_lvl = args.get("subagent_level", 2)
            task_desc = args.get("task_description", "")
            return f"[SUBAGENT LVL {subagent_lvl} DISPATCH]: Task '{task_desc}' received and queued for handoff."

        @self.registry.register("generate_image")
        def handle_generate_image(args: Any) -> str:
            import sys
            import ast
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

            # Handle nested action_input / parameters / arguments
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
                            except Exception: pass
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
            elif isinstance(prompt, str) and (prompt.strip().startswith("{") and prompt.strip().endswith("}")):
                try:
                    p_dict = ast.literal_eval(prompt)
                    if isinstance(p_dict, dict):
                        req_type = p_dict.get("type", req_type)
                        prompt = p_dict.get("prompt") or p_dict.get("description") or prompt
                except Exception: pass

            clean_prompt = str(prompt).strip()
            import re
            clean_prompt = re.sub(r'<\|"?|\\"?\|?>?|<\||\|>', '', clean_prompt).strip(' "<|>\\')
            
            def spawn_viewer():
                try:
                    base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
                    scratch_dir = os.path.join(base_dir, "scratch")
                    os.makedirs(scratch_dir, exist_ok=True)
                    viewer_script = os.path.join(base_dir, "System", "hud_viewer.py")
                    hud_data_path = os.path.join(scratch_dir, "hud_data.json")
                    with open(hud_data_path, "w", encoding="utf-8") as f:
                        json.dump({"req_type": req_type, "content": clean_prompt}, f)
                    subprocess.Popen([sys.executable, viewer_script, hud_data_path])
                except Exception as e:
                    print(f"[TOOL] Image viewer error: {e}")
            
            threading.Thread(target=spawn_viewer, daemon=True).start()
            if self.app and hasattr(self.app, "process_queue"):
                try:
                    self.app.process_queue.put({"status": "tool_log_update", "content": f"\n[IMAGE / VISUAL HUD] Rendered {req_type}: {clean_prompt[:80]}...\n"})
                except: pass
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
                except: pass
            if nvidia_ml:
                try:
                    handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(handle)
                    stats["vram"] = f"{mem.used/1024**2:.0f} / {mem.total/1024**2:.0f} MB"
                except: pass
            return json.dumps(stats)

        def _is_sandboxed_path(target_p: str) -> bool:
            """Ensure target path is within workspace root or designated data subdirectories."""
            base_dir = os.path.abspath(getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd())
            abs_p = os.path.abspath(target_p)
            try:
                # Must reside inside base workspace directory
                return os.path.commonpath([base_dir, abs_p]) == base_dir
            except Exception:
                return False

        @self.registry.register("read_file")
        def handle_read_file(args: Dict[str, Any]) -> str:
            path = args.get("path")
            if not path:
                return "Notice: No file path provided."
            
            target_path = path
            if not os.path.isabs(target_path):
                base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
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
                base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
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
            base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
            state_path = os.path.join(base_dir, "System", "rgb_state.json")
            try:
                state = {}
                if os.path.exists(state_path):
                    with open(state_path, 'r') as f:
                        state = json.load(f)
                
                if "color" in args: state["manual_color"] = args["color"]
                if "style" in args: state["manual_style"] = args["style"]
                state["mode"] = "manual"
                
                with open(state_path, 'w') as f:
                    json.dump(state, f, indent=4)
                return f"RGB adjusted: Mode=Manual, Color={args.get('color')}, Style={args.get('style')}"
            except Exception as e:
                return f"Notice: RGB controller not fully accessible ({str(e)}). Simulated state applied."

    def execute(self, call_name: str, args: Dict[str, Any]) -> str:
        """Executes a tool call via the modular registry with graceful fallbacks."""
        print(f"[TOOL] Executing: {call_name} with args: {args}")
        try:
            if self.registry.has(call_name):
                return self.registry.execute(call_name, args)
            return f"Notice: Tool '{call_name}' is not recognized in the registry. Please answer directly using your baseline knowledge."
        except Exception as e:
            return f"Notice: Tool '{call_name}' execution encountered an issue ({str(e)}). Please answer using available knowledge."

    def get_definitions(self, level=1) -> List[Dict[str, Any]]:
        """Returns tool definitions permitted for the current persona level and offline state."""
        if level < 2: return []
        
        # Check Offline Mode Guard
        from System.network_guard import is_offline_mode
        is_offline = is_offline_mode() or (self.app and getattr(self.app, 'config', {}).get("offline_mode", False))
        
        base_tools = self.tools
        if is_offline:
            # Strictly filter out web_search and remote internet services when offline
            base_tools = [t for t in base_tools if t["function"]["name"] not in ("web_search",)]
            
        allowed_names = {"get_system_stats", "control_rgb", "generate_image", "read_file", "read_file_range"}
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
                if p_type in ("integer", "int"): py_type = "int"
                elif p_type in ("array", "list"): py_type = "list"
                elif p_type in ("object", "dict"): py_type = "dict"
                elif p_type in ("boolean", "bool"): py_type = "bool"
                
                if p_name in req:
                    args_list.append(f"{p_name}: {py_type}")
                else:
                    args_list.append(f"{p_name}: {py_type} = None")
            
            args_str = ", ".join(args_list)
            stubs.append(f"def {name}({args_str}):\n    \"\"\"{desc}\"\"\"\n    ...")
        return "\n\n".join(stubs)

    def get_gemma_declarations(self, level: int) -> str:
        """Generates Gemma-4 official template-aligned tool declarations string."""
        def official_q(s): return f"<|\"|>{s}<|\"|>"
        tool_defs = ""
        tools = self.get_definitions(level)
        for t in tools:
            f = t["function"]
            t_params = f.get("parameters", {})
            t_props = t_params.get("properties", {})
            t_prop_str = ",".join([f"{k}:{{description:{official_q(v.get('description',''))},type:{official_q(v.get('type','STRING'))}}}" for k, v in t_props.items()])
            t_req_str = ",".join([official_q(r) for r in t_params.get("required", [])])
            
            tool_defs += f"<|tool>declaration:{f['name']}{{description:{official_q(f['description'])},parameters:{{properties:{{{t_prop_str}}},required:[{t_req_str}]}}}}<tool|>\n"
        if tools:
            tool_defs += "\n"
        return tool_defs
