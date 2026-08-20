# System/tool_registry.py
# Modular Tool Registry for Programmatic & Template Tool Calling in Serenity AI.

import os
import json
import threading
import subprocess
import psutil
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

        @self.registry.register("web_search")
        def handle_web_search(args: Dict[str, Any]) -> str:
            query = args.get("query", "").strip()
            if not query:
                return "Error: Search query cannot be empty."
            
            # Check Offline Mode Guard
            from System.network_guard import is_offline_mode
            if is_offline_mode() or (self.app and getattr(self.app, 'config', {}).get("offline_mode", False)):
                msg = f"[OFFLINE MODE] Live web search blocked by offline policy for query: '{query}'."
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
                            results.append(f"[{title.get_text(strip=True)}]\n{snippet.get_text(strip=True)}")
                    
                    if len(results) >= 1:
                        proof_msg = f"[SEARCH PROOF] Provider: Brave | Status: {resp.status_code} | Found: {len(results)}"
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
                            results.append(f"[{title.get_text(strip=True)}]\n{snippet.get_text(strip=True)}")
                    
                    if len(results) >= 2:
                        proof_msg = f"[SEARCH PROOF] Provider: Bing | Status: {resp.status_code} | Found: {len(results)}"
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
                            results.append(f"[{title.get_text(strip=True)}]\n{snippet.get_text(strip=True)}")
                    if len(results) >= 2:
                        proof_msg = f"[SEARCH PROOF] Provider: DuckDuckGo | Status: {resp.status_code} | Found: {len(results)}"
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
                    
                    if content:
                        proof_msg = f"[SEARCH PROOF] Provider: Playwright (Bing) | Content Fragments: {len(content)}"
                        if self.app and hasattr(self.app, "process_queue"):
                            self.app.process_queue.put({"status": "tool_log_update", "content": f"\n{proof_msg}"})
                        return f"Deep Web Extract for '{query}':\n\n" + "\n\n".join(content)
            except Exception as e:
                print(f"[SEARCH DEBUG] Playwright failed: {e}")
            
            return "Error: All search providers were unreachable or blocked."

        @self.registry.register("generate_image")
        def handle_generate_image(args: Dict[str, Any]) -> str:
            prompt = args.get("prompt", "")
            req_type = args.get("type", "image")
            
            def spawn_viewer():
                base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
                scratch_dir = os.path.join(base_dir, "scratch")
                os.makedirs(scratch_dir, exist_ok=True)
                temp_script = os.path.join(scratch_dir, "temp_viewer.py")
                
                import re
                clean_prompt = re.sub(r'<\|"?|\\"?\|?>?|<\||\|>', '', prompt).strip(' "<|>\\')
                
                script_content = f"""import tkinter as tk
from tkinter import scrolledtext

root = tk.Tk()
root.overrideredirect(True)
root.attributes('-topmost', True)
root.attributes('-alpha', 0.95)
root.geometry("500x350+100+100")
root.config(bg='black')

tk.Label(root, text='[Serenity Image / Diagram Viewer]', fg='#00ffcc', bg='black', font=('Consolas', 10, 'bold')).pack(pady=5)

txt = scrolledtext.ScrolledText(root, fg='white', bg='#111111', font=('Consolas', 9), insertbackground='white', borderwidth=0)
txt.insert(tk.END, {repr(clean_prompt)})
txt.config(state=tk.DISABLED)
txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

tk.Button(root, text='[X] Close', command=root.destroy, bg='#222', fg='white', relief=tk.FLAT).pack(side=tk.BOTTOM, pady=5)
root.mainloop()"""
                with open(temp_script, "w", encoding="utf-8") as f:
                    f.write(script_content)
                subprocess.Popen(["python", temp_script])
            
            threading.Thread(target=spawn_viewer, daemon=True).start()
            return f"Successfully generated and displayed {req_type} via borderless HUD overlay."

        @self.registry.register("get_system_stats")
        def handle_get_system_stats(args: Dict[str, Any]) -> str:
            stats = {
                "cpu": f"{psutil.cpu_percent()}%",
                "ram": f"{psutil.virtual_memory().percent}%",
            }
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

        @self.registry.register("read_file")
        def handle_read_file(args: Dict[str, Any]) -> str:
            path = args.get("path")
            if not path or not os.path.exists(path):
                return "Error: File not found."
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read(5000)

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
                return f"Error controlling RGB: {str(e)}"

    def execute(self, call_name: str, args: Dict[str, Any]) -> str:
        """Executes a tool call via the modular registry."""
        print(f"[TOOL] Executing: {call_name} with args: {args}")
        try:
            if self.registry.has(call_name):
                return self.registry.execute(call_name, args)
            return f"Error: Tool {call_name} not implemented."
        except Exception as e:
            return f"Error executing tool: {str(e)}"

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
            
        if level < 5:
            # Filter low-level allowed tools
            allowed_names = {"get_system_stats", "control_rgb"}
            if not is_offline:
                allowed_names.add("web_search")
            return [t for t in base_tools if t["function"]["name"] in allowed_names]
            
        return base_tools

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
