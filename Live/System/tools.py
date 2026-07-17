import os
import time
import subprocess
import urllib.parse
import traceback
import psutil
import shutil
import threading
import json

# Vector Storage
HAS_VECTOR_DB = False
try:
    import chromadb
    from sentence_transformers import SentenceTransformer
    HAS_VECTOR_DB = True
except ImportError:
    pass

class MemoryEngine:
    def __init__(self, db_path=None):
        self.db_path = db_path or os.path.join(os.environ.get("USERPROFILE", ""), ".serenity_memory")
        if not os.path.exists(self.db_path):
            os.makedirs(self.db_path)
        
        if not HAS_VECTOR_DB:
            print("Memory Engine Init Failed: chromadb or sentence-transformers not installed.")
            return
            
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            self.collection = self.client.get_or_create_collection(name="serenity_memories")
        except Exception as e:
            print(f"Memory Engine Init Failed: {e}")

    def save(self, text, metadata=None):
        try:
            # Use 12 cores for embeddings is handled by sentence_transformers naturally on CPU if available
            import torch
            torch.set_num_threads(12)
            
            # Simple unique ID
            mem_id = f"mem_{int(time.time() * 1000)}"
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[mem_id]
            )
            return True
        except Exception as e:
            print(f"Memory Save Failed: {e}")
            return False

    def recall(self, query, n_results=3):
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            if results and results['documents']:
                return "\n".join(results['documents'][0])
            return ""
        except Exception as e:
            print(f"Memory Recall Failed: {e}")
            return ""

class SystemMonitor:
    def __init__(self):
        self.nvml_loaded = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_loaded = True
        except: pass

    def get_stats(self) -> str:
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        stats = f"System Load: CPU {cpu}%, RAM {ram}%."
        
        if self.nvml_loaded:
            try:
                import pynvml
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats += f" GPU {util.gpu}%, VRAM {mem.used/1024**3:.1f}/{mem.total/1024**3:.1f}GB."
            except: pass
        return stats

class BrowserTools:
    def __init__(self):
        # Default fallback
        self.user_data = "C:/Users/ccrg6/AppData/Local/Google/Chrome/User Data"
        self.profile = "GHZ"
        
        # VLC standard installation paths
        self.vlc_paths = [
            r"C:\Program Files\VideoLAN\VLC\vlc.exe",
            r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe"
        ]

    def _get_vlc_path(self):
        for path in self.vlc_paths:
            if os.path.exists(path):
                return path
        return None

    def _check_chrome_lock(self):
        """Returns a temporary path if Chrome is locked, otherwise return original."""
        lock_file = os.path.join(self.user_data, "SingletonLock")
        if os.path.exists(lock_file):
            # Chrome is open. Attempt to copy to a temp directory to bypass lock.
            temp_path = os.path.join(os.environ.get("TEMP", ""), "serenity_chrome_context")
            # We don't want to copy the whole 10GB profile, just the minimal shell
            # but for a simple fix, let's just warn or use a fresh context.
            return None # Force fresh context or inform
        return self.user_data

    def search_web(self, query: str) -> str:
        """Playwright-powered search using the persistent GHZ profile."""
        try:
            from playwright.sync_api import sync_playwright
            
            with sync_playwright() as p:
                lock_context_path = self._check_chrome_lock()
                
                # If locked, we launch a standard browser to avoid Target Closed
                if not lock_context_path:
                    browser = p.chromium.launch(headless=True, args=["--disable-gpu"])
                    page = browser.new_page()
                else:
                    context = p.chromium.launch_persistent_context(
                        self.user_data,
                        channel="chrome",
                        headless=True,
                        args=["--profile-directory=" + self.profile, "--disable-gpu"]
                    )
                    page = context.new_page()
                
                url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
                page.goto(url, timeout=30000)
                
                # Extract snippets
                snippets = page.locator('div.vv779b, div.VwiC3b').all_text_contents()
                
                if lock_context_path: context.close()
                else: browser.close()
                
                # Also visually open for the user to see (using system default)
                subprocess.Popen(["start", url], shell=True)
                
                if snippets:
                    return "Web Search Results:\n" + "\n".join(snippets[:3])
                return "I've opened the search results in your browser."
        except Exception as e:
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            subprocess.Popen(["start", url], shell=True)
            return f"Opened search in browser. (Playwright error: {e})"

    def play_media(self, query: str) -> str:
        """Play media via VLC or YouTube Music Playwright integration."""
        try:
            user_profile = os.environ.get("USERPROFILE", "")
            music_dirs = [
                os.path.join(user_profile, "Music"),
                os.path.join(user_profile, "Downloads"),
                os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "System", "Media")
            ]
            
            vlc_exe = self._get_vlc_path()
            target_file = None
            
            if vlc_exe:
                for m_dir in music_dirs:
                    if not os.path.exists(m_dir): continue
                    for root, dirs, files in os.walk(m_dir):
                        if root[len(m_dir):].count(os.sep) > 2: continue
                        for file in files:
                            if file.lower().endswith((".mp3", ".flac", ".wav", ".m4a", ".mp4")):
                                tokens = query.lower().split()
                                if all(t in file.lower() for t in tokens):
                                    target_file = os.path.join(root, file)
                                    break
                        if target_file: break
                    if target_file: break
            
            if target_file and vlc_exe:
                subprocess.Popen([vlc_exe, target_file])
                return f"Playing local file '{os.path.basename(target_file)}' via VLC."
                
            # YT Music fallback
            yt_url = f"https://music.youtube.com/search?q={urllib.parse.quote(query)}"
            subprocess.Popen(["start", yt_url], shell=True)
            return f"I couldn't find '{query}' locally, so I opened it in YouTube Music."
        except Exception as e:
            return f"Media playback failed: {e}"

