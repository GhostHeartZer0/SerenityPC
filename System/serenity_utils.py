# serenity_utils.py
# Helper classes for logging, UI components, and error handling.

import tkinter as tk
from tkinter import messagebox, ttk
import sys
import os
import subprocess
import time
import traceback
import threading
import faulthandler
import psutil
try:
    import pynvml as nvidia_ml
except ImportError:
    nvidia_ml = None
from PIL import Image, ImageTk
from serenity_resources import ANIMATION_SEQUENCE, MEDIA_DIR, THEME

def awaken_live_agent(agent_path):
    """Launches the Live Agent as an independent background process and returns the handle."""
    if not os.path.exists(agent_path):
        print(f"Error: Could not find Live Agent at {agent_path}")
        return None
    
    try:
        # Using subprocess.Popen keeps it running in the background.
        if sys.platform == "win32":
            CREATE_NO_WINDOW = 0x08000000
            proc = subprocess.Popen([sys.executable, agent_path], creationflags=CREATE_NO_WINDOW)
        else:
            proc = subprocess.Popen([sys.executable, agent_path])
        return proc
    except Exception as e:
        print(f"Failed to awaken Live Agent: {e}")
        return None

SPAM_PATTERNS = [
    "is not marked as EOG",
    "create_tensor: loading tensor",
    "llama_model_loader: dumping metadata",
    "llm_load_print_meta:",
    "llm_load_tensors:",
    "ggml_cuda_init: found",
    "ggml_backend_cuda_buffer_type_alloc_buffer",
    "load_tensors: loading",
    "loaded:",
    "Llama.generate:",
    "chat template"
]

class WidgetLogger:
    """A class to redirect stdout/stderr to a tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
        self.original_stream = sys.__stdout__ if tag == "stdout" else sys.__stderr__

    def write(self, text):
        # Filtering spam patterns
        if any(p in text for p in SPAM_PATTERNS):
            return

        if not tk._default_root or not self.widget or not hasattr(self.widget, 'winfo_exists') or not self.widget.winfo_exists():
            try: self.original_stream.write(text)
            except: pass
            return
            
        try: self.widget.after_idle(self._write_to_widget, text)
        except tk.TclError:
            try: self.original_stream.write(text)
            except: pass

    def _write_to_widget(self, text):
        try:
            if hasattr(self.widget, 'winfo_exists') and self.widget.winfo_exists():
                is_scrolled_up = self.widget.yview()[1] < 0.99
                
                self.widget.config(state='normal')
                self.widget.insert(tk.END, text, (self.tag,))
                
                # Trim widget memory overhead (keep max ~3000 lines)
                if int(float(self.widget.index('end-1c'))) > 3000:
                    self.widget.delete("1.0", "500.0")
                
                if not is_scrolled_up:
                    self.widget.see(tk.END)
                    
                self.widget.config(state='disabled')
        except tk.TclError: pass

    def flush(self):
        try: self.original_stream.flush()
        except: pass

class FileAndWidgetLogger:
    """Redirects stdout/stderr to both a widget and a file."""
    def __init__(self, widget, log_file, tag="stderr"):
        self.widget_logger = WidgetLogger(widget, tag)
        self.log_file = log_file
        self.original_stream = sys.__stderr__
        try:
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n--- Log session started at {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
        except Exception as e:
            self.original_stream.write(f"FATAL: Failed to initialize log file {self.log_file}: {e}\n")

    def write(self, text):
        # Filtering spam patterns
        if any(p in text for p in SPAM_PATTERNS):
            return

        self.widget_logger.write(text)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f: f.write(text)
        except Exception as e:
            try:
                self.original_stream.write(f"FATAL: Failed to write to log file: {e}\n")
                self.original_stream.write(text + "\n")
            except: pass

    def flush(self):
        self.widget_logger.flush()


class LoadingScreen:
    """A visual loading splash screen that runs before the engine initializes."""
    def __init__(self, root):
        self.root = tk.Toplevel(root)
        self.root.overrideredirect(True)
        self.animation_frames = []
        self.current_frame_index = 0
        self.animation_id = None
        self.width, self.height = 350, 350

        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x = (screen_width // 2) - (self.width // 2)
        y = (screen_height // 2) - (self.height // 2)
        self.root.geometry(f'{self.width}x{self.height}+{x}+{y}')

        self.canvas = tk.Canvas(self.root, width=self.width, height=self.height, bg="#000000", highlightthickness=0)
        self.canvas.pack()
        self.load_animation_images()
        self.canvas.create_text(self.width / 2, self.height - 40, text="Serenity is Awakening...", font=("Open Sans", 11, "bold"), fill="#FFFFFF")
        self.canvas.create_text(self.width / 2, self.height - 20, text="Loading... please wait. This'll only take a minute or two.", font=("Open Sans", 9, "italic"), fill="#00FFCC")
        self.root.lift()
        self.root.attributes("-topmost", True)

    def load_animation_images(self):
        try:
            image_folder = MEDIA_DIR
            if not os.path.isdir(image_folder):
                print(f"Loading Screen: Media folder not found at {image_folder}", file=sys.stderr)
                return

            for state in ANIMATION_SEQUENCE:
                filename = f"{state}.png"
                if state == "serene_serenity" or state == "idle_nemo": filename = "Serene_Serenity.jpg"
                img_path = os.path.join(image_folder, filename)
                if not os.path.exists(img_path): continue
                with Image.open(img_path) as img:
                    img.thumbnail((self.width, self.height), Image.Resampling.LANCZOS)
                    self.animation_frames.append(ImageTk.PhotoImage(img))
        except Exception as e:
            print(f"Animation load error: {e}", file=sys.stderr)
            
    def start_animation(self):
        if self.animation_frames:
            self.update_animation()

    def update_animation(self):
        if not self.animation_frames or not self.root.winfo_exists(): return
        frame = self.animation_frames[self.current_frame_index]
        self.canvas.delete("anim")
        self.canvas.create_image(self.width/2, self.height/2 - 20, image=frame, tags="anim")
        self.current_frame_index = (self.current_frame_index + 1) % len(self.animation_frames)
        self.animation_id = self.root.after(1000, self.update_animation)

    def stop_and_destroy(self):
        try: self.root.after_cancel(self.animation_id)
        except: pass
        self.animation_id = None
        if self.root.winfo_exists(): self.root.destroy()

class HardwareProfile:
    """Auto-detects CPU cores and RAM to optimize inference and background tasking."""
    
    @staticmethod
    def initialize_gpu_acceleration():
        """Finds the latest CUDA toolkit installation and adds it to the DLL search path."""
        if sys.platform != "win32": return
        
        import glob
        cuda_path = None
        base_install = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
        
        if os.path.exists(base_install):
            versions = glob.glob(os.path.join(base_install, "v*")) 
            if versions:
                cuda_path = os.path.join(sorted(versions)[-1], "bin")
        
        if not cuda_path:
            cuda_path = os.environ.get('CUDA_PATH') or os.environ.get('CUDA_HOME')
            if cuda_path and not cuda_path.endswith("bin"):
                cuda_path = os.path.join(cuda_path, "bin")

        if cuda_path and os.path.isdir(cuda_path) and hasattr(os, 'add_dll_directory'):
            try:
                os.add_dll_directory(cuda_path)
                # Also add the lib/x64 path where cublas lives
                lib_path = os.path.join(os.path.dirname(cuda_path), "lib", "x64")
                if os.path.exists(lib_path) and os.path.isdir(lib_path):
                    os.add_dll_directory(lib_path)
                print(f"[HARDWARE] Apex CUDA Link Established: {cuda_path}")
            except Exception as e:
                print(f"[HARDWARE] CUDA DLL Link Failed: {e}")
    
    @staticmethod
    def get_cpu_info():
        """Returns physical and logical core counts."""
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
            logical = psutil.cpu_count(logical=True)
            return {"physical": physical, "logical": logical}
        except:
            return {"physical": 4, "logical": 8} # Fallback

    @staticmethod
    def get_total_ram_gb():
        """Returns total system RAM in GB."""
        try:
            import psutil
            total_bytes = psutil.virtual_memory().total
            return total_bytes / (1024 ** 3)
        except:
            return 8.0 # Fallback

    @staticmethod
    def get_dynamic_core_mask():
        try:
            import psutil
            total = psutil.cpu_count(logical=True)
            if total is None: return None
            if total < 4: return list(range(total))
            if total == 4: return list(range(3)) # Reserve 1
            return list(range(total - 2)) # Reserve 2 for 5+
        except: return None

    @staticmethod
    def get_dynamic_e_core_mask():
        try:
            import psutil
            total = psutil.cpu_count(logical=True)
            if total is None: return None
            if total < 4: return list(range(total))
            if total == 4: return [3] # Focus background on the 1 reserved core
            return [total - 2, total - 1] # Focus background on the 2 reserved cores
        except: return None

    @staticmethod
    def pin_to_p_cores():
        """
        Dynamically assign heavy Inference/Vision tasks to P-core threads/main cores.
        """
        try:
            import psutil
            p = psutil.Process()
            mask = HardwareProfile.get_dynamic_core_mask() or list(range(psutil.cpu_count(logical=True)))
            p.cpu_affinity(mask)
            print("[HARDWARE] Pinning Active (Main Cores)")
        except Exception as e:
            print(f"[HARDWARE] Pinning failed: {e}")

    @staticmethod
    def pin_to_e_cores():
        """Assign background 'Crusher' tasks to reserved/E-core threads."""
        try:
            import psutil
            p = psutil.Process()
            mask = HardwareProfile.get_dynamic_e_core_mask() or list(range(psutil.cpu_count(logical=True)))
            p.cpu_affinity(mask)
            print("[HARDWARE] Background/E-Core Pinning Active")
        except Exception as e:
            print(f"[HARDWARE] background pinning failed: {e}")

    @staticmethod
    def release_cores():
        """Resets affinity to all available cores."""
        try:
            import psutil
            p = psutil.Process()
            count = psutil.cpu_count()
            if count:
                p.cpu_affinity(list(range(count)))
        except: pass

    @staticmethod
    def set_priority(level="normal"):
        try:
            import psutil
            import os
            p = psutil.Process(os.getpid())
            if level == "high":
                p.nice(psutil.HIGH_PRIORITY_CLASS if os.name == 'nt' else -10)
            elif level == "above_normal":
                p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS if os.name == 'nt' else -5)
                print("[APEX] Priority elevated to ABOVE_NORMAL.")
            else:
                p.nice(psutil.NORMAL_PRIORITY_CLASS if os.name == 'nt' else 0)
        except Exception as e:
            print(f"[HARDWARE] Priority set failed: {e}")


class MediaProcessor:
    """Consolidated engine for auto-watching and batch processing media."""
    def __init__(self, chatbot_app):
        self.app = chatbot_app
        self.stop_event = chatbot_app.stop_process
        self._watcher_thread = None

    def start_watcher(self, folder_path):
        if self._watcher_thread and self._watcher_thread.is_alive():
            return
        self._watcher_thread = threading.Thread(target=self._watcher_loop, args=(folder_path,), daemon=True)
        self._watcher_thread.start()
        print(f"[MEDIA] Watcher started on: {folder_path}")

    def _watcher_loop(self, folder_path):
        """Background file watching pinned to E-cores."""
        HardwareProfile.pin_to_e_cores()
        HardwareProfile.set_priority("normal")
        known_files = set(os.listdir(folder_path))
        while not self.stop_event.is_set():
            try:
                current_files = set(os.listdir(folder_path))
                new_files = current_files - known_files
                for f in new_files:
                    if f.endswith((".mp4", ".mkv", ".avi", ".mov")):
                        time.sleep(2) # Finish writing
                        full_path = os.path.join(folder_path, f)
                        self.app.root.after(0, lambda p=full_path: self.app.initiate_vision_analysis("video", p, "Analyze recent gameplay."))
                known_files = current_files
                time.sleep(5)
            except: time.sleep(10)

    @staticmethod
    def get_adaptive_chunk_size(current_size, success):
        """Reduces chunk size by 20% on failure."""
        if not success:
            new_size = int(current_size * 0.8)
            return max(new_size, 1)
        return current_size

class SystemMonitor:
    """Handles GPU/CPU stats monitoring in a separate thread."""
    def __init__(self, chatbot_app):
        self.app = chatbot_app
        self.gpu_handle = None
        self.stop_event = chatbot_app.stop_process
        self.process_queue = chatbot_app.process_queue

    def start(self):
        if nvidia_ml:
            try:
                nvidia_ml.nvmlInit()
                self.gpu_handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
            except:
                self.gpu_handle = None
            
        threading.Thread(target=self._stats_loop, daemon=True).start()

    @staticmethod
    def _get_cpu_temp():
        try:
            if hasattr(psutil, "sensors_temperatures"):
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        if entries and entries[0].current > 0:
                            return f"{entries[0].current:.0f}°C"
        except: pass
        try:
            import wmi
            w = wmi.WMI(namespace="root\\wmi")
            temp_info = w.MSAcpi_ThermalZoneTemperature()
            if temp_info:
                celsius = (temp_info[0].CurrentTemperature / 10.0) - 273.15
                if 0 < celsius < 120:
                    return f"{celsius:.0f}°C"
        except: pass
        try:
            import wmi
            for ns in ["root\\LibreHardwareMonitor", "root\\OpenHardwareMonitor"]:
                try:
                    w = wmi.WMI(namespace=ns)
                    sensors = w.Sensor()
                    for s in sensors:
                        if s.SensorType == 'Temperature' and 'cpu' in s.Name.lower():
                            return f"{s.Value:.0f}°C"
                except: pass
        except: pass
        return "N/A"

    @staticmethod
    def _get_cpu_power():
        try:
            import wmi
            for ns in ["root\\LibreHardwareMonitor", "root\\OpenHardwareMonitor"]:
                try:
                    w = wmi.WMI(namespace=ns)
                    sensors = w.Sensor()
                    for s in sensors:
                        if s.SensorType == 'Power' and 'cpu' in s.Name.lower():
                            return f"{s.Value:.1f}W"
                except: pass
        except: pass
        return "N/A"

    @staticmethod
    def _get_shared_vram_used_bytes():
        try:
            import wmi
            w = wmi.WMI(namespace="root\\cimv2")
            perf = w.Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory()
            if perf:
                total_shared = sum(int(getattr(p, 'SharedUsage', 0)) for p in perf)
                if total_shared > 0:
                    return total_shared
        except: pass
        try:
            import win32com.client
            wmi_obj = win32com.client.GetObject("winmgmts:\\\\.\\root\\cimv2")
            perf = wmi_obj.ExecQuery("SELECT SharedUsage FROM Win32_PerfFormattedData_GPUPerformanceCounters_GPUAdapterMemory")
            if perf:
                total_shared = sum(int(getattr(p, 'SharedUsage', 0)) for p in perf)
                if total_shared > 0:
                    return total_shared
        except: pass
        return 0


    def _stats_loop(self):
        while not self.stop_event.is_set():
            try:
                stats = {}
                
                # System Stats (CPU, RAM, Temp, Power)
                p = psutil.Process()
                with p.oneshot():
                    vm = psutil.virtual_memory()
                    stats["CPU"] = f"{psutil.cpu_percent():.1f}%"
                    stats["RAM"] = f"{vm.used / (1024**2):.0f} / {vm.total / (1024**2):.0f} MB"
                
                stats["CPU Temp"] = SystemMonitor._get_cpu_temp()
                stats["CPU Power"] = SystemMonitor._get_cpu_power()
                
                # GPU Stats (NVML & Shared VRAM)
                if nvidia_ml and self.gpu_handle:
                    try:
                        util = nvidia_ml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
                        mem = nvidia_ml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
                        temp = nvidia_ml.nvmlDeviceGetTemperature(self.gpu_handle, nvidia_ml.NVML_TEMPERATURE_GPU)
                        
                        ded_used_gb = mem.used / (1024**3)
                        ded_total_gb = mem.total / (1024**3)
                        
                        shared_total_gb = vm.total / (2 * 1024**3)
                        shared_used_bytes = SystemMonitor._get_shared_vram_used_bytes()
                        shared_used_gb = shared_used_bytes / (1024**3)
                        
                        tot_used_gb = ded_used_gb + shared_used_gb
                        tot_total_gb = ded_total_gb + shared_total_gb
                        
                        stats["GPU Use"] = f"{util.gpu}%"
                        stats["VRAM"] = f"{mem.used / (1024**2):.0f} / {mem.total / (1024**2):.0f} MB"
                        stats["Shared VRAM"] = f"{shared_used_gb:.2f} / {shared_total_gb:.1f} GB"
                        stats["Total VRAM"] = f"{tot_used_gb:.2f} / {tot_total_gb:.1f} GB"
                        stats["GPU Temp"] = f"{temp}°C"
                        
                        try:
                            pwr_mw = nvidia_ml.nvmlDeviceGetPowerUsage(self.gpu_handle)
                            stats["Power"] = pwr_mw / 1000.0
                        except: pass
                    except: pass
                
                self.process_queue.put({"status": "stats_update", "stats": stats})
            except: pass
            time.sleep(2)

def log_uncaught_exception(exc_type, exc_value, exc_traceback):
    """Catches fatal app crashes and saves them to error_log.txt"""
    error_log_file = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0] if hasattr(sys, 'frozen') else __file__)), "Logs", "error_log.txt")
    try:
        os.makedirs(os.path.dirname(error_log_file), exist_ok=True)
        with open(error_log_file, "a", encoding='utf-8') as f:
            f.write("\n--- UNCAUGHT EXCEPTION ---\n")
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("".join(traceback.format_exception(exc_type, exc_value, exc_traceback)))
            f.write("--- END EXCEPTION ---\n")
    except: pass
    print("--- UNCAUGHT EXCEPTION ---", file=sys.__stderr__)
    traceback.print_exception(exc_type, exc_value, exc_traceback, file=sys.__stderr__)
    
    try:
        root = tk.Tk(); root.withdraw()
        messagebox.showerror("Fatal Error", f"A critical error occurred. Please check {os.path.basename(error_log_file)} for details.")
        root.destroy()
    except: pass
def enable_fault_debugging():
    """Enables faulthandler to capture segfaults and hard crashes."""
    try:
        # Use a dedicated file for faulthandler to avoid mixing with regular logs
        import os, faulthandler
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs")
        if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
        fault_log = open(os.path.join(log_dir, "fault_log.txt"), "a", encoding="utf-8")
        faulthandler.enable(file=fault_log, all_threads=True)
        print("[DEBUG] Faulthandler enabled. Hard crashes will be logged to Logs/fault_log.txt")
    except Exception as e:
        print(f"[DEBUG] Failed to enable faulthandler: {e}")


# --- GIL-free Thread-Safety Utilities (Python 3.13/3.14+) ---
class ThreadSafeDict(dict):
    """
    A thread-safe dictionary subclass wrapper designed for GIL-free Python (3.13/3.14+).
    Uses a reentrant lock to synchronize all read, write, and deletion operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def __getitem__(self, key):
        with self._lock:
            return super().__getitem__(key)

    def __setitem__(self, key, value):
        with self._lock:
            super().__setitem__(key, value)

    def __delitem__(self, key):
        with self._lock:
            super().__delitem__(key)

    def __contains__(self, key):
        with self._lock:
            return super().__contains__(key)

    def get(self, key, default=None):
        with self._lock:
            return super().get(key, default)

    def setdefault(self, key, default=None):
        with self._lock:
            return super().setdefault(key, default)

    def pop(self, key, default=None):
        with self._lock:
            return super().pop(key, default)

    def popitem(self):
        with self._lock:
            return super().popitem()

    def clear(self):
        with self._lock:
            super().clear()

    def update(self, *args, **kwargs):
        with self._lock:
            super().update(*args, **kwargs)

    def keys(self):
        with self._lock:
            return list(super().keys())

    def values(self):
        with self._lock:
            return list(super().values())

    def items(self):
        with self._lock:
            return list(super().items())

    def copy(self):
        with self._lock:
            return ThreadSafeDict(super().copy())

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __repr__(self):
        with self._lock:
            return super().__repr__()


class ThreadSafeList(list):
    """
    A thread-safe list subclass wrapper designed for GIL-free Python (3.13/3.14+).
    Uses a reentrant lock to synchronize all read, write, and iteration operations.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._lock = threading.RLock()

    def append(self, item):
        with self._lock:
            super().append(item)

    def extend(self, iterable):
        with self._lock:
            super().extend(iterable)

    def insert(self, index, item):
        with self._lock:
            super().insert(index, item)

    def remove(self, item):
        with self._lock:
            super().remove(item)

    def pop(self, index=-1):
        with self._lock:
            return super().pop(index)

    def clear(self):
        with self._lock:
            super().clear()

    def __getitem__(self, index):
        with self._lock:
            if isinstance(index, slice):
                return ThreadSafeList(super().__getitem__(index))
            return super().__getitem__(index)

    def __setitem__(self, index, value):
        with self._lock:
            super().__setitem__(index, value)

    def __delitem__(self, index):
        with self._lock:
            super().__delitem__(index)

    def __len__(self):
        with self._lock:
            return super().__len__()

    def __iter__(self):
        with self._lock:
            return iter(list(super().__iter__()))

    def __repr__(self):
        with self._lock:
            return super().__repr__()

    def copy(self):
        with self._lock:
            return ThreadSafeList(super().copy())


class ThinkingDisplay(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, bg=THEME["bg_color"], *args, **kwargs)
        self.label = tk.Label(self, text="Thinking...", font=("Open Sans", 10, "italic"), 
                            fg=THEME["electric_blue"], bg=THEME["bg_color"])
        self.label.pack(side=tk.LEFT, padx=5)
        self.progress = ttk.Progressbar(self, mode='indeterminate', length=150)
        self.progress.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
    
    def start(self):
        if not self.winfo_exists(): return
        self.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)
        self.progress.start(15)
        
    def stop(self):
        if not self.winfo_exists(): return
        self.progress.stop()
        self.pack_forget()
        self.label.config(text="Thinking...")

    def update_status(self, text):
        if not self.winfo_exists(): return
        self.label.config(text=text)
