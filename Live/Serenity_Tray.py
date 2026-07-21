import pystray, os, sys, subprocess, socket
from datetime import datetime
from PIL import Image

base_dir = os.path.dirname(os.path.abspath(__file__))
log_file = os.path.join(base_dir, "Logs", "tray_debug.txt")
os.makedirs(os.path.dirname(log_file), exist_ok=True)

# Single Instance Logic
# We move the HUD/Engine trigger ABOVE the Tray lock so that clicking the .bat
# while the tray is already running still attempts to bring up the HUD.
def quick_launch():
    # Attempt to spawn HUD (UI has its own singleton lock, so it's safe)
    ui_script = os.path.join(base_dir, "serenity_live.py")
    if os.path.exists(ui_script):
        pyw = sys.executable
        if "python.exe" in pyw.lower() and "pythonw.exe" not in pyw.lower():
            pyw = pyw.lower().replace("python.exe", "pythonw.exe")
        subprocess.Popen([pyw, ui_script], cwd=base_dir)

quick_launch()

_lock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
try:
    _lock.bind(("127.0.0.1", 47200))
except socket.error:
    sys.exit(0)  # Tray manager already running, exit silently after poking HUD

# Robustly find the non-windowed python.exe (even if named python3.11.exe etc)
python_exe = sys.executable
if "pythonw" in python_exe.lower():
    python_exe = python_exe.lower().replace("pythonw", "python")

def log_sys_external(text):
    """Helper for tray-side logging to the shared SysLog.txt."""
    try:
        log_path = os.path.join(base_dir, "Logs", "SysLog.txt")
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [TRAY] {text}\n")
    except: pass

def launch_engine(icon, item=None):
    # 1. ALWAYS attempt to launch UI script (it has its own internal singleton lock)
    ui_script = os.path.join(base_dir, "serenity_live.py")
    if os.path.exists(ui_script):
        pyw = sys.executable
        if "python.exe" in pyw.lower() and "pythonw.exe" not in pyw.lower():
            pyw = pyw.lower().replace("python.exe", "pythonw.exe")
        subprocess.Popen([pyw, ui_script], cwd=base_dir)

    # 2. Check if engine is already serving before trying to spawn a new one
    try:
        import requests
        r = requests.get("http://127.0.0.1:8001/diagnose", headers={"x-api-key": "REVOKED"}, timeout=1)
        if r.status_code == 200:
            log_sys_external("Engine already online, skipping backend spawn.")
            return
    except Exception:
        pass

    engine_script = os.path.join(base_dir, "Engine", "t5_server.py")
    if not os.path.exists(engine_script): 
        log_sys_external("Engine script missing, cannot launch.")
        return

    # Load core from params.json
    active_core = "med"
    params_path = os.path.join(base_dir, "System", "params.json")
    if os.path.exists(params_path):
        try:
            import json
            with open(params_path, "r") as f:
                active_core = json.load(f).get("active_core", "med")
        except: pass

    log_sys_external(f"Initiating Engine startup (Core: {active_core})")
    env = os.environ.copy()
    env["SERENITY_CORE"] = active_core
    env["SERENITY_SPAWNED_BY_UI"] = "0" # Meaning tray spawned
    
    # Use the non-windowed python_exe defined at module level
    subprocess.Popen([python_exe, engine_script], cwd=base_dir, env=env, creationflags=0x08000000)

def exit_action(icon, item):
    icon.stop()
    sys.exit(0)

icon_path = os.path.join(base_dir, "System", "transcendent_serenity_ws_hq.ico")
image = Image.open(icon_path) if os.path.exists(icon_path) else Image.new('RGB', (64, 64), color='blue')
menu = pystray.Menu(
    pystray.MenuItem("Start Serenity", launch_engine, default=True), 
    pystray.MenuItem("Exit", exit_action)
)
icon = pystray.Icon("Serenity", image, "Serenity AI", menu)

def on_ready(icon):
    """Called by pystray once the tray icon is fully initialized."""
    icon.visible = True
    launch_engine(icon, None)

icon.run(setup=on_ready)
