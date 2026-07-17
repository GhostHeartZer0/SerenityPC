import tkinter as tk
from tkinter import ttk, colorchooser
import json
import os

# Configuration Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # s:/SerenityPC
STATE_PATH = os.path.join(BASE_DIR, "System", "rgb_state.json")
OVERRIDE_PATH = os.path.join(BASE_DIR, "System", "rgb_overrides.json")

THEME = {
    "bg": "#0f0f0f",
    "trim": "#1a1a1a",
    "electric_blue": "#00ffcc",
    "fg": "#ffffff",
    "button_bg": "#2d2d2d",
    "accent": "#ff3366"
}

class RGBPanel(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Serenity RGB Control")
        self.geometry("400x650")
        self.configure(bg=THEME["bg"])
        self.resizable(False, False)
        self.attributes("-topmost", True)
        
        self.state = self.load_state()
        self.setup_ui()

    def load_state(self):
        default = {"mode": "auto", "manual_color": [255, 0, 0], "manual_color2": [0, 0, 255], 
                   "manual_style": "Steady", "speed": 50, "brightness": 100}
        if not os.path.exists(STATE_PATH): return default
        try:
            with open(STATE_PATH, 'r') as f:
                data = json.load(f)
                # Migration: ensure color2 and speed exist
                for k, v in default.items():
                    if k not in data: data[k] = v
                return data
        except: return default

    def save_state(self):
        try:
            with open(STATE_PATH, 'w') as f:
                json.dump(self.state, f, indent=4)
        except Exception as e: print(f"Error saving RGB state: {e}")

    def setup_ui(self):
        # Header
        header = tk.Label(self, text="MYSTIC LIGHT ENGINE", font=("Consolas", 14, "bold"), 
                         bg=THEME["bg"], fg=THEME["electric_blue"], pady=10)
        header.pack(fill=tk.X)

        # Tabbed Control (Optional, but let's keep it single page for now)
        main_container = tk.Frame(self, bg=THEME["bg"])
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=5)

        # Mode Selection
        mode_frame = tk.Frame(main_container, bg=THEME["trim"], pady=10)
        mode_frame.pack(fill=tk.X, pady=5)
        
        tk.Label(mode_frame, text="OPERATING MODE", font=("Consolas", 10), 
                 bg=THEME["trim"], fg="#888888").pack()
        
        self.mode_var = tk.StringVar(value=self.state.get("mode", "auto"))
        modes = [("AUTO (Thermal/Persona)", "auto"), ("MANUAL OVERRIDE", "manual")]
        for text, val in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, value=val, 
                          command=self.update_mode, bg=THEME["trim"], fg=THEME["fg"], 
                          selectcolor=THEME["bg"], activebackground=THEME["trim"]).pack(pady=2)

        # Customization Section
        tk.Label(main_container, text="CUSTOMIZATION", font=("Consolas", 10), 
                 bg=THEME["bg"], fg="#888888").pack(pady=(15, 5))

        # Color 1
        c1_frame = tk.Frame(main_container, bg=THEME["bg"])
        c1_frame.pack(fill=tk.X, pady=2)
        self.color1_preview = tk.Frame(c1_frame, height=25, width=40, bg="red")
        self.color1_preview.pack(side=tk.LEFT, padx=5)
        tk.Button(c1_frame, text="PICK COLOR 1", command=lambda: self.pick_color(1),
                  bg=THEME["button_bg"], fg=THEME["fg"], relief=tk.FLAT, font=("Consolas", 9)).pack(fill=tk.X)

        # Color 2
        c2_frame = tk.Frame(main_container, bg=THEME["bg"])
        c2_frame.pack(fill=tk.X, pady=2)
        self.color2_preview = tk.Frame(c2_frame, height=25, width=40, bg="blue")
        self.color2_preview.pack(side=tk.LEFT, padx=5)
        tk.Button(c2_frame, text="PICK COLOR 2 (FADE/SPIN)", command=lambda: self.pick_color(2),
                  bg=THEME["button_bg"], fg=THEME["fg"], relief=tk.FLAT, font=("Consolas", 9)).pack(fill=tk.X)
        
        self.update_previews()

        # Effect Style
        tk.Label(main_container, text="EFFECT ANIMATION", font=("Consolas", 10), 
                 bg=THEME["bg"], fg="#888888").pack(pady=(15, 5))
        
        styles = ["Steady", "Breathing", "Rainbow", "Spin", "Spiral", "ColorFade", "Music"]
        self.style_var = tk.StringVar(value=self.state.get("manual_style", "Steady"))
        style_menu = ttk.Combobox(main_container, textvariable=self.style_var, values=styles, state="readonly")
        style_menu.pack(fill=tk.X, pady=5)
        style_menu.bind("<<ComboboxSelected>>", self.update_style)

        # Speed Slider
        tk.Label(main_container, text="EFFECT SPEED", font=("Consolas", 10), 
                 bg=THEME["bg"], fg="#888888").pack(pady=(15, 5))
        self.speed_slider = tk.Scale(main_container, from_=1, to=100, orient=tk.HORIZONTAL,
                                    bg=THEME["bg"], fg=THEME["fg"], troughcolor=THEME["trim"],
                                    highlightthickness=0, command=self.update_speed)
        self.speed_slider.set(self.state.get("speed", 50))
        self.speed_slider.pack(fill=tk.X)

        # Persona Override Section
        tk.Label(main_container, text="PERSONA COLOR OVERRIDE", font=("Consolas", 10), 
                 bg=THEME["bg"], fg="#888888").pack(pady=(15, 5))
        tk.Button(main_container, text="SET CURRENT PERSONA COLOR", 
                  command=self.set_persona_override, bg=THEME["accent"], fg=THEME["fg"], 
                  relief=tk.FLAT, font=("Consolas", 10, "bold")).pack(fill=tk.X, pady=10)

        # Footer
        footer = tk.Label(self, text="Real-time Synchronization Active", font=("Consolas", 8), 
                         bg=THEME["bg"], fg="#444444", pady=10)
        footer.pack(side=tk.BOTTOM)

    def update_previews(self):
        c1 = self.state.get("manual_color", [255, 0, 0])
        self.color1_preview.configure(bg=f'#{c1[0]:02x}{c1[1]:02x}{c1[2]:02x}')
        c2 = self.state.get("manual_color2", [0, 0, 255])
        self.color2_preview.configure(bg=f'#{c2[0]:02x}{c2[1]:02x}{c2[2]:02x}')

    def pick_color(self, num):
        key = "manual_color" if num == 1 else "manual_color2"
        curr = self.state.get(key, [255, 255, 255])
        hex_color = f'#{curr[0]:02x}{curr[1]:02x}{curr[2]:02x}'
        color = colorchooser.askcolor(initialcolor=hex_color, title=f"Pick Color {num}")
        if color[0]:
            self.state[key] = [int(c) for c in color[0]]
            self.state["mode"] = "manual"
            self.mode_var.set("manual")
            self.update_previews()
            self.save_state()

    def update_mode(self):
        self.state["mode"] = self.mode_var.get()
        self.save_state()

    def update_style(self, event=None):
        self.state["manual_style"] = self.style_var.get()
        self.state["mode"] = "manual"
        self.mode_var.set("manual")
        self.save_state()

    def update_speed(self, val):
        self.state["speed"] = int(val)
        self.save_state()

    def set_persona_override(self):
        # Read current level from config
        try:
            with open(os.path.join(BASE_DIR, "System\config.json"), 'r') as f:
                config = json.load(f)
            level = config.get("active_persona_level", 1)
            
            # Use current Manual Color 1 as the override
            c1 = self.state.get("manual_color", [255, 0, 0])
            
            overrides = {}
            if os.path.exists(OVERRIDE_PATH):
                with open(OVERRIDE_PATH, 'r') as f: overrides = json.load(f)
            
            overrides[str(level)] = c1
            with open(OVERRIDE_PATH, 'w') as f: json.dump(overrides, f, indent=4)
            
            tk.messagebox.showinfo("Success", f"Persona Level {level} color set to current Color 1!")
        except Exception as e:
            tk.messagebox.showerror("Error", f"Failed to set override: {e}")

if __name__ == "__main__":
    import tkinter.messagebox
    root = tk.Tk()
    root.withdraw()
    panel = RGBPanel(root)
    root.mainloop()
