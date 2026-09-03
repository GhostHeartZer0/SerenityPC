"""System/hud_viewer.py
Standalone visual prompt & diagram HUD viewer for SerenityPC.
Eliminates runtime Python script generation/execution.
"""

import sys
import json
import tkinter as tk
from tkinter import scrolledtext

def display_hud(req_type: str, content: str):
    root = tk.Tk()
    root.title("Serenity Visual HUD")
    root.attributes('-topmost', True)
    root.attributes('-alpha', 0.96)
    root.geometry("560x420+120+120")
    root.config(bg='#0d0d12')

    hdr = tk.Frame(root, bg='#161622', pady=6)
    hdr.pack(fill=tk.X)
    tk.Label(hdr, text='✨ Serenity Visual & Diagram Viewer', fg='#00ffcc', bg='#161622', font=('Segoe UI', 10, 'bold')).pack(side=tk.LEFT, padx=10)
    tk.Label(hdr, text=f'[{req_type.upper()}]', fg='#8888aa', bg='#161622', font=('Consolas', 9)).pack(side=tk.RIGHT, padx=10)

    txt = scrolledtext.ScrolledText(root, fg='#e0e0ff', bg='#12121a', font=('Consolas', 10), insertbackground='white', borderwidth=0, wrap=tk.WORD)
    txt.insert(tk.END, content)
    txt.config(state=tk.DISABLED)
    txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    btn_f = tk.Frame(root, bg='#0d0d12', pady=6)
    btn_f.pack(fill=tk.X, padx=12)
    tk.Button(btn_f, text='Close HUD', command=root.destroy, bg='#252535', fg='#00ffcc', font=('Segoe UI', 9, 'bold'), relief=tk.FLAT, padx=12).pack(side=tk.RIGHT)
    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) > 2:
        req_t = sys.argv[1]
        payload = sys.argv[2]
    elif len(sys.argv) == 2:
        try:
            data = json.loads(sys.argv[1])
            req_t = data.get("type", "VISUAL")
            payload = data.get("prompt", "")
        except Exception:
            req_t = "VISUAL"
            payload = sys.argv[1]
    else:
        req_t = "VISUAL"
        payload = "No content provided."

    display_hud(req_t, payload)
