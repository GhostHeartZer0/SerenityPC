import tkinter as tk
from serenity_resources import THEME

def add_synthesis_toggle(parent, config_dict):
    """
    Adds the 'Enable Master Overview in Tactical Mode' checkbox to the settings UI.
    Expects a dictionary-like object (config_dict) to store/retrieve the boolean value.
    """
    var = tk.BooleanVar(value=config_dict.get("synthesis_in_tactical_mode", False))
    
    def on_toggle():
        config_dict["synthesis_in_tactical_mode"] = var.get()
        print(f"[SETTINGS] Master Overview in Tactical: {'ENABLED' if var.get() else 'DISABLED'}")

    cb = tk.Checkbutton(
        parent, 
        text="Enable Master Overview in Tactical Mode", 
        variable=var,
        command=on_toggle,
        bg=THEME["bg_color"], 
        fg=THEME["electric_blue"],
        selectcolor=THEME["widget_bg_color"],
        activebackground=THEME["bg_color"],
        activeforeground=THEME["electric_blue"],
        font=("Open Sans", 10)
    )
    cb.pack(anchor="w", padx=20, pady=5)
    
    return var # Return to keep reference if needed
