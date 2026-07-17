import os, re, time, sys
import tkinter as tk
from tkinter import scrolledtext

# Add project root to path to import serenity_resources
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from serenity_resources import THEME, DOCS_DIR

class LoreBookApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Serenity Prime - The Chronicles")
        self.root.geometry("700x900")
        self.root.config(bg=THEME["bg_color"])

        # Path to the hidden chronicle in the Docs folder
        self.chronicle_path = os.path.join(DOCS_DIR, ".prime_chronicles.txt")

        self.setup_ui()
        self.load_lore()

    def setup_ui(self):
        header = tk.Frame(self.root, bg=THEME["bg_color"], pady=20)
        header.pack(fill=tk.X)
        tk.Label(header, text="THE PRIME CHRONICLES", font=("Georgia", 22, "bold"), 
                 bg=THEME["bg_color"], fg=THEME["electric_blue"]).pack()
        
        # Main reading area
        self.display = scrolledtext.ScrolledText(self.root, font=("Georgia", 13), 
                                                bg=THEME["widget_bg_color"], fg=THEME["fg_color"], 
                                                relief=tk.FLAT, padx=30, pady=30, wrap=tk.WORD)
        self.display.pack(fill=tk.BOTH, expand=True, padx=25, pady=(0, 20))
        
        # Formatting tags
        self.display.tag_config("timestamp", foreground="#555555", font=("Consolas", 10))
        self.display.tag_config("lore", spacing1=10, spacing3=15)
        self.display.tag_config("divider", foreground=THEME["trim_color"])

        footer = tk.Frame(self.root, bg=THEME["bg_color"], pady=10)
        footer.pack(fill=tk.X)
        tk.Button(footer, text="Refresh Chronicles", command=self.load_lore, 
                  bg=THEME["trim_color"], fg=THEME["electric_blue"], relief=tk.FLAT,
                  padx=20, pady=5).pack()

    def load_lore(self):
        self.display.config(state='normal')
        self.display.delete('1.0', tk.END)
        
        if not os.path.exists(self.chronicle_path):
            self.display.insert(tk.END, "\n[ The Chronicles are currently empty. Serenity Prime has not yet shared her secrets. ]", "timestamp")
        else:
            try:
                with open(self.chronicle_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                
                # Newest entries at the top
                for line in reversed(lines):
                    if not line.strip(): continue
                    
                    # Regex to separate [Timestamp] from the message
                    match = re.match(r'\[(.*?)\] (.*)', line)
                    if match:
                        ts, content = match.groups()
                        self.display.insert(tk.END, f"Recorded on {ts}\n", "timestamp")
                        self.display.insert(tk.END, f"{content}\n", "lore")
                        self.display.insert(tk.END, f"{'━' * 50}\n", "divider")
            except Exception as e:
                self.display.insert(tk.END, f"Error opening Chronicles: {e}")
        
        self.display.config(state='disabled')
        self.display.see('1.0')

if __name__ == "__main__":
    root = tk.Tk()
    app = LoreBookApp(root)
    root.mainloop()