# serenity_utils.py
# Helper classes for logging, UI components, and error handling.

import tkinter as tk
from tkinter import messagebox
import sys
import os
import time
import traceback
from PIL import Image, ImageTk
from serenity_resources import ANIMATION_SEQUENCE

class WidgetLogger:
    """A class to redirect stdout/stderr to a tkinter Text widget."""
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag
        self.original_stream = sys.__stdout__ if tag == "stdout" else sys.__stderr__

    def write(self, text):
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
                # Check if the user has scrolled up in the logs
                is_scrolled_up = self.widget.yview()[1] < 0.99
                
                self.widget.config(state='normal')
                self.widget.insert(tk.END, text, (self.tag,))
                
                # Only snap to the bottom if they were already at the bottom
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
    def __init__(self, parent):
        self.root = tk.Toplevel(parent)
        self.root.title("Loading Serenity AI")
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
        self.canvas.create_text(self.width / 2, self.height - 30, text="Serenity is Awakening...", font=("Open Sans", 12, "italic"), fill="#FFFFFF")
        self.root.lift()
        self.root.attributes("-topmost", True)

    def load_animation_images(self):
        try:
            base_dir = os.path.dirname(os.path.abspath(sys.argv[0] if hasattr(sys, 'frozen') else __file__))
            image_folder = os.path.join(base_dir, 'Media')
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
            print(f"Error loading startup images: {e}", file=sys.stderr)

    def start_animation(self):
        if not self.animation_frames: return
        self.canvas_image = self.canvas.create_image(self.width / 2, self.height / 2, anchor="center", image=self.animation_frames[0])
        self._animate_next_frame()

    def _animate_next_frame(self):
        if not self.root.winfo_exists() or not self.animation_frames: return
        self.current_frame_index = (self.current_frame_index + 1) % len(self.animation_frames)
        self.canvas.itemconfig(self.canvas_image, image=self.animation_frames[self.current_frame_index])
        delay = 2000 if self.current_frame_index == len(self.animation_frames) - 1 else 300
        self.animation_id = self.root.after(delay, self._animate_next_frame)

    def stop_and_destroy(self):
        if self.animation_id:
            try: self.root.after_cancel(self.animation_id)
            except: pass
            self.animation_id = None
        if self.root.winfo_exists(): self.root.destroy()

def log_uncaught_exception(exc_type, exc_value, exc_traceback):
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