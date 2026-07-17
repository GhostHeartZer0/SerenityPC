import os
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox

def open_crusher():
    root = tk.Tk()
    root.withdraw() # Hide the main window
    
    # Open file dialog
    input_video = filedialog.askopenfilename(
        title="Select Video to Crush",
        filetypes=[
            ("Video Files", "*.mp4 *.mkv *.avi *.mov *.wmv *.flv"),
            ("All Files", "*.*")
        ]
    )
    
    if not input_video:
        print("No video selected. Exiting.")
        return
        
    if not os.path.exists(input_video):
        messagebox.showerror("Error", f"File not found:\n{input_video}")
        return

    # Determine output filename
    dir_name = os.path.dirname(input_video)
    base_name = os.path.basename(input_video)
    name, ext = os.path.splitext(base_name)
    output_video = os.path.join(dir_name, f"{name}_crushed{ext}")

    # Ensure output doesn't overwrite input if somehow named the same
    counter = 1
    while os.path.exists(output_video):
        output_video = os.path.join(dir_name, f"{name}_crushed_{counter}{ext}")
        counter += 1

    print(f"Input:  {input_video}")
    print(f"Output: {output_video}")
    print("Crushing video to 5 FPS and 480p resolution...")

    # ffmpeg command: scale to 480p height (auto width divisible by 2), and 5 fps
    cmd = [
        "ffmpeg",
        "-y",                 # Overwrite output
        "-i", input_video,    # Input file
        "-vf", "scale=-2:480,fps=5", # Scaling and framerate filter
        "-c:v", "libx264",    # Video codec (standard)
        "-preset", "fast",    # Encoding speed
        "-crf", "28",         # Constant Rate Factor (compression quality)
        "-c:a", "copy",       # Copy audio without re-encoding
        output_video
    ]
    
    try:
        # Run ffmpeg
        subprocess.run(cmd, check=True)
        messagebox.showinfo("Success", f"Video crushed successfully!\n\nSaved to:\n{output_video}")
    except FileNotFoundError:
        messagebox.showerror("Error", "FFmpeg not found. Please ensure ffmpeg is installed and available in your system's PATH.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"FFmpeg processing failed.\nExit code: {e.returncode}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{str(e)}")

if __name__ == "__main__":
    open_crusher()
