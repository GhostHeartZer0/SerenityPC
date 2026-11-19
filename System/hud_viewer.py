# System/hud_viewer.py
# Serenity Visual, Image, Animated GIF, Video & Diagram HUD Overlay.
# Dynamically inherits UI/Mono fonts and Theme colors from System/config.json (TBD #111).

import json
import os
import shutil
import subprocess
import sys
import tkinter as tk
from tkinter import filedialog, scrolledtext

try:
    from PIL import Image, ImageSequence, ImageTk
except ImportError:
    Image = None
    ImageSequence = None
    ImageTk = None

try:
    import cv2
except ImportError:
    cv2 = None


def _load_dynamic_ui_config():
    """Loads dynamic font family and theme settings from System/config.json instead of hardcoded strings."""
    cfg_candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"),
        os.path.join(os.getcwd(), "System", "config.json"),
    ]
    ui_font = "Segoe UI"
    mono_font = "Consolas"
    bg_color = "#0d0d12"
    hdr_bg = "#161622"
    accent = "#00ffcc"
    fg_color = "#e0e0ff"

    for p in cfg_candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ui_font = data.get("ui_font") or ui_font
                mono_font = data.get("mono_font") or mono_font
                break
            except Exception:
                pass

    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from serenity_resources import THEME
        bg_color = THEME.get("bg_color", bg_color)
        hdr_bg = THEME.get("widget_bg_color", hdr_bg)
        accent = THEME.get("electric_blue", accent)
        fg_color = THEME.get("fg_color", fg_color)
    except Exception:
        pass

    return {
        "ui_font": ui_font,
        "mono_font": mono_font,
        "bg_color": bg_color,
        "hdr_bg": hdr_bg,
        "accent": accent,
        "fg_color": fg_color,
    }


def main():
    if len(sys.argv) < 2:
        return
    data_path = sys.argv[1]
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return

    req_type = str(data.get("req_type", "IMAGE")).upper()
    content = str(data.get("content", ""))
    file_path = str(data.get("file_path", "") or "").strip()
    fps = int(data.get("fps", 8) or 8)

    style = _load_dynamic_ui_config()
    ui_font = style["ui_font"]
    mono_font = style["mono_font"]

    root = tk.Tk()
    root.title("Serenity Visual & Media HUD")
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.97)
    root.geometry("640x580+120+90")
    root.config(bg=style["bg_color"])

    hdr = tk.Frame(root, bg=style["hdr_bg"], pady=6)
    hdr.pack(fill=tk.X)
    tk.Label(
        hdr,
        text="✨ Serenity Visual, GIF & Video HUD",
        fg=style["accent"],
        bg=style["hdr_bg"],
        font=(ui_font, 10, "bold")
    ).pack(side=tk.LEFT, padx=10)
    tk.Label(
        hdr,
        text=f"[{req_type}]",
        fg="#8888aa",
        bg=style["hdr_bg"],
        font=(mono_font, 9)
    ).pack(side=tk.RIGHT, padx=10)

    has_media_file = bool(file_path and os.path.isfile(file_path))
    ext = os.path.splitext(file_path)[1].lower() if has_media_file else ""

    if has_media_file and Image is not None and ImageTk is not None:
        canvas_frame = tk.Frame(root, bg=style["bg_color"])
        canvas_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=(10, 4))

        canvas = tk.Canvas(canvas_frame, bg="#08080c", highlightthickness=1, highlightbackground=style["accent"])
        canvas.pack(fill=tk.BOTH, expand=True)

        anim_state = {"frames": [], "idx": 0, "after_id": None, "cap": None}

        def _fit_pil(pil_img, max_w=580, max_h=380):
            iw, ih = pil_img.size
            if iw <= 0 or ih <= 0:
                return pil_img
            scale = min(max_w / float(iw), max_h / float(ih), 1.0)
            nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
            resample = getattr(Image, "Resampling", Image).LANCZOS
            return pil_img.resize((nw, nh), resample)

        if ext == ".gif" and ImageSequence is not None:
            try:
                with Image.open(file_path) as gif_im:
                    for frame in ImageSequence.Iterator(gif_im):
                        fitted = _fit_pil(frame.convert("RGBA"))
                        anim_state["frames"].append(ImageTk.PhotoImage(fitted))
            except Exception as e:
                print(f"[HUD] GIF load warning: {e}")

            def _step_gif():
                if not anim_state["frames"] or not canvas.winfo_exists():
                    return
                idx = anim_state["idx"] % len(anim_state["frames"])
                canvas.delete("all")
                cw = max(300, canvas.winfo_width())
                ch = max(220, canvas.winfo_height())
                canvas.create_image(cw // 2, ch // 2, image=anim_state["frames"][idx], anchor="center")
                anim_state["idx"] = (idx + 1) % len(anim_state["frames"])
                anim_state["after_id"] = root.after(max(40, int(1000 / max(1, fps))), _step_gif)

            root.after(80, _step_gif)

        elif ext in (".mp4", ".avi", ".mov", ".mkv") and cv2 is not None:
            cap = cv2.VideoCapture(file_path)
            anim_state["cap"] = cap
            v_fps = cap.get(cv2.CAP_PROP_FPS)
            if v_fps and v_fps > 1:
                fps = int(v_fps)

            def _step_video():
                if not canvas.winfo_exists() or anim_state["cap"] is None:
                    return
                ret, frame = anim_state["cap"].read()
                if not ret:
                    anim_state["cap"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = anim_state["cap"].read()
                if ret and frame is not None:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_f = _fit_pil(Image.fromarray(rgb))
                    tk_img = ImageTk.PhotoImage(pil_f)
                    anim_state["current_tk"] = tk_img
                    canvas.delete("all")
                    cw = max(300, canvas.winfo_width())
                    ch = max(220, canvas.winfo_height())
                    canvas.create_image(cw // 2, ch // 2, image=tk_img, anchor="center")
                anim_state["after_id"] = root.after(max(30, int(1000 / max(1, fps))), _step_video)

            root.after(80, _step_video)

        else:
            try:
                pil_static = _fit_pil(Image.open(file_path).convert("RGBA"))
                tk_static = ImageTk.PhotoImage(pil_static)
                anim_state["frames"] = [tk_static]

                def _draw_static(event=None):
                    if not canvas.winfo_exists():
                        return
                    canvas.delete("all")
                    cw = max(300, canvas.winfo_width())
                    ch = max(220, canvas.winfo_height())
                    canvas.create_image(cw // 2, ch // 2, image=anim_state["frames"][0], anchor="center")

                canvas.bind("<Configure>", _draw_static)
                root.after(60, _draw_static)
            except Exception as e:
                print(f"[HUD] Static image load warning: {e}")

        # Compact prompt caption box below canvas
        txt = scrolledtext.ScrolledText(
            root, fg=style["fg_color"], bg="#12121a", font=(mono_font, 9),
            height=4, borderwidth=0, wrap=tk.WORD
        )
        txt.insert(tk.END, f"Prompt: {content}\nFile: {file_path}")
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.X, padx=12, pady=(2, 6))
    else:
        txt = scrolledtext.ScrolledText(
            root, fg=style["fg_color"], bg="#12121a", font=(mono_font, 10),
            insertbackground="white", borderwidth=0, wrap=tk.WORD
        )
        txt.insert(tk.END, content)
        txt.config(state=tk.DISABLED)
        txt.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

    btn_f = tk.Frame(root, bg=style["bg_color"], pady=6)
    btn_f.pack(fill=tk.X, padx=12)

    if has_media_file:
        def _save_copy():
            dest = filedialog.asksaveasfilename(
                parent=root,
                title="Save Generated Media As",
                initialfile=os.path.basename(file_path),
                defaultextension=ext or ".png"
            )
            if dest:
                try:
                    shutil.copy2(file_path, dest)
                except Exception as e:
                    print(f"[HUD] Save copy failed: {e}")

        def _open_folder():
            folder = os.path.dirname(os.path.abspath(file_path))
            if os.path.isdir(folder):
                try:
                    if sys.platform.startswith("win"):
                        os.startfile(folder)
                    elif sys.platform == "darwin":
                        subprocess.Popen(["open", folder])
                    else:
                        subprocess.Popen(["xdg-open", folder])
                except Exception:
                    pass

        tk.Button(
            btn_f, text="💾 Save Copy...", command=_save_copy,
            bg="#252535", fg=style["accent"], font=(ui_font, 9, "bold"), relief=tk.FLAT, padx=10
        ).pack(side=tk.LEFT, padx=(0, 6))
        tk.Button(
            btn_f, text="📂 Open Folder", command=_open_folder,
            bg="#252535", fg=style["fg_color"], font=(ui_font, 9), relief=tk.FLAT, padx=10
        ).pack(side=tk.LEFT)

    tk.Button(
        btn_f, text="Close HUD", command=root.destroy,
        bg="#252535", fg=style["accent"], font=(ui_font, 9, "bold"), relief=tk.FLAT, padx=12
    ).pack(side=tk.RIGHT)
    root.mainloop()


if __name__ == "__main__":
    main()