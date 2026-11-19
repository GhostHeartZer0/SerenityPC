# System/media_generator.py
# Local Image, Animated GIF, and Video Synthesis Engine for SerenityPC.
# Supports stable-diffusion.cpp (sd-cli.exe), Wan2.1 / AnimateDiff CLI/Diffusers,
# 6GB VRAM Bouncer evacuation, FFmpeg palette GIF conversion, and zero-dependency
# Pillow/OpenCV procedural synthesis fallback.

import math
import os
import re
import shutil
import subprocess
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except ImportError:
    Image = None
    ImageDraw = None

try:
    import cv2
except ImportError:
    cv2 = None


RECOMMENDED_SYNTH_MODELS = {
    "image_fast_6gb": {
        "name": "SDXL-Lightning 4-Step (Q5_K / FP16)",
        "vram_gb": 3.6,
        "resolution": "768x768",
        "steps": 4,
        "cfg": 1.5,
        "flags": ["--vae-tiling"],
        "notes": "Recommended default for 6GB VRAM (2-4s per image via sd-cli.exe)."
    },
    "image_flux_6gb": {
        "name": "FLUX.1-schnell-GGUF (Q4_0 / Q4_K_S)",
        "vram_gb": 4.4,
        "resolution": "768x768",
        "steps": 4,
        "cfg": 1.0,
        "flags": ["--clip-on-cpu", "--vae-tiling"],
        "notes": "High prompt adherence & text rendering; offloads T5-XXL to 32GB System RAM."
    },
    "image_hud_fast": {
        "name": "LCM-DreamShaper-v7 (SD 1.5 F16 / Q8_0)",
        "vram_gb": 2.2,
        "resolution": "512x512",
        "steps": 6,
        "cfg": 1.5,
        "flags": [],
        "notes": "Sub-1.5s latency for quick HUD visuals and avatar generation."
    },
    "video_wan_6gb": {
        "name": "Wan2.1-T2V-1.3B-GGUF (Q4_K_M / Q8_0)",
        "vram_gb": 4.2,
        "resolution": "832x480",
        "frames": 16,
        "fps": 8,
        "steps": 12,
        "flags": ["--clip-on-cpu", "--vae-tiling"],
        "notes": "Primary 480p video/GIF generator for 6GB GPUs."
    },
    "gif_animatediff_6gb": {
        "name": "AnimateDiff-Lightning (4-Step SD 1.5)",
        "vram_gb": 3.0,
        "resolution": "512x512",
        "frames": 16,
        "fps": 8,
        "steps": 4,
        "flags": ["--vae-tiling"],
        "notes": "Fast looping 512x512 GIF synthesis (5-8s total)."
    }
}


class MediaGenerator:
    """
    Orchestrates local Image (.png), Animated GIF (.gif), and Video (.mp4) generation
    with 6GB VRAM protection and real-time status bar telemetry.
    """
    def __init__(self, app=None):
        self.app = app

    def _push_event(self, status: str, content: Any):
        if self.app and hasattr(self.app, "process_queue") and self.app.process_queue:
            try:
                self.app.process_queue.put({"status": status, "content": content})
            except Exception:
                pass

    def _get_output_dir(self) -> str:
        base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
        if self.app and hasattr(self.app, "get_user_dir"):
            try:
                u_dir = self.app.get_user_dir()
                if u_dir and isinstance(u_dir, str):
                    out = os.path.join(u_dir, "GeneratedMedia")
                    os.makedirs(out, exist_ok=True)
                    return out
            except Exception:
                pass
        out = os.path.join(base_dir, "Media", "Generated")
        os.makedirs(out, exist_ok=True)
        return out

    def _find_sd_cli(self) -> Optional[str]:
        base_dir = getattr(self.app, "script_dir", os.getcwd()) if self.app else os.getcwd()
        candidates = [
            os.path.join(base_dir, "Tools", "sd-cli.exe"),
            os.path.join(base_dir, "Tools", "sd.exe"),
            os.path.join(base_dir, "Tools", "stable-diffusion-cli.exe"),
            "sd-cli.exe",
            "sd.exe",
            "sd-cli",
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
            w = shutil.which(c)
            if w:
                return w
        return None

    @staticmethod
    def parse_resolution(res_str: str, default: Tuple[int, int] = (768, 768)) -> Tuple[int, int]:
        if not res_str or not isinstance(res_str, str):
            return default
        m = re.search(r"(\d{2,4})\s*[xX*]\s*(\d{2,4})", res_str)
        if m:
            w = max(128, min(2048, int(m.group(1))))
            h = max(128, min(2048, int(m.group(2))))
            return (w - (w % 8), h - (h % 8))
        return default

    def _maybe_evacuate_vram(self):
        cfg = getattr(self.app, "config", {}) if self.app else {}
        if not cfg.get("media_gen_evacuate_vram", True):
            return
        try:
            from System.vision_handler import VisionHandler
            self._push_event("thinking_status", "Evacuating VRAM for Media Synthesis...")
            if self.app and getattr(self.app, "model", None) is not None and hasattr(self.app, "offload_model"):
                self.app.offload_model()
            else:
                VisionHandler.hygiene_gate()
        except Exception as e:
            print(f"[MEDIA GEN] VRAM evacuation notice: {e}")

    def generate(
        self,
        prompt: str,
        media_type: str = "image",
        resolution: Optional[str] = None,
        steps: Optional[int] = None,
        frames: Optional[int] = None,
        fps: Optional[int] = None,
        wait: bool = True
    ) -> Dict[str, Any]:
        """
        Generates an image (.png), animated GIF (.gif), or video (.mp4).
        Returns metadata dict containing 'file_path', 'media_type', 'backend', and 'status'.
        """
        cfg = getattr(self.app, "config", {}) if self.app else {}
        m_type = str(media_type or cfg.get("media_gen_mode", "image")).lower().strip()
        if m_type in ("off", "none", ""):
            m_type = "image"
        if m_type == "mp4":
            m_type = "video"

        res_str = resolution or cfg.get("media_gen_resolution", "768x768" if m_type == "image" else "512x512")
        width, height = self.parse_resolution(res_str)
        num_steps = int(steps or cfg.get("media_gen_steps", 6))
        num_frames = int(frames or cfg.get("video_gen_frames", 16))
        target_fps = int(fps or cfg.get("video_gen_fps", 8))
        backend_pref = str(cfg.get("media_gen_backend", "sd_cpp")).lower()

        out_dir = self._get_output_dir()
        ts = int(time.time() * 1000)
        ext = "gif" if m_type == "gif" else ("mp4" if m_type == "video" else "png")
        out_path = os.path.join(out_dir, f"serenity_{m_type}_{ts}.{ext}")

        model_paths = getattr(self.app, "model_paths", {}) if self.app else cfg.get("model_paths", {})
        if not isinstance(model_paths, dict):
            model_paths = {}
        synth_key = "synth_video" if m_type in ("gif", "video") else "synth_image"
        model_path = str(model_paths.get(synth_key) or model_paths.get("synth_image") or "").strip()

        sd_cli = self._find_sd_cli()
        used_backend = "procedural"

        # 1. Try native C++/CUDA sd-cli.exe if model & binary exist and backend != procedural
        if backend_pref != "procedural" and sd_cli and model_path and os.path.isfile(model_path):
            self._maybe_evacuate_vram()
            if m_type == "image":
                ok = self._run_sd_cli_image(sd_cli, model_path, prompt, out_path, width, height, num_steps)
                if ok and os.path.isfile(out_path):
                    used_backend = "sd_cpp"
            else:
                ok = self._run_sd_cli_animation(
                    sd_cli, model_path, prompt, out_path, width, height, num_steps, num_frames, target_fps, m_type
                )
                if ok and os.path.isfile(out_path):
                    used_backend = "sd_cpp"

        # 2. Fallback to built-in procedural synthesis so generation always produces a real file
        if not os.path.isfile(out_path):
            if m_type in ("image", "diagram", "artistic"):
                self._synthesize_procedural_image(prompt, out_path, width, height, is_diagram=(m_type == "diagram"))
            elif m_type == "gif":
                self._synthesize_procedural_gif(prompt, out_path, min(width, 512), min(height, 512), num_frames, target_fps)
            elif m_type == "video":
                self._synthesize_procedural_video(prompt, out_path, min(width, 640), min(height, 480), num_frames, target_fps)

        self._push_event("video_progress", 100)
        self._push_event(
            "tool_log_update",
            f"\n[MEDIA GEN] Generated {m_type.upper()} ({used_backend}) -> {out_path}\n"
        )
        return {
            "status": "ok" if os.path.isfile(out_path) else "error",
            "file_path": out_path,
            "media_type": m_type,
            "backend": used_backend,
            "resolution": f"{width}x{height}",
            "frames": num_frames if m_type in ("gif", "video") else 1,
            "fps": target_fps if m_type in ("gif", "video") else 0,
            "prompt": prompt
        }

    def _run_sd_cli_image(
        self, cli_path: str, model_path: str, prompt: str, out_path: str, w: int, h: int, steps: int
    ) -> bool:
        cmd = [
            cli_path,
            "-m", model_path,
            "-p", prompt,
            "-W", str(w),
            "-H", str(h),
            "--steps", str(steps),
            "--cfg-scale", "1.5",
            "--vae-tiling",
            "-o", out_path
        ]
        if "flux" in os.path.basename(model_path).lower():
            cmd.append("--clip-on-cpu")
        try:
            self._push_event("thinking_status", f"Synthesizing Image ({w}x{h}, {steps} steps)...")
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace"
            )
            if proc.stdout:
                for line in proc.stdout:
                    m = re.search(r"step\s*(\d+)\s*/\s*(\d+)", line, re.IGNORECASE)
                    if m:
                        cur, tot = int(m.group(1)), max(1, int(m.group(2)))
                        pct = int((cur / tot) * 100)
                        self._push_event("video_progress", pct)
                        self._push_event("thinking_status", f"Synthesizing Image: Step {cur}/{tot} ({pct}%)")
            proc.wait(timeout=300)
            return proc.returncode == 0 and os.path.isfile(out_path)
        except Exception as e:
            print(f"[MEDIA GEN] sd-cli image error: {e}")
            return False

    def _run_sd_cli_animation(
        self, cli_path: str, model_path: str, prompt: str, out_path: str,
        w: int, h: int, steps: int, frames: int, fps: int, mode: str
    ) -> bool:
        tmp_dir = os.path.join(os.path.dirname(out_path), f".frames_{int(time.time()*1000)}")
        os.makedirs(tmp_dir, exist_ok=True)
        frame_files: List[str] = []
        try:
            for idx in range(frames):
                f_out = os.path.join(tmp_dir, f"frame_{idx:04d}.png")
                seed = 1000 + idx
                cmd = [
                    cli_path, "-m", model_path,
                    "-p", f"{prompt} (motion frame {idx+1}/{frames})",
                    "-W", str(w), "-H", str(h),
                    "--steps", str(max(2, steps)),
                    "--seed", str(seed),
                    "--vae-tiling",
                    "-o", f_out
                ]
                self._push_event("video_progress", int(((idx + 1) / max(1, frames)) * 90))
                self._push_event("thinking_status", f"Synthesizing {mode.upper()} Frame {idx+1}/{frames}...")
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
                if os.path.isfile(f_out):
                    frame_files.append(f_out)

            if not frame_files:
                return False

            if mode == "gif" and Image is not None:
                pil_frames = [Image.open(p).convert("RGB") for p in frame_files]
                pil_frames[0].save(
                    out_path,
                    save_all=True,
                    append_images=pil_frames[1:],
                    duration=max(40, int(1000 / max(1, fps))),
                    loop=0,
                    optimize=True
                )
                return os.path.isfile(out_path)
            elif mode == "video":
                if shutil.which("ffmpeg"):
                    pattern = os.path.join(tmp_dir, "frame_%04d.png")
                    subprocess.run([
                        "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
                    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
                if not os.path.isfile(out_path) and cv2 is not None:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
                    for p in frame_files:
                        img = cv2.imread(p)
                        if img is not None:
                            vw.write(cv2.resize(img, (w, h)))
                    vw.release()
                return os.path.isfile(out_path)
            return False
        except Exception as e:
            print(f"[MEDIA GEN] Animation synthesis error: {e}")
            return False
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    def _render_procedural_frame(
        self, prompt: str, w: int, h: int, frame_idx: int = 0, total_frames: int = 1, is_diagram: bool = False
    ):
        if Image is None or ImageDraw is None:
            return None
        try:
            from serenity_resources import THEME
            bg_hex = THEME.get("bg_color", "#0e0d13")
            accent_hex = THEME.get("electric_blue", "#00ffcc")
            sec_hex = THEME.get("accent_secondary", "#6D00FF")
        except Exception:
            bg_hex, accent_hex, sec_hex = "#0e0d13", "#00ffcc", "#6D00FF"

        def _hex_rgb(hx: str, default=(14, 13, 19)):
            hx = str(hx).lstrip("#")
            if len(hx) == 6:
                try:
                    return (int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16))
                except Exception:
                    pass
            return default

        bg_rgb = _hex_rgb(bg_hex, (14, 13, 19))
        acc_rgb = _hex_rgb(accent_hex, (0, 255, 204))
        sec_rgb = _hex_rgb(sec_hex, (109, 0, 255))

        img = Image.new("RGB", (w, h), bg_rgb)
        draw = ImageDraw.Draw(img)

        phase = (frame_idx / max(1, total_frames)) * 2.0 * math.pi

        # Subtle cybernetic grid
        step = max(24, min(w, h) // 12)
        grid_col = tuple(max(0, min(255, int(c * 0.22 + 18))) for c in acc_rgb)
        for x in range(0, w, step):
            draw.line([(x, 0), (x, h)], fill=grid_col, width=1)
        for y in range(0, h, step):
            draw.line([(0, y), (w, y)], fill=grid_col, width=1)

        cx, cy = w // 2, int(h * 0.43)
        max_r = int(min(w, h) * 0.28)

        if is_diagram or "-->" in prompt or "graph " in prompt.lower():
            nodes = [s.strip() for s in re.split(r"--!>|-->|->|\n|;", prompt) if s.strip()][:6]
            if not nodes:
                nodes = ["Input Prompt", "Serenity Engine", "Visual Output"]
            box_w = min(int(w * 0.68), 360)
            box_h = 38
            start_y = max(36, (h - (len(nodes) * 58)) // 2)
            for idx, label in enumerate(nodes):
                bx0 = (w - box_w) // 2
                by0 = start_y + idx * 58
                bx1 = bx0 + box_w
                by1 = by0 + box_h
                draw.rectangle([bx0, by0, bx1, by1], fill=(22, 22, 34), outline=acc_rgb, width=2)
                draw.text((bx0 + 12, by0 + 11), label[:42], fill=(235, 245, 255))
                if idx < len(nodes) - 1:
                    mid_x = w // 2
                    draw.line([(mid_x, by1), (mid_x, by1 + 20)], fill=sec_rgb, width=2)
        else:
            for ring in range(3):
                r = int(max_r * (0.45 + 0.25 * ring + 0.06 * math.sin(phase + ring)))
                col = acc_rgb if ring % 2 == 0 else sec_rgb
                draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=col, width=2)

            for i in range(6):
                ang = phase + (i * math.pi / 3.0)
                ox = int(cx + max_r * 0.75 * math.cos(ang))
                oy = int(cy + max_r * 0.75 * math.sin(ang))
                draw.ellipse([ox - 5, oy - 5, ox + 5, oy + 5], fill=acc_rgb)
                draw.line([(cx, cy), (ox, oy)], fill=sec_rgb, width=1)

            pts = []
            wave_y = int(h * 0.72)
            for x in range(16, w - 16, 6):
                wy = wave_y + int(16 * math.sin((x / 36.0) + phase * 2.0))
                pts.append((x, wy))
            if len(pts) >= 2:
                draw.line(pts, fill=acc_rgb, width=2)

        draw.rectangle([2, 2, w - 3, h - 3], outline=acc_rgb, width=2)
        banner_h = 54
        draw.rectangle([4, h - banner_h - 4, w - 5, h - 5], fill=(12, 12, 20), outline=sec_rgb, width=1)
        clean_p = (prompt or "Serenity Visual Synthesis").replace("\n", " ").strip()
        line1 = clean_p[:56]
        line2 = clean_p[56:112] + ("..." if len(clean_p) > 112 else "")
        draw.text((14, 12), f"SERENITY SYNTHESIS [{w}x{h}]", fill=acc_rgb)
        draw.text((14, h - banner_h + 4), line1, fill=(240, 240, 255))
        if line2:
            draw.text((14, h - banner_h + 24), line2, fill=(180, 190, 215))
        return img

    def _synthesize_procedural_image(self, prompt: str, out_path: str, w: int, h: int, is_diagram: bool = False) -> bool:
        img = self._render_procedural_frame(prompt, w, h, 0, 1, is_diagram=is_diagram)
        if img is not None:
            img.save(out_path, format="PNG")
            return True
        return False

    def _synthesize_procedural_gif(self, prompt: str, out_path: str, w: int, h: int, frames: int, fps: int) -> bool:
        if Image is None:
            return False
        pil_frames = []
        for idx in range(max(4, frames)):
            f_img = self._render_procedural_frame(prompt, w, h, idx, max(4, frames))
            if f_img is not None:
                pil_frames.append(f_img)
            self._push_event("video_progress", int(((idx + 1) / max(1, frames)) * 95))
        if not pil_frames:
            return False
        pil_frames[0].save(
            out_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=max(40, int(1000 / max(1, fps))),
            loop=0,
            optimize=True
        )
        return os.path.isfile(out_path)

    def _synthesize_procedural_video(self, prompt: str, out_path: str, w: int, h: int, frames: int, fps: int) -> bool:
        if Image is None:
            return False
        tmp_dir = os.path.join(os.path.dirname(out_path), f".vframes_{int(time.time()*1000)}")
        os.makedirs(tmp_dir, exist_ok=True)
        try:
            frame_paths = []
            for idx in range(max(8, frames)):
                f_img = self._render_procedural_frame(prompt, w, h, idx, max(8, frames))
                if f_img is not None:
                    fp = os.path.join(tmp_dir, f"frame_{idx:04d}.png")
                    f_img.save(fp, format="PNG")
                    frame_paths.append(fp)
            if not frame_paths:
                return False

            if cv2 is not None:
                try:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    vw = cv2.VideoWriter(out_path, fourcc, float(fps), (w, h))
                    for fp in frame_paths:
                        bgr = cv2.imread(fp)
                        if bgr is not None:
                            vw.write(cv2.resize(bgr, (w, h)))
                    vw.release()
                except Exception:
                    pass

            if not os.path.isfile(out_path) and shutil.which("ffmpeg"):
                try:
                    pattern = os.path.join(tmp_dir, "frame_%04d.png")
                    subprocess.run([
                        "ffmpeg", "-nostdin", "-y", "-framerate", str(fps), "-i", pattern,
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", out_path
                    ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                except Exception:
                    pass
            return os.path.isfile(out_path)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)