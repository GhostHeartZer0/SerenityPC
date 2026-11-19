import os
import re
import shutil
import subprocess
import tempfile
import threading
import time


class DiffusionCLIWrapper:
    """
    Entropy-Bound Diffusion LLM CLI Wrapper (llama-diffusion-cli.exe).
    Includes real-time step telemetry, ETA time-grounding, Gemma turn formatting,
    BOS/EOS token helpers, ANSI visual frame parser, and context/token safety clamping.
    """
    def __init__(self, app_instance, model_path, n_gpu_layers, n_ctx, **kwargs):
        self.app = app_instance
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = int(n_ctx) if n_ctx else 4096
        self.cpu_moe = kwargs.get("cpu_moe", True)
        self.kwargs = kwargs
        self.chat_handler = kwargs.get("chat_handler", None)
        self.process = None

    def token_bos(self) -> int:
        return 2

    def token_eos(self) -> int:
        return 1

    def _push_event(self, status, content):
        if self.app and hasattr(self.app, "process_queue"):
            self.app.process_queue.put({"status": status, "content": content})
        else:
            if status == "thinking_status":
                print(f"[{status}] {content}", end="\r")

    def _is_stopped(self):
        if self.app and hasattr(self.app, "stop_process"):
            return self.app.stop_process.is_set()
        return False

    def __call__(self, prompt, stream=True, echo=False, **kwargs):
        fd, temp_path = tempfile.mkstemp(suffix=".txt")
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(prompt)

        exe_path = None
        base_dir = getattr(self.app, 'BASE_DIR', os.getcwd()) if self.app else os.getcwd()
        home_dir = os.path.expanduser("~")

        search_paths = [
            os.path.join(home_dir, "llama.cpp", "build", "bin", "Release", "llama-diffusion-cli.exe"),
            os.path.join(home_dir, "llama.cpp", "build", "bin", "llama-diffusion-cli.exe"),
            os.path.join(base_dir, "llama-b9592", "llama-diffusion-cli.exe"),
            os.path.join(base_dir, "Tools", "llama-diffusion-cli.exe"),
            "llama-diffusion-cli.exe",
            "llama-diffusion-cli"
        ]

        for path in search_paths:
            if os.path.exists(path) or shutil.which(path):
                exe_path = path if os.path.exists(path) else shutil.which(path)
                break

        if not exe_path:
            self._push_event("diag_log_update", "ERROR: llama-diffusion-cli not found in standard paths!")
            yield {"choices": [{"text": "", "delta": {"content": ""}}]}
            try:
                os.remove(temp_path)
            except Exception:
                pass
            return

        raw_max_tokens = kwargs.get("max_tokens", 512)
        try:
            n_predict = max(128, min(int(raw_max_tokens), 2048))
        except (ValueError, TypeError):
            n_predict = 512

        eb_max_steps = kwargs.get("diffusion_eb_max_steps", self.kwargs.get("diffusion_eb_max_steps", 48))
        eb_entropy_bound = kwargs.get("diffusion_eb_entropy_bound", self.kwargs.get("diffusion_eb_entropy_bound", 0.1))
        diff_steps = kwargs.get("diffusion_steps", self.kwargs.get("diffusion_steps", 48))

        # Calculate context size tailored to prompt + generation
        estimated_prompt_tokens = int(len(prompt) / 3.0) + 512
        needed_ctx = estimated_prompt_tokens + n_predict
        actual_ctx = max(needed_ctx, 1024)
        if self.n_ctx and self.n_ctx > 0:
            actual_ctx = max(actual_ctx, self.n_ctx)

        cmd = [
            exe_path,
            "-m", self.model_path,
            "-ngl", str(self.n_gpu_layers),
            "-c", str(actual_ctx),
            "-ub", str(actual_ctx),
            "-f", temp_path,
            "-n", str(n_predict),
            "--diffusion-visual",
            "--diffusion-eb-max-steps", str(eb_max_steps),
            "--diffusion-eb-entropy-bound", str(eb_entropy_bound),
            "--diffusion-steps", str(diff_steps),
            "--diffusion-kv-cache", "on"
        ]

        # For MoE models (e.g. DiffusionGemma 26B-A4B), pin experts to CPU to prevent
        # Windows WDDM from spilling >6GB VRAM allocations into 9GB Shared System Memory.
        is_moe = any(k in os.path.basename(self.model_path).lower() for k in ["a4b", "moe", "diffusiongemma"])
        if self.cpu_moe or is_moe:
            cmd.append("-cmoe")

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        start_time = time.time()
        step_times = []
        last_step_time = [start_time]

        def _read_stderr():
            for line in iter(self.process.stderr.readline, ''):
                line_str = line.strip()
                if not line_str:
                    continue
                step_match = re.search(r'diffusion step:\s*(\d+)(?:/(\d+))?', line_str, re.IGNORECASE)
                if step_match:
                    now = time.time()
                    step_delta = now - last_step_time[0]
                    last_step_time[0] = now
                    step_times.append(step_delta)
                    current_step_num = int(step_match.group(1))
                    total_steps = int(step_match.group(2)) if step_match.group(2) else 0

                    avg_step_s = sum(step_times[-5:]) / len(step_times[-5:]) if step_times else 0.0
                    eta_str = ""
                    if total_steps > current_step_num and avg_step_s > 0:
                        rem_steps = total_steps - current_step_num
                        eta_s = rem_steps * avg_step_s
                        eta_str = f" [ETA: {eta_s:.1f}s @ {avg_step_s*1000:.0f}ms/step]"

                    step_text = f"Denoising: Step {current_step_num}/{total_steps if total_steps else '?'}{eta_str}"
                    self._push_event("thinking_status", step_text)
                elif any(k in line_str.lower() for k in ["diffusion_params:", "total time:", "diffusioneb:", "diffusion_eb:", "error:"]):
                    self._push_event("diag_log_update", line_str)

        err_thread = threading.Thread(target=_read_stderr, daemon=True)
        err_thread.start()

        raw_pattern = re.compile(r'(\x1b\[\d+A|\x1b\[2J\x1b\[H|\x1b\[2J|\x1b\[H)')
        ansi_escape_pattern = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')

        buffer = ""
        last_clean_frame = ""

        while True:
            if self._is_stopped():
                self.process.terminate()
                break

            chunk = self.process.stdout.read(64)
            if not chunk:
                break

            buffer += chunk
            parts = raw_pattern.split(buffer)
            while len(parts) > 2:
                frame_raw = parts.pop(0)
                _sep = parts.pop(0)
                clean_frame = ansi_escape_pattern.sub('', frame_raw).strip()
                clean_lines = [l for l in clean_frame.splitlines() if not any(k in l.lower() for k in ['total time:', 'throughput:', 'diffusion step:', 'time per step:'])]
                frame_content = "\n".join(clean_lines).strip()
                if frame_content:
                    last_clean_frame = frame_content
                    self._push_event("streaming_replace", frame_content)
            buffer = parts[0]

        if buffer:
            clean_frame = ansi_escape_pattern.sub('', buffer).strip()
            clean_lines = [l for l in clean_frame.splitlines() if not any(k in l.lower() for k in ['total time:', 'throughput:', 'diffusion step:', 'time per step:'])]
            frame_content = "\n".join(clean_lines).strip()
            if frame_content:
                last_clean_frame = frame_content

        self.process.wait()
        try:
            os.remove(temp_path)
        except Exception:
            pass

        elapsed_total = time.time() - start_time
        if last_clean_frame:
            self._push_event("diag_log_update", f"[DIFFUSION] Denoising finished in {elapsed_total:.2f}s ({len(step_times)} steps).")
            self._push_event("streaming_replace", "")
            yield {"choices": [{"text": last_clean_frame, "delta": {"content": last_clean_frame}}]}
        else:
            yield {"choices": [{"text": "", "delta": {"content": ""}}]}

    def create_chat_completion(self, messages, stream=True, **kwargs):
        prompt = ""
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            prompt += f"<start_of_turn>{role}\n{content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"

        gen = self(prompt, stream=stream, **kwargs)
        if stream:
            return gen
        else:
            final_text = ""
            for chunk in gen:
                choice = chunk.get("choices", [{}])[0]
                text_part = choice.get("text")
                if text_part is None:
                    text_part = choice.get("delta", {}).get("content", "")
                final_text += text_part
            return {"choices": [{"message": {"content": final_text}}]}