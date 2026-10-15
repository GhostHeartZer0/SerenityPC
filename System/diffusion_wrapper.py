import os
import shutil
import subprocess
import tempfile

class DiffusionCLIWrapper:
    def __init__(self, app_instance, model_path, n_gpu_layers, n_ctx, **kwargs):
        self.app = app_instance
        self.model_path = model_path
        self.n_gpu_layers = n_gpu_layers
        self.n_ctx = n_ctx
        self.kwargs = kwargs
        self.chat_handler = kwargs.get("chat_handler", None)
        self.process = None

    def _push_event(self, status, content):
        if self.app and hasattr(self.app, "process_queue"):
            self.app.process_queue.put({"status": status, "content": content})
        else:
            # Fallback if running headless (e.g., Wringer.py)
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
            yield {"choices": [{"text": ""}]}
            try: os.remove(temp_path)
            except: pass
            return

        n_predict = kwargs.get("max_tokens", 2048)
        
        # Calculate context size exactly tailored to the prompt + generation
        estimated_prompt_tokens = int(len(prompt) / 3.0) + 512
        actual_ctx = estimated_prompt_tokens + n_predict
        
        # Safety cap to prevent OOM
        actual_ctx = min(actual_ctx, self.n_ctx)

        cmd = [
            exe_path,
            "-m", self.model_path,
            "-ngl", str(self.n_gpu_layers),
            "-c", str(actual_ctx),
            "-ub", str(actual_ctx),
            "-f", temp_path,
            "-n", str(n_predict),
            "--diffusion-visual"
        ]

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace'
        )

        buf = ""
        current_frame = ""
        last_clean_frame = ""
        import re, time
        frame_pattern = re.compile(r'(\033\[2J\033\[H|\033\[\d+A\033\[J|\033\[2J|\033\[H)')
        
        start_time = time.time()
        step_times = []
        last_step_time = start_time
        current_step_num = 0
        total_steps = 0

        while True:
            if self._is_stopped():
                self.process.terminate()
                break

            char = self.process.stdout.read(1)
            if not char:
                if current_frame:
                    clean_frame = re.sub(r'\033\[[0-9;?]*[A-Za-z]', '', current_frame)
                    lines = clean_frame.strip().split('\n')
                    clean_lines = []
                    for line in lines:
                        if "diffusion step:" in line:
                            self._push_event("thinking_status", f"Denoising: {line.replace('\r', '').strip()}")
                        elif any(k in line for k in ["diffusion_params:", "total time:", "diffusioneb:", "diffusion_eb:"]):
                            self._push_event("diag_log_update", line.strip())
                        else:
                            clean_lines.append(line)
                    frame_content = "\n".join(clean_lines).strip()
                    if frame_content:
                        last_clean_frame = frame_content
                break

            current_frame += char

            if char in ('H', 'J', '\n'):
                match = frame_pattern.search(current_frame)
                if match:
                    raw_frame = current_frame[:match.start()]
                    current_frame = current_frame[match.end():]
                    
                    if raw_frame:
                        clean_frame = re.sub(r'\033\[[0-9;?]*[A-Za-z]', '', raw_frame)
                        lines = clean_frame.strip().split('\n')
                        clean_lines = []
                        for line in lines:
                            # Step and time-grounding telemetry
                            step_match = re.search(r'diffusion step:\s*(\d+)(?:/(\d+))?', line, re.IGNORECASE)
                            if step_match:
                                now = time.time()
                                step_delta = now - last_step_time
                                last_step_time = now
                                step_times.append(step_delta)
                                current_step_num = int(step_match.group(1))
                                if step_match.group(2):
                                    total_steps = int(step_match.group(2))
                                
                                avg_step_s = sum(step_times[-5:]) / len(step_times[-5:]) if step_times else 0.0
                                eta_str = ""
                                if total_steps > current_step_num and avg_step_s > 0:
                                    rem_steps = total_steps - current_step_num
                                    eta_s = rem_steps * avg_step_s
                                    eta_str = f" [ETA: {eta_s:.1f}s @ {avg_step_s*1000:.0f}ms/step]"
                                
                                step_text = f"Denoising: Step {current_step_num}/{total_steps if total_steps else '?'}{eta_str}"
                                self._push_event("thinking_status", step_text)
                            elif any(k in line for k in ["diffusion_params:", "total time:", "diffusioneb:", "diffusion_eb:"]):
                                self._push_event("diag_log_update", line.strip())
                            else:
                                clean_lines.append(line)

                        frame_content = "\n".join(clean_lines).strip()
                        if frame_content:
                            last_clean_frame = frame_content
                            self._push_event("streaming_replace", frame_content)

        self.process.wait()
        try: os.remove(temp_path)
        except: pass

        elapsed_total = time.time() - start_time
        if last_clean_frame:
            self._push_event("diag_log_update", f"[DIFFUSION] Denoising finished in {elapsed_total:.2f}s ({len(step_times)} steps).")
            self._push_event("streaming_replace", "")
            yield {"choices": [{"text": last_clean_frame}]}
        else:
            yield {"choices": [{"text": ""}]}

    def create_chat_completion(self, messages, stream=True, **kwargs):
        prompt = ""
        for m in messages:
            prompt += f"<|turn>{m['role']}\n{m['content']}<turn|>\n"
        prompt += "<|turn>model\n"
        
        gen = self(prompt, stream=stream, **kwargs)
        if stream:
            return gen
        else:
            final_text = ""
            for chunk in gen:
                final_text += chunk["choices"][0]["text"]
            return {"choices": [{"message": {"content": final_text}}]}
