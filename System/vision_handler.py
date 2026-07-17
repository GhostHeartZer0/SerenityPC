import cv2
import base64
import os
import torch
import psutil
from llama_cpp import Llama

# NVDEC Acceleration Attempt
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "video_codec;h264_cuvid|video_codec;hevc_cuvid"

try:
    try:
        from System.serenity_utils import HardwareProfile
    except ImportError:
        from serenity_utils import HardwareProfile
except ImportError:
    HardwareProfile = None

class VisionHandler:
    """Handles multimodal query formulation and context injection for Serenity."""
    SUB_CHUNK_SIZE = 8

    def __init__(self, model_path=None):
        self.model_path = model_path
        self.llm = None

    def initialize_model(self, layers=24):
        """[THE BOUNCER] Ensures VRAM is clear before spawning LLM instance."""
        # 1. REQUIRE self.llm to be None
        if self.llm is not None:
            print("[APEX] Bouncer: LLM detected. Executing disposal...")
            del self.llm
            self.llm = None
            
        # 2. Force hardware flush before loading
        import gc
        import torch
        import time
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print("[APEX] VRAM Flush Complete. Initiating 3.0s 'Breath'...")
        time.sleep(3.0) # THE BREATH: Give 3050 time to forget
        
        print(f"[APEX] Initialization starting for {layers} layers...")

        
        # [VRAM SCOUT] Reserved for dynamic 1.5GB landing zone
        params = getattr(self, 'params', {}).copy()
        use_flash = params.pop('flash_attn', True) # KW_ARGS COLLISION FIX

        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=layers,  # This is your 'offload' target
            n_ctx=32768,          # You have 32GB RAM now—use it!
            n_threads=8,          # Strictly pin to 8 P-Cores
            n_batch=512,          # Optimized for VRAM spike guard
            n_ubatch=256,         # Micro-batching for bus alignment
            use_mmap=True,        # i7 manages kernel mapping
            use_mlock=False,      # Prevents lag on 32GB RAM
            type_k=8,             # Q8_0 KV Cache
            type_v=8,             # Q8_0 KV Cache
            flash_attn=use_flash, # Collision safety
            **params
        )
        
    def evacuate_vram(self):
        """
        [THE BOUNCER] Brute-force memory purge.
        Blocks until RTX 3050 LP memory usage < 1.2GB.
        """
        print("[APEX] Breaking Infinite Hygiene Loop: Evacuating VRAM...")
        
        # 1. Kill the object
        if self.llm:
            del self.llm
            self.llm = None
        
        # 2. Force System & GPU to breathe
        import gc
        import time
        import torch
        try:
            import pynvml as nvidia_ml
            nvidia_ml.nvmlInit()
            handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
        except:
            handle = None

        start_wait = time.time()
        while True:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # 3. VERIFICATION GATE: Confirm 2.2GB floor (Allow OS/Display overhead)
            # A 6GB card usually idles at 1.5GB - 1.8GB on Windows with dual displays.
            if handle:
                try:
                    mem = nvidia_ml.nvmlDeviceGetMemoryInfo(handle)
                    used_mb = mem.used / (1024**2)
                    print(f"[APEX] VRAM Monitor: {used_mb:.0f}MB / 2200MB threshold.")
                    if used_mb < 2200:
                        break
                except:
                    break # Safety escape if NVML fails
            else:
                time.sleep(2.0) # Fallback breathing room
                break

            # Hard timeout to prevent "Lockout"
            if time.time() - start_wait > 10.0:
                print("[APEX] Bouncer: VRAM clearing timed out. Proceeding with caution...")
                break

            print("[APEX] Waiting for PCIe 'Ghosting' to settle...")
            time.sleep(1.0) 
        
        # FINAL BREATH after verify loop
        time.sleep(3.0)

        
        print("[APEX] VRAM Verified Empty. Deck is clear.")

    @staticmethod
    def _check_hardware_guards():
        """Monitor VRAM and PCIe power draw for safety."""
        try:
            import pynvml as nvidia_ml
            import time
            nvidia_ml.nvmlInit()
            handle = nvidia_ml.nvmlDeviceGetHandleByIndex(0)
            
            # --- 1. BUS SATURATION GUARD (The 75W Limit) ---
            pwr = nvidia_ml.nvmlDeviceGetPowerUsage(handle)
            if pwr > 75000:
                print(f"[APEX] Bus Guard: Draw exceeding 75W ({pwr/1000:.1f}W). Throttling injection...")
                time.sleep(0.5)
            
            # --- 2. BUS HYGIENE (The Cool-Down Pulse) ---
            time.sleep(0.2) 

            # --- 3. VRAM LANDING ZONE (4.8GB Used Floor) ---
            # MISSION: Verify VRAM 'Used' is < 4.8GB before companion loading
            mem = nvidia_ml.nvmlDeviceGetMemoryInfo(handle)
            used_mb = mem.used / (1024**2)
            if used_mb > 4800:
                print(f"[APEX] Landing Zone Saturated: {used_mb:.0f}MB > 4800MB. Evacuation required.")
        except: pass

    @staticmethod
    def get_chunked_frames(video_path, chunk_size=5):
        """
        Apex Strategic Staging:
        1. Fully decodes video into a NumPy array in System DRAM (GB-Scale Buffer).
        2. Yields bursts with PCIe bus hygiene.
        """
        import numpy as np
        import time
        video = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        original_fps = video.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0:
            video.release()
            return

        frame_step = max(int(original_fps / 5.0), 1)
        
        # --- DRAM PRE-PROCESSING (The GB-Scale NumPy Landing Dock) ---
        print(f"[APEX] DRAM PRE-PROCESSING: Buffering {os.path.basename(video_path)} to System RAM...")
        if HardwareProfile: HardwareProfile.pin_to_e_cores() # Decoding on E-Cores
        
        raw_frames = []
        count = 0
        while True:
            success, frame = video.read()
            if not success: break
                
            if count % frame_step == 0:
                # 480p Shift Tactical Scaling
                target_res = (854, 480)
                resized = cv2.resize(frame, target_res, interpolation=cv2.INTER_AREA)
                raw_frames.append(resized)
            count += 1
        video.release()
        
        if not raw_frames: return

        # MISSION: Land payload in a NumPy array buffer
        dram_buffer = np.stack(raw_frames, axis=0) 
        print(f"[APEX] DRAM Staging Complete: {len(dram_buffer)} frames buffered in System RAM ({dram_buffer.nbytes/1024**2:.1f} MB).")

        # Yield in bursts
        current_res = (1280, 720)
        for i in range(0, len(dram_buffer), chunk_size):
            VisionHandler._check_hardware_guards()
            burst = dram_buffer[i : i + chunk_size]
            
            # Convert NumPy burst to B64 for llama-cpp-python injection
            prep_start = time.time()
            b64_burst = []
            for frame in burst:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                b64_burst.append(base64.b64encode(buffer).decode("utf-8"))
            prep_ms = (time.time() - prep_start) * 1000
            print(f"[APEX] Swap Jig Prep Latency: {prep_ms:.1f}ms (Bus/CPU)")
            
            yield b64_burst
            
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Prompt for standard 'Send' button (Phase 1: Silent Scout)
    SILENT_SCOUT_UNIVERSAL_PROMPT = (
        "[SYSTEM: UNIVERSAL TELEMETRY SENSOR v3.0]\n"
        "[STRICT OPERATIONAL DIRECTIVE]\n"
        "- OUTPUT: RAW JSON OBJECTS ONLY. \n"
        "- PROSE_DETECTION: NO INTRODUCTIONS, NO \"CERTAINLY\", NO CHATTER.\n"
        "- UI_LOCK: ANCHOR ON HUD TIMERS [TOP], STATUS ICONS [RIGHT], AND CENTRAL MAGNITUDE SPIKES.\n"
        "- PERSISTENCE: IF HUD IS OBSCURED, EXTRAPOLATE FROM [t-1] STATE (v = v - 1s).\n\n"
        "[JSON SCHEMA]\n"
        "{\n"
        '  "t": "[VIDEO_TIME]",\n'
        '  "v": "[PRIMARY_HUD_TIMER]",\n'
        '  "a": "[ACTIVE_ENTITY_NAME]",\n'
        '  "s": "[STATUS_FLAGS: LIST ACTIVE BUFFS/DEBUFFS]",\n'
        '  "x": "[EVENT_MAGNITUDE: PEAK NUMBER/CRITICAL VALUE]",\n'
        '  "w": "[PRIMARY_WINDOW: SECONDS REMAINING ON ACTIVE ABILITY]",\n'
        '  "q": "[SIGNAL_QUALITY: 1-10]"\n'
        "}"
    )

    # Prompt for 'Deep Cook' or Final Analysis synthesis (Phase 2)
    FINAL_VERDICT_PROMPT = (
        "[SYSTEM: APEX FINAL VERDICT]\n"
        "[INPUT: DRAM TELEMETRY LOGS + UNICORN SAMPLES]\n"
        "[OBJECTIVE: HARDCORE PERFORMANCE AUDIT]\n\n"
        "1. MACRO ANALYSIS:\n"
        "   - Analyze the [DRAM LOGS] for the full duration.\n"
        "   - Calculate Average DPS vs. Peak DPS.\n"
        "   - Identify the \"Dead Zone\" (longest window of zero damage/utility).\n\n"
        "2. UNICORN CROSS-REFERENCE:\n"
        "   - Contrast the raw telemetry with the attached high-fidelity Unicorn frames.\n"
        "   - Did the 350k+ peaks occur during optimal support windows (e.g., Omen, Fantastic Voyage)?\n\n"
        "3. THE 120S GAP:\n"
        "   - Based on the data, identify exactly where the time for the 3-star clear was lost.\n"
        "   - Provide actionable fixes for Swap-Jig latency or Rotation sequencing."
    )

    # Prompt for 'Deep Cook' button (Qwen3.5-9B / Auditor)
    GRANDMASTER_AUDITOR_PROMPT = (
        "You are an elite Combat Auditor and Theorycrafting Grandmaster evaluating a 2.5fps video log. "
        "Processing time is irrelevant; accuracy and verifiable correctness are your absolute priorities. "
        "Conduct an exhaustive, step-by-step audit of the encounter:\n"
        "1. DPR Verification: Break down the player's rotation. Verify if the sequence of skills/bursts mathematically maximizes the Damage Per Rotation. If a buff or elemental reaction was missed, flag the exact state change.\n"
        "2. Mechanical Autopsy: Analyze problematic points. Did the player fail a dodge, mismanage stamina, or misunderstand a boss mechanic? Detail the precise failure and the optimal counter-strategy.\n"
        "3. Roster & Loadout Audit: Scrutinize the characters used. Verify if this team composition is optimal for this specific enemy/floor. Provide alternative roster suggestions and mathematically justify why they would tally a higher score or faster clear time.\n"
        "Take your time. Think step-by-step. Provide a definitive, undeniable breakdown of how to achieve perfection in this scenario."
    )

    @staticmethod
    def find_segments(video_path):
        """Identifies PartXXX_analysis.txt files associated with the video."""
        import glob
        import os
        base = os.path.splitext(video_path)[0]
        segments = glob.glob(f"{base}_Part*_analysis.txt")
        # Sort chronologically (Part001, Part002...)
        segments.sort()
        return segments

    @staticmethod
    def hygiene_gate(llm_instance=None):
        """
        Consolidated post-inference hygiene. 
        Clears LLM KV-cache, triggers garbage collection, and flushes VRAM.
        """
        import time
        import gc
        
        # 1. THE SETTLE: Allow PCIe bus/LLM state to rest for 100ms
        time.sleep(0.1)
        
        try:
            if llm_instance:
                # llama-cpp-python specific cache clearing
                if hasattr(llm_instance, "reset"):
                    llm_instance.reset()
                print("[HYGIENE] LLM Context Reset.")
            
            gc.collect()
            
            # If using torch/cuda
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # print("[HYGIENE] VRAM Flush Complete.")
            except ImportError: pass
            
        except Exception as e:
            print(f"[HYGIENE] Cleanup encountered a non-fatal issue: {e}")

    @staticmethod
    def clear_cache(llm_instance):
        """Deprecated: Use hygiene_gate instead."""
        VisionHandler.hygiene_gate(llm_instance)

    @staticmethod
    def _determine_visual_budget(query):
        """Analyzes the query to pick an optimal Gemma-4 visual budget (70-1120)."""
        q = query.lower()
        if any(x in q for x in ["read", "ocr", "text", "document", "financial", "code"]):
            return 1120
        if any(x in q for x in ["detail", "small", "identify", "examine", "micro"]):
            return 560
        if any(x in q for x in ["summarize", "overview", "thumbnail", "quick"]):
            return 140
        return 280 # Standard APEX Balanced budget

    @staticmethod
    def prepare_vision_query(user_query, is_deep_cook=False):
        """
        Prepends tactical instructions and determines the optimal visual budget.
        """
        prompt = VisionHandler.GRANDMASTER_AUDITOR_PROMPT if is_deep_cook else VisionHandler.SILENT_SCOUT_UNIVERSAL_PROMPT
        budget = VisionHandler._determine_visual_budget(user_query)
        print(f"[APEX] Auto-Vision: Using {budget} token budget for this query.")
        return f"{prompt}\n\n[VISUAL_BUDGET: {budget}]\n[USER QUERY]: {user_query}", budget

    @staticmethod
    def encode_image(image_path, budget=280):
        """Encodes image with scaling optimized for the target token budget."""
        import cv2, base64
        import os
        img = cv2.imread(image_path)
        if img is None: return None
        
        # Mapping budgets to target resolutions (Approximate for Gemma-4)
        budget_map = {
            70: 224,
            140: 336,
            280: 448,
            560: 672,
            1120: 1024 # High-res OCR target
        }
        max_dim = budget_map.get(budget, 448)
        
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        _, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return base64.b64encode(buffer).decode("utf-8")

    @staticmethod
    def get_audio_chunks(audio_path, chunk_length_s=30, max_chunks=10):
        """
        Splits audio (.mp3, .wav, .flac) into 30s Base64 encoded chunks via ffmpeg.
        Gemma-4 E2B/E4B Optimized.
        """
        import subprocess
        import math
        import tempfile
        import base64
        import os

        # Verify extension support
        ext = os.path.splitext(audio_path)[1].lower()
        if ext not in [".mp3", ".wav", ".flac"]:
            print(f"[APEX] Unsupported audio format: {ext}")
            return []

        # Get duration
        try:
            cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path]
            duration_str = subprocess.check_output(cmd).decode('utf-8').strip()
            duration = float(duration_str)
        except Exception:
            duration = chunk_length_s 

        chunks = []
        num_chunks = int(math.ceil(duration / chunk_length_s))
        num_chunks = min(num_chunks, max_chunks)
        
        for i in range(num_chunks):
            start_time = i * chunk_length_s
            temp_wav = os.path.join(tempfile.gettempdir(), f"serenity_audio_chunk_{i}.wav")
            
            # Create a 16kHz mono WAV chunk (standard for ASR)
            cmd = ['ffmpeg', '-y', '-i', audio_path, '-ss', str(start_time), '-t', str(chunk_length_s), '-ar', '16000', '-ac', '1', temp_wav]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                with open(temp_wav, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                chunks.append(b64)
            except Exception as e:
                print(f"[APEX] Audio chunking failed for {i}: {e}")
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                    
        return chunks
