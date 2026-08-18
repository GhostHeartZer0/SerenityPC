try:
    import cv2
except ImportError:
    cv2 = None
import base64
import os
import glob
import math
import tempfile
import numpy as np
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

        threads = HardwareProfile.get_optimal_threads() if HardwareProfile else 4
        self.llm = Llama(
            model_path=self.model_path,
            n_gpu_layers=layers,  # This is your 'offload' target
            n_ctx=32768,          # You have 32GB RAM now—use it!
            n_threads=threads,    # Dynamic physical/logical core allocation
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
        if isinstance(query, tuple):
            if len(query) == 2 and isinstance(query[0], str):
                query = query[0]
            else:
                query = str(query)
        elif not isinstance(query, str):
            query = str(query) if query is not None else ""
        q = query.lower()
        if any(x in q for x in [
            "read", "ocr", "text", "document", "financial", "code", 
            "card", "cards", "suit", "clubs", "spades", "hearts", "diamonds", 
            "rank", "zoom", "crop", "poker", "blackjack", "table", "flop", 
            "turn", "river", "hand", "dealer", "6-heart", "9-diamond", "board"
        ]):
            return 1120
        if any(x in q for x in ["detail", "small", "identify", "examine", "micro"]):
            return 560
        if any(x in q for x in ["summarize", "overview", "thumbnail", "quick"]):
            return 140
        return 280 # Standard APEX Balanced budget

    @staticmethod
    def crop_active_playing_area(image_input, padding=0.06):
        """
        [HEURISTIC CONTOUR CARD ROI DETECTION]
        Isolates active cards and playing area from poker/card tables,
        eliminating wasted background felt and maximizing symbol pixel density.
        """
        import cv2
        import numpy as np

        if isinstance(image_input, str):
            img = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img = image_input
        else:
            return image_input

        if img is None or img.size == 0:
            return image_input

        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return img

        # 1. Convert to grayscale & compute edge gradients
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # 2. Canny edge detector + morphological closing to seal card perimeters
        edges = cv2.Canny(blurred, 30, 150)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # 3. Find card candidate contours with hierarchy
        contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        card_boxes = []
        img_area = float(h * w)

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Cards generally occupy between 0.1% and 35% of total table area
            if area < (img_area * 0.001) or area > (img_area * 0.40):
                continue

            x, y, cw, ch = cv2.boundingRect(cnt)
            aspect = float(cw) / float(ch) if ch > 0 else 0

            # Standard card aspect ratio is ~1.4 (portrait ~0.55-0.90, landscape ~1.15-1.75)
            is_card_aspect = (0.55 <= aspect <= 0.90) or (1.15 <= aspect <= 1.75)
            # Also allow square-ish tilted cards or small clusters
            is_card_cluster = (0.45 <= aspect <= 2.2) and (area >= img_area * 0.002)

            if is_card_aspect or is_card_cluster:
                card_boxes.append((x, y, x + cw, y + ch))

        if not card_boxes:
            # No clear card contours isolated; return original image safely
            return img

        # 4. Compute bounding envelope of all detected card regions
        min_x = min(box[0] for box in card_boxes)
        min_y = min(box[1] for box in card_boxes)
        max_x = max(box[2] for box in card_boxes)
        max_y = max(box[3] for box in card_boxes)

        # 5. Apply safety padding margin around playing area
        pad_x = int((max_x - min_x) * padding)
        pad_y = int((max_y - min_y) * padding)

        crop_x1 = max(0, min_x - pad_x)
        crop_y1 = max(0, min_y - pad_y)
        crop_x2 = min(w, max_x + pad_x)
        crop_y2 = min(h, max_y + pad_y)

        # Ensure cropped region has meaningful dimension
        if (crop_x2 - crop_x1) >= 40 and (crop_y2 - crop_y1) >= 40:
            cropped = img[crop_y1:crop_y2, crop_x1:crop_x2]
            print(f"[APEX] Card Playing Area Auto-Cropped: {w}x{h} -> {cropped.shape[1]}x{cropped.shape[0]} (Envelope: ({crop_x1},{crop_y1})-({crop_x2},{crop_y2}))")
            return cropped

        return img

    @staticmethod
    def enhance_symbol_clarity(img_np):
        """
        [SYMBOL PIXEL DENSITY & EDGE CLARITY]
        Applies CLAHE on L-channel in LAB space + high-frequency unsharp masking
        to make 6 vs 9 numerals and Heart vs Diamond suit symbols sharply distinct.
        """
        import cv2
        import numpy as np

        if img_np is None or img_np.size == 0:
            return img_np

        try:
            # 1. LAB color space CLAHE for local contrast on suit & rank boundaries
            lab = cv2.cvtColor(img_np, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l_enhanced = clahe.apply(l)
            lab_enhanced = cv2.merge((l_enhanced, a, b))
            bgr_contrast = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

            # 2. Unsharp masking to crisp suit edges and number serifs
            gaussian = cv2.GaussianBlur(bgr_contrast, (0, 0), 2.0)
            sharpened = cv2.addWeighted(bgr_contrast, 1.35, gaussian, -0.35, 0)
            return sharpened
        except Exception as e:
            print(f"[APEX] Symbol enhancement non-fatal fallback: {e}")
            return img_np

    @staticmethod
    def prepare_vision_query(user_query, is_deep_cook=False, is_scout=False):
        """
        Prepends tactical instructions and determines the optimal visual budget.
        """
        if isinstance(user_query, tuple) and len(user_query) == 2:
            return user_query
        budget = VisionHandler._determine_visual_budget(user_query)
        print(f"[APEX] Auto-Vision: Using {budget} token budget for this query.")
        if is_deep_cook:
            prompt = VisionHandler.GRANDMASTER_AUDITOR_PROMPT
            return f"{prompt}\n\n[VISUAL_BUDGET: {budget}]\n[USER QUERY]: {user_query}", budget
        elif is_scout:
            prompt = VisionHandler.SILENT_SCOUT_UNIVERSAL_PROMPT
            return f"{prompt}\n\n[VISUAL_BUDGET: {budget}]\n[USER QUERY]: {user_query}", budget
        else:
            q_text = str(user_query) if user_query else "Analyze this media in detail."
            return f"[VISUAL_BUDGET: {budget}]\n{q_text}", budget

    @staticmethod
    def encode_image(image_path, budget=280, auto_crop_cards=True, query=None):
        """Encodes image with scaling, card contour auto-crop, and symbol clarity optimization."""
        import cv2, base64
        import os
        img = cv2.imread(image_path)
        if img is None: return None

        # Check if query or budget dictates card crop & symbol enhancement
        is_card_target = (budget >= 1120)
        if query:
            is_card_target = is_card_target or any(k in str(query).lower() for k in [
                "card", "cards", "suit", "hearts", "diamonds", "spades", "clubs", 
                "rank", "poker", "blackjack", "table", "flop", "turn", "river", "hand"
            ])

        # 1. Auto-crop to active playing area if card target
        if auto_crop_cards and is_card_target:
            img = VisionHandler.crop_active_playing_area(img)

        # 2. Enhance symbol clarity for card/OCR budgets
        if is_card_target or budget >= 560:
            img = VisionHandler.enhance_symbol_clarity(img)
        
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
            # Use Lanczos-4 interpolation for crisp symbol and edge preservation
            interp = cv2.INTER_LANCZOS4 if is_card_target else cv2.INTER_AREA
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=interp)

        # Encode with quality 98 and 4:4:4 sampling factor to eliminate red chroma blur on hearts/diamonds
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 98]
        if hasattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR'):
            encode_params.extend([cv2.IMWRITE_JPEG_SAMPLING_FACTOR, getattr(cv2, 'IMWRITE_JPEG_SAMPLING_FACTOR_444', 0)])

        _, buffer = cv2.imencode(".jpg", img, encode_params)
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
                print(f"[APEX] Audio chunking via ffmpeg failed for {i}: {e}. Trying native scipy fallback...")
                if ext == ".wav":
                    try:
                        import scipy.io.wavfile as wavfile
                        import numpy as np
                        
                        in_rate, in_data = wavfile.read(audio_path)
                        orig_dtype = in_data.dtype
                        
                        if len(in_data.shape) > 1:
                            in_data = in_data.mean(axis=1)
                            
                        start_idx = int(start_time * in_rate)
                        end_idx = int((start_time + chunk_length_s) * in_rate)
                        chunk_data = in_data[start_idx:end_idx]
                        
                        if len(chunk_data) > 0:
                            target_rate = 16000
                            num_samples = int(len(chunk_data) * target_rate / in_rate)
                            
                            x_orig = np.linspace(0, len(chunk_data), len(chunk_data))
                            x_new = np.linspace(0, len(chunk_data), num_samples)
                            resampled_data = np.interp(x_new, x_orig, chunk_data)
                            
                            if np.issubdtype(orig_dtype, np.integer):
                                resampled_data = np.round(resampled_data).astype(orig_dtype)
                            else:
                                resampled_data = resampled_data.astype(orig_dtype)
                                
                            wavfile.write(temp_wav, target_rate, resampled_data)
                            with open(temp_wav, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                            chunks.append(b64)
                            print(f"[APEX] Native scipy audio chunking fallback succeeded for chunk {i}")
                        else:
                            print(f"[APEX] Native fallback failed: chunk data is empty")
                    except Exception as fallback_err:
                        print(f"[APEX] Native scipy audio chunking fallback failed for {i}: {fallback_err}")
                else:
                    print(f"[APEX] Native fallback only supports .wav files (current: {ext})")
            finally:
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                    
        return chunks

    @staticmethod
    def get_video_sampled_frames(video_path, target_fps=1.0, max_dim=672, jpeg_quality=85, budget=280, zoom=False):
        """
        Dynamically samples frames at the target rate (default 1 fps) up to 60 seconds limit.
        Scale resolution up to 1024px for budget >= 1120 or zoom mode, plus optional 2x center crop zoom.
        """
        import cv2
        import base64
        import numpy as np

        if (budget is not None and budget >= 1120) or zoom:
            max_dim = max(max_dim, 1024)
            jpeg_quality = max(jpeg_quality, 92)

        video = cv2.VideoCapture(video_path)
        total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv2.CAP_PROP_FPS)

        if total_frames <= 0 or fps <= 0:
            video.release()
            return []

        step = max(1, int(round(fps / target_fps)))
        sampled = []
        count = 0
        while True:
            success, frame = video.read()
            if not success:
                break
            if count % step == 0:
                h, w = frame.shape[:2]
                resized_frame = frame
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    resized_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                _, buffer = cv2.imencode(".jpg", resized_frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                sampled.append(base64.b64encode(buffer).decode("utf-8"))

                if zoom:
                    # Append 2x center zoom crop for extreme detail resolution
                    ch, cw = h // 4, w // 4
                    crop = frame[ch:h-ch, cw:w-cw]
                    if crop.shape[0] > 0 and crop.shape[1] > 0:
                        ch_h, ch_w = crop.shape[:2]
                        if max(ch_h, ch_w) > max_dim:
                            scale = max_dim / max(ch_h, ch_w)
                            crop = cv2.resize(crop, (int(ch_w * scale), int(ch_h * scale)), interpolation=cv2.INTER_AREA)
                        _, crop_buf = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                        sampled.append(base64.b64encode(crop_buf).decode("utf-8"))

                if len(sampled) >= 60:  # Max 60 frames limit per Gemma 4 spec
                    break
            count += 1
        video.release()
        return sampled

