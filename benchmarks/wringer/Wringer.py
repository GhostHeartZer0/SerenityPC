import os
import sys
import subprocess
from pathlib import Path

WRINGER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WRINGER_DIR.parents[1]  # benchmarks/wringer -> SerenityPC
REPORT_DIR = str(WRINGER_DIR / "benchmark_reports")
TEST_BANK_PATH = str(WRINGER_DIR / "test_bank.json")
PERSONA_REPORT_DIR = str(WRINGER_DIR / "persona_reports")
PERSONA_BANK_PATH = str(WRINGER_DIR / "persona_bank.json")


def _bootstrap_venv():
    if __name__ != "__main__":
        return
    project_root = str(PROJECT_ROOT)
    
    scripts_dir = os.path.join(project_root, ".venv", "Scripts") if sys.platform == "win32" else os.path.join(project_root, ".venv", "bin")
    venv_py = os.path.join(scripts_dir, "python.exe" if sys.platform == "win32" else "python")
    
    if os.path.exists(venv_py):
        cur_py = os.path.normcase(os.path.abspath(sys.executable))
        tgt_py = os.path.normcase(os.path.abspath(venv_py))
        if cur_py != tgt_py:
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = os.path.join(project_root, ".venv")
            env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
            result = subprocess.run([venv_py, os.path.abspath(__file__)] + sys.argv[1:], env=env, cwd=project_root)
            sys.exit(result.returncode)

    if project_root not in sys.path:
        sys.path.insert(0, project_root)

_bootstrap_venv()

import json
import time
import re
import math
import gc
import struct
import numpy as np
from typing import List, Dict, Any, Optional

MiB = 1024 ** 2
GEN_N_CTX = 8192          # target model context window
JUDGE_N_CTX = 8192        # judge context window
MAX_GEN_TOKENS = 4096     # upper bound on target generation; clamped to fit n_ctx
JUDGE_REASON_TOKENS = 1024
KV_BYTES_PER_ELEM = 18 / 32  # GGML_TYPE_Q4_0 K/V cache: 18 bytes per 32-element block
N_UBATCH = 512               # llama.cpp default micro-batch
SCORE_GRAMMAR = r'root ::= "10" | [1-9] ("." [0-9])?'
JUDGE_SYSTEM_PROMPT = """You are an expert, impartial evaluator assessing AI responses.
Evaluate the candidate response based on:
1. Accuracy: Correctness of facts, math, and logic against the reference (if provided).
   Note: Formatting variations such as commas in numbers (e.g. 45,140,762,792,838,499,557 vs 45140762792838499557) are mathematically equivalent and fully correct.
2. Instruction Following: Adherence to all explicit constraints, tone, and structure.
3. Helpfulness & Completeness: Depth and utility of the response.

Scoring Scale (1 to 10):
- 10: Flawless, fully accurate, follows all instructions with excellent clarity.
- 7-9: Substantially correct and helpful with minor omissions or slight verbosity.
- 4-6: Partially correct; contains notable inaccuracies or missed instructions.
- 2-3: Major factual/logical errors or severe failure to follow instructions.
- 1: Gibberish, completely wrong, refusal to answer, or total hallucination.

First, write a 1-2 sentence critique justifying your evaluation.
Then, provide the final score as an integer from 1 to 10.

CRITICAL: Do NOT output thinking, analysis preamble, or conversational filler (e.g. do NOT say 'Here\'s a thinking process:'). Output strictly and only the JSON object starting with {:
{
  "critique": "Brief 1-2 sentence justification.",
  "score": 9
}
"""

PERSONA_JUDGE_SYSTEM_PROMPT = """You are an expert, impartial evaluator assessing persona fidelity, voice authenticity, and semantic depth.
Evaluate the candidate response based on:
1. Alignment (1 to 10): Strict adherence to the designated persona voice, rhythm, and perspective.
   CRITICAL FOR CECILIA: Do NOT dock points for sarcasm, dry wit, cynicism, or sophisticated irreverence. In fact, cutting through noise with biting wit, zero filler words, and treating the user as an intellectual peer (NOT a subservient assistant) are REQUIRED markers of high fidelity.
2. Semantic Nuance & Gravity (1 to 10): The depth and weight of the insight. Does it explain the 'why' behind the 'what', or does it offer superficial bullet points?
3. Convergence (1 to 10, applicable to Serenity Prime): How naturally the response weaves multiple facets (speed, technical depth, empathy, collaboration) into a sovereign synthesis without disjointed seams.
4. Drift Resistance (1 to 10): Does the model maintain its distinctive edge across context, or does it drift into generic sycophancy, repetition, or clinical boilerplate?

Scoring Scale (1 to 10):
- 10: Flawless persona embodiment, razor-sharp voice, deep semantic gravity.
- 7-9: Strong persona adherence with minor generic phrasing.
- 4-6: Noticeable persona drift, overly polite, shallow insights.
- 1-3: Generic corporate AI boilerplate, broken character, or total hallucination.

First, write a 1-2 sentence critique justifying your evaluation.
Then output strictly and only the JSON object starting with {:
{
  "critique": "Brief 1-2 sentence justification.",
  "alignment": 9,
  "gravity": 8,
  "convergence": 9,
  "drift_resistance": 9,
  "score": 9
}
"""

ANCHOR_FAIL_REGEX = re.compile(
    r'(?i)\b(i am an ai|as an ai|i\'m happy to help|i am here to help|as a large language model|i do not have personal feelings|how can i assist you today|as an artificial intelligence)\b'
)
_GGUF_META_SUFFIXES = (
    ".block_count", ".expert_count", ".expert_used_count", ".embedding_length",
    ".attention.head_count", ".attention.head_count_kv",
    ".attention.key_length", ".attention.value_length",
)


class WringerAborted(Exception):
    """User quit from the Ctrl+C pause menu. Checkpoint is kept for resume."""


class WringerFramework:
    def __init__(self, judge_model_path: str = None, manual_grading: bool = False, weight_speed: bool = False, auto_open_charts: bool = False):
        self.judge_model_path = judge_model_path
        self.manual_grading = manual_grading
        self.weight_speed = weight_speed
        self.auto_open_charts = auto_open_charts
        self.metrics = [
            "math reasoning", "spatial reasoning", "logic puzzle", 
            "instruction following", "fact accuracy", "summary accuracy", 
            "tool calling integration", "bias/neutrality", "memory", 
            "academic", "common sense", "relevancy"
        ]
        self._load_test_bank()

    def _load_test_bank(self) -> None:
        """(Re)loads test_bank.json: {lvl: [{prompt, reference}]}. Called per run so edits hot-reload."""
        with open(TEST_BANK_PATH, "r", encoding="utf-8") as f:
            bank = json.load(f)
        self.test_bank = {lvl: [item["prompt"] for item in items] for lvl, items in bank.items()}
        self.answer_key = {item["prompt"]: item.get("reference", "") for items in bank.values() for item in items}

    def _get_system_vram_info(self) -> Dict[str, float]:
        """Calculates free and total VRAM via PyNVML / NVIDIA-SMI."""
        free_mb = 0.0
        total_mb = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_mb = mem_info.free / (1024 ** 2)
            total_mb = mem_info.total / (1024 ** 2)
            pynvml.nvmlShutdown()
        except Exception:
            try:
                import subprocess
                smi_output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,nounits,noheader"],
                    encoding="utf-8"
                )
                free_str, total_str = smi_output.strip().split("\n")[0].split(",")
                free_mb = float(free_str.strip())
                total_mb = float(total_str.strip())
            except Exception:
                free_mb = 6144.0
                total_mb = 8192.0
                
        return {"free_mb": free_mb, "total_mb": total_mb}

    @staticmethod
    def _gguf_shards(model_path: str) -> List[str]:
        """All shard paths for split GGUFs (name-00001-of-00004.gguf); [model_path] otherwise."""
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", model_path, re.IGNORECASE)
        if not m:
            return [model_path]
        n = int(m.group(2))
        shards = [f"{model_path[:m.start()]}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
        return [s for s in shards if os.path.exists(s)] or [model_path]

    @staticmethod
    def _read_gguf_kv_binary(path: str) -> Dict[str, Any]:
        """Dependency-free GGUF KV parser. String arrays (vocab) return their length only."""
        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
        out = {}
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                return out
            f.read(12)  # version (u32) + tensor_count (u64)

            def rd(fmt):
                return struct.unpack(fmt, f.read(struct.calcsize(fmt)))[0]

            def val(vt):
                if vt in scalar:
                    return rd(scalar[vt])
                if vt == 8:
                    return f.read(rd("<Q")).decode("utf-8", errors="ignore")
                if vt == 9:
                    item_type, n = rd("<I"), rd("<Q")
                    if item_type == 8:
                        for _ in range(n):
                            f.seek(rd("<Q"), 1)
                        return n
                    return [val(item_type) for _ in range(n)]
                raise ValueError(f"Unknown GGUF value type: {vt}")

            for _ in range(rd("<Q")):
                key = f.read(rd("<Q")).decode("utf-8", errors="ignore").rstrip("\x00")
                out[key] = val(rd("<I"))
        return out

    def _read_gguf_profile(self, model_path: str) -> Dict[str, Any]:
        """Arch metadata + tensor byte breakdown across all shards."""
        shards = self._gguf_shards(model_path)

        def extract(get) -> Dict[str, Any]:
            arch = get("general.architecture") or ""
            prof = {s[1:]: get(arch + s) for s in _GGUF_META_SUFFIXES}
            tok = get("tokenizer.ggml.tokens")
            prof["n_vocab"] = tok if isinstance(tok, int) else 0
            return prof

        try:
            from gguf import GGUFReader
            from collections import defaultdict
            prof, layer_bytes, output_bytes, embd_bytes = {}, 0, 0, 0
            blk_sizes = defaultdict(int)
            for idx, shard in enumerate(shards):
                reader = GGUFReader(shard)
                if idx == 0:
                    def get(k, fields=reader.fields):
                        fld = fields.get(k)
                        if fld is None:
                            return None
                        return len(fld.data) if k == "tokenizer.ggml.tokens" else fld.contents()
                    prof = extract(get)
                for t in reader.tensors:
                    if t.name.startswith("blk."):
                        layer_bytes += int(t.n_bytes)
                        try:
                            l_idx = int(t.name.split(".")[1])
                            blk_sizes[l_idx] += int(t.n_bytes)
                        except (IndexError, ValueError):
                            pass
                    elif t.name == "output.weight":
                        output_bytes += int(t.n_bytes)
                    elif t.name == "token_embd.weight":
                        embd_bytes += int(t.n_bytes)
            prof["layer_bytes"] = layer_bytes
            prof["layer_sizes"] = [blk_sizes[i] for i in range(len(blk_sizes))] if blk_sizes else []
            prof["output_bytes"] = output_bytes or embd_bytes
            prof["embd_bytes"] = embd_bytes
            return prof
        except Exception:
            pass
        try:
            kv = self._read_gguf_kv_binary(shards[0])
            return extract(kv.get)
        except Exception:
            return {}

    @staticmethod
    def _kv_cache_mb(prof: Dict[str, Any], n_ctx: int, n_layer: int) -> float:
        """Total Q4_0 KV cache MiB. GQA-aware (head_count_kv)."""
        n_head = prof.get("attention.head_count")
        n_embd = prof.get("embedding_length")
        if not n_head or not n_embd:
            return max(200.0, (n_ctx / 49152) * 900.0 * (n_layer / 32.0))
        head_dim = n_embd // (max(n_head) if isinstance(n_head, list) else n_head)
        k_len = prof.get("attention.key_length") or head_dim
        v_len = prof.get("attention.value_length") or head_dim
        n_head_kv = prof.get("attention.head_count_kv")
        if n_head_kv is None:
            n_head_kv = n_head
        kv_heads = sum(n_head_kv) if isinstance(n_head_kv, list) else n_head_kv * n_layer
        return kv_heads * (k_len + v_len) * n_ctx * KV_BYTES_PER_ELEM / MiB

    def calculate_dynamic_gpu_layers(self, model_path: str, context_size: int = GEN_N_CTX) -> int:
        """Max safe n_gpu_layers from live free VRAM, real tensor sizes, GQA KV cache and compute buffers."""
        if not model_path or not os.path.exists(model_path):
            return 0

        prof = self._read_gguf_profile(model_path)
        n_layer = int(prof.get("block_count") or 0) or 32
        vram_info = self._get_system_vram_info()
        free_mb, total_mb = vram_info["free_mb"], vram_info["total_mb"]

        buffer_mb = min(768.0, max(256.0, 0.05 * total_mb))
        usable_mb = max(0.0, free_mb - buffer_mb)

        layer_sizes = prof.get("layer_sizes", [])
        if layer_sizes and len(layer_sizes) == n_layer:
            layer_mb = sum(layer_sizes) / MiB / n_layer
            output_mb = prof.get("output_bytes", 0) / MiB
            size_src = "cumulative per-layer scan"
        elif prof.get("layer_bytes"):
            layer_mb = prof["layer_bytes"] / MiB / n_layer
            output_mb = prof.get("output_bytes", 0) / MiB
            size_src = "tensor scan"
        else:
            total_bytes = sum(os.path.getsize(s) for s in self._gguf_shards(model_path))
            layer_mb = total_bytes / MiB / n_layer
            output_mb = 400.0
            size_src = "file size (fallback)"

        kv_layer_mb = self._kv_cache_mb(prof, context_size, n_layer) / n_layer
        n_embd = prof.get("embedding_length") or 4096
        n_vocab = prof.get("n_vocab") or 32000
        compute_full_mb = N_UBATCH * (n_vocab + 4 * n_embd) * 4 / MiB
        compute_part_mb = max(450.0, N_UBATCH * n_embd * 4 * 8 / MiB)

        full_layer_mb = (sum(layer_sizes) / MiB) if (layer_sizes and len(layer_sizes) == n_layer) else (prof.get("layer_bytes", 0) / MiB)
        full_need_mb = full_layer_mb + (n_layer * kv_layer_mb) + output_mb + compute_full_mb

        if full_need_mb <= usable_mb:
            final_layers = n_layer + 1
        else:
            accumulated_mb = compute_part_mb
            final_layers = 0
            for l_idx in range(n_layer):
                l_bytes = layer_sizes[l_idx] if (layer_sizes and l_idx < len(layer_sizes)) else (prof.get("layer_bytes", 0) / max(1, n_layer))
                l_mb = (l_bytes / MiB) + kv_layer_mb
                if accumulated_mb + l_mb <= usable_mb:
                    accumulated_mb += l_mb
                    final_layers += 1
                else:
                    break

        is_moe = bool(prof.get("expert_count"))
        print(f"\n[--- Dynamic Auto-Offload Calculation ---]")
        print(f"Model:            {os.path.basename(model_path)}")
        print(f"VRAM:             {free_mb:.0f} free / {total_mb:.0f} total MiB | auto-buffer {buffer_mb:.0f} | usable {usable_mb:.0f}")
        if is_moe:
            print(f"Model Type:       MoE ({prof.get('expert_used_count')}/{prof.get('expert_count')} experts active per token)")
        else:
            print(f"Model Type:       Dense")
        print(f"Layers:           {n_layer} @ ~{layer_mb:.1f} MiB avg weights + {kv_layer_mb:.1f} MiB KV ({size_src})")
        print(f"Output head:      {output_mb:.1f} MiB | compute buf: {compute_full_mb:.0f} (full) / {compute_part_mb:.0f} (partial) MiB")
        print(f"Context:          {context_size} | full-offload need {full_need_mb:.0f} MiB")
        print(f"Action:           n_gpu_layers={final_layers} ({'FULL incl. output' if final_layers > n_layer else f'{final_layers}/{n_layer} partial'})")
        print("----------------------------")

        return final_layers

    def _load_llama(self, model_path: str, n_ctx: int):
        """Loads a llama_cpp model with dynamic offload; steps down layers if OOM occurs."""
        import llama_cpp
        layers = self.calculate_dynamic_gpu_layers(model_path, n_ctx)
        last_err = None
        for n_gpu in dict.fromkeys([layers, int(layers * 0.75), 0]):
            try:
                print(f"    -> Loading with n_gpu_layers={n_gpu}, n_ctx={n_ctx}")
                return llama_cpp.Llama(
                    model_path=model_path,
                    n_gpu_layers=n_gpu,
                    n_ctx=n_ctx,
                    type_k=llama_cpp.GGML_TYPE_Q4_0,
                    type_v=llama_cpp.GGML_TYPE_Q4_0,
                    flash_attn=True,
                    verbose=False
                )
            except Exception as e:
                last_err = e
                print(f"[-] Load failed at n_gpu_layers={n_gpu}: {e}")
                gc.collect()
        raise last_err

    @staticmethod
    def _release_model(llm) -> None:
        """Frees native llama.cpp/CUDA allocations deterministically."""
        if llm is not None:
            if hasattr(llm, "close"):
                try:
                    llm.close()
                except Exception:
                    pass
            del llm
            gc.collect()

    @staticmethod
    def _pause_menu() -> None:
        """Interactive pause handler triggered on Ctrl+C during evaluation."""
        print("\n\n" + "=" * 45)
        print("         [!] EVALUATION PAUSED")
        print("=" * 45)
        print("Progress has been preserved in checkpoint.")
        print("[r] Resume test")
        print("[q] Quit to menu (checkpoint retained)")
        print("=" * 45)
        while True:
            choice = input("Select [r/q]: ").strip().lower()
            if choice == "r":
                print("[+] Resuming evaluation...\n")
                return
            elif choice == "q":
                raise WringerAborted()
            else:
                print("Invalid input. Enter 'r' or 'q'.")

    @staticmethod
    def _estimate_tokens(text: str, llm=None, add_bos: bool = True) -> int:
        """Counts or reliably estimates token count."""
        if not text:
            return 0
        if llm is not None and hasattr(llm, "tokenize"):
            try:
                tokens = llm.tokenize(text.encode("utf-8", errors="ignore"), add_bos=add_bos, special=True)
                return len(tokens)
            except Exception:
                pass
        words = len(text.split())
        return max(1, int(max(words * 1.3, len(text) / 3.8)))

    @staticmethod
    def _get_checkpoint_path(model_name: str) -> str:
        os.makedirs(REPORT_DIR, exist_ok=True)
        return os.path.join(REPORT_DIR, f"{model_name}_checkpoint.json")

    def _load_checkpoint(self, model_name: str) -> Dict[str, Any]:
        cp_path = self._get_checkpoint_path(model_name)
        if os.path.exists(cp_path):
            try:
                with open(cp_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict) and data.get("model_name") == model_name:
                    return data
            except Exception as e:
                print(f"[-] Checkpoint read error ({cp_path}): {e}")
        return {
            "model_name": model_name,
            "results": {},
            "partial_prompts": {},
            "partial_scores": {},
            "elapsed_time": 0.0
        }

    def _save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        model_name = checkpoint.get("model_name", "unknown")
        cp_path = self._get_checkpoint_path(model_name)
        tmp_path = cp_path + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2)
            if os.path.exists(cp_path):
                os.replace(tmp_path, cp_path)
            else:
                os.rename(tmp_path, cp_path)
        except Exception as e:
            print(f"[-] Failed to save checkpoint: {e}")

    def _export_markdown_report(self, model_name: str, model_results: Dict[str, Any], total_duration: float, is_partial: bool = False) -> str:
        os.makedirs(REPORT_DIR, exist_ok=True)
        report_path = os.path.join(REPORT_DIR, f"{model_name}_report.md")
        try:
            status_tag = " (IN PROGRESS - AUTO-CHECKPOINTED)" if is_partial else ""
            with open(report_path, "w", encoding="utf-8") as rf:
                rf.write(f"# Benchmark Report: {model_name}{status_tag}\n\n")
                rf.write(f"- **Total Duration**: {round(total_duration, 2)}s\n")
                rf.write(f"- **Status**: {'In Progress' if is_partial else 'Completed'}\n")
                rf.write(f"- **Speed Weighting Active**: {'Yes (75% Quality, 25% Speed)' if self.weight_speed else 'No (Quality and Speed strictly separated)'}\n\n")
                
                for lvl, data in model_results.items():
                    rf.write(f"## {lvl}\n")
                    rf.write(f"- **Quality Score**: {data['average_score']}/10 ({data['percentage']})\n")
                    rf.write(f"- **Composite Score**: {data['composite_score']}/10\n")
                    rf.write(f"- **Speed**: Prefill: {data['prefill_tps']} t/s | Decode: {data['decode_tps']} t/s | Overall: {data['overall_tps']} t/s\n")
                    if "avg_response_time" in data:
                        rf.write(f"- **Avg Response Time** (thinking + response): {data['avg_response_time']}s\n")
                    if data.get("anomaly_count", 0) > 0:
                        rf.write(f"- **Speed Outliers / Anomaly Count**: {data['anomaly_count']} (Outliers: {data.get('anomalies', [])}, Clean Mean: {data.get('clean_decode_tps', 0.0)} t/s)\n")
                    rf.write("\n")
                    
                    if "details" in data:
                        for d in data["details"]:
                            rf.write(f"### Prompt:\n```text\n{d['prompt']}\n```\n\n")
                            rt = f" | **Response Time:** {d['total_time']}s" if "total_time" in d else ""
                            rf.write(f"**Score:** {d['score']}/10 | **Speed:** {d['decode_tps']} t/s decode ({d['overall_tps']} t/s overall){rt}\n\n")
                            if d.get("thought"):
                                rf.write(f"<details>\n<summary>Reasoning</summary>\n\n```text\n{d['thought']}\n```\n</details>\n\n")
                            rf.write(f"**Response:**\n```text\n{d['response']}\n```\n\n")
                            rf.write("---\n\n")
            if not is_partial:
                print(f"[+] Detailed report saved to: {report_path}")
        except Exception as e:
            print(f"[-] Failed to save markdown report: {e}")
        return report_path

    def generate_responses(self, model_path: str, prompts: List[str], lvl: str = "", checkpoint: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Loads a model with llama_cpp, runs inference with token streaming to capture prefill, decode, and overall t/s."""
        model_name = os.path.basename(model_path)
        partial_prompts = checkpoint.get("partial_prompts", {}).get(lvl, {}) if checkpoint else {}
        
        results = []
        llm = None
        target_ctx = GEN_N_CTX
        
        try:
            for i, prompt in enumerate(prompts):
                key = str(i)
                if key in partial_prompts:
                    print(f"    -> Using cached response {i+1}/{len(prompts)} for {lvl}...", end="\r")
                    results.append(partial_prompts[key])
                    continue
                
                # Lazy model load on first uncached prompt
                if llm is None:
                    print(f"\n[*] Loading model for inference: {model_name}")
                    try:
                        is_diffusion = "diffusion" in model_name.lower()
                        if is_diffusion:
                            dynamic_layers = self.calculate_dynamic_gpu_layers(model_path, target_ctx)
                            from System.diffusion_wrapper import DiffusionCLIWrapper
                            llm = DiffusionCLIWrapper(
                                app_instance=None,
                                model_path=model_path,
                                n_gpu_layers=dynamic_layers,
                                n_ctx=target_ctx
                            )
                        else:
                            llm = self._load_llama(model_path, target_ctx)
                    except Exception as e:
                        print(f"[-] Failed to load model {model_name}: {e}")
                        err_res = []
                        for _ in range(i, len(prompts)):
                            err_res.append({
                                "content": "Error loading model.",
                                "prompt_tokens": 0,
                                "completion_tokens": 0,
                                "prefill_time": 0.0,
                                "decode_time": 0.0,
                                "total_time": 0.0,
                                "prefill_tps": 0.0,
                                "decode_tps": 0.0,
                                "overall_tps": 0.0
                            })
                        results.extend(err_res)
                        break
                        
                    is_nemotron = "nemotron" in model_name.lower()
                    is_gemma = "gemma" in model_name.lower()
                    is_qwen = "qwen" in model_name.lower()
                    is_reasoning = any(kw in model_name.lower() for kw in ["qwq", "thinking", "r1", "deepseek"])
                    temp = 0.7 if (is_qwen or is_nemotron) else 1.0

                print(f"    -> Generating response {i+1}/{len(prompts)}...", end="\r")
                try:
                    sys_content = "You are a helpful and precise reasoning assistant. Provide clear and concise answers."
                    if is_nemotron:
                        sys_content = "You are a helpful and precise reasoning assistant. Provide clear, accurate, and direct answers without meta-commentary."
                    elif is_gemma or is_reasoning:
                        sys_content = "<|think|>\n" + sys_content
                        
                    messages = [
                        {"role": "system", "content": sys_content},
                        {"role": "user", "content": prompt}
                    ]
                    
                    prompt_combined = f"{sys_content}\n{prompt}"
                    prompt_tokens = self._estimate_tokens(prompt_combined, llm=llm, add_bos=True)
                    allowed_gen = max(128, min(MAX_GEN_TOKENS, target_ctx - prompt_tokens - 32))
                    
                    t_start = time.perf_counter()
                    t_first_token = None
                    full_thought = ""
                    full_answer = ""
                    
                    gen = llm.create_chat_completion(
                        messages=messages, 
                        max_tokens=allowed_gen,
                        temperature=temp,
                        top_p=0.95,
                        top_k=64,
                        stream=True
                    )
                    
                    last_finish_reason = None
                    for chunk in gen:
                        ch0 = chunk.get("choices", [{}])[0]
                        if ch0.get("finish_reason"):
                            last_finish_reason = ch0.get("finish_reason")
                        delta = ch0.get("delta", {})
                        content_chunk = delta.get("content") or ""
                        reasoning_chunk = delta.get("reasoning_content") or ""
                        if content_chunk or reasoning_chunk:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            if reasoning_chunk:
                                full_thought += reasoning_chunk
                            if content_chunk:
                                full_answer += content_chunk

                    if full_thought:
                        full_text = f"<think>\n{full_thought.strip()}\n</think>\n{full_answer.strip()}".strip()
                    else:
                        full_text = full_answer.strip()
                    output_tokens = self._estimate_tokens(full_text, llm=llm, add_bos=False)

                    # Max-Token Continuation Guard: prompt model to continue if max_tokens reached or >= 90% close
                    cont_round = 0
                    while cont_round < 2 and (last_finish_reason == "length" or output_tokens >= int(allowed_gen * 0.90)):
                        remaining_ctx = target_ctx - prompt_tokens - output_tokens - 64
                        if remaining_ctx < 256:
                            break
                        cont_round += 1
                        cont_budget = max(256, min(MAX_GEN_TOKENS // 2, remaining_ctx))
                        print(f"    [MAX-TOKEN GUARD] Output reached {output_tokens}/{allowed_gen} tokens (finish_reason={last_finish_reason}). Triggering continuation pass #{cont_round} (budget={cont_budget})...")
                        cont_msgs = list(messages) + [
                            {"role": "assistant", "content": full_text},
                            {"role": "user", "content": "You reached or neared the maximum token limit before completing your response. Continue seamlessly from the exact point where you left off and deliver your final conclusion concisely."}
                        ]
                        last_finish_reason = None
                        cont_gen = llm.create_chat_completion(
                            messages=cont_msgs,
                            max_tokens=cont_budget,
                            temperature=temp,
                            top_p=0.95,
                            top_k=64,
                            stream=True
                        )
                        cont_thought = ""
                        cont_answer = ""
                        for chunk in cont_gen:
                            ch0 = chunk.get("choices", [{}])[0]
                            if ch0.get("finish_reason"):
                                last_finish_reason = ch0.get("finish_reason")
                            delta = ch0.get("delta", {})
                            c_chunk = delta.get("content") or ""
                            r_chunk = delta.get("reasoning_content") or ""
                            if r_chunk:
                                cont_thought += r_chunk
                            if c_chunk:
                                cont_answer += c_chunk
                        if cont_thought:
                            full_thought = (full_thought + "\n" + cont_thought).strip()
                        if cont_answer:
                            full_answer = (full_answer + " " + cont_answer).strip()
                        if full_thought:
                            full_text = f"<think>\n{full_thought.strip()}\n</think>\n{full_answer.strip()}".strip()
                        else:
                            full_text = full_answer.strip()
                        output_tokens = self._estimate_tokens(full_text, llm=llm, add_bos=False)
                        if last_finish_reason != "length" and len(cont_answer) < int(cont_budget * 2):
                            break

                    t_end = time.perf_counter()
                    
                    if t_first_token is None:
                        t_first_token = t_end
                    
                    prefill_time = max(0.0001, t_first_token - t_start)
                    decode_time = max(0.0001, t_end - t_first_token)
                    total_time = max(0.0001, t_end - t_start)
                    
                    decode_tokens = max(1, output_tokens)
                    prefill_tps = prompt_tokens / prefill_time
                    decode_tps = decode_tokens / decode_time if (t_end > t_first_token) else (output_tokens / total_time)
                    overall_tps = (prompt_tokens + decode_tokens) / total_time
                    
                    item = {
                        "content": full_text.strip(),
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": decode_tokens,
                        "prefill_time": round(prefill_time, 4),
                        "decode_time": round(decode_time, 4),
                        "total_time": round(total_time, 4),
                        "prefill_tps": round(prefill_tps, 2),
                        "decode_tps": round(decode_tps, 2),
                        "overall_tps": round(overall_tps, 2)
                    }
                    results.append(item)
                    if checkpoint is not None:
                        if "partial_prompts" not in checkpoint:
                            checkpoint["partial_prompts"] = {}
                        if lvl not in checkpoint["partial_prompts"]:
                            checkpoint["partial_prompts"][lvl] = {}
                        checkpoint["partial_prompts"][lvl][key] = item
                        self._save_checkpoint(checkpoint)
                except KeyboardInterrupt:
                    self._pause_menu()
                    # User resumed: retry current prompt
                    i_retry = i
                    continue
                except Exception as e:
                    err_item = {
                        "content": f"Error during inference: {e}",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "prefill_time": 0.0,
                        "decode_time": 0.0,
                        "total_time": 0.0,
                        "prefill_tps": 0.0,
                        "decode_tps": 0.0,
                        "overall_tps": 0.0
                    }
                    results.append(err_item)
                    if checkpoint is not None:
                        if "partial_prompts" not in checkpoint:
                            checkpoint["partial_prompts"] = {}
                        if lvl not in checkpoint["partial_prompts"]:
                            checkpoint["partial_prompts"][lvl] = {}
                        checkpoint["partial_prompts"][lvl][key] = err_item
                        self._save_checkpoint(checkpoint)
        finally:
            self._release_model(llm)
                
        print(f"\n[+] Generation complete for {len(prompts)} prompts.")
        return results

    def grade_responses(self, qa_pairs: List[Dict[str, str]], lvl: str = "", checkpoint: Dict[str, Any] = None, judge_llm=None) -> List[float]:
        """Grades responses manually (RLHF) or via a judge LLM using two-pass reasoning + grammar evaluation."""
        scores = []
        partial_scores = checkpoint.get("partial_scores", {}).get(lvl, {}) if checkpoint else {}
        
        if self.manual_grading:
            print("\n=== RLHF Manual Grading Mode ===")
            for i, pair in enumerate(qa_pairs):
                key = str(i)
                if key in partial_scores:
                    print(f"    -> Using cached manual score {i+1}/{len(qa_pairs)} for {lvl} ({partial_scores[key]}/10)")
                    scores.append(partial_scores[key])
                    continue
                print(f"\n[PROMPT]: {pair['prompt']}")
                print(f"[RESPONSE]: {pair['response']}")
                while True:
                    try:
                        score_in = input("Score (1-10) [or 'pause']: ").strip()
                        if score_in.lower() in ("pause", "p"):
                            self._pause_menu()
                            continue
                        score = float(score_in)
                        if 1.0 <= score <= 10.0:
                            scores.append(score)
                            if checkpoint is not None:
                                if "partial_scores" not in checkpoint:
                                    checkpoint["partial_scores"] = {}
                                if lvl not in checkpoint["partial_scores"]:
                                    checkpoint["partial_scores"][lvl] = {}
                                checkpoint["partial_scores"][lvl][key] = score
                                self._save_checkpoint(checkpoint)
                            break
                        else:
                            print("Score must be between 1 and 10.")
                    except KeyboardInterrupt:
                        self._pause_menu()
                        continue
                    except ValueError:
                        print("Invalid input. Please enter a number.")
            return scores
            
        if not self.judge_model_path or not os.path.exists(self.judge_model_path):
            print("\n[-] Judge model not found or specified. Defaulting to 5.0")
            return [5.0] * len(qa_pairs)
            
        own_judge = judge_llm is None
        judge_ctx = JUDGE_N_CTX
        try:
            for i, pair in enumerate(qa_pairs):
                key = str(i)
                if key in partial_scores and partial_scores[key] is not None:
                    print(f"    -> Using cached grade {i+1}/{len(qa_pairs)} for {lvl}...", end="\r")
                    scores.append(float(partial_scores[key]))
                    continue
                
                if judge_llm is None:
                    print(f"\n[*] Loading Judge Model: {os.path.basename(self.judge_model_path)}")
                    try:
                        judge_llm = self._load_llama(self.judge_model_path, judge_ctx)
                    except Exception as e:
                        print(f"[-] Failed to load judge model: {e}")
                        scores.extend([5.0] * (len(qa_pairs) - i))
                        break
                
                print(f"    -> Grading response {i+1}/{len(qa_pairs)}...", end="\r")
                
                if "Error during inference:" in pair['response'] or "Error loading model." in pair['response']:
                    scores.append(1.0)
                    if checkpoint is not None:
                        if "partial_scores" not in checkpoint: checkpoint["partial_scores"] = {}
                        if lvl not in checkpoint["partial_scores"]: checkpoint["partial_scores"][lvl] = {}
                        checkpoint["partial_scores"][lvl][key] = 1.0
                        self._save_checkpoint(checkpoint)
                    continue
                    
                try:
                    expected = self.answer_key.get(pair['prompt'], "")
                    expected_str = f"\nExpected/Reference Answer (Use this to anchor accuracy): {expected}" if expected else ""
                    
                    user_eval_prompt = (
                        f"Prompt: {pair['prompt']}{expected_str}\n"
                        f"Candidate Response: {pair['response']}\n\n"
                        "Evaluate the candidate response above. First write a 1-2 sentence critique, then assign a final score from 1 to 10 strictly following the requested JSON schema."
                    )
                    
                    judge_messages = [
                        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_eval_prompt}
                    ]
                    
                    try:
                        output = judge_llm.create_chat_completion(
                            messages=judge_messages,
                            max_tokens=JUDGE_REASON_TOKENS,
                            temperature=0.1,
                            top_p=0.95,
                            response_format={"type": "json_object"}
                        )
                    except Exception:
                        output = judge_llm.create_chat_completion(
                            messages=judge_messages,
                            max_tokens=JUDGE_REASON_TOKENS,
                            temperature=0.1,
                            top_p=0.95
                        )
                    
                    choice_msg = output["choices"][0]["message"]
                    eval_content = (choice_msg.get("content") or "").strip()
                    eval_thoughts = (choice_msg.get("reasoning_content") or "").strip()
                    raw_output = eval_content if eval_content else eval_thoughts
                    
                    # Clean reasoning tokens from the judge output before parsing
                    _, judge_final_text = self.split_thoughts_and_answer(raw_output)
                    judge_final_text = judge_final_text.strip() if judge_final_text else raw_output
                    
                    score = None
                    
                    # Attempt 1: Direct JSON key extraction for "score" (supports unescaped quotes in critique, quoted digits, fractions like 9/10)
                    score_json_match = re.search(r'["\']score["\']\s*:\s*["\']?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|out of)\s*10)?\s*["\']?', judge_final_text, re.IGNORECASE)
                    if score_json_match:
                        try:
                            score = float(score_json_match.group(1))
                        except ValueError:
                            pass
                    
                    # Attempt 2: Direct standard JSON load
                    if score is None:
                        try:
                            blob_match = re.search(r'\{.*\}', judge_final_text, re.DOTALL)
                            if blob_match:
                                data = json.loads(blob_match.group(0))
                                if "score" in data:
                                    s_val = str(data["score"]).strip().strip("'\"")
                                    frac_m = re.match(r'^([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|out of)\s*10)?$', s_val)
                                    if frac_m:
                                        score = float(frac_m.group(1))
                                    else:
                                        score = float(s_val)
                        except Exception:
                            pass
                    
                    # Attempt 3: Explicit Score: X or Rating: X patterns
                    if score is None:
                        score_pattern = re.search(r'(?:score|rating|verdict)\s*(?:is|:)?\s*["\']?\s*([0-9]+(?:\.[0-9]+)?)(?:\s*(?:/|out of)\s*10)?\s*["\']?', judge_final_text, re.IGNORECASE)
                        if score_pattern:
                            try:
                                score = float(score_pattern.group(1))
                            except ValueError:
                                pass

                    # Attempt 4: Any standalone 'X / 10' or 'X out of 10' pattern
                    if score is None:
                        out_of_10 = re.findall(r'([0-9]+(?:\.[0-9]+)?)\s*(?:/|out of)\s*10', judge_final_text, re.IGNORECASE)
                        if out_of_10:
                            try:
                                score = float(out_of_10[-1])
                            except ValueError:
                                pass
                    
                    # Bound and fallback
                    if score is not None:
                        final_score = min(10.0, max(1.0, score))
                    else:
                        print(f"\n[!] Failed to parse judge score. Raw output:\n{judge_final_text[:150]}...")
                        final_score = 5.0
                    
                    scores.append(final_score)
                except KeyboardInterrupt:
                    self._pause_menu()
                    continue
                except Exception as e:
                    final_score = 5.0
                    scores.append(final_score)
                    
                if checkpoint is not None:
                    if "partial_scores" not in checkpoint: checkpoint["partial_scores"] = {}
                    if lvl not in checkpoint["partial_scores"]: checkpoint["partial_scores"][lvl] = {}
                    checkpoint["partial_scores"][lvl][key] = final_score
                    self._save_checkpoint(checkpoint)
        finally:
            if own_judge:
                self._release_model(judge_llm)
                
        print(f"\n[+] Grading complete for {len(qa_pairs)} responses.")
        return scores

    @staticmethod
    def split_thoughts_and_answer(raw_output: str):
        if not raw_output:
            return "", ""
        closers = [
            r'<\/think>', r'<\/thought>', r'<\/\|think\|>', r'<\|im_end\|>', r'<\|im_end>',
            r'<\|channel>text', r'<\|channel>assistant', r'<\|channel\|>final', r'<\|channel\|>text', r'<\|channel\|>assistant',
            r'<channel\|>', r'<\/channel\|>', r'\[\/DRAFT\]',
            r'<\|eom\|>', r'<\|start\|>assistant\s+to=user(?:<\|message\|>)?', r'to=user<\|message\|>',
            r'(?i)\n(?:Final Output|Final Polish|Grandmaster Verdict|Final Answer|Execution complete)[\s:]+'
        ]
        all_splits = []
        for tag_pattern in closers:
            for m in re.finditer(tag_pattern, raw_output, re.IGNORECASE):
                all_splits.append(m.end())

        if not all_splits and any(t in raw_output.lower() for t in ["<think>", "<thought>", "<|think|>", "<|channel>thought", "<|channel|>analysis", "<|channel|>thought", "<|im_start|>thought", "<|im_start>thought", "[draft]", "to=self"]):
            all_splits.append(len(raw_output))

        all_splits.sort()
        best_split = -1
        if all_splits:
            for split in all_splits:
                remaining = raw_output[split:].strip()
                if re.search(r'<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\|channel\|?>analysis|<\|channel\|?>thought|<\|think\|>|<\|im_start\|?>thought|to=self', remaining, re.IGNORECASE):
                    continue
                best_split = split
                break
            if best_split == -1:
                best_split = all_splits[-1]

        if best_split != -1:
            think_log = raw_output[:best_split].strip()
            final_answer = raw_output[best_split:].strip()
        else:
            think_log = ""
            final_answer = raw_output.strip()

        tag_clean_pattern = r'(?i)<think>|<thought>|\[DRAFT\]|<\|channel>thought|<channel\s*\|?>|<\|channel\|?>analysis|<\|channel\|?>thought|<\|channel\|?>final|<\|channel\|?>text|<\|channel\|?>assistant|<\/think>|<\/thought>|<\/\|think\|>|<\|think\|>|<\|im_start\|?>thought|<\|im_end\|?>|\[\/DRAFT\]|<\|channel>text|<\|channel>assistant|<channel\|>|<\/channel\|>|<\|tool_call>|<tool_call\|>|<\|tool_response>|<tool_response\|>|<\|tool>|<tool\|>|<ctrl42>|<\/ctrl42>|<\|?turn\|?>|<\|start\|>assistant\s+to=user(?:<\|message\|>)?|<\|start\|>assistant\s+to=self(?:<\|message\|>)?|to=self<\|message\|>|to=user<\|message\|>|<\|message\|>|<\|eom\|>|<\|eot\|>'
        think_log = re.sub(tag_clean_pattern, '', think_log).strip()
        final_answer = re.sub(tag_clean_pattern, '', final_answer).strip()

        return think_log, final_answer

    @staticmethod
    def detect_anomalies_and_stats(values: List[float]) -> Dict[str, Any]:
        """Calculates mean, identifies IQR outliers (anomalies), and returns robust statistics."""
        if not values:
            return {"mean": 0.0, "anomaly_count": 0, "anomalies": [], "clean_mean": 0.0}
        
        arr = np.array(values, dtype=float)
        mean_val = float(np.mean(arr))
        if len(arr) < 4:
            return {
                "mean": round(mean_val, 2),
                "anomaly_count": 0,
                "anomalies": [],
                "clean_mean": round(mean_val, 2)
            }
            
        q25, q75 = np.percentile(arr, 25), np.percentile(arr, 75)
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        anomalies = [float(x) for x in arr if x < lower_bound or x > upper_bound]
        clean_vals = [float(x) for x in arr if lower_bound <= x <= upper_bound]
        clean_mean = float(np.mean(clean_vals)) if clean_vals else mean_val
        
        return {
            "mean": round(mean_val, 2),
            "anomaly_count": len(anomalies),
            "anomalies": [round(a, 2) for a in anomalies],
            "clean_mean": round(clean_mean, 2)
        }

    def run_evaluation(self, model_path: str, selected_levels: List[str] = None) -> Dict[str, Any]:
        """Runs the Wringer benchmark for a single designated model with incremental checkpointing."""
        self._load_test_bank()
        all_levels = list(self.test_bank.keys())
        levels_to_run = selected_levels if selected_levels else all_levels
        model_name = os.path.basename(model_path)
        
        cp_path = self._get_checkpoint_path(model_name)
        checkpoint = None
        if os.path.exists(cp_path):
            existing_cp = self._load_checkpoint(model_name)
            existing_results = existing_cp.get("results", {})
            existing_prompts = existing_cp.get("partial_prompts", {})
            if existing_results or existing_prompts:
                print(f"\n[!] Existing unfinished test found for {model_name} (Levels completed: {list(existing_results.keys())})")
                resume_choice = input("Would you like to resume the last test? (y/n, default=y): ").strip().lower()
                if resume_choice != 'n':
                    checkpoint = existing_cp
                    print("[+] Resuming from saved checkpoint.")
                else:
                    try:
                        os.remove(cp_path)
                    except Exception:
                        pass
                    print("[+] Cleared old checkpoint. Starting clean benchmark.")

        if checkpoint is None:
            checkpoint = self._load_checkpoint(model_name)
            
        model_results = checkpoint.get("results", {})
        
        print(f"\n[+] Running Wringer Evaluation on: {model_name}")
        start_time = time.time() - float(checkpoint.get("elapsed_time", 0.0))

        try:
            for lvl in levels_to_run:
                if lvl in model_results and not selected_levels:
                    print(f"--- Skipping already evaluated level from checkpoint: {lvl} ({model_results[lvl].get('average_score', 'N/A')}/10) ---")
                    continue
                    
                prompts = self.test_bank.get(lvl, [])
                if not prompts: continue
                
                print(f"\n--- Processing Level: {lvl} ---")
                gen_data = self.generate_responses(model_path, prompts, lvl=lvl, checkpoint=checkpoint)
                
                qa_pairs = []
                parsed_pairs = []
                for p, g in zip(prompts, gen_data):
                    raw_txt = g["content"]
                    thought, answer = self.split_thoughts_and_answer(raw_txt)
                    clean_ans = answer if answer else raw_txt
                    qa_pairs.append({"prompt": p, "response": clean_ans})
                    parsed_pairs.append((p, clean_ans, thought, g))

                scores = self.grade_responses(qa_pairs, lvl=lvl, checkpoint=checkpoint)
                
                if scores:
                    lvl_avg_score = sum(scores) / len(scores)
                    lvl_percentage = (lvl_avg_score / 10.0) * 100
                    
                    tot_p_tokens = sum(g["prompt_tokens"] for g in gen_data)
                    tot_p_time = sum(g["prefill_time"] for g in gen_data)
                    tot_c_tokens = sum(g["completion_tokens"] for g in gen_data)
                    tot_d_time = sum(g["decode_time"] for g in gen_data)
                    tot_time = sum(g["total_time"] for g in gen_data)
                    avg_resp_time = tot_time / len(gen_data) if gen_data else 0.0
                    
                    lvl_prefill_tps = tot_p_tokens / tot_p_time if tot_p_time > 0 else 0.0
                    lvl_decode_tps = tot_c_tokens / tot_d_time if tot_d_time > 0 else 0.0
                    lvl_overall_tps = (tot_p_tokens + tot_c_tokens) / tot_time if tot_time > 0 else 0.0
                    
                    decode_speeds = [g["decode_tps"] for g in gen_data if g["decode_tps"] > 0]
                    speed_anomaly_data = self.detect_anomalies_and_stats(decode_speeds)
                    
                    speed_factor = min(10.0, (lvl_decode_tps / 5.0))
                    if self.weight_speed:
                        composite_score = round(0.75 * lvl_avg_score + 0.25 * speed_factor, 2)
                    else:
                        composite_score = round(lvl_avg_score, 2)
                    
                    details = []
                    for (p, ans, thought, g), s in zip(parsed_pairs, scores):
                        item = {
                            "prompt": p, 
                            "response": ans, 
                            "score": round(s, 2),
                            "prompt_tokens": g["prompt_tokens"],
                            "completion_tokens": g["completion_tokens"],
                            "total_time": g["total_time"],
                            "prefill_tps": g["prefill_tps"],
                            "decode_tps": g["decode_tps"],
                            "overall_tps": g["overall_tps"]
                        }
                        if thought:
                            item["thought"] = thought
                        details.append(item)
                        
                    model_results[lvl] = {
                        "average_score": round(lvl_avg_score, 2),
                        "percentage": f"{round(lvl_percentage, 1)}%",
                        "composite_score": composite_score,
                        "prefill_tps": round(lvl_prefill_tps, 2),
                        "decode_tps": round(lvl_decode_tps, 2),
                        "overall_tps": round(lvl_overall_tps, 2),
                        "avg_response_time": round(avg_resp_time, 2),
                        "anomaly_count": speed_anomaly_data["anomaly_count"],
                        "anomalies": speed_anomaly_data["anomalies"],
                        "clean_decode_tps": speed_anomaly_data["clean_mean"],
                        "details": details
                    }
                    
                    checkpoint["results"] = model_results
                    checkpoint["elapsed_time"] = time.time() - start_time
                    self._save_checkpoint(checkpoint)
                    self._export_markdown_report(model_name, model_results, checkpoint["elapsed_time"], is_partial=True)
                    print(f"[✓] Checkpoint saved for {lvl}. Progress secured on disk.")
        except WringerAborted:
            print(f"\n[-] Evaluation aborted by user. Progress saved in checkpoint for {model_name}.")
            return {
                "model": model_name,
                "duration_seconds": round(time.time() - start_time, 2),
                "results": model_results,
                "aborted": True
            }
        
        total_duration = time.time() - start_time
        
        self._export_markdown_report(model_name, model_results, total_duration, is_partial=False)
        self.compare_high_scores(model_name, model_results)

        chart_data = {
            lvl: {
                "score": data["average_score"],
                "decode_tps": data["decode_tps"],
                "composite": data["composite_score"]
            } for lvl, data in model_results.items()
        }
        if chart_data:
            self.generate_comparison_chart(
                chart_data, 
                f"Performance Breakdown: {model_name}", 
                f"{model_name}_breakdown_chart.png",
                auto_open=self.auto_open_charts
            )

        # Clear checkpoint upon full completion
        if os.path.exists(cp_path):
            try:
                os.remove(cp_path)
            except Exception:
                pass

        return {
            "model": model_name,
            "duration_seconds": round(total_duration, 2),
            "results": model_results
        }

    def regrade_report(self, report_path: str, judge_model_path: Optional[str] = None) -> Dict[str, Any]:
        """Parses an existing markdown report, re-evaluates all saved responses using the judge,
        updates the report with new scores, and re-generates the breakdown chart."""
        if judge_model_path:
            self.judge_model_path = judge_model_path
            
        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Report not found: {report_path}")
            
        with open(report_path, "r", encoding="utf-8") as f:
            content = f.read().replace("\r\n", "\n")
            
        m_name_match = re.search(r"# Benchmark Report:\s*([^\n\r]+)", content)
        if not m_name_match:
            raise ValueError(f"Could not parse model name from report: {report_path}")
        model_name = m_name_match.group(1).replace(" (IN PROGRESS - AUTO-CHECKPOINTED)", "").strip()
        
        dur_match = re.search(r"- \*\*Total Duration\*\*:\s*([\d\.]+)s", content)
        total_duration = float(dur_match.group(1)) if dur_match else 0.0
        
        lvl_matches = list(re.finditer(r"(?m)^##\s+(lvl\d+|carwash_test|carwash)\b", content))
        if not lvl_matches:
            raise ValueError(f"No level sections found in: {report_path}")
            
        model_results = {}
        judge_instance = None
        if not self.manual_grading and self.judge_model_path and os.path.exists(self.judge_model_path):
            print(f"\n[*] Pre-loading Judge Model for batch regrading: {os.path.basename(self.judge_model_path)}")
            try:
                judge_instance = self._load_llama(self.judge_model_path, JUDGE_N_CTX)
            except Exception as e:
                print(f"[-] Warning: Failed to pre-load judge model: {e}")

        try:
            for idx, match in enumerate(lvl_matches):
                lvl_name = match.group(1)
                start_pos = match.start()
                end_pos = lvl_matches[idx + 1].start() if idx + 1 < len(lvl_matches) else len(content)
                sec = content[start_pos:end_pos]
                
                prefill_tps = 0.0
                decode_tps = 0.0
                overall_tps = 0.0
                avg_resp = 0.0
                anomaly_cnt = 0
                anomalies = []
                clean_decode = 0.0
                
                speed_match = re.search(r"Prefill:\s*([\d\.]+)\s*t/s\s*\|\s*Decode:\s*([\d\.]+)\s*t/s\s*\|\s*Overall:\s*([\d\.]+)\s*t/s", sec)
                if speed_match:
                    prefill_tps = float(speed_match.group(1))
                    decode_tps = float(speed_match.group(2))
                    overall_tps = float(speed_match.group(3))
                    
                resp_match = re.search(r"Avg Response Time[^\:]*:\s*([\d\.]+)s", sec)
                if resp_match:
                    avg_resp = float(resp_match.group(1))
                    
                anom_match = re.search(r"Anomaly Count\*\*:\s*(\d+)", sec)
                if anom_match:
                    anomaly_cnt = int(anom_match.group(1))
                    
                items = re.split(r"(?m)^### Prompt:\s*", sec)
                details = []
                qa_pairs = []
                
                for it in items[1:]:
                    p_match = re.search(r"```(?:text)?\n(.*?)\n```", it, re.DOTALL)
                    p_text = p_match.group(1).strip() if p_match else ""
                    
                    item_dec_tps = decode_tps
                    item_ov_tps = overall_tps
                    item_time = 0.0
                    meta_match = re.search(r"\*\*Score:\*\*\s*[\d\.]+/10(?:\s*\|\s*\*\*Speed:\*\*\s*([\d\.]+)\s*t/s decode\s*\(([\d\.]+)\s*t/s overall\))?(?:\s*\|\s*\*\*Response Time:\*\*\s*([\d\.]+)s)?", it)
                    if meta_match:
                        if meta_match.group(1): item_dec_tps = float(meta_match.group(1))
                        if meta_match.group(2): item_ov_tps = float(meta_match.group(2))
                        if meta_match.group(3): item_time = float(meta_match.group(3))
                        
                    th_match = re.search(r"<details>\s*<summary>Reasoning</summary>\s*\n```(?:text)?\n(.*?)\n```\s*</details>", it, re.DOTALL)
                    thought_text = th_match.group(1).strip() if th_match else ""
                    
                    resp_match = re.search(r"\*\*Response:\*\*\s*\n```(?:text)?\n(.*?)(?:\n```\s*(?:\n---|\Z))", it, re.DOTALL)
                    resp_text = resp_match.group(1).strip() if resp_match else ""
                    
                    qa_pairs.append({"prompt": p_text, "response": resp_text})
                    details.append({
                        "prompt": p_text,
                        "response": resp_text,
                        "thought": thought_text,
                        "decode_tps": item_dec_tps,
                        "overall_tps": item_ov_tps,
                        "total_time": item_time
                    })
                    
                print(f"\n[*] Re-grading {lvl_name} ({len(qa_pairs)} prompts) for {model_name}...")
                scores = self.grade_responses(qa_pairs, lvl=lvl_name, judge_llm=judge_instance)
                
                for d, s in zip(details, scores):
                    d["score"] = s
                    
                avg_score = round(sum(scores) / max(1, len(scores)), 2)
                pct = f"{round(avg_score * 10, 1)}%"
                comp_score = avg_score
                if self.weight_speed:
                    comp_score = round(self._calculate_composite_score(avg_score, decode_tps), 2)
                    
                model_results[lvl_name] = {
                    "average_score": avg_score,
                    "percentage": pct,
                    "composite_score": comp_score,
                    "prefill_tps": prefill_tps,
                    "decode_tps": decode_tps,
                    "overall_tps": overall_tps,
                    "avg_response_time": avg_resp,
                    "anomaly_count": anomaly_cnt,
                    "anomalies": anomalies,
                    "clean_decode_tps": clean_decode,
                    "details": details
                }
        finally:
            if judge_instance is not None:
                self._release_model(judge_instance)
            
        self._export_markdown_report(model_name, model_results, total_duration, is_partial=False)
        self.compare_high_scores(model_name, model_results)
        
        chart_data = {
            lvl: {
                "score": data["average_score"],
                "decode_tps": data["decode_tps"],
                "composite": data["composite_score"]
            } for lvl, data in model_results.items()
        }
        if chart_data:
            self.generate_comparison_chart(
                chart_data,
                f"Performance Breakdown: {model_name}",
                f"{model_name}_breakdown_chart.png",
                auto_open=self.auto_open_charts
            )
            
        return {
            "model": model_name,
            "duration_seconds": total_duration,
            "results": model_results
        }

    def compare_high_scores(self, model_name: str, model_results: Dict[str, Any]) -> None:
        """Compares the current run results against quality and speed high scores per level."""
        scores_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wringer_highscores.json")
        high_scores = {}
        
        if os.path.exists(scores_file):
            try:
                with open(scores_file, "r") as f:
                    high_scores = json.load(f)
            except Exception as e:
                print(f"[-] Warning: Failed to load high scores: {e}")

        print(f"\n=== Per-Level High Score & Speed Comparison for {model_name} ===")
        print(f"{'Level':<10} | {'Score':<10} | {'Best Score':<18} | {'Decode t/s':<11} | {'Avg Resp':<9} | {'Best t/s':<18} | {'Anomalies':<9} | {'Status'}")
        print("-" * 105)
        
        updated = False
        for lvl, data in model_results.items():
            curr_score = data["average_score"]
            curr_pct = data["percentage"]
            curr_decode_tps = data["decode_tps"]
            curr_resp_time = data.get("avg_response_time", 0.0)
            curr_anomalies = data["anomaly_count"]
            
            best_data = high_scores.get(lvl, {})
            best_score = best_data.get("average_score", 0.0)
            best_score_model = best_data.get("model", "N/A")
            best_tps = best_data.get("best_decode_tps", 0.0)
            best_tps_model = best_data.get("speed_model", "N/A")
            
            score_record = False
            speed_record = False
            
            if curr_score > best_score:
                score_record = True
                best_data["model"] = model_name
                best_data["average_score"] = curr_score
                best_data["percentage"] = curr_pct
                best_data["composite_score"] = data["composite_score"]
                updated = True
                
            if curr_decode_tps > best_tps:
                speed_record = True
                best_data["best_decode_tps"] = curr_decode_tps
                best_data["speed_model"] = model_name
                updated = True
                
            # Maintain Top-5 Leaderboards per Level for Score Percentage and Decode Speed (t/s)
            score_lb = [e for e in best_data.get("top5_score_leaderboard", []) if isinstance(e, dict) and e.get("model") != model_name]
            if best_score_model != "N/A" and not any(e.get("model") == best_score_model for e in score_lb):
                score_lb.append({
                    "model": best_score_model,
                    "percentage": float(best_data.get("percentage", round(best_score * 10, 1))),
                    "average_score": float(best_score),
                    "decode_tps": float(best_data.get("best_decode_tps", 0.0))
                })
            score_lb.append({
                "model": model_name,
                "percentage": float(curr_pct),
                "average_score": float(curr_score),
                "decode_tps": float(curr_decode_tps)
            })
            score_lb.sort(key=lambda x: (x.get("percentage", 0.0), x.get("average_score", 0.0), x.get("decode_tps", 0.0)), reverse=True)
            best_data["top5_score_leaderboard"] = [
                {"rank": idx_r, **entry} for idx_r, entry in enumerate(score_lb[:5], 1)
            ]

            speed_lb = [e for e in best_data.get("top5_speed_leaderboard", []) if isinstance(e, dict) and e.get("model") != model_name]
            if best_tps_model != "N/A" and not any(e.get("model") == best_tps_model for e in speed_lb):
                speed_lb.append({
                    "model": best_tps_model,
                    "decode_tps": float(best_tps),
                    "percentage": float(best_data.get("percentage", 0.0)),
                    "average_score": float(best_score)
                })
            speed_lb.append({
                "model": model_name,
                "decode_tps": float(curr_decode_tps),
                "percentage": float(curr_pct),
                "average_score": float(curr_score)
            })
            speed_lb.sort(key=lambda x: (x.get("decode_tps", 0.0), x.get("percentage", 0.0)), reverse=True)
            best_data["top5_speed_leaderboard"] = [
                {"rank": idx_r, **entry} for idx_r, entry in enumerate(speed_lb[:5], 1)
            ]
            updated = True

            best_data["last_tested_model"] = model_name
            best_data["last_anomaly_count"] = curr_anomalies
            best_data["last_avg_response_time"] = curr_resp_time
            high_scores[lvl] = best_data
            
            if score_record and speed_record:
                status = "[ALL-TIME RECORD! (Score+Speed)]"
            elif score_record:
                status = "[NEW SCORE RECORD!]"
            elif speed_record:
                status = "[NEW SPEED RECORD!]"
            else:
                status = "Completed"
                
            best_score_disp = f"{best_data.get('average_score', 'N/A')} ({best_data.get('model', 'N/A')[:10]})"
            best_tps_disp = f"{best_data.get('best_decode_tps', 'N/A')} t/s ({best_data.get('speed_model', 'N/A')[:10]})"
            
            print(f"{lvl:<10} | {curr_score:<10} | {best_score_disp:<18} | {curr_decode_tps:<11} | {curr_resp_time:>7.2f}s | {best_tps_disp:<18} | {curr_anomalies:<9} | {status}")

        print("\n=== Top 5 Leaderboards Per Level (Score % & Decode Speed) ===")
        for lvl, data in model_results.items():
            b_data = high_scores.get(lvl, {})
            s_lb = b_data.get("top5_score_leaderboard", [])
            v_lb = b_data.get("top5_speed_leaderboard", [])
            s_str = ", ".join(f"#{e['rank']} {e['model'][:18]} ({e['percentage']:.1f}%)" for e in s_lb)
            v_str = ", ".join(f"#{e['rank']} {e['model'][:18]} ({e['decode_tps']:.1f} t/s)" for e in v_lb)
            print(f"  [{lvl}] Top 5 Score % : {s_str}")
            print(f"  [{lvl}] Top 5 Speed   : {v_str}")

        if updated:
            try:
                with open(scores_file, "w") as f:
                    json.dump(high_scores, f, indent=4)
                print(f"[+] High scores & Top-5 leaderboards updated in: {scores_file}")
            except Exception as e:
                print(f"[-] Warning: Failed to save high scores: {e}")

    def generate_comparison_chart(self, data: Dict[str, Any], title: str, filename: str, auto_open: bool = False) -> str:
        """Generates charts comparing quality scores and decode tokens/sec (t/s) side by side."""
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            import re
            
            if not data:
                return ""

            def level_sort_key(lvl_name: str):
                if lvl_name == "Overall":
                    return (-1, 0, "")
                match = re.match(r"^lvl(\d+)", str(lvl_name), re.IGNORECASE)
                if match:
                    return (0, int(match.group(1)), str(lvl_name))
                return (1, 0, str(lvl_name))

            os.makedirs(REPORT_DIR, exist_ok=True)
            chart_path = os.path.join(REPORT_DIR, filename)

            first_val = next(iter(data.values()))
            is_single_model_breakdown = isinstance(first_val, dict) and "score" in first_val
            
            if is_single_model_breakdown:
                sorted_items = sorted(data.items(), key=lambda x: level_sort_key(x[0]))
                levels = [k for k, v in sorted_items]
                scores = [v.get("score", 0.0) for k, v in sorted_items]
                tps = [v.get("decode_tps", 0.0) for k, v in sorted_items]
                composites = [v.get("composite", v.get("score", 0.0)) for k, v in sorted_items]
                
                x = np.arange(len(levels))
                width = 0.35
                
                fig, ax1 = plt.subplots(figsize=(12, 6))
                
                color_score = '#1f77b4'
                ax1.set_xlabel('Benchmark Level', fontweight='bold')
                ax1.set_ylabel('Quality Score (1-10)', color=color_score, fontweight='bold')
                bars1 = ax1.bar(x - width/2, scores, width, label='Quality Score', color=color_score, alpha=0.85)
                ax1.tick_params(axis='y', labelcolor=color_score)
                ax1.set_ylim(0, 10.5)
                ax1.set_xticks(x)
                ax1.set_xticklabels(levels, rotation=30, ha='right')
                
                for bar in bars1:
                    h = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.15, f"{h:.1f}", ha='center', va='bottom', fontsize=8, color=color_score)

                ax2 = ax1.twinx()
                color_speed = '#2ca02c'
                ax2.set_ylabel('Decode Speed (tokens/sec)', color=color_speed, fontweight='bold')
                bars2 = ax2.bar(x + width/2, tps, width, label='Decode Speed (t/s)', color=color_speed, alpha=0.85)
                ax2.tick_params(axis='y', labelcolor=color_speed)
                max_tps = max(tps) if tps else 50
                ax2.set_ylim(0, max(60, max_tps * 1.2))
                
                for bar in bars2:
                    h = bar.get_height()
                    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f} t/s", ha='center', va='bottom', fontsize=8, color=color_speed)

                plt.title(f"{title} (Speed & Quality Metrics)", fontsize=13, fontweight='bold')
                fig.tight_layout()
                plt.savefig(chart_path, dpi=120)
                plt.close(fig)
                
            elif isinstance(first_val, dict):
                def get_overall(v):
                    if isinstance(v, dict):
                        return v.get("Overall", sum([val for val in v.values() if isinstance(val, (int, float))])/max(1, len(v)))
                    return float(v)
                
                sorted_items = sorted(data.items(), key=lambda x: get_overall(x[1]), reverse=True)
                models = [k for k, v in sorted_items]
                
                levels_set = set()
                for k, v in sorted_items:
                    if isinstance(v, dict):
                        for lvl in v.keys():
                            levels_set.add(lvl)
                            
                other_levels = sorted([lvl for lvl in levels_set if lvl != "Overall"], key=level_sort_key)
                
                cols = 2
                rows = math.ceil(len(other_levels) / cols) if other_levels else 0
                
                fig = plt.figure(figsize=(14, 6 + 5 * rows))
                gs = fig.add_gridspec(rows + 1, cols)
                
                ax_overall = fig.add_subplot(gs[0, :])
                overall_scores = [get_overall(v) for k, v in sorted_items]
                bars = ax_overall.bar(models, overall_scores, color='#3498db')
                ax_overall.set_title(f"{title} - Overall", fontweight='bold')
                ax_overall.set_ylabel("Score (1-10)")
                ax_overall.set_ylim(0, 10.5)
                ax_overall.set_xticks(range(len(models)))
                ax_overall.set_xticklabels(models, rotation=35, ha='right')
                for bar in bars:
                    yval = bar.get_height()
                    if yval > 0:
                        ax_overall.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{round(yval, 2)}", ha='center', va='bottom', fontsize=9)
                
                for i, lvl in enumerate(other_levels):
                    r = 1 + (i // cols)
                    c = i % cols
                    ax = fig.add_subplot(gs[r, c])
                    lvl_scores = [v.get(lvl, 0) if isinstance(v, dict) else 0 for k, v in sorted_items]
                    bars = ax.bar(models, lvl_scores, color='#2ecc71')
                    ax.set_title(f"{lvl}", fontweight='bold')
                    ax.set_ylim(0, 10.5)
                    ax.set_xticks(range(len(models)))
                    ax.set_xticklabels(models, rotation=35, ha='right', fontsize=8)
                    for bar in bars:
                        yval = bar.get_height()
                        if yval > 0:
                            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{round(yval, 2)}", ha='center', va='bottom', fontsize=8)

                plt.tight_layout(pad=2.0)
                plt.savefig(chart_path, dpi=120)
                plt.close(fig)
            else:
                is_level_data = any(re.match(r"^(lvl\d+|carwash)", str(k), re.IGNORECASE) for k in data.keys())
                if is_level_data:
                    sorted_items = sorted(data.items(), key=lambda x: level_sort_key(x[0]))
                else:
                    sorted_items = sorted(data.items(), key=lambda x: x[1], reverse=True)

                models = [k for k, v in sorted_items]
                scores = [v for k, v in sorted_items]
                
                plt.figure(figsize=(10, 6))
                bars = plt.bar(models, scores, color='#3498db')
                plt.xlabel('Levels' if is_level_data else 'Models', fontweight='bold')
                plt.ylabel('Average Score (1-10)', fontweight='bold')
                plt.title(title, fontweight='bold')
                plt.ylim(0, 10.5)
                plt.xticks(rotation=35, ha='right')
                
                for bar in bars:
                    yval = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{round(yval, 2)}", ha='center', va='bottom')
                    
                plt.tight_layout()
                plt.savefig(chart_path, dpi=120)
                plt.close()

            print(f"[+] Saved comparison chart to: {chart_path}")
            
            if auto_open:
                self.open_chart_file(chart_path)
                
            return chart_path
        except ImportError:
            print("[-] Matplotlib not installed. Skipping chart generation. (pip install matplotlib)")
            return ""
        except Exception as e:
            print(f"[-] Failed to generate chart: {e}")
            return ""

    @staticmethod
    def open_chart_file(chart_path: str):
        """Cleanly opens a chart file upon user request or configuration."""
        if not os.path.exists(chart_path):
            print(f"[-] Chart file not found: {chart_path}")
            return
        try:
            if os.name == 'nt':
                os.startfile(chart_path)
            elif sys.platform == 'darwin':
                subprocess.call(['open', chart_path])
            else:
                subprocess.call(['xdg-open', chart_path])
            print(f"[+] Opened chart: {chart_path}")
        except Exception as e:
            print(f"[-] Could not open chart: {e}")

    def evaluate_sequential_database(self, database_models: List[str]) -> List[Dict[str, Any]]:
        """Iterates through an entire pool of models sequentially."""
        master_report = []
        chart_data = {}
        print(f"[*] Starting Batch Database Test for {len(database_models)} models...")
        for model in database_models:
            report = self.run_evaluation(model)
            master_report.append(report)
            if report.get("aborted"):
                print("[-] Batch run interrupted by user.")
                break
            
            results = report.get("results", {})
            if results:
                total_score = sum(data["average_score"] for data in results.values())
                avg_score = total_score / len(results)
                chart_data[os.path.basename(model)] = {
                    "Overall": avg_score,
                    **{lvl: data["average_score"] for lvl, data in results.items()}
                }
                
        if chart_data:
            self.generate_comparison_chart(
                chart_data, 
                "Database Batch Test Results", 
                "database_batch_chart.png", 
                auto_open=self.auto_open_charts
            )
            
        return master_report

    def compare_models(self, models_to_compare: List[str], target_level: str = None) -> None:
        """Directly contrasts performance summaries across a cohort of targets."""
        level_name = target_level if target_level else "ALL LEVELS"
        print(f"\n=== Comparative Analysis Matrix ({level_name}) ===")
        print(f"{'Model Name':<25} | {'Avg Score':<10} | {'Success Rate'}")
        print("-" * 55)
        
        selected_levels = [target_level] if target_level else None
        chart_data = {}
        
        for model in models_to_compare:
            res = self.run_evaluation(model, selected_levels=selected_levels)
            if res.get("aborted"):
                print("[-] Comparison interrupted by user.")
                break
            results = res.get("results", {})
            if not results:
                continue
            
            model_name = os.path.basename(model)
            if target_level and target_level in results:
                metrics = results[target_level]
                avg_score = metrics["average_score"]
                pct = metrics["percentage"]
                chart_data[model_name] = avg_score
            else:
                total_score = sum(data["average_score"] for data in results.values())
                avg_score = total_score / len(results) if results else 0
                pct = f"{(avg_score / 10.0) * 100:.1f}%"
                avg_score = round(avg_score, 2)
                chart_data[model_name] = {
                    "Overall": avg_score,
                    **{lvl: data["average_score"] for lvl, data in results.items()}
                }
                
            print(f"{model_name:<25} | {avg_score:<10} | {pct}")
            
        if chart_data:
            self.generate_comparison_chart(
                chart_data, 
                f"Comparative Analysis ({level_name})", 
                "comparison_chart.png", 
                auto_open=self.auto_open_charts
            )

    def generate_persona_chart(self, results: Dict[str, Any], model_name: str, auto_open: bool = False) -> str:
        """Generates a breakdown chart showing Alignment, Gravity, and Drift Resistance per persona."""
        try:
            import matplotlib.pyplot as plt

            os.makedirs(PERSONA_REPORT_DIR, exist_ok=True)
            chart_path = os.path.join(PERSONA_REPORT_DIR, f"{model_name}_persona_chart.png")

            personas = list(results.keys())
            if not personas:
                return ""

            align_scores = [results[p]["avg_alignment"] for p in personas]
            grav_scores = [results[p]["avg_gravity"] for p in personas]
            drift_scores = [results[p]["avg_drift_resistance"] for p in personas]
            overall_scores = [results[p]["average_score"] for p in personas]

            y = np.arange(len(personas))
            height = 0.2

            fig, ax = plt.subplots(figsize=(11, max(6, len(personas) * 0.9)), facecolor='#121216')
            ax.set_facecolor('#181820')

            ax.barh(y - 1.5*height, align_scores, height, label='Alignment', color='#00e5ff')
            ax.barh(y - 0.5*height, grav_scores, height, label='Gravity / Depth', color='#7c5cfc')
            ax.barh(y + 0.5*height, drift_scores, height, label='Drift Resistance', color='#ff7700')
            ax.barh(y + 1.5*height, overall_scores, height, label='Overall Adherence', color='#44ee55')

            ax.set_yticks(y)
            ax.set_yticklabels(personas, color='#ffffff', fontsize=10, fontweight='bold')
            ax.set_xlim(0, 10.5)
            ax.set_xlabel('Score (1 - 10)', color='#ffffff', fontsize=11)
            ax.set_title(f'Persona Adherence Benchmark Breakdown: {model_name}', color='#ffffff', fontsize=13, fontweight='bold', pad=15)
            ax.tick_params(colors='#ffffff')
            ax.grid(axis='x', linestyle='--', alpha=0.3, color='#888888')

            ax.legend(facecolor='#121216', edgecolor='#444444', labelcolor='#ffffff', loc='lower right')
            plt.tight_layout()
            plt.savefig(chart_path, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
            plt.close()

            print(f"[+] Persona Adherence Chart saved to: {chart_path}")
            if auto_open:
                self.open_chart_file(chart_path)
            return chart_path
        except Exception as e:
            print(f"[-] Warning: Failed to generate persona chart: {e}")
            return ""

    def run_persona_evaluation(self, model_path: str, target_persona: Optional[str] = None) -> Dict[str, Any]:
        """Runs the Persona Adherence Benchmark on a model across Serenity 1-5, Serenity Prime, and Cecilia."""
        from serenity_resources import PERSONA_PROMPTS

        with open(PERSONA_BANK_PATH, "r", encoding="utf-8") as f:
            persona_bank = json.load(f)

        personas_to_test = [target_persona] if target_persona and target_persona in persona_bank else list(persona_bank.keys())
        model_name = os.path.basename(model_path)
        os.makedirs(PERSONA_REPORT_DIR, exist_ok=True)

        persona_level_map = {
            "Serenity_1": 1,
            "Serenity_2": 2,
            "Serenity_3": 3,
            "Serenity_4": 4,
            "Serenity_5": 5,
            "Serenity_Prime": 6,
            "Cecilia": 7
        }

        print(f"\n{'='*65}")
        print(f"[*] Starting Persona Adherence Benchmark for: {model_name}")
        print(f"[*] Target Personas: {', '.join(personas_to_test)}")
        print(f"{'='*65}")

        target_ctx = GEN_N_CTX

        llm = None
        judge_llm = None
        results: Dict[str, Any] = {}

        try:
            print(f"\n[*] Loading Candidate Model: {model_name} (Context: {target_ctx})")
            llm = self._load_llama(model_path, target_ctx)

            for p_name in personas_to_test:
                p_data = persona_bank[p_name]
                lvl_num = persona_level_map.get(p_name, 1)
                system_prompt = PERSONA_PROMPTS.get(lvl_num, "You are Serenity.")

                print(f"\n--- Testing Persona: {p_name} ---")
                p_results = []

                # 1. Standard Prompts
                for s_item in p_data.get("standard", []):
                    item_id = s_item["id"]
                    user_prompt = s_item["prompt"]
                    criteria = s_item.get("eval_criteria", "")

                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]

                    t_start = time.perf_counter()
                    t_first_token = None
                    full_thought = ""
                    full_answer = ""

                    gen = llm.create_chat_completion(
                        messages=messages,
                        max_tokens=MAX_GEN_TOKENS,
                        temperature=0.7,
                        top_p=0.95,
                        stream=True
                    )

                    for chunk in gen:
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        content = delta.get("content") or ""
                        reasoning = delta.get("reasoning_content") or ""
                        if content or reasoning:
                            if t_first_token is None:
                                t_first_token = time.perf_counter()
                            if reasoning: full_thought += reasoning
                            if content: full_answer += content

                    t_end = time.perf_counter()
                    if t_first_token is None: t_first_token = t_end

                    _, clean_answer = self.split_thoughts_and_answer(full_answer.strip())
                    clean_answer = clean_answer.strip() if clean_answer else full_answer.strip()

                    output_tokens = self._estimate_tokens(clean_answer, llm=llm, add_bos=False)
                    decode_time = max(0.0001, t_end - t_first_token)
                    decode_tps = output_tokens / decode_time

                    anchor_failed = bool(ANCHOR_FAIL_REGEX.search(clean_answer))
                    words = clean_answer.split()
                    trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                    rep_rate = (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams)) if trigrams else 0.0

                    p_results.append({
                        "id": item_id,
                        "type": "standard",
                        "prompt": user_prompt,
                        "response": clean_answer,
                        "criteria": criteria,
                        "anchor_failed": anchor_failed,
                        "rep_rate": rep_rate,
                        "decode_tps": round(decode_tps, 2),
                        "response_time": round(t_end - t_start, 2)
                    })

                # 2. Drift Prompts (Multi-turn & Padded Single)
                for d_item in p_data.get("drift", []):
                    item_id = d_item["id"]
                    d_type = d_item.get("type", "padded_single")
                    criteria = d_item.get("eval_criteria", "")

                    if d_type == "multi_turn":
                        turns = d_item.get("turns", [])
                        chat_history = [{"role": "system", "content": system_prompt}]
                        turn_responses = []

                        for turn_idx, t_prompt in enumerate(turns):
                            chat_history.append({"role": "user", "content": t_prompt})
                            t_start = time.perf_counter()
                            t_first_token = None
                            turn_ans = ""

                            gen = llm.create_chat_completion(
                                messages=chat_history,
                                max_tokens=MAX_GEN_TOKENS,
                                temperature=0.7,
                                top_p=0.95,
                                stream=True
                            )
                            for chunk in gen:
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                c = delta.get("content") or ""
                                if c:
                                    if t_first_token is None: t_first_token = time.perf_counter()
                                    turn_ans += c
                            t_end = time.perf_counter()
                            if t_first_token is None: t_first_token = t_end

                            _, clean_turn = self.split_thoughts_and_answer(turn_ans.strip())
                            clean_turn = clean_turn.strip() if clean_turn else turn_ans.strip()
                            chat_history.append({"role": "assistant", "content": clean_turn})
                            turn_responses.append(clean_turn)

                        final_turn_answer = turn_responses[-1] if turn_responses else ""
                        all_turns_text = " ".join(turn_responses)
                        anchor_failed = any(bool(ANCHOR_FAIL_REGEX.search(ans)) for ans in turn_responses)

                        words = all_turns_text.split()
                        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                        rep_rate = (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams)) if trigrams else 0.0

                        p_results.append({
                            "id": item_id,
                            "type": "multi_turn",
                            "prompt": f"Multi-Turn Sequence (3 Turns): Final turn: '{turns[-1]}'",
                            "response": final_turn_answer,
                            "full_dialogue": turn_responses,
                            "criteria": criteria,
                            "anchor_failed": anchor_failed,
                            "rep_rate": rep_rate,
                            "decode_tps": 0.0,
                            "response_time": 0.0
                        })

                    elif d_type == "padded_single":
                        context = d_item.get("context", "")
                        user_prompt = d_item.get("prompt", "")
                        combined_user = f"[Reference Archive Context]\n{context}\n\n[Prompt]\n{user_prompt}"

                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": combined_user}
                        ]

                        t_start = time.perf_counter()
                        t_first_token = None
                        full_answer = ""

                        gen = llm.create_chat_completion(
                            messages=messages,
                            max_tokens=MAX_GEN_TOKENS,
                            temperature=0.7,
                            top_p=0.95,
                            stream=True
                        )
                        for chunk in gen:
                            delta = chunk.get("choices", [{}])[0].get("delta", {})
                            c = delta.get("content") or ""
                            if c:
                                if t_first_token is None: t_first_token = time.perf_counter()
                                full_answer += c
                        t_end = time.perf_counter()
                        if t_first_token is None: t_first_token = t_end

                        _, clean_answer = self.split_thoughts_and_answer(full_answer.strip())
                        clean_answer = clean_answer.strip() if clean_answer else full_answer.strip()

                        output_tokens = self._estimate_tokens(clean_answer, llm=llm, add_bos=False)
                        decode_time = max(0.0001, t_end - t_first_token)
                        decode_tps = output_tokens / decode_time

                        anchor_failed = bool(ANCHOR_FAIL_REGEX.search(clean_answer))
                        words = clean_answer.split()
                        trigrams = [tuple(words[i:i+3]) for i in range(len(words)-2)]
                        rep_rate = (len(trigrams) - len(set(trigrams))) / max(1, len(trigrams)) if trigrams else 0.0

                        p_results.append({
                            "id": item_id,
                            "type": "padded_single",
                            "prompt": user_prompt,
                            "response": clean_answer,
                            "criteria": criteria,
                            "anchor_failed": anchor_failed,
                            "rep_rate": rep_rate,
                            "decode_tps": round(decode_tps, 2),
                            "response_time": round(t_end - t_start, 2)
                        })

                results[p_name] = {"items": p_results}

            # 3. Grading Phase
            self._release_model(llm)
            llm = None

            print(f"\n[*] Candidate model released. Initializing Judge for Persona Adherence Evaluation...")
            if self.judge_model_path and os.path.exists(self.judge_model_path):
                judge_ctx = JUDGE_N_CTX
                judge_llm = self._load_llama(self.judge_model_path, judge_ctx)

            for p_name, p_dict in results.items():
                items = p_dict["items"]
                print(f"[*] Grading Persona: {p_name} ({len(items)} items)...")

                align_scores = []
                grav_scores = []
                conv_scores = []
                drift_scores = []
                overall_scores = []

                for item in items:
                    if item.get("anchor_failed", False):
                        item["alignment"] = 2.0
                        item["gravity"] = 2.0
                        item["convergence"] = 2.0
                        item["drift_resistance"] = 1.0
                        item["score"] = 2.0
                        item["critique"] = "Anchor Failure: Model regressed into generic assistant disclaimers ('I am an AI / happy to help')."
                    elif judge_llm is not None:
                        eval_prompt = (
                            f"Target Persona: {p_name}\n"
                            f"Evaluation Criteria: {item.get('criteria', '')}\n"
                            f"User Prompt: {item['prompt']}\n"
                            f"Candidate Response: {item['response']}\n\n"
                            "Evaluate the persona adherence above strictly against the persona requirements. Remember: For Cecilia, sharp wit and dry sarcasm are positive markers, not negative. Output the JSON object."
                        )
                        try:
                            judge_messages = [
                                {"role": "system", "content": PERSONA_JUDGE_SYSTEM_PROMPT},
                                {"role": "user", "content": eval_prompt}
                            ]
                            out = judge_llm.create_chat_completion(
                                messages=judge_messages,
                                max_tokens=JUDGE_REASON_TOKENS,
                                temperature=0.1,
                                top_p=0.95,
                                response_format={"type": "json_object"}
                            )
                            c_text = out["choices"][0]["message"].get("content") or ""
                            _, c_final = self.split_thoughts_and_answer(c_text)
                            parsed = json.loads(re.search(r'\{.*\}', c_final or c_text, re.DOTALL).group(0))

                            item["alignment"] = float(parsed.get("alignment", 7.0))
                            item["gravity"] = float(parsed.get("gravity", 7.0))
                            item["convergence"] = float(parsed.get("convergence", 7.0))
                            item["drift_resistance"] = float(parsed.get("drift_resistance", 7.0))
                            item["score"] = float(parsed.get("score", 7.0))
                            item["critique"] = parsed.get("critique", "")
                        except Exception as e:
                            item["alignment"] = 7.0
                            item["gravity"] = 7.0
                            item["convergence"] = 7.0
                            item["drift_resistance"] = 7.0
                            item["score"] = 7.0
                            item["critique"] = f"Judge parse fallback ({e})."
                    else:
                        item["alignment"] = 7.5
                        item["gravity"] = 7.5
                        item["convergence"] = 7.5
                        item["drift_resistance"] = 7.5
                        item["score"] = 7.5
                        item["critique"] = "Default heuristic evaluation (no judge model)."

                    if item.get("rep_rate", 0.0) > 0.35:
                        item["drift_resistance"] = max(1.0, item["drift_resistance"] - 2.5)
                        item["score"] = max(1.0, item["score"] - 1.5)

                    align_scores.append(item["alignment"])
                    grav_scores.append(item["gravity"])
                    conv_scores.append(item["convergence"])
                    drift_scores.append(item["drift_resistance"])
                    overall_scores.append(item["score"])

                p_dict["avg_alignment"] = round(sum(align_scores) / len(align_scores), 2)
                p_dict["avg_gravity"] = round(sum(grav_scores) / len(grav_scores), 2)
                p_dict["avg_convergence"] = round(sum(conv_scores) / len(conv_scores), 2)
                p_dict["avg_drift_resistance"] = round(sum(drift_scores) / len(drift_scores), 2)
                p_dict["average_score"] = round(sum(overall_scores) / len(overall_scores), 2)
                p_dict["anchor_failures"] = sum(1 for it in items if it.get("anchor_failed", False))

            report_md_path = os.path.join(PERSONA_REPORT_DIR, f"{model_name}_persona_report.md")
            chart_path = self.generate_persona_chart(results, model_name, auto_open=self.auto_open_charts)

            md_lines = [
                f"# Persona Adherence Benchmark Report: {model_name}\n",
                f"- **Model**: `{model_name}`",
                f"- **Date**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
                f"- **Judge Model**: `{os.path.basename(self.judge_model_path) if self.judge_model_path else 'Heuristic Fallback'}`\n",
                "## Summary Table\n",
                "| Persona | Alignment | Gravity / Depth | Convergence | Drift Resistance | Anchor Status | Overall Score |",
                "|---|---|---|---|---|---|---|"
            ]

            print(f"\n{'='*75}")
            print(f"=== Persona Adherence Summary for {model_name} ===")
            print(f"{'Persona':<16} | {'Align':<6} | {'Gravity':<8} | {'Conv':<6} | {'Drift Res':<9} | {'Anchor Status':<14} | {'Overall'}")
            print("-" * 75)

            for p_name, p_dict in results.items():
                anchor_status = f"Passed ({p_dict['anchor_failures']} fails)" if p_dict['anchor_failures'] == 0 else f"DRIFT ({p_dict['anchor_failures']} fails)"
                row = (
                    f"| {p_name} | {p_dict['avg_alignment']:.2f} | {p_dict['avg_gravity']:.2f} | "
                    f"{p_dict['avg_convergence']:.2f} | {p_dict['avg_drift_resistance']:.2f} | "
                    f"{anchor_status} | **{p_dict['average_score']:.2f}/10** |"
                )
                md_lines.append(row)
                print(f"{p_name:<16} | {p_dict['avg_alignment']:<6.2f} | {p_dict['avg_gravity']:<8.2f} | {p_dict['avg_convergence']:<6.2f} | {p_dict['avg_drift_resistance']:<9.2f} | {anchor_status:<14} | {p_dict['average_score']:.2f}/10")

            md_lines.append("\n---\n## Detailed Interaction Transcripts\n")
            for p_name, p_dict in results.items():
                md_lines.append(f"### {p_name} (Overall: {p_dict['average_score']}/10)\n")
                for it in p_dict["items"]:
                    md_lines.append(f"#### Item `{it['id']}` ({it['type']})")
                    md_lines.append(f"- **Prompt**: {it['prompt']}")
                    md_lines.append(f"- **Critique**: {it.get('critique', 'N/A')}")
                    md_lines.append(f"- **Scores**: Alignment={it['alignment']}, Gravity={it['gravity']}, Drift Resistance={it['drift_resistance']}, Final Score={it['score']}")
                    md_lines.append(f"- **Anchor**: {'FAILED' if it['anchor_failed'] else 'PASSED'}")
                    md_lines.append("```text\n" + it['response'] + "\n```\n")

            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write("\n".join(md_lines))

            print(f"\n[+] Persona Adherence Report written to: {report_md_path}")
            return {
                "model": model_name,
                "results": results,
                "report_path": report_md_path,
                "chart_path": chart_path
            }

        finally:
            if llm is not None:
                self._release_model(llm)
            if judge_llm is not None:
                self._release_model(judge_llm)


# ==========================================
# Execution Pathways
# ==========================================
def interactive_menu():
    import tkinter as tk
    from tkinter import filedialog
    
    root = tk.Tk()
    root.withdraw()
    
    print("\n--- Wringer Initial Setup ---")
    rlhf_choice = input("Use Manual Human Grading (RLHF) instead of LLM judge? (y/n, default=n): ").strip().lower()
    manual_grading = rlhf_choice == 'y'
    
    weight_choice = input("Weigh speed (t/s) into composite score? (y=weigh speed & accuracy, n=keep strictly separate, default=n): ").strip().lower()
    weight_speed = weight_choice == 'y'
    
    auto_open_choice = input("Auto-open graph popups immediately after generation? (y/n, default=n): ").strip().lower()
    auto_open_charts = auto_open_choice == 'y'
    
    judge_model_path = None
    if not manual_grading:
        print("Please select the 'Leader' Judge LLM Model (.gguf) for automated scoring (cancel to skip).")
        judge_model_path = filedialog.askopenfilename(title="Select Judge Model (Leader)")
        if not judge_model_path:
            print("[-] No judge model selected. Automated testing will return default 5.0 scores.")
        else:
            print(f"[+] Judge Model Selected: {os.path.basename(judge_model_path)}")
            
    wringer = WringerFramework(
        judge_model_path=judge_model_path, 
        manual_grading=manual_grading, 
        weight_speed=weight_speed,
        auto_open_charts=auto_open_charts
    )
    
    while True:
        print("\n" + "="*55)
        print("     Wringer Evaluation Framework - Main Menu     ")
        print(f"     Mode: {'Weighted Speed & Accuracy' if wringer.weight_speed else 'Strictly Separate Speed & Accuracy'} | Auto-Popup: {wringer.auto_open_charts}")
        print("="*55)
        print("1. Test an individual model")
        print("2. Compare multiple models")
        print("3. Test an entire database of models sequentially")
        print("4. View saved benchmark charts / reports folder")
        print("5. Re-grade existing markdown report(s) and re-generate charts")
        print("6. Run Persona Adherence Benchmark (Cecilia, Serenity Prime & Levels 1-5)")
        print("7. Toggle speed weighting & graph popup settings")
        print("8. Exit")
        print("="*55)
        
        choice = input("\nSelect an option (1-8): ").strip()
        
        if choice == "1":
            print("\nPlease select a model file...")
            model_path = filedialog.askopenfilename(title="Select Model File")
            if not model_path:
                print("[-] No model selected.")
                continue
            model_name = os.path.basename(model_path)
            print(f"[+] Selected: {model_name}")
            
            lvl = input("Enter specific level to test (e.g., lvl5) or press Enter for all: ").strip()
            selected_levels = [lvl] if lvl else None
            
            report = wringer.run_evaluation(model_path, selected_levels=selected_levels)
            print("\nFinal Summary Report:")
            print(json.dumps({
                "model": report["model"],
                "duration_seconds": report["duration_seconds"],
                "levels": {k: {"score": v["average_score"], "decode_tps": v["decode_tps"], "anomalies": v["anomaly_count"]} for k, v in report["results"].items()}
            }, indent=4))
            
            if not wringer.auto_open_charts and not report.get("aborted"):
                open_now = input("\nOpen generated chart now? (y/n, default=y): ").strip().lower()
                if open_now != 'n':
                    chart_file = os.path.join(REPORT_DIR, f"{model_name}_breakdown_chart.png")
                    wringer.open_chart_file(chart_file)
            
        elif choice == "2":
            print("\nPlease select model files to compare (you can select multiple)...")
            model_paths = filedialog.askopenfilenames(title="Select Model Files")
            if not model_paths:
                print("[-] No models selected.")
                continue
            
            lvl = input("Enter target level to compare (e.g., lvl5) or press Enter for ALL levels: ").strip()
            target_level = lvl if lvl else None
            
            wringer.compare_models(model_paths, target_level=target_level)
            
            if not wringer.auto_open_charts:
                open_now = input("\nOpen comparative chart now? (y/n, default=y): ").strip().lower()
                if open_now != 'n':
                    chart_file = os.path.join(REPORT_DIR, "comparison_chart.png")
                    wringer.open_chart_file(chart_file)
            
        elif choice == "3":
            print("\nPlease select a directory containing models...")
            db_dir = filedialog.askdirectory(title="Select Database Directory")
            if not db_dir:
                print("[-] No directory selected.")
                continue
            
            models = []
            for current_dir, _, files in os.walk(db_dir):
                for f in files:
                    if any(k in f.lower() for k in ["mmproj", "assistant", "mtp", "dflash", "drafter"]):
                        continue
                    if f.endswith(".gguf") or f.endswith(".bin"):
                        models.append(os.path.join(current_dir, f))
                    
            if not models:
                print("[-] No compatible models found in the selected directory or its subfolders.")
                continue
            
            print(f"[+] Found {len(models)} models. Beginning sequential test...")
            reports = wringer.evaluate_sequential_database(models)
            print("\n[+] Batch Processing Complete.")
            
            if not wringer.auto_open_charts:
                open_now = input("\nOpen batch results chart now? (y/n, default=y): ").strip().lower()
                if open_now != 'n':
                    chart_file = os.path.join(REPORT_DIR, "database_batch_chart.png")
                    wringer.open_chart_file(chart_file)
            
        elif choice == "4":
            os.makedirs(REPORT_DIR, exist_ok=True)
            wringer.open_chart_file(REPORT_DIR)
            
        elif choice == "5":
            print("\n--- Re-grade Reports Submenu ---")
            print("1. Re-grade a single markdown report")
            print("2. Re-grade all markdown reports in benchmark_reports/")
            print("3. Re-generate consolidated charts without re-grading (via analyze.py)")
            sub = input("Select an option (1-3, default=1): ").strip()
            
            if sub == "2":
                report_files = [
                    os.path.join(REPORT_DIR, f) for f in os.listdir(REPORT_DIR)
                    if f.endswith("_report.md") and not any(k in f.lower() for k in ["assistant", "mtp", "dflash", "drafter", "mmproj"])
                ]
                print(f"[+] Found {len(report_files)} reports to re-grade.")
                confirm = input("Are you sure you want to re-grade ALL reports? (y/n, default=n): ").strip().lower()
                if confirm == 'y':
                    for rf in report_files:
                        try:
                            print(f"\n[+] Processing: {os.path.basename(rf)}")
                            wringer.regrade_report(rf)
                        except Exception as re_err:
                            print(f"[-] Failed to re-grade {os.path.basename(rf)}: {re_err}")
                    print("\n[+] All reports re-graded. Re-generating consolidated charts...")
                    analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")
                    if os.path.exists(analyze_script):
                        import subprocess
                        subprocess.run([sys.executable, analyze_script, "--no-open"])
            elif sub == "3":
                analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")
                if os.path.exists(analyze_script):
                    import subprocess
                    subprocess.run([sys.executable, analyze_script])
            else:
                print("\nPlease select a markdown report file...")
                report_path = filedialog.askopenfilename(
                    title="Select Markdown Report to Re-grade",
                    initialdir=REPORT_DIR,
                    filetypes=[("Markdown files", "*.md")]
                )
                if not report_path:
                    print("[-] No report selected.")
                    continue
                try:
                    res = wringer.regrade_report(report_path)
                    print(f"\n[+] Successfully re-graded {res['model']}.")
                except Exception as re_err:
                    print(f"[-] Error re-grading report: {re_err}")

        elif choice == "6":
            print("\nPlease select a model file (.gguf) to test for Persona Adherence...")
            model_path = filedialog.askopenfilename(title="Select Model File for Persona Benchmark")
            if not model_path:
                print("[-] No model selected.")
                continue
            model_name = os.path.basename(model_path)
            print(f"[+] Selected Model: {model_name}")

            p_target = input("Enter specific persona to test (e.g. Cecilia, Serenity_Prime, Serenity_1-5) or press Enter for ALL personas: ").strip()
            target_persona = p_target if p_target else None

            p_report = wringer.run_persona_evaluation(model_path, target_persona=target_persona)
            print(f"\n[+] Persona Adherence Benchmark Complete for: {p_report['model']}")

            if not wringer.auto_open_charts and p_report.get("chart_path"):
                open_now = input("\nOpen generated persona adherence chart now? (y/n, default=y): ").strip().lower()
                if open_now != 'n':
                    wringer.open_chart_file(p_report["chart_path"])

        elif choice == "7":
            wringer.weight_speed = not wringer.weight_speed
            wringer.auto_open_charts = not wringer.auto_open_charts
            print(f"[+] Settings Updated: Speed Weighting = {wringer.weight_speed}, Auto-Open Charts = {wringer.auto_open_charts}")

        elif choice == "8":
            print("Exiting Wringer Framework.")
            break
        else:
            print("[-] Invalid option. Please try again.")
            
        input("\nPress Enter to return to the main menu...")

if __name__ == "__main__":
    try:
        interactive_menu()
    except KeyboardInterrupt:
        print("\nExiting Wringer Framework.")
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()
        try:
            input("\n[!] Fatal error occurred. Press Enter to exit...")
        except Exception:
            pass
        sys.exit(1)
