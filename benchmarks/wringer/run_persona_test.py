# Runner script for Persona Adherence Benchmark
import os
import sys
import argparse
from pathlib import Path

# Add project root
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from benchmarks.wringer.Wringer import WringerFramework

def main():
    parser = argparse.ArgumentParser(description="Run Persona Adherence Benchmark")
    parser.add_argument("--model", type=str, default="S:/LLM/gemma-4-unc/E2B/Huihui-gemma-4-E2B-it-qat-q4_0-unquantized-abliterated-Q4_K.gguf", help="Candidate model path")
    parser.add_argument("--persona", type=str, default="Cecilia", help="Target persona name or ALL")
    parser.add_argument("--judge", type=str, default=None, help="Judge model path (optional)")
    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    target_persona = None if args.persona.upper() == "ALL" else args.persona
    judge_path = os.path.abspath(args.judge) if args.judge else None

    print(f"[*] Candidate Model: {model_path}")
    print(f"[*] Target Persona: {target_persona or 'ALL'}")
    print(f"[*] Judge Model: {judge_path or 'Heuristic Rule-Based'}")

    wringer = WringerFramework(judge_model_path=judge_path, auto_open_charts=False)
    result = wringer.run_persona_evaluation(model_path, target_persona=target_persona)
    print("\n[+] Benchmark finished successfully!")
    print(f"[+] Report: {result['report_path']}")
    print(f"[+] Chart: {result['chart_path']}")

if __name__ == "__main__":
    main()
