import os
import sys
import subprocess
import argparse
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def _bootstrap_venv():
    scripts_dir = os.path.join(PROJECT_ROOT, ".venv", "Scripts") if sys.platform == "win32" else os.path.join(PROJECT_ROOT, ".venv", "bin")
    venv_py = os.path.join(scripts_dir, "python.exe" if sys.platform == "win32" else "python")
    if os.path.exists(venv_py):
        cur_py = os.path.normcase(os.path.abspath(sys.executable))
        tgt_py = os.path.normcase(os.path.abspath(venv_py))
        if cur_py != tgt_py:
            env = os.environ.copy()
            env["VIRTUAL_ENV"] = os.path.join(PROJECT_ROOT, ".venv")
            env["PATH"] = scripts_dir + os.pathsep + env.get("PATH", "")
            result = subprocess.run([venv_py, os.path.abspath(__file__)] + sys.argv[1:], env=env, cwd=PROJECT_ROOT)
            sys.exit(result.returncode)

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

_bootstrap_venv()

from benchmarks.wringer.Wringer import WringerFramework, REPORT_DIR

DEFAULT_JUDGE = r"S:\LLM\NVIDIA-Nemotron-3.5-Lightning-30B-A3B-IQ4_XS.gguf"


def is_pid_running(pid: int) -> bool:
    import subprocess
    cmd = f"Get-Process -Id {pid} -ErrorAction SilentlyContinue"
    out = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True)
    return bool(out.stdout.strip())


def main():
    parser = argparse.ArgumentParser(description="Batch re-grade markdown reports and regenerate charts.")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help="Path to judge GGUF model")
    parser.add_argument("--wait-pid", type=int, default=None, help="PID of active benchmark to wait for before starting")
    parser.add_argument("--target", default=None, help="Specific report filename or model name to re-grade (default: all)")
    args = parser.parse_args()

    if args.wait_pid:
        print(f"[*] Waiting for PID {args.wait_pid} to terminate before starting re-grading...")
        while is_pid_running(args.wait_pid):
            time.sleep(30)
        print(f"[+] PID {args.wait_pid} terminated. Proceeding with batch re-grading.")

    if not os.path.exists(args.judge):
        print(f"[-] Judge model not found: {args.judge}")
        sys.exit(1)

    wringer = WringerFramework(judge_model_path=args.judge, manual_grading=False, auto_open_charts=False)

    reports = []
    if args.target:
        target_path = args.target if os.path.isabs(args.target) else os.path.join(REPORT_DIR, args.target)
        if not target_path.endswith("_report.md"):
            target_path += "_report.md"
        reports = [target_path]
    else:
        for f in os.listdir(REPORT_DIR):
            if f.endswith("_report.md") and not any(k in f.lower() for k in ["assistant", "mtp", "dflash", "drafter", "mmproj"]):
                m_base = f.replace("_report.md", "")
                if os.path.exists(os.path.join(REPORT_DIR, f"{m_base}_checkpoint.json")):
                    print(f"[*] Skipping in-progress model with active checkpoint: {m_base}")
                    continue
                reports.append(os.path.join(REPORT_DIR, f))

    print(f"[+] Found {len(reports)} report(s) to re-grade using {os.path.basename(args.judge)}.")

    for i, rf in enumerate(reports):
        name = os.path.basename(rf)
        print(f"\n[{i+1}/{len(reports)}] Re-grading: {name}")
        try:
            wringer.regrade_report(rf)
            print(f"[+] Successfully updated {name} and regenerated breakdown chart.")
        except Exception as e:
            print(f"[-] Error re-grading {name}: {e}")

    print("\n[+] Re-grading complete. Updating consolidated charts...")
    import subprocess
    analyze_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py")
    subprocess.run([sys.executable, analyze_script, "--no-open"])
    print("[+] All breakdown charts and consolidated charts regenerated successfully.")


if __name__ == "__main__":
    main()
