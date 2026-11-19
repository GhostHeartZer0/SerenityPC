import json
import math
import os
import re
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def level_sort_key(lvl_name: str) -> Tuple[int, int, str]:
    if lvl_name == "Overall":
        return (-1, 0, "")
    match = re.match(r"^lvl(\d+)", str(lvl_name), re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), str(lvl_name))
    return (1, 0, str(lvl_name))


def _open_file(path: str) -> None:
    if os.name == "nt":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.call(["open", path])
    else:
        subprocess.call(["xdg-open", path])


def collect_results(report_dir: str) -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    if not os.path.isdir(report_dir):
        return results

    for f in sorted(os.listdir(report_dir)):
        if not f.endswith(".md"):
            continue
        model = f.replace("_report.md", "")
        if any(k in model.lower() for k in ("assistant", "mtp", "dflash", "drafter", "mmproj")):
            continue

        path = os.path.join(report_dir, f)
        with open(path, "r", encoding="utf-8") as file:
            content = file.read()

        levels = re.findall(r"## (lvl\d+|carwash_test) \(Average: ([\d\.]+)/10, ([\d\.]+)%\)", content)
        if not levels:
            levels = re.findall(
                r"## (lvl\d+|carwash_test)\s*\n-\s*\*\*Quality Score\*\*:\s*([\d\.]+)/10\s*\(([\d\.]+)%\)",
                content,
            )
        if not levels:
            raw_levels = re.findall(r"## (lvl\d+|carwash_test)\s*\n-\s*\*\*Quality Score\*\*:\s*([\d\.]+)/10", content)
            levels = [(lvl, score, str(round(float(score) * 10, 1))) for lvl, score in raw_levels]
        if not levels:
            continue

        results[model] = {}
        speed_matches = dict(re.findall(r"## (lvl\d+|carwash_test)[\s\S]*?Decode:\s*([\d\.]+)\s*t/s", content))
        for lvl, avg, pct in levels:
            tps_val = float(speed_matches.get(lvl, 0.0)) if lvl in speed_matches else 0.0
            results[model][lvl] = {"avg": float(avg), "pct": float(pct), "tps": tps_val}

    for m in results:
        avgs = [v["avg"] for v in results[m].values()]
        overall = sum(avgs) / len(avgs)
        results[m]["overall"] = overall
    return results


def _apply_dark_style(fig, ax, title: str, ylabel: str, ylim: Tuple[float, float]) -> None:
    fig.patch.set_facecolor("#141418")
    ax.set_facecolor("#1e1e24")
    ax.set_title(title, color="#f0f0f0", fontsize=12, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color="#d0d0d0", fontsize=10)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#888899")
    ax.tick_params(colors="#d0d0d0")
    for spine in ax.spines.values():
        spine.set_color("#3a3a44")


def _bold_highscore_bars(ax, bars, values: List[float], models: List[str], fmt: str, offset: float, tick_fs: int) -> None:
    max_val = max((v for v in values if v > 0), default=0.0)
    high_indices = {i for i, v in enumerate(values) if max_val > 0 and math.isclose(v, max_val, rel_tol=1e-6, abs_tol=1e-6)}
    for i, (bar, val) in enumerate(zip(bars, values)):
        is_hi = i in high_indices
        if is_hi:
            bar.set_facecolor("#00ffcc")
            bar.set_edgecolor("#ffd700")
            bar.set_linewidth(2.0)
        if val > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                val + offset,
                fmt.format(val),
                ha="center",
                va="bottom",
                fontsize=tick_fs + (1 if is_hi else 0),
                fontweight="bold" if is_hi else "normal",
                color="#ffd700" if is_hi else "#d8dce6",
            )
    ax.set_xticks(range(len(models)))
    tick_labels = ax.set_xticklabels(models, rotation=40, ha="right", fontsize=tick_fs)
    for i, lbl in enumerate(tick_labels):
        if i in high_indices:
            lbl.set_fontweight("bold")
            lbl.set_color("#ffd700")
        else:
            lbl.set_fontweight("normal")
            lbl.set_color("#c8ccd4")


def generate_consolidated_chart(results: Dict[str, Dict[str, Any]], report_dir: str, open_chart: bool = False) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    sorted_models = sorted(results.items(), key=lambda x: x[1]["overall"], reverse=True)
    models = [m for m, _ in sorted_models]
    other_levels = sorted(
        list({lvl for m_data in results.values() for lvl in m_data.keys() if lvl != "overall"}),
        key=level_sort_key,
    )

    cols = 2
    rows = math.ceil(len(other_levels) / cols) if other_levels else 0
    fig = plt.figure(figsize=(15, 6 + 4.8 * rows), facecolor="#141418", constrained_layout=True)
    gs = gridspec.GridSpec(rows + 1, cols, figure=fig)

    ax_overall = fig.add_subplot(gs[0, :])
    overall_scores = [results[m]["overall"] for m in models]
    bars = ax_overall.bar(models, overall_scores, color="#3b82f6", edgecolor="#1d4ed8", linewidth=0.8)
    _apply_dark_style(fig, ax_overall, "Benchmark Analysis - Overall (Out of 10)", "Score (/10)", (0, 10.8))
    _bold_highscore_bars(ax_overall, bars, overall_scores, models, "{:.2f}", 0.15, 9)

    for i, lvl in enumerate(other_levels):
        r, c = 1 + (i // cols), i % cols
        ax = fig.add_subplot(gs[r, c])
        lvl_scores = [results[m].get(lvl, {}).get("avg", 0.0) if isinstance(results[m].get(lvl), dict) else 0.0 for m in models]
        bars = ax.bar(models, lvl_scores, color="#4f7cac", edgecolor="#2b4c7e", linewidth=0.8)
        _apply_dark_style(fig, ax, f"{lvl.upper()} (/10)", "Score (/10)", (0, 10.8))
        _bold_highscore_bars(ax, bars, lvl_scores, models, "{:.2f}", 0.15, 8)

    chart_path = os.path.join(report_dir, "analyze_fallback_consolidated.png")
    plt.savefig(chart_path, dpi=140, facecolor="#141418")
    plt.close(fig)
    print(f"[+] Consolidated chart saved to: {chart_path}")
    if open_chart:
        _open_file(chart_path)
    return chart_path


def generate_vertical_pct_chart(results: Dict[str, Dict[str, Any]], report_dir: str, open_chart: bool = False) -> str:
    """Generates vertical % comparison chart (0-100%) with highscores bolded."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt

    sorted_models = sorted(results.items(), key=lambda x: x[1]["overall"], reverse=True)
    models = [m for m, _ in sorted_models]
    other_levels = sorted(
        list({lvl for m_data in results.values() for lvl in m_data.keys() if lvl != "overall"}),
        key=level_sort_key,
    )

    cols = 2
    rows = math.ceil(len(other_levels) / cols) if other_levels else 0
    fig = plt.figure(figsize=(15, 6.2 + 5.0 * rows), facecolor="#141418", constrained_layout=True)
    gs = gridspec.GridSpec(rows + 1, cols, figure=fig)

    ax_overall = fig.add_subplot(gs[0, :])
    overall_pcts = [round(float(results[m]["overall"]) * 10.0, 1) for m in models]
    bars = ax_overall.bar(models, overall_pcts, color="#3b82f6", edgecolor="#1d4ed8", linewidth=0.8)
    _apply_dark_style(fig, ax_overall, "Vertical % Comparison - Overall (Highscores Bolded)", "Score (%)", (0, 109.0))
    _bold_highscore_bars(ax_overall, bars, overall_pcts, models, "{:.1f}%", 1.5, 9)

    for i, lvl in enumerate(other_levels):
        r, c = 1 + (i // cols), i % cols
        ax = fig.add_subplot(gs[r, c])
        lvl_pcts = [
            float(results[m][lvl].get("pct", results[m][lvl].get("avg", 0.0) * 10.0))
            if isinstance(results[m].get(lvl), dict)
            else 0.0
            for m in models
        ]
        bars = ax.bar(models, lvl_pcts, color="#4a7ab5", edgecolor="#2b4c7e", linewidth=0.8)
        _apply_dark_style(fig, ax, f"{lvl.upper()} (%)", "Score (%)", (0, 109.0))
        _bold_highscore_bars(ax, bars, lvl_pcts, models, "{:.1f}%", 1.5, 8)

    chart_path = os.path.join(report_dir, "analyze_vertical_pct_comparison.png")
    plt.savefig(chart_path, dpi=140, facecolor="#141418")
    plt.close(fig)
    print(f"[+] Vertical % comparison chart saved to: {chart_path}")
    if open_chart:
        _open_file(chart_path)
    return chart_path


def main(argv: Optional[List[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_reports")
    if "--report-dir" in args:
        idx = args.index("--report-dir")
        if idx + 1 < len(args):
            report_dir = args[idx + 1]
    results = collect_results(report_dir)

    if not results:
        print("No benchmark reports found.")
        return 0

    sorted_models = sorted(results.items(), key=lambda x: x[1]["overall"], reverse=True)
    all_display_levels = sorted(
        list({lvl for m_data in results.values() for lvl in m_data.keys() if lvl != "overall"}),
        key=level_sort_key,
    )

    print("### Benchmark Analysis")
    print("| Model | Overall Score | " + " | ".join(all_display_levels) + " |")
    print("|-------|---------------|" + "|".join(["---"] * len(all_display_levels)) + "|")
    for m, data in sorted_models:
        row = f"| {m} | {data['overall']:.2f}/10 ({data['overall'] * 10:.1f}%) |"
        for lvl in all_display_levels:
            if lvl in data:
                row += f" {data[lvl]['avg']:.2f} ({data[lvl]['pct']:.1f}%) |"
            else:
                row += " - |"
        print(row)

    print("\n### Top 5 Highscore Leaderboards Per Level (Score % & Decode Speed)")
    for lvl in all_display_levels:
        lvl_entries = [(m, results[m][lvl]) for m in results if lvl in results[m]]
        top5_score = sorted(lvl_entries, key=lambda x: (x[1]["pct"], x[1]["avg"]), reverse=True)[:5]
        top5_speed = sorted([e for e in lvl_entries if e[1].get("tps", 0) > 0], key=lambda x: x[1]["tps"], reverse=True)[:5]
        print(f"\n#### {lvl.upper()}")
        print("Top 5 Score Percentage:")
        for rank, (m_name, m_d) in enumerate(top5_score, 1):
            print(f"  {rank}. {m_name} — {m_d['pct']:.1f}% ({m_d['avg']:.2f}/10)")
        if top5_speed:
            print("Top 5 Decode Speed:")
            for rank, (m_name, m_d) in enumerate(top5_speed, 1):
                print(f"  {rank}. {m_name} — {m_d['tps']:.2f} t/s ({m_d['pct']:.1f}%)")

    no_open = "--no-open" in args
    only_vertical_pct = "--vertical-pct" in args
    try:
        if not only_vertical_pct:
            generate_consolidated_chart(results, report_dir, open_chart=not no_open)
        generate_vertical_pct_chart(results, report_dir, open_chart=(not no_open and only_vertical_pct))
    except ImportError:
        print("\n[-] Matplotlib not installed. Skipping chart generation.")
    except Exception as e:
        print(f"\n[-] Failed to generate chart: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
