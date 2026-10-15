import os, re, json

report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'benchmark_reports')
results = {}
for f in os.listdir(report_dir):
    if not f.endswith('.md'): continue
    model = f.replace('_report.md', '')
    
    # Filter out 'assistant', 'MTP', 'dflash', and 'mmproj' models
    if any(k in model.lower() for k in ["assistant", "mtp", "dflash", "drafter", "mmproj"]):
        continue
        
    path = os.path.join(report_dir, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        levels = re.findall(r'## (lvl\d+|carwash_test) \(Average: ([\d\.]+)/10, ([\d\.]+)%\)', content)
        if not levels:
            levels = re.findall(r'## (lvl\d+|carwash_test)\s*\n-\s*\*\*Quality Score\*\*:\s*([\d\.]+)/10\s*\(([\d\.]+)%\)', content)
        if not levels:
            raw_levels = re.findall(r'## (lvl\d+|carwash_test)\s*\n-\s*\*\*Quality Score\*\*:\s*([\d\.]+)/10', content)
            levels = [(lvl, score, str(round(float(score) * 10, 1))) for lvl, score in raw_levels]
        if not levels: continue
        results[model] = {}
        speed_matches = dict(re.findall(r'## (lvl\d+|carwash_test)[\s\S]*?Decode:\s*([\d\.]+)\s*t/s', content))
        for lvl, avg, pct in levels:
            tps_val = float(speed_matches.get(lvl, 0.0)) if lvl in speed_matches else 0.0
            results[model][lvl] = {'avg': float(avg), 'pct': float(pct), 'tps': tps_val}

# Calculate overall averages
chart_data = {}
for m in results:
    avgs = [v['avg'] for v in results[m].values()]
    overall = sum(avgs) / len(avgs)
    results[m]['overall'] = overall
    chart_data[m] = {
        "Overall": overall,
        **{lvl: v['avg'] for lvl, v in results[m].items() if lvl != 'overall'}
    }

# Sort by overall
sorted_models = sorted(results.items(), key=lambda x: x[1]['overall'], reverse=True)

if not results:
    print("No benchmark reports found.")
    import sys
    sys.exit(0)

def level_sort_key(lvl_name: str):
    if lvl_name == "Overall":
        return (-1, 0, "")
    match = re.match(r"^lvl(\d+)", str(lvl_name), re.IGNORECASE)
    if match:
        return (0, int(match.group(1)), str(lvl_name))
    return (1, 0, str(lvl_name))

all_display_levels = sorted(list({lvl for m_data in results.values() for lvl in m_data.keys() if lvl != 'overall'}), key=level_sort_key)

print("### Benchmark Analysis")
print("| Model | Overall Score | " + " | ".join(all_display_levels) + " |")
print("|-------|---------------|" + "|".join(["---"] * len(all_display_levels)) + "|")

for m, data in sorted_models:
    row = f"| {m} | {data['overall']:.2f}/10 |"
    for lvl in all_display_levels:
        if lvl in data:
            row += f" {data[lvl]['avg']:.2f} |"
        else:
            row += " - |"
    print(row)


print("\n### Top 5 Highscore Leaderboards Per Level (Score % & Decode Speed)")
for lvl in all_display_levels:
    lvl_entries = [(m, results[m][lvl]) for m in results if lvl in results[m]]
    top5_score = sorted(lvl_entries, key=lambda x: (x[1]['pct'], x[1]['avg']), reverse=True)[:5]
    top5_speed = sorted([e for e in lvl_entries if e[1].get('tps', 0) > 0], key=lambda x: x[1]['tps'], reverse=True)[:5]
    print(f"\n#### {lvl.upper()}")
    print("Top 5 Score Percentage:")
    for rank, (m_name, m_d) in enumerate(top5_score, 1):
        print(f"  {rank}. {m_name} — {m_d['pct']:.1f}% ({m_d['avg']:.2f}/10)")
    if top5_speed:
        print("Top 5 Decode Speed:")
        for rank, (m_name, m_d) in enumerate(top5_speed, 1):
            print(f"  {rank}. {m_name} — {m_d['tps']:.2f} t/s ({m_d['pct']:.1f}%)")

# Fallback chart generation
if chart_data:
    try:
        import matplotlib.pyplot as plt
        import sys
        
        sorted_items = sorted(chart_data.items(), key=lambda x: x[1].get("Overall", 0), reverse=True)
        models = [k for k, v in sorted_items]
        
        def level_sort_key(lvl_name: str):
            if lvl_name == "Overall":
                return (-1, 0, "")
            match = re.match(r"^lvl(\d+)", str(lvl_name), re.IGNORECASE)
            if match:
                return (0, int(match.group(1)), str(lvl_name))
            return (1, 0, str(lvl_name))

        levels = []
        for k, v in sorted_items:
            for lvl in v.keys():
                if lvl not in levels:
                    levels.append(lvl)
        
        other_levels = sorted([lvl for lvl in levels if lvl != "Overall"], key=level_sort_key)
        
        import math
        import matplotlib.gridspec as gridspec

        cols = 2
        rows = math.ceil(len(other_levels) / cols) if other_levels else 0
        
        fig = plt.figure(figsize=(14, 6 + 5 * rows))
        gs = gridspec.GridSpec(rows + 1, cols, figure=fig)
        
        # Overall chart spanning the top row
        ax_overall = fig.add_subplot(gs[0, :])
        overall_scores = [v.get("Overall", 0) for k, v in sorted_items]
        bars = ax_overall.bar(models, overall_scores, color='skyblue')
        ax_overall.set_title("Benchmark Analysis - Overall")
        ax_overall.set_ylabel("Score (out of 10)")
        ax_overall.set_ylim(0, 10)
        ax_overall.set_xticks(range(len(models)))
        ax_overall.set_xticklabels(models, rotation=45, ha='right')
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                ax_overall.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{round(yval, 2)}", ha='center', va='bottom', fontsize=9)
        
        # Subplots for individual levels
        for i, lvl in enumerate(other_levels):
            r = 1 + (i // cols)
            c = i % cols
            ax = fig.add_subplot(gs[r, c])
            lvl_scores = [v.get(lvl, 0) for k, v in sorted_items]
            bars = ax.bar(models, lvl_scores, color='lightgreen')
            ax.set_title(f"{lvl}")
            ax.set_ylim(0, 10)
            ax.set_xticks(range(len(models)))
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=8)
            for bar in bars:
                yval = bar.get_height()
                if yval > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{round(yval, 2)}", ha='center', va='bottom', fontsize=8)

        plt.tight_layout(pad=2.0)
        chart_path = os.path.join(report_dir, "analyze_fallback_consolidated.png")
        plt.savefig(chart_path)
        plt.close(fig)
        print(f"[+] Consolidated fallback chart saved to: {chart_path}")
        
        if '--no-open' not in sys.argv:
            if os.name == 'nt':
                os.startfile(chart_path)
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.call(['open', chart_path])
            else:
                import subprocess
                subprocess.call(['xdg-open', chart_path])
            
    except ImportError:
        print("\n[-] Matplotlib not installed. Skipping chart generation.")
    except Exception as e:
        print(f"\n[-] Failed to generate fallback chart: {e}")
