import os, re, json
report_dir = r'c:\Users\ccrg6\Desktop\Desktop\Hub\SerenityPC\System\tests\benchmark_reports'
results = {}
for f in os.listdir(report_dir):
    if not f.endswith('.md'): continue
    model = f.replace('_report.md', '')
    path = os.path.join(report_dir, f)
    with open(path, 'r', encoding='utf-8') as file:
        content = file.read()
        levels = re.findall(r'## (lvl\d+|carwash_test) \(Average: ([\d\.]+)/10, ([\d\.]+)%\)', content)
        if not levels: continue
        results[model] = {}
        for lvl, avg, pct in levels:
            results[model][lvl] = {'avg': float(avg), 'pct': float(pct)}

# Calculate overall averages
chart_data = {}
for m in results:
    avgs = [v['avg'] for v in results[m].values()]
    overall = sum(avgs) / len(avgs)
    results[m]['overall'] = overall
    chart_data[m] = overall

# Sort by overall
sorted_models = sorted(results.items(), key=lambda x: x[1]['overall'], reverse=True)

if not results:
    print("No benchmark reports found.")
    import sys
    sys.exit(0)

print("### Benchmark Analysis")
print("| Model | Overall Score | " + " | ".join(sorted(list(results[list(results.keys())[0]].keys())[:-1])) + " |")
print("|-------|---------------|" + "|".join(["---"] * (len(list(results.keys())[0]) - 1)) + "|")

for m, data in sorted_models:
    row = f"| {m} | {data['overall']:.2f}/10 |"
    for lvl in sorted([k for k in data.keys() if k != 'overall']):
        row += f" {data[lvl]['avg']:.2f} |"
    print(row)

# Fallback chart generation
if chart_data:
    try:
        import matplotlib.pyplot as plt
        import sys
        
        plt.figure(figsize=(12, 6))
        models = list(chart_data.keys())
        scores = list(chart_data.values())
        
        bars = plt.bar(models, scores, color='lightgreen')
        plt.xlabel('Models')
        plt.ylabel('Average Score (out of 10)')
        plt.title('Benchmark Analysis - Overall Scores')
        plt.ylim(0, 10)
        plt.xticks(rotation=45, ha='right')
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{round(yval, 2)}", ha='center', va='bottom')
            
        plt.tight_layout()
        chart_path = os.path.join(report_dir, "analyze_fallback_chart.png")
        plt.savefig(chart_path)
        plt.close()
        print(f"\n[+] Fallback chart saved to: {chart_path}")
        
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
