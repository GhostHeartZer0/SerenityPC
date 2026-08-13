from Wringer import WringerFramework
import os

wringer = WringerFramework(manual_grading=True)
data = {"Model A": 8.5, "Model B": 9.2, "Model C": 7.0}
wringer.generate_comparison_chart(data, "Test Chart", "test_chart.png")

report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_reports")
chart_path = os.path.join(report_dir, "test_chart.png")

if os.path.exists(chart_path):
    print("Chart successfully generated at:", chart_path)
else:
    print("Failed to find generated chart.")
