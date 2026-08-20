"""
Test: Telemetry Population & Formatting (VRAM, Total RAM, CPU Temp, CPU Power)
Validates:
1. SystemMonitor._get_cpu_temp() returns formatted temperature reading.
2. SystemMonitor._get_cpu_power() returns formatted power wattage reading.
3. SystemMonitor._get_shared_vram_used_bytes() and adapter counters return valid integers.
4. SystemMonitor._stats_loop emits full telemetry dictionary with all 10 keys.
5. main.py _update_stats_display formats standard and graph bar modes correctly.
"""
import sys
import os
import time
import queue
import threading
import unittest
import tkinter as tk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.serenity_utils import SystemMonitor
from main import ChatbotApp

class TestTelemetryPopulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = tk.Tk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        try:
            cls.root.destroy()
        except Exception:
            pass

    def test_cpu_temp_format(self):
        """Verify CPU Temp returns valid reading or N/A."""
        temp = SystemMonitor._get_cpu_temp()
        self.assertIsInstance(temp, str)
        self.assertTrue(temp.endswith("°C") or temp == "N/A", f"Unexpected CPU Temp format: {temp}")

    def test_cpu_power_format(self):
        """Verify CPU Power returns valid wattage string or N/A."""
        power = SystemMonitor._get_cpu_power()
        self.assertIsInstance(power, str)
        self.assertTrue(power.endswith("W") or power == "N/A", f"Unexpected CPU Power format: {power}")

    def test_shared_vram_and_adapter_counters(self):
        """Verify shared VRAM byte count and GPU adapter counter fallbacks."""
        shared_bytes = SystemMonitor._get_shared_vram_used_bytes()
        self.assertIsInstance(shared_bytes, int)
        self.assertGreaterEqual(shared_bytes, 0)

        ded, sh = SystemMonitor._get_gpu_adapter_counters()
        self.assertIsInstance(ded, int)
        self.assertIsInstance(sh, int)
        self.assertGreaterEqual(ded, 0)
        self.assertGreaterEqual(sh, 0)

    def test_stats_loop_all_telemetry_keys(self):
        """Verify SystemMonitor thread emits stats_update with all 10 telemetry keys."""
        class MockApp:
            def __init__(self):
                self.stop_process = threading.Event()
                self.process_queue = queue.Queue()

        mock_app = MockApp()
        mon = SystemMonitor(mock_app)
        mon.start()
        
        # Wait for at least one stats update
        msg = None
        for _ in range(25):
            if not mock_app.process_queue.empty():
                msg = mock_app.process_queue.get()
                if msg.get("status") == "stats_update":
                    break
            time.sleep(0.2)
        mock_app.stop_process.set()

        self.assertIsNotNone(msg, "SystemMonitor did not emit a stats_update message within timeout")
        stats = msg.get("stats", {})
        
        expected_keys = [
            "CPU", "RAM", "CPU Temp", "CPU Power",
            "GPU Use", "VRAM", "Shared VRAM", "Total VRAM",
            "GPU Temp", "Power"
        ]
        for k in expected_keys:
            self.assertIn(k, stats, f"Missing telemetry key: {k}")
            self.assertIsNotNone(stats[k], f"Telemetry value for {k} was None")

    def test_update_stats_display_modes(self):
        """Verify UI stats label rendering in both standard text and graph bar mode."""
        app = ChatbotApp.__new__(ChatbotApp)
        app.root = self.root
        app.config = {"monitor_graph_mode": False}
        app.stats_labels = {}

        stats_to_show = [
            ("GPU Use", "GPU Use"), ("CPU", "CPU Use"),
            ("VRAM", "VRAM"), ("Total VRAM", "Total VRAM"),
            ("Shared VRAM", "Shared VRAM"), ("RAM", "Total RAM"),
            ("GPU Temp", "GPU Temp"), ("CPU Temp", "CPU Temp"),
            ("Power", "GPU Power"), ("CPU Power", "CPU Power")
        ]

        frame = tk.Frame(self.root)
        frame.pack()
        for key, _ in stats_to_show:
            app.stats_labels[key] = tk.Label(frame, text="N/A")

        sample_stats = {
            "CPU": "15.0%",
            "RAM": "8192 / 32768 MB",
            "CPU Temp": "35°C",
            "CPU Power": 24.5,
            "GPU Use": "40%",
            "VRAM": "1024 / 6144 MB",
            "Shared VRAM": "0.10 / 16.0 GB",
            "Total VRAM": "1.10 / 22.0 GB",
            "GPU Temp": "42°C",
            "Power": 15.2
        }

        # 1. Standard mode
        app.config["monitor_graph_mode"] = False
        app._update_stats_display(sample_stats)
        self.assertEqual(app.stats_labels["CPU"].cget("text"), "15.0%")
        self.assertEqual(app.stats_labels["RAM"].cget("text"), "8192 / 32768 MB")
        self.assertEqual(app.stats_labels["CPU Temp"].cget("text"), "35°C")
        self.assertEqual(app.stats_labels["CPU Power"].cget("text"), "24.5W")
        self.assertEqual(app.stats_labels["Power"].cget("text"), "15.2W")

        # 2. Graph mode
        app.config["monitor_graph_mode"] = True
        app._update_stats_display(sample_stats)
        self.assertIn("█", app.stats_labels["CPU"].cget("text"))
        self.assertIn("█", app.stats_labels["GPU Use"].cget("text"))

        frame.destroy()

if __name__ == "__main__":
    unittest.main()
