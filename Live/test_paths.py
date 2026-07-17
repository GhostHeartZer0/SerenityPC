
import os
import sys

# Mock classes to check directory logic
class MockAgent:
    def __init__(self):
        current_dir = os.path.abspath(os.path.dirname(__file__))
        parts = current_dir.replace('\\', '/').split('/')
        if "Live" in parts:
            idx = len(parts) - 1 - list(reversed(parts)).index("Live")
            self.live_dir = "/".join([parts[i] for i in range(idx + 1)])
        else:
            self.live_dir = current_dir
        self.logs_dir = os.path.join(self.live_dir, "Logs")

print(f"Agent live_dir: {MockAgent().live_dir}")
print(f"Agent logs_dir: {MockAgent().logs_dir}")
print(f"Current Working Dir: {os.getcwd()}")
print(f"__file__: {__file__}")
