import threading
import time
import traceback
import sys
import os

class UIWatchdog(threading.Thread):
    """
    A background thread that monitors the responsiveness of the main Tkinter thread.
    If the main thread is unresponsive for a defined timeout, it dumps stack traces 
    to a debug log file.
    """
    def __init__(self, root, timeout=5.0, check_interval=1.0):
        super().__init__(daemon=True)
        self.root = root
        self.timeout = timeout
        self.check_interval = check_interval
        self.last_heartbeat = time.time()
        self.stop_event = threading.Event()
        self.freeze_count = 0
        self.log_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Logs", "freeze_debug.txt")

    def run(self):
        print(f"[WATCHDOG] UI Watchdog active (Timeout: {self.timeout}s).")
        while not self.stop_event.is_set():
            # Signal the main thread to update the heartbeat via after_idle
            try:
                self.root.after_idle(self._update_heartbeat)
            except Exception as e:
                print(f"[WATCHDOG] Could not signal main thread: {e}")
                break
            
            time.sleep(self.check_interval)
            
            # Check if we've exceeded the timeout since the last heartbeat
            elapsed = time.time() - self.last_heartbeat
            if elapsed > self.timeout:
                self.freeze_count += 1
                self._report_freeze(elapsed)
                # Avoid spamming multiple reports for the same long freeze
                time.sleep(self.timeout)

    def _update_heartbeat(self):
        """Called by the main thread to prove it is still alive and processing events."""
        self.last_heartbeat = time.time()

    def _report_freeze(self, duration):
        """Dumps current stack traces for all threads to help identify the block."""
        try:
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
            msg = f"\n!!! UI FREEZE DETECTED !!! Duration: {duration:.2f}s | Timestamp: {timestamp}\n"
            print(msg, file=sys.__stderr__) # Print to real stderr, not the redirected one
            
            os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*80}\n")
                f.write(f"FREEZE DETECTED AT: {timestamp}\n")
                f.write(f"Measured Duration: {duration:.2f}s\n")
                f.write(f"Freeze Event Count: {self.freeze_count}\n")
                f.write(f"{'='*80}\n\n")
                
                f.write("--- THREAD STACK DUMP ---\n")
                # sys._current_frames() returns a mapping of thread IDs to stack frames
                for thread_id, frame in sys._current_frames().items():
                    f.write(f"\nThread ID: {thread_id}")
                    if thread_id == threading.main_thread().ident:
                        f.write(" [MAIN THREAD - LIKELY BLOCKED]")
                    f.write("\n")
                    f.write("".join(traceback.format_stack(frame)))
                    f.write("-" * 40 + "\n")
                
                f.write("\n--- END OF DUMP ---\n")
                f.flush()
        except Exception as e:
            print(f"[WATCHDOG] Failed to generate freeze report: {e}", file=sys.__stderr__)

    def stop(self):
        """Gracefully stops the watchdog."""
        self.stop_event.set()
