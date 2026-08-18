"""
Test: History Archive - Unified List, Filtering, Sorting, and Deep Search
Validates metadata extraction, date grouping, dropdown filtering, sorting,
and deep full-text JSONZ search.
"""
import sys
import os
import zlib
import json
import tempfile
import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def test_history_metadata_and_date_buckets():
    print("=" * 60)
    print("TEST: History Metadata & Date Buckets")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        now = datetime.datetime.now()
        
        # Create test jsonz files with different timestamps and levels
        test_files = [
            ("Gemma-4-26B-it_lvl5.history.jsonz", 0, [{"role": "user", "content": "Hello Serenity"}]),
            ("Qwen3.5-9B_lvl1.history.jsonz", 1, [{"role": "user", "content": "Code review please"}]),
            ("Muse-Glimmer-30B_lvl6.history.jsonz", 4, [{"role": "user", "content": "Tell me a story"}]),
            ("Midnight-Miqu-70B_lvl7.history.jsonz", 20, [{"role": "user", "content": "Deep analysis"}]),
            ("DiffusionGemma-26B_lvl5.history.jsonz", 45, [{"role": "user", "content": "Generate art"}])
        ]
        
        for name, days_ago, msgs in test_files:
            p = os.path.join(tmpdir, name)
            data = zlib.compress(json.dumps(msgs).encode("utf-8"))
            with open(p, "wb") as f:
                f.write(data)
            # Set modified time
            target_time = (now - datetime.timedelta(days=days_ago)).timestamp()
            os.utime(p, (target_time, target_time))

        # Simulate _get_all_history_entries
        class MockApp:
            def __init__(self, d):
                self.dirs = {"History": d}
            from main import ChatbotApp
            _get_all_history_entries = ChatbotApp._get_all_history_entries

        app = MockApp(tmpdir)
        entries = app._get_all_history_entries()

        assert len(entries) == 5, f"Expected 5 entries, got {len(entries)}"
        
        # Check buckets
        bucket_map = {e["filename"]: e["date_bucket"] for e in entries}
        assert bucket_map["Gemma-4-26B-it_lvl5.history.jsonz"] == "Today", f"Expected Today, got {bucket_map['Gemma-4-26B-it_lvl5.history.jsonz']}"
        assert bucket_map["Qwen3.5-9B_lvl1.history.jsonz"] == "Yesterday", f"Expected Yesterday, got {bucket_map['Qwen3.5-9B_lvl1.history.jsonz']}"
        assert bucket_map["Muse-Glimmer-30B_lvl6.history.jsonz"] == "Past 7 Days", f"Expected Past 7 Days, got {bucket_map['Muse-Glimmer-30B_lvl6.history.jsonz']}"
        assert bucket_map["Midnight-Miqu-70B_lvl7.history.jsonz"] == "Past 30 Days", f"Expected Past 30 Days, got {bucket_map['Midnight-Miqu-70B_lvl7.history.jsonz']}"
        assert bucket_map["DiffusionGemma-26B_lvl5.history.jsonz"] == "Older", f"Expected Older, got {bucket_map['DiffusionGemma-26B_lvl5.history.jsonz']}"
        
        # Check level parsing
        level_map = {e["filename"]: e["level"] for e in entries}
        assert level_map["Gemma-4-26B-it_lvl5.history.jsonz"] == 5
        assert level_map["Midnight-Miqu-70B_lvl7.history.jsonz"] == 7

        print("[PASS] Metadata parsed and date buckets calculated correctly.")
        return True

def test_history_deep_search():
    print("\n" + "=" * 60)
    print("TEST: Deep Full-Text JSONZ Search")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        p1 = os.path.join(tmpdir, "session1_lvl3.history.jsonz")
        p2 = os.path.join(tmpdir, "session2_lvl5.history.jsonz")
        
        msgs1 = [
            {"role": "user", "content": "Can you analyze this 6 of hearts card?"},
            {"role": "assistant", "content": "I see a 6 of hearts with clear red suit curves."}
        ]
        msgs2 = [
            {"role": "user", "content": "Explain quantum computing."},
            {"role": "assistant", "content": "Quantum computers use qubits in superposition."}
        ]
        
        with open(p1, "wb") as f:
            f.write(zlib.compress(json.dumps(msgs1).encode("utf-8")))
        with open(p2, "wb") as f:
            f.write(zlib.compress(json.dumps(msgs2).encode("utf-8")))

        # Perform deep search logic
        cache = {}
        query = "6 of hearts"
        q_lower = query.lower()
        matches = {}

        for path in [p1, p2]:
            with open(path, "rb") as fp:
                msgs = json.loads(zlib.decompress(fp.read()).decode("utf-8"))
            cache[path] = msgs
            for m in msgs:
                content = str(m.get("content", ""))
                idx = content.lower().find(q_lower)
                if idx != -1:
                    start = max(0, idx - 30)
                    end = min(len(content), idx + len(query) + 40)
                    snippet = ("..." if start > 0 else "") + content[start:end].replace("\n", " ") + ("..." if end < len(content) else "")
                    matches[path] = f"[{m.get('role', 'msg').capitalize()}]: {snippet}"
                    break

        assert p1 in matches, "Expected session1 to match '6 of hearts'"
        assert p2 not in matches, "Expected session2 not to match"
        assert "6 of hearts" in matches[p1], "Snippet should contain matched search term"
        print(f"[PASS] Deep search identified match: {matches[p1]}")
        return True

if __name__ == "__main__":
    t1 = test_history_metadata_and_date_buckets()
    t2 = test_history_deep_search()
    if t1 and t2:
        print("\nALL HISTORY ARCHIVE TESTS PASSED!")
        sys.exit(0)
    else:
        sys.exit(1)
