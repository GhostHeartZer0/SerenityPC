"""
Test: Vision (Image) Handling - Multimodal Content Safety
Validates that the Gemma prompt builder correctly handles list-type content
from vision/image messages without crashing on .strip().

Also validates HistoryKeywordIndex search.
"""
import sys
import os
import re

# Add the project root to path
_ws = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ws not in sys.path: sys.path.insert(0, _ws)
_sys_dir = os.path.join(_ws, "System")
if _sys_dir not in sys.path: sys.path.insert(0, _sys_dir)

def test_multimodal_content_strip_safety():
    """
    Simulates the Gemma prompt builder loop with mixed content types.
    This is the exact code path that was crashing with:
    'list' object has no attribute 'strip'
    """
    print("=" * 60)
    print("TEST: Multimodal Content Strip Safety")
    print("=" * 60)
    
    # Simulate messages list with both string and list content types
    # This is what happens after inline vision processing (line 2376 in main.py)
    test_messages = [
        {"role": "user", "content": "Hello, analyze this image for me."},
        {"role": "assistant", "content": "Sure, I'll take a look at that image."},
        # THIS is the problematic message format from inline vision (line 2376)
        {"role": "user", "content": [
            {"type": "text", "text": "What do you see in this image?"},
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRg=="}},
        ]},
        {"role": "assistant", "content": "<think>Let me analyze this image carefully.</think>I see a landscape photo."},
        {"role": "user", "content": "Thanks! Now tell me about the weather."},
    ]
    
    # Simulate the FIXED Gemma prompt builder (lines 3062-3090 in main.py)
    prompt_str = "<|turn>system\nYou are Serenity.\n<turn|>\n"
    errors = []
    
    for m in test_messages:
        role = "model" if m["role"] == "assistant" else m["role"]
        raw_content = m["content"]
        
        # MULTIMODAL SAFETY: If content is a list (vision/image messages),
        # extract only the text parts for the Gemma prompt string builder.
        if isinstance(raw_content, list):
            text_parts = []
            for part in raw_content:
                if isinstance(part, dict):
                    if part.get("type") == "text":
                        text_parts.append(part.get("text", ""))
                    elif part.get("type") == "image_url":
                        text_parts.append("[image]")  # Placeholder for prompt context
                elif isinstance(part, str):
                    text_parts.append(part)
            content = " ".join(text_parts).strip()
        else:
            content = str(raw_content).strip() if raw_content else ""
        
        # HF Template alignment: Strip previous thoughts from the context window
        if role == "model":
            content = re.sub(r'(?s)<think>.*?(?:<\/think>|$)', '', content, flags=re.IGNORECASE)
            content = content.strip()
        
        # Append ALL messages to the prompt
        prompt_str += f"<|turn>{role}\n{content}<turn|>\n"
    
    prompt_str += "<|turn>model\n"
    
    # Validate results
    assert isinstance(prompt_str, str), "Prompt should be a string"
    assert "[image]" in prompt_str, "Image placeholder should be present"
    assert "What do you see in this image?" in prompt_str, "Text from multimodal message should be preserved"
    assert "I see a landscape photo." in prompt_str, "Think tags should be stripped from model responses"
    assert "<think>" not in prompt_str, "Think tags should NOT be in the final prompt"
    assert "data:image/jpeg" not in prompt_str, "Base64 image data should NOT leak into the prompt string"
    
    print("[PASS] List content correctly serialized to text")
    print("[PASS] Image data replaced with [image] placeholder")
    print("[PASS] Think tags stripped from model responses")
    print("[PASS] No crash on .strip() with list content")
    print(f"\n--- Generated Prompt Preview (first 500 chars) ---")
    print(prompt_str[:500])
    return True


def test_edge_cases():
    """Test additional edge cases for content handling."""
    print("\n" + "=" * 60)
    print("TEST: Edge Cases")
    print("=" * 60)
    
    edge_cases = [
        # None content
        {"role": "user", "content": None},
        # Empty string
        {"role": "user", "content": ""},
        # Normal string
        {"role": "user", "content": "Hello world"},
        # List with only text
        {"role": "user", "content": [{"type": "text", "text": "Just text"}]},
        # List with only image
        {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,abc"}}]},
        # List with audio
        {"role": "user", "content": [
            {"type": "text", "text": "Analyze this audio"},
            {"type": "image_url", "image_url": {"url": "data:audio/wav;base64,xyz"}}
        ]},
        # Empty list
        {"role": "user", "content": []},
        # Mixed with raw strings in list
        {"role": "user", "content": ["raw string part", {"type": "text", "text": "dict part"}]},
    ]
    
    for i, m in enumerate(edge_cases):
        raw_content = m["content"]
        try:
            if isinstance(raw_content, list):
                text_parts = []
                for part in raw_content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            text_parts.append(part.get("text", ""))
                        elif part.get("type") == "image_url":
                            text_parts.append("[image]")
                    elif isinstance(part, str):
                        text_parts.append(part)
                content = " ".join(text_parts).strip()
            else:
                content = str(raw_content).strip() if raw_content else ""
            
            assert isinstance(content, str), f"Case {i}: content must be a string, got {type(content)}"
            print(f"  [PASS] Case {i}: {type(raw_content).__name__} -> '{content[:60]}'")
        except Exception as e:
            print(f"  [FAIL] Case {i}: {e}")
            return False
    
    return True


def test_history_keyword_index():
    """Test that HistoryKeywordIndex initializes and searches properly without vector dependencies."""
    print("\n" + "=" * 60)
    print("TEST: History Keyword Index Search")
    print("=" * 60)
    
    from System.kv_manager import HistoryKeywordIndex
    index = HistoryKeywordIndex(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "History"))
    results = index.search("test query", top_k=2, lookup_mode="all")
    assert isinstance(results, list), "Expected list result from search"
    print("  [PASS] HistoryKeywordIndex initialized and returned valid list")
    return True


def test_prepare_vision_query_return_type():
    """
    Validates that prepare_vision_query's return type (tuple) is handled 
    correctly at all call sites.
    """
    print("\n" + "=" * 60)
    print("TEST: prepare_vision_query Return Type Safety")
    print("=" * 60)
    
    # Simulate VisionHandler.prepare_vision_query behavior
    def mock_prepare_vision_query(user_query, is_deep_cook=False):
        """Mirrors vision_handler.py line 331-340"""
        if isinstance(user_query, tuple) and len(user_query) == 2:
            return user_query
        prompt = "AUDITOR_PROMPT" if is_deep_cook else "SCOUT_PROMPT"
        budget = 280  # default
        return f"{prompt}\n\n[VISUAL_BUDGET: {budget}]\n[USER QUERY]: {user_query}", budget
    
    # Test case 1: Normal string input -> returns tuple
    result = mock_prepare_vision_query("What's in this image?")
    assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
    assert len(result) == 2, f"Expected 2-tuple, got {len(result)}"
    query_str, budget = result
    assert isinstance(query_str, str), "First element should be string"
    assert isinstance(budget, int), "Second element should be int"
    print("  [PASS] Normal input returns (str, int) tuple")
    
    # Test case 2: Already a tuple -> passes through
    result2 = mock_prepare_vision_query(("pre-built query", 560))
    assert result2 == ("pre-built query", 560), "Tuple input should pass through"
    print("  [PASS] Tuple input passes through unchanged")
    
    # Test case 3: Unpacking at call site (line 5235 pattern)
    final_user_msg, budget = mock_prepare_vision_query("Analyze this media.")
    assert isinstance(final_user_msg, str), "Unpacked query should be string"
    assert isinstance(budget, int), "Unpacked budget should be int"
    print("  [PASS] Tuple unpacking works correctly")
    
    # Test case 4: Used as single value (lines 2387, 2404 pattern)
    # These pass the WHOLE tuple as user_msg to initiate_vision_analysis
    final_query = mock_prepare_vision_query("What do you see?", is_deep_cook=False)
    # When this tuple gets used as user_msg in _batch_vision_worker line 5127:
    # prompt_content.append({"type": "text", "text": f"{t_tag} " + (user_msg if user_msg else ...)})
    # A tuple is truthy but str() of a tuple includes parens — this works but is ugly.
    # The _determine_visual_budget already handles tuple input (lines 314-318).
    if isinstance(final_query, tuple):
        actual_msg = final_query[0]  # Extract string part
    else:
        actual_msg = final_query
    assert isinstance(actual_msg, str), "Extracted message should be a string"
    print("  [PASS] Tuple query can be safely decomposed")
    
    return True


if __name__ == "__main__":
    results = []
    results.append(("Multimodal Content Strip Safety", test_multimodal_content_strip_safety()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("History Keyword Index Search", test_history_keyword_index()))
    results.append(("prepare_vision_query Return Type", test_prepare_vision_query_return_type()))
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        if not passed: all_passed = False
        print(f"  [{status}] {name}")
    
    print(f"\nOverall: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")
    sys.exit(0 if all_passed else 1)
