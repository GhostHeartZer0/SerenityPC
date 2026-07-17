import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from System.kv_manager import KVManager
from System.tri_attention_core import TriAttentionScorer
import logging

logging.basicConfig(level=logging.INFO)

def test_message_pruning():
    print("--- Test: Message Array Pruning ---")
    manager = KVManager(max_context_tokens=100, prune_ratio=0.5) 
    
    # Create fake messages (each char is roughly 1/4 token, so ~40 chars = 10 tokens)
    # We will make 12 messages of 200 chars each (50 tokens each) = 600 tokens total
    # This heavily exceeds the 100 token max_context_tokens.
    messages = [
        {"role": "system", "content": "You are a highly capable AI. " * 5}, # msg 0 (System)
        {"role": "user", "content": "Hello! " * 20}, # msg 1
        {"role": "assistant", "content": "Hello user! " * 20}, # msg 2
        {"role": "user", "content": "Question 1? " * 15}, # msg 3
        {"role": "assistant", "content": "Ok here. " * 25}, # msg 4
        {"role": "user", "content": "Let's explore AI. " * 12}, # msg 5
        {"role": "assistant", "content": "It is fun. " * 30}, # msg 6
        {"role": "user", "content": "Wait. " * 30}, # msg 7
        {"role": "assistant", "content": "I am waiting. " * 20}, # msg 8
        {"role": "user", "content": "Last question... " * 10}, # msg 9
    ]
    
    start_tokens = manager._get_messages_token_count(messages)
    pruned = manager.enforce_kv_budget(messages)
    end_tokens = manager._get_messages_token_count(pruned)
    
    print(f"Start Tokens: {start_tokens} | Pruned Tokens: {end_tokens}")
    print(f"Original length: {len(messages)} | Pruned length: {len(pruned)}")
    
    # Verify System is kept
    assert pruned[0]["role"] == "system", "System prompt was dropped!"
    # Verify latest is kept
    assert pruned[-1]["content"] == messages[-1]["content"], "Latest turn was dropped!"
    
    print("Message Pruning passes.\n")

def test_string_pruning():
    print("--- Test: String Pruning (Deep Cook) ---")
    manager = KVManager(max_context_tokens=150, prune_ratio=0.4)
    
    # Create a large chunked string
    chunks = []
    for i in range(10):
        # 400 chars ~ 100 tokens per chunk
        chunks.append(f"Cycle {i} Log:\n" + "Blah " * 100)
        
    full_string = "\n\n".join(chunks)
    
    start_tokens = manager._estimate_tokens(full_string)
    pruned_string = manager.enforce_string_kv_budget(full_string)
    end_tokens = manager._estimate_tokens(pruned_string)
    
    print(f"Start Tokens: {start_tokens} | Pruned Tokens: {end_tokens}")
    assert "Cycle 9" in pruned_string, "Failed to keep the most recent reasoning cycle!"
    
    print("String Pruning passes.\n")

if __name__ == "__main__":
    test_message_pruning()
    test_string_pruning()
    print("All tests passed successfully.")
