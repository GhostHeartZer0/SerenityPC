import numpy as np

class GgufDraftModel:
    """Wraps a secondary Llama model to act as a speculator draft model for llama-cpp-python."""
    def __init__(self, draft_model_path, n_gpu_layers=0, n_ctx=2048):
        from llama_cpp import Llama
        # Load the assistant draft model with lightweight settings
        self.draft_llm = Llama(
            model_path=draft_model_path,
            n_gpu_layers=n_gpu_layers,
            n_ctx=n_ctx,
            n_threads=4, # Use fewer threads for draft model
            n_batch=512,
            verbose=False
        )
        self.num_pred_tokens = 5  # Standard speculative sequence length

    def __call__(self, input_ids, **kwargs):
        # input_ids is a numpy array of tokens generated so far
        tokens_list = input_ids.tolist()
        
        # Reset and pre-evaluate up to n_past
        self.draft_llm.reset()
        self.draft_llm.eval(tokens_list)
        
        # Sample next tokens greedily from draft model
        drafted_tokens = []
        for _ in range(self.num_pred_tokens):
            tok = self.draft_llm.sample(temp=0.0) # Greedy sampling
            drafted_tokens.append(tok)
            self.draft_llm.eval([tok])
            
        return np.array(drafted_tokens, dtype=np.intc)
