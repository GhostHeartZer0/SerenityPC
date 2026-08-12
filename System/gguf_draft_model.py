import numpy as np

class GgufDraftModel:
    """Wraps a secondary Llama model to act as a speculator draft model for llama-cpp-python."""
    def __init__(self, draft_model_path, n_gpu_layers=0, n_ctx=2048):
        from llama_cpp import Llama
        try:
            from System.serenity_utils import HardwareProfile
            draft_threads = HardwareProfile.get_optimal_threads(is_draft=True)
        except Exception:
            draft_threads = 2

        # Load the assistant draft model with lightweight settings
        try:
            self.draft_llm = Llama(
                model_path=draft_model_path,
                n_gpu_layers=n_gpu_layers,
                n_ctx=n_ctx,
                n_threads=draft_threads, # Dynamic lightweight thread allocation
                n_batch=512,
                verbose=False
            )
        except Exception as err:
            from System.serenity_utils import patch_gguf_architecture
            if patch_gguf_architecture(draft_model_path, new_arch="llama"):
                self.draft_llm = Llama(
                    model_path=draft_model_path,
                    n_gpu_layers=n_gpu_layers,
                    n_ctx=n_ctx,
                    n_threads=draft_threads,
                    n_batch=512,
                    verbose=False
                )
            else:
                raise err
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
