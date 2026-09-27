import os
import unittest
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler
import llama_cpp.llama_chat_format as lcf
from main import ChatbotApp

class TestChatFormatRecovery(unittest.TestCase):
    def test_format_recovery(self):
        app = ChatbotApp.__new__(ChatbotApp)
        path = 'C:/Users/ccrg6/Desktop/Hub/Serenities/SerenityPC/Models/E4B/gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf'
        proj = 'C:/Users/ccrg6/Desktop/Hub/Serenities/SerenityPC/Models/E4B/mmproj-BF16.gguf'
        if not os.path.exists(path) or not os.path.exists(proj):
            self.skipTest("Test model or projector not found.")
            
        app.model_path = path
        
        # 1. Model loaded with projector
        ch = Llava15ChatHandler(clip_model_path=proj, verbose=False)
        m = Llama(model_path=path, chat_handler=ch, n_ctx=512, vocab_only=True, verbose=False)
        app.model = m
        app._cached_chat_handler = ch
        
        # Initial ensure
        app._ensure_model_chat_format(m)
        self.assertEqual(m.chat_format, 'chat_template.default')
        
        # 2. Text-only turn: chat_handler detached
        m.chat_handler = None
        app._ensure_model_chat_format(m)
        self.assertEqual(m.chat_format, 'chat_template.default')
        
        # Handler resolution check (must never raise KeyError: None)
        handler = m.chat_handler or m._chat_handlers.get(m.chat_format) or lcf.get_chat_completion_handler(m.chat_format)
        self.assertTrue(callable(handler))
        
        # 3. Multimodal turn: chat_handler restored
        self.assertTrue(app._ensure_chat_handler(interactive=False))
        self.assertIs(m.chat_handler, ch)

if __name__ == "__main__":
    unittest.main()
