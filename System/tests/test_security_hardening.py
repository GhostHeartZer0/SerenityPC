import os, sys, unittest
from unittest.mock import MagicMock, patch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from System.tool_registry import GemmaToolRegistry

class TestSecurityHardeningAndSash(unittest.TestCase):
    def setUp(self):
        self.workspace_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.mock_app = MagicMock()
        self.mock_app.script_dir = self.workspace_root
        self.mock_app.config = {'offline_mode': False}
        self.tool_reg = GemmaToolRegistry(self.mock_app)

    def test_sandboxed_file_read_allowed(self):
        p = os.path.join(self.workspace_root, 'TODO.txt')
        res = self.tool_reg.execute('read_file', {'path': p})
        self.assertNotIn('[SECURITY RESTRICTION]', res)
        self.assertIn('TODO', res)

    def test_sandboxed_file_read_blocked(self):
        p = os.path.abspath(os.path.join(self.workspace_root, '..', 'out.txt'))
        res = self.tool_reg.execute('read_file', {'path': p})
        self.assertIn('[SECURITY RESTRICTION]', res)

    def test_sandboxed_file_range_blocked(self):
        p = os.path.abspath(os.path.join(self.workspace_root, '..', 'out.txt'))
        res = self.tool_reg.execute('read_file_range', {'path': p, 'start': 1, 'end': 5})
        self.assertIn('[SECURITY RESTRICTION]', res)

    def test_redos_protection(self):
        from main import ChatbotApp
        app = MagicMock()
        app.script_dir = self.workspace_root
        app.state = {'staged_attachments': []}
        huge_input = '@[' + ('a' * 500) + ']' * 200 + ' normal'
        res = ChatbotApp._parse_and_stage_filename_imports(app, huge_input)
        self.assertIn('normal', res)

    @patch('subprocess.Popen')
    def test_launch_lore_book_security(self, mock_popen):
        from main import ChatbotApp
        app = MagicMock()
        app.script_dir = self.workspace_root
        ChatbotApp.launch_lore_book(app)
        mock_popen.assert_called_once()

if __name__ == '__main__':
    unittest.main()
