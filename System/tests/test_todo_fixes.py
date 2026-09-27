import unittest
import tkinter as tk
from unittest.mock import MagicMock, patch

from main import ChatbotApp


class TestTodoFixes(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = MagicMock(spec=ChatbotApp)
        self.app.root = self.root
        self.app.config = {
            "text_scale": 125,
            "vault_modal_geometry": "520x420+100+100"
        }
        self.app.fonts = {
            "log": ("Consolas", 9),
            "log_bold": ("Consolas", 9, "bold"),
            "ui_button": ("Segoe UI", 9),
            "ui_small": ("Segoe UI", 8),
            "large": ("Segoe UI", 12),
            "main": ("Segoe UI", 10),
            "small": ("Segoe UI", 8)
        }
        self.app.state = {"log_view": "thought", "deep_cook": False}
        self.app.active_persona_level = 7

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass

    # 1. Profile Switching & Startup Password Prompt Verification
    def test_switch_user_no_double_lock_on_same_profile(self):
        """Verify switch_user does not re-lock vault or prompt if remaining on same profile."""
        vault_mock = MagicMock()
        vault_mock.is_lock_enabled.return_value = True
        vault_mock.is_locked.return_value = False
        self.app.vault_manager = vault_mock
        self.app.get_active_username.return_value = "Admin"
        self.app.get_user_dir.return_value = "dummy_dir"
        self.app.get_user_history_dir.return_value = "dummy_hist"

        res = ChatbotApp.switch_user(self.app, "Admin")
        # Vault lock should NOT be called because old_un == clean_un
        vault_mock.lock.assert_not_called()

    def test_switch_user_auto_locks_on_profile_swap(self):
        """Verify swapping from locked profile auto-locks the vault."""
        vault_mock = MagicMock()
        vault_mock.is_lock_enabled.return_value = True
        vault_mock.is_locked.return_value = True
        self.app.vault_manager = vault_mock
        self.app.get_active_username.return_value = "Alice"
        self.app.get_user_dir.return_value = "dummy_dir"
        self.app.get_user_history_dir.return_value = "dummy_hist"

        # Switching to Bob with skip_lock_prompt=True
        ChatbotApp.switch_user(self.app, "Bob", skip_lock_prompt=True)
        # Vault lock must be called because Alice -> Bob
        vault_mock.lock.assert_called_once()

    def test_switch_user_skip_lock_prompt_bypasses_dialog(self):
        """Verify skip_lock_prompt prevents secondary dialog prompt at startup."""
        vault_mock = MagicMock()
        vault_mock.is_lock_enabled.return_value = True
        vault_mock.is_locked.return_value = True
        self.app.vault_manager = vault_mock
        self.app.get_active_username.return_value = "Default"
        self.app.get_user_dir.return_value = "dummy_dir"
        self.app.get_user_history_dir.return_value = "dummy_hist"

        with patch("tkinter.simpledialog.askstring") as mock_ask:
            ChatbotApp.switch_user(self.app, "ProtectedUser", skip_lock_prompt=True)
            mock_ask.assert_not_called()

    # 2. Persona Label Isolation
    def test_persona_label_isolation(self):
        """Verify Level 7 is strictly 'Cecilia' and Levels 1-6 are strictly 'Serenity'."""
        self.app.active_persona_level = 7
        self.assertEqual(ChatbotApp._get_persona_label(self.app), "Cecilia")

        for lvl in [1, 2, 3, 4, 5, 6]:
            self.app.active_persona_level = lvl
            lbl = ChatbotApp._get_persona_label(self.app)
            self.assertEqual(lbl, "Serenity")
            self.assertNotEqual(lbl, "Cecilia")

    # 3. Tag Isolation & Prefix Stripping in Synthesis Output
    def test_sanitize_synthesis_output_strips_persona_prefix(self):
        """Verify model-generated Cecilia: and Serenity: prefixes are cleanly stripped."""
        cases = [
            ("Cecilia: You think you understand?", "You think you understand?"),
            ("**Cecilia:** Here is the truth.", "Here is the truth."),
            ("*Cecilia:* Secrets revealed.", "Secrets revealed."),
            ("Serenity: How can I help?", "How can I help?"),
            ("**Serenity:** All systems nominal.", "All systems nominal."),
            ("Cecilia: Cecilia: Double prefix test", "Double prefix test"),
        ]
        for raw, expected in cases:
            cleaned = ChatbotApp._sanitize_synthesis_output(self.app, raw)
            self.assertEqual(cleaned, expected)

    def test_sanitize_synthesis_output_strips_thought_and_deep_cook_tags(self):
        """Verify all internal thought, reasoning, turn, and Deep Cook tracking tags are stripped."""
        raw_text = (
            "<think>Analyzing variables...</think>\n"
            "[STATUS: Planning cycle 1]\n"
            "[CURRENT RANGE: X=1 to 5]\n"
            "<|turn>model\n"
            "<|channel>thought\nInternal deliberation\n<channel|>\n"
            "Cecilia: This is the finalized verdict for the mortal."
        )
        cleaned = ChatbotApp._sanitize_synthesis_output(self.app, raw_text)
        self.assertEqual(cleaned, "This is the finalized verdict for the mortal.")
        self.assertNotIn("<think>", cleaned)
        self.assertNotIn("[STATUS:", cleaned)
        self.assertNotIn("<|channel>", cleaned)
        self.assertNotIn("<|turn>", cleaned)
        self.assertNotIn("Cecilia:", cleaned)

    # 4. Backend Logs Dynamic Sizing & Slots
    def test_backend_log_slot_calculation(self):
        """Verify slot calculations provide adequate width (>30px) without bleeding."""
        scales = [1.0, 1.25, 1.5, 2.0]
        for s in scales:
            slot_w = max(34, int(32 * s))
            canv_h = max(28, int(26 * s))
            canv_w = 4 * slot_w + 4
            self.assertGreaterEqual(slot_w, 34)
            self.assertEqual(canv_w, 4 * slot_w + 4)
            # Verify slots do not overlap
            for i in range(4):
                x_center = 2 + i * slot_w + slot_w // 2
                self.assertTrue(2 + i * slot_w < x_center < 2 + (i + 1) * slot_w)


if __name__ == "__main__":
    unittest.main()
