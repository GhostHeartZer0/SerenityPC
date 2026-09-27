import unittest
import os
import shutil
import tempfile
import json
import zlib
import tkinter as tk
from unittest.mock import MagicMock, patch

from System.vault_manager import VaultManager
from main import ChatbotApp


class TestVaultHistoryUnlock(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.system_dir = os.path.join(self.test_dir, "System")
        self.history_dir = os.path.join(self.test_dir, "History")
        self.users_dir = os.path.join(self.test_dir, "Users")
        os.makedirs(self.system_dir, exist_ok=True)
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.users_dir, exist_ok=True)

        self.vault = VaultManager(history_dir=self.history_dir, state_dir=self.system_dir)
        self.password = "TestMasterPassword123!"
        success, msg = self.vault.set_password(self.password)
        self.assertTrue(success, f"Failed to set password: {msg}")

        self.root = tk.Tk()
        self.root.withdraw()

    def tearDown(self):
        try:
            self.root.destroy()
        except Exception:
            pass
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_set_user_context_preserves_unlocked_session_key(self):
        """Verify set_user_context does not relock when state file is unchanged."""
        # Unlock vault
        self.assertTrue(self.vault.unlock(self.password))
        self.assertFalse(self.vault.is_locked())
        cached_key = self.vault._session_key
        self.assertIsNotNone(cached_key)

        # Call set_user_context with same state context
        user_dir = os.path.join(self.users_dir, "TestUser")
        user_hist_dir = os.path.join(self.history_dir, "TestUser")
        self.vault.set_user_context(user_dir=user_dir, history_dir=user_hist_dir)

        # Must still be unlocked with the same key
        self.assertFalse(self.vault.is_locked())
        self.assertEqual(self.vault._session_key, cached_key)

    def test_encrypted_history_file_read_after_unlock(self):
        """Verify reading encrypted history archive succeeds without reprompt after unlock."""
        self.assertTrue(self.vault.unlock(self.password))
        
        # Write encrypted history archive
        enc_file = os.path.join(self.history_dir, "test_lvl4.history.encz")
        sample_messages = [
            {"role": "user", "content": "Hello Serenity"},
            {"role": "assistant", "content": "Hello user"}
        ]
        self.vault.write_history_messages(enc_file, sample_messages)
        self.assertTrue(os.path.exists(enc_file))

        # Re-read messages multiple times
        read_back_1 = self.vault.read_history_messages(enc_file)
        self.assertEqual(read_back_1, sample_messages)

        # Simulate context update like switch_user
        user_dir = os.path.join(self.users_dir, "TestUser")
        self.vault.set_user_context(user_dir, self.history_dir)
        read_back_2 = self.vault.read_history_messages(enc_file)
        self.assertEqual(read_back_2, sample_messages)

    def test_history_unlock_modal_close_exits_history_not_app(self):
        """Verify closing unlock modal during history access exits history without calling root.destroy()."""
        app = MagicMock(spec=ChatbotApp)
        app.root = self.root
        app.config = {"text_scale": 100}
        app.fonts = {
            "large": ("Segoe UI", 12),
            "ui_label": ("Segoe UI", 9),
            "small": ("Segoe UI", 8),
            "ui_button": ("Segoe UI", 9),
            "main": ("Segoe UI", 10),
            "ui_small": ("Segoe UI", 8)
        }
        app.get_active_username.return_value = "TestUser"
        app.list_user_profiles.return_value = ["Default", "TestUser"]
        app.active_tab = "history"
        app.vault_manager = self.vault
        app._vault_modal_open = False
        app._back_history = MagicMock()
        app.show_active_chat = MagicMock()

        # Modal opened with on_unlock_callback (simulating _load_selected_history)
        cb_called = False
        def unlock_cb():
            nonlocal cb_called
            cb_called = True

        # Call show_vault_unlock_modal
        ChatbotApp.show_vault_unlock_modal(app, on_unlock_callback=unlock_cb)
        self.assertTrue(app._vault_modal_open)

        # Trigger WM_DELETE_WINDOW (_on_close_modal)
        # Find the open Toplevel window
        toplevels = [w for w in self.root.winfo_children() if isinstance(w, tk.Toplevel)]
        self.assertTrue(len(toplevels) > 0)
        win = toplevels[0]

        # Find Cancel button inside toplevel
        cancel_btn = None
        for child in win.winfo_children():
            if isinstance(child, tk.Frame):
                for sub in child.winfo_children():
                    if isinstance(sub, tk.Button) and sub.cget("text") == "Cancel":
                        cancel_btn = sub
                        break
        self.assertIsNotNone(cancel_btn, "Cancel button not found in unlock modal")

        with patch.object(self.root, "destroy") as mock_destroy:
            cancel_btn.invoke()
            mock_destroy.assert_not_called()
            app._back_history.assert_called_once()
            app.show_active_chat.assert_called_once()

        self.assertFalse(cb_called)


if __name__ == "__main__":
    unittest.main()
