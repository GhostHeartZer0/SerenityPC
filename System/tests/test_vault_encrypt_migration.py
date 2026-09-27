# System/tests/test_vault_encrypt_migration.py
import os
import shutil
import tempfile
import unittest
import zlib
import json

from System.vault_manager import VaultManager

class TestVaultEncryptMigration(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.history_dir = os.path.join(self.tmp_dir, "History")
        self.state_dir = os.path.join(self.tmp_dir, "State")
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)
        self.vault = VaultManager(history_dir=self.history_dir, state_dir=self.state_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_encrypt_vault_creates_password_if_none(self):
        sample_file = os.path.join(self.history_dir, "chat_lvl3.history.jsonz")
        data = [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there"}]
        with open(sample_file, "wb") as f:
            f.write(zlib.compress(json.dumps(data).encode("utf-8")))

        self.assertFalse(self.vault.is_lock_enabled())
        success, msg = self.vault.encrypt_vault("MasterPassword123")
        self.assertTrue(success, msg)
        self.assertTrue(self.vault.is_lock_enabled())

        enc_file = os.path.join(self.history_dir, "chat_lvl3.history.encz")
        self.assertTrue(os.path.exists(enc_file))
        self.assertFalse(os.path.exists(sample_file))

        read_msgs = self.vault.read_history_messages(enc_file)
        self.assertEqual(read_msgs, data)

    def test_encrypt_vault_migrates_when_already_enabled(self):
        self.vault.set_password("MasterPassword123")
        self.assertTrue(self.vault.is_lock_enabled())

        sample_file = os.path.join(self.history_dir, "manual_lvl5.history.jsonz")
        data = [{"role": "user", "content": "Explain relativity"}]
        with open(sample_file, "wb") as f:
            f.write(zlib.compress(json.dumps(data).encode("utf-8")))

        success, msg = self.vault.encrypt_vault("MasterPassword123")
        self.assertTrue(success, msg)

        enc_file = os.path.join(self.history_dir, "manual_lvl5.history.encz")
        self.assertTrue(os.path.exists(enc_file))
        self.assertFalse(os.path.exists(sample_file))

        read_msgs = self.vault.read_history_messages(enc_file)
        self.assertEqual(read_msgs, data)

    def test_encrypt_vault_wrong_password_fails(self):
        self.vault.set_password("MasterPassword123")
        success, msg = self.vault.encrypt_vault("WrongPassword")
        self.assertFalse(success)
        self.assertIn("verification failed", msg.lower())

if __name__ == "__main__":
    unittest.main()
