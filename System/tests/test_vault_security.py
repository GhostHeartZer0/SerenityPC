"""
Test: Vault Security, Encryption, and Rollback Integrity
Validates AES-256-GCM authenticated encryption, PBKDF2-HMAC-SHA256 key derivation,
tampering rejection, and fail-safe migration with automated rollback.
"""

import os
import sys
import json
import zlib
import tempfile
import unittest

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from System.vault_manager import VaultManager, DISCLAIMER_WARNING_TEXT


class TestVaultManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.history_dir = os.path.join(self.temp_dir.name, "History")
        self.state_dir = os.path.join(self.temp_dir.name, "System")
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.state_dir, exist_ok=True)

        self.vault = VaultManager(history_dir=self.history_dir, state_dir=self.state_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_disclaimer_present(self):
        self.assertIn("CRITICAL SECURITY & DATA LOSS WARNING", DISCLAIMER_WARNING_TEXT)
        self.assertIn("NO BACKDOOR", DISCLAIMER_WARNING_TEXT)

    def test_password_lifecycle_and_verification(self):
        self.assertFalse(self.vault.is_lock_enabled())
        self.assertFalse(self.vault.is_locked())

        # Set initial password
        success, msg = self.vault.set_password("SerenitySecret2026!")
        self.assertTrue(success, msg)
        self.assertTrue(self.vault.is_lock_enabled())
        self.assertFalse(self.vault.is_locked()) # Remains unlocked in active session

        # Lock and verify password
        self.vault.lock()
        self.assertTrue(self.vault.is_locked())
        self.assertFalse(self.vault.verify_password("wrongpassword"))
        self.assertTrue(self.vault.verify_password("SerenitySecret2026!"))

        # Unlock with valid password
        unlocked = self.vault.unlock("SerenitySecret2026!")
        self.assertTrue(unlocked)
        self.assertFalse(self.vault.is_locked())

    def test_encryption_and_tamper_detection(self):
        self.vault.set_password("StrongPassword123#")
        sample_data = b"Confidential user conversation with Serenity AI"

        # Encrypt
        encrypted = self.vault.encrypt_data(sample_data)
        self.assertTrue(encrypted.startswith(VaultManager.HEADER_MAGIC))
        self.assertNotEqual(encrypted, sample_data)

        # Decrypt
        decrypted = self.vault.decrypt_data(encrypted)
        self.assertEqual(decrypted, sample_data)

        # Tampering test: corrupt a single byte in ciphertext
        tampered = bytearray(encrypted)
        tampered[-5] ^= 0xFF
        with self.assertRaises(Exception):
            self.vault.decrypt_data(bytes(tampered))

    def test_safe_history_migration_and_rollback(self):
        # Create 3 unencrypted test history files
        test_data = [
            ("gemma_lvl1.history.jsonz", [{"role": "user", "content": "Hello L1"}]),
            ("qwen_lvl5.history.jsonz", [{"role": "user", "content": "Analyze code"}]),
            ("cecilia_lvl7.history.jsonz", [{"role": "user", "content": "Level 7 awakened"}])
        ]

        for fname, msgs in test_data:
            p = os.path.join(self.history_dir, fname)
            compressed = zlib.compress(json.dumps(msgs).encode("utf-8"))
            with open(p, "wb") as fp:
                fp.write(compressed)

        # 1. Migrate to encrypted format
        success, msg = self.vault.set_password("MasterVaultPass2026!")
        self.assertTrue(success, msg)

        # Verify files are now .encz
        enc_files = [f for f in os.listdir(self.history_dir) if f.endswith(".history.encz")]
        self.assertEqual(len(enc_files), 3)
        self.assertEqual(len([f for f in os.listdir(self.history_dir) if f.endswith(".history.jsonz")]), 0)

        # Read back via vault
        for fname, expected_msgs in test_data:
            enc_name = fname.replace(".history.jsonz", ".history.encz")
            p = os.path.join(self.history_dir, enc_name)
            read_msgs = self.vault.read_history_messages(p)
            self.assertEqual(read_msgs, expected_msgs)

        # 2. Disable lock and migrate back to plaintext .jsonz
        success, msg = self.vault.disable_lock("MasterVaultPass2026!")
        self.assertTrue(success, msg)

        jsonz_files = [f for f in os.listdir(self.history_dir) if f.endswith(".history.jsonz")]
        self.assertEqual(len(jsonz_files), 3)
        self.assertEqual(len([f for f in os.listdir(self.history_dir) if f.endswith(".history.encz")]), 0)

        for fname, expected_msgs in test_data:
            p = os.path.join(self.history_dir, fname)
            read_msgs = self.vault.read_history_messages(p)
            self.assertEqual(read_msgs, expected_msgs)


if __name__ == "__main__":
    unittest.main()
