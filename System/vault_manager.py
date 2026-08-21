"""
Serenity Vault Security & Cryptographic Storage Manager.
Provides AES-256-GCM authenticated encryption for chat history archives,
PBKDF2-HMAC-SHA256 master key derivation, startup lock authentication,
and fail-safe transaction-backed archive migration with automatic rollback.
"""

import os
import sys
import json
import zlib
import shutil
import secrets
import hashlib
import datetime
from typing import Optional, Dict, Any, List, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    CRYPTO_AVAILABLE = True
except ImportError:
    AESGCM = None
    CRYPTO_AVAILABLE = False


DISCLAIMER_WARNING_TEXT = (
    "======================================================================\n"
    "⚠️ CRITICAL SECURITY & DATA LOSS WARNING ⚠️\n"
    "======================================================================\n"
    "SERENITY PC USES INDUSTRY-GRADE AES-256-GCM AUTHENTICATED ENCRYPTION.\n"
    "THERE IS NO BACKDOOR, MASTER RECOVERY KEY, OR CLOUD RESET MECHANISM.\n\n"
    "IF YOU LOSE OR FORGET YOUR MASTER PASSWORD, ALL ENCRYPTED CHAT ARCHIVES\n"
    "WILL BE PERMANENTLY AND IRREVERSIBLY UNREADABLE. NEITHER SERENITY NOR\n"
    "ANY DEVELOPER CAN RECOVER YOUR ENCRYPTED DATA WITHOUT THE PASSWORD.\n\n"
    "PROCEED ONLY IF YOU HAVE RECORDED OR MEMORIZED YOUR PASSWORD SAFELY.\n"
    "======================================================================"
)


class VaultManager:
    """Manages master password verification, AES-256-GCM encryption, and safe file migration."""
    
    PBKDF2_ITERATIONS = 250_000
    SALT_SIZE = 16
    NONCE_SIZE = 12
    HEADER_MAGIC = b"SERENITY_ENCZ_V1"

    def __init__(self, history_dir: str, state_dir: str):
        self.history_dir = os.path.abspath(history_dir)
        self.state_dir = os.path.abspath(state_dir)
        self.state_file = os.path.join(self.state_dir, "vault_state.json")
        
        self._session_key: Optional[bytes] = None
        self._is_locked: bool = True
        self._state: Dict[str, Any] = self._load_state()

        if self._state.get("lock_enabled", False):
            self._is_locked = True
            self._auto_unlock_from_env()
        else:
            self._is_locked = False

    def _find_env_file(self) -> Optional[str]:
        """Locates the .env file in workspace root or state parent directory."""
        candidates = [
            os.path.join(self.state_dir, ".env"),
            os.path.join(os.path.dirname(self.state_dir), ".env"),
            os.path.join(os.getcwd(), ".env"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def get_env_storage_key(self, env_path: Optional[str] = None) -> Optional[str]:
        """Reads encrypted storage key / password from .env or os.environ."""
        target = env_path or self._find_env_file()
        if target and os.path.isfile(target):
            try:
                with open(target, "r", encoding="utf-8") as fp:
                    for line in fp:
                        line = line.strip()
                        if not line or line.startswith("#") or "=" not in line:
                            continue
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k in ("SERENITY_ENCRYPTED_STORAGE_KEY", "SERENITY_STORAGE_KEY", "SERENITY_VAULT_KEY", "SERENITY_VAULT_PASSWORD") and v:
                            return v
            except Exception as e:
                print(f"[VAULT] Warning: Failed reading .env file: {e}", file=sys.stderr)
        for var in ("SERENITY_ENCRYPTED_STORAGE_KEY", "SERENITY_STORAGE_KEY", "SERENITY_VAULT_KEY", "SERENITY_VAULT_PASSWORD"):
            if val := os.environ.get(var):
                return val.strip().strip("'\"")
        return None

    def _auto_unlock_from_env(self):
        """Attempts auto-unlocking if a valid encrypted storage key or master password is provided in .env."""
        if not self.is_lock_enabled() or not self.is_crypto_available():
            return
        env_val = self.get_env_storage_key()
        if not env_val:
            return
        try:
            # Check if hex 32-byte key
            if len(env_val) == 64 and all(c in "0123456789abcdefABCDEF" for c in env_val):
                candidate_key = bytes.fromhex(env_val)
                verifier = self._compute_verifier(candidate_key)
                if verifier == self._state.get("verifier_hash", ""):
                    self._session_key = candidate_key
                    self._is_locked = False
                    print("[VAULT] Serenity Vault unlocked automatically via .env storage key.")
                    return
            # Check if password
            if self.verify_password(env_val):
                salt = bytes.fromhex(self._state["salt_hex"])
                self._session_key = self.derive_key(env_val, salt)
                self._is_locked = False
                print("[VAULT] Serenity Vault unlocked automatically via .env master password.")
        except Exception as e:
            print(f"[VAULT] Note: .env key auto-unlock did not match: {e}", file=sys.stderr)

    @classmethod
    def sync_env_storage_key(cls, key_or_password: str, env_path: Optional[str] = None) -> bool:
        """Safely updates or appends the SERENITY_ENCRYPTED_STORAGE_KEY in .env."""
        target = env_path or os.path.join(os.getcwd(), ".env")
        lines = []
        key_found = False
        key_name = "SERENITY_ENCRYPTED_STORAGE_KEY"
        if os.path.isfile(target):
            try:
                with open(target, "r", encoding="utf-8") as fp:
                    lines = fp.readlines()
            except Exception:
                lines = []

        new_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped.startswith("#") and "=" in stripped:
                k = stripped.split("=", 1)[0].strip()
                if k in (key_name, "SERENITY_STORAGE_KEY", "SERENITY_VAULT_KEY", "SERENITY_VAULT_PASSWORD"):
                    new_lines.append(f"{key_name}={key_or_password}\n")
                    key_found = True
                    continue
            new_lines.append(line)

        if not key_found:
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"
            new_lines.append(f"{key_name}={key_or_password}\n")

        try:
            with open(target, "w", encoding="utf-8") as fp:
                fp.writelines(new_lines)
            return True
        except Exception as e:
            print(f"[VAULT] Failed to sync .env storage key: {e}", file=sys.stderr)
            return False

    def _load_state(self) -> Dict[str, Any]:
        """Loads vault configuration state from disk."""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as fp:
                    return json.load(fp)
            except Exception as e:
                print(f"[VAULT] Warning: Failed to read vault state: {e}", file=sys.stderr)
        return {
            "lock_enabled": False,
            "salt_hex": "",
            "verifier_hash": "",
            "auto_lock_seconds": 0,
            "last_migration": None
        }

    def _save_state(self):
        """Persists vault configuration state to disk."""
        os.makedirs(self.state_dir, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as fp:
            json.dump(self._state, fp, indent=2)

    def is_crypto_available(self) -> bool:
        return CRYPTO_AVAILABLE and AESGCM is not None

    def is_lock_enabled(self) -> bool:
        return bool(self._state.get("lock_enabled", False))

    def is_locked(self) -> bool:
        return self.is_lock_enabled() and self._is_locked

    def get_auto_lock_seconds(self) -> int:
        return int(self._state.get("auto_lock_seconds", 0))

    def set_auto_lock_seconds(self, seconds: int):
        self._state["auto_lock_seconds"] = max(0, int(seconds))
        self._save_state()

    def derive_key(self, password: str, salt: bytes) -> bytes:
        """Derives a 256-bit AES key using PBKDF2-HMAC-SHA256 with 250,000 iterations."""
        return hashlib.pbkdf2_hmac(
            "sha256", 
            password.encode("utf-8"), 
            salt, 
            self.PBKDF2_ITERATIONS
        )

    def _compute_verifier(self, key: bytes) -> str:
        """Computes a secure one-way verifier hash to validate password without storing key."""
        return hashlib.sha256(key + b"SERENITY_VAULT_VERIFIER_TAG").hexdigest()

    def verify_password(self, password: str) -> bool:
        """Validates the password against the stored verifier hash."""
        if not self.is_lock_enabled():
            return True
            
        salt_hex = self._state.get("salt_hex", "")
        stored_verifier = self._state.get("verifier_hash", "")
        if not salt_hex or not stored_verifier:
            return False
            
        salt = bytes.fromhex(salt_hex)
        key = self.derive_key(password, salt)
        verifier = self._compute_verifier(key)
        
        if hmac_compare := getattr(hashlib, 'compare_digest', None):
            return hmac_compare(verifier, stored_verifier)
        return verifier == stored_verifier

    def unlock(self, password: str) -> bool:
        """Attempts to unlock the vault and caches the AES master key in session memory."""
        if not self.is_lock_enabled():
            self._is_locked = False
            return True

        if not self.is_crypto_available():
            raise RuntimeError("Cryptography library is required for vault decryption.")

        salt_hex = self._state.get("salt_hex", "")
        stored_verifier = self._state.get("verifier_hash", "")
        if not salt_hex or not stored_verifier:
            return False

        salt = bytes.fromhex(salt_hex)
        key = self.derive_key(password, salt)
        verifier = self._compute_verifier(key)

        if hmac_compare := getattr(hashlib, 'compare_digest', None):
            valid = hmac_compare(verifier, stored_verifier)
        else:
            valid = (verifier == stored_verifier)

        if valid:
            self._session_key = key
            self._is_locked = False
            print("[VAULT] Serenity Vault unlocked successfully.")
            return True
        return False

    def lock(self):
        """Immediately locks the vault and purges the session key from memory."""
        if self.is_lock_enabled():
            self._session_key = None
            self._is_locked = True
            print("[VAULT] Serenity Vault locked.")

    def set_password(self, new_password: str, current_password: Optional[str] = None) -> Tuple[bool, str]:
        """
        Enables lock or updates master password.
        Re-encrypts all existing histories with full backup & rollback safety.
        """
        if not self.is_crypto_available():
            return False, "Cryptography package (AESGCM) is not available."

        if not new_password or len(new_password) < 4:
            return False, "Master password must be at least 4 characters long."

        was_enabled = self.is_lock_enabled()
        old_key = self._session_key

        if was_enabled:
            if not current_password or not self.verify_password(current_password):
                return False, "Current master password verification failed."
            old_salt = bytes.fromhex(self._state["salt_hex"])
            old_key = self.derive_key(current_password, old_salt)

        # 1. Derive new key & verifier
        new_salt = secrets.token_bytes(self.SALT_SIZE)
        new_key = self.derive_key(new_password, new_salt)
        new_verifier = self._compute_verifier(new_key)

        # 2. Transactional migration with rollback
        success, msg = self._migrate_vault_files(
            source_key=old_key, 
            target_key=new_key, 
            to_encrypted=True
        )
        if not success:
            return False, f"Migration failed. Rollback executed: {msg}"

        # 3. Update state
        self._state["lock_enabled"] = True
        self._state["salt_hex"] = new_salt.hex()
        self._state["verifier_hash"] = new_verifier
        self._state["last_migration"] = datetime.datetime.now().isoformat()
        self._save_state()

        self._session_key = new_key
        self._is_locked = False
        return True, "Master password set and history archives encrypted successfully."

    def disable_lock(self, current_password: str) -> Tuple[bool, str]:
        """
        Disables vault lock and decrypts all `.encz` archives back to `.jsonz`.
        Full rollback safety included.
        """
        if not self.is_lock_enabled():
            return True, "Vault lock is already disabled."

        if not self.verify_password(current_password):
            return False, "Current master password verification failed."

        salt = bytes.fromhex(self._state["salt_hex"])
        current_key = self.derive_key(current_password, salt)

        # Decrypt all files back to .jsonz
        success, msg = self._migrate_vault_files(
            source_key=current_key, 
            target_key=None, 
            to_encrypted=False
        )
        if not success:
            return False, f"Decryption migration failed. Rollback executed: {msg}"

        self._state["lock_enabled"] = False
        self._state["salt_hex"] = ""
        self._state["verifier_hash"] = ""
        self._state["last_migration"] = datetime.datetime.now().isoformat()
        self._save_state()

        self._session_key = None
        self._is_locked = False
        return True, "Vault lock disabled and all archives decrypted to plaintext format."

    def encrypt_data(self, data_bytes: bytes, custom_key: Optional[bytes] = None) -> bytes:
        """
        Encrypts arbitrary bytes using AES-256-GCM.
        Returns: HEADER_MAGIC (16B) + NONCE (12B) + CIPHERTEXT (with appended 16B GCM tag).
        """
        key = custom_key or self._session_key
        if not key:
            raise PermissionError("Vault is locked or master key is missing.")
        if not self.is_crypto_available():
            raise RuntimeError("Cryptography library is required for AES-256-GCM.")

        aesgcm = AESGCM(key)
        nonce = secrets.token_bytes(self.NONCE_SIZE)
        ciphertext = aesgcm.encrypt(nonce, data_bytes, self.HEADER_MAGIC)
        return self.HEADER_MAGIC + nonce + ciphertext

    def decrypt_data(self, payload_bytes: bytes, custom_key: Optional[bytes] = None) -> bytes:
        """
        Decrypts an AES-256-GCM payload.
        Verifies magic header and GCM authentication tag.
        """
        key = custom_key or self._session_key
        if not key:
            raise PermissionError("Vault is locked or master key is missing.")
        if not self.is_crypto_available():
            raise RuntimeError("Cryptography library is required for AES-256-GCM.")

        if not payload_bytes.startswith(self.HEADER_MAGIC):
            raise ValueError("Payload does not contain valid Serenity Vault encryption header.")

        offset = len(self.HEADER_MAGIC)
        nonce = payload_bytes[offset : offset + self.NONCE_SIZE]
        ciphertext = payload_bytes[offset + self.NONCE_SIZE :]

        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, self.HEADER_MAGIC)

    def read_history_messages(self, path: str) -> List[Dict[str, Any]]:
        """
        Safely reads and decrypts history messages directly in memory.
        Supports both plaintext `.history.jsonz` and encrypted `.history.encz`.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"History file not found: {path}")

        with open(path, "rb") as fp:
            raw_data = fp.read()

        if path.endswith(".encz") or raw_data.startswith(self.HEADER_MAGIC):
            if self.is_locked() or not self._session_key:
                raise PermissionError("Vault is locked. Master password required to read encrypted archive.")
            decrypted_compressed = self.decrypt_data(raw_data)
            decompressed = zlib.decompress(decrypted_compressed).decode("utf-8")
            return json.loads(decompressed)
        else:
            decompressed = zlib.decompress(raw_data).decode("utf-8")
            return json.loads(decompressed)

    def write_history_messages(self, path: str, messages: List[Dict[str, Any]]):
        """
        Safely compresses and encrypts history messages to disk.
        If vault is active, writes to `.history.encz`.
        """
        json_bytes = json.dumps(messages).encode("utf-8")
        compressed = zlib.compress(json_bytes)

        if self.is_lock_enabled() and self._session_key:
            target_path = path if path.endswith(".encz") else path.replace(".jsonz", ".encz")
            encrypted_payload = self.encrypt_data(compressed)
            with open(target_path, "wb") as fp:
                fp.write(encrypted_payload)
            # Remove old unencrypted file if path changed
            if target_path != path and os.path.exists(path):
                try: os.remove(path)
                except: pass
        else:
            target_path = path if path.endswith(".jsonz") else path.replace(".encz", ".jsonz")
            with open(target_path, "wb") as fp:
                fp.write(compressed)

    def _migrate_vault_files(
        self, 
        source_key: Optional[bytes], 
        target_key: Optional[bytes], 
        to_encrypted: bool
    ) -> Tuple[bool, str]:
        """
        [TRANSACTION-SAFE BATCH MIGRATION WITH AUTOMATIC BACKUP & ROLLBACK]
        Guarantees zero data loss:
        1. Backs up all history files to a temporary directory.
        2. Encrypts or decrypts every file and verifies that the output decrypts/parses accurately.
        3. If any failure occurs, restores the entire backup and aborts cleanly.
        """
        if not os.path.exists(self.history_dir):
            return True, "No history directory found."

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = os.path.join(self.history_dir, "backups", f"migration_backup_{timestamp}")
        os.makedirs(backup_dir, exist_ok=True)

        try:
            # 1. Discover all history files (root + user subdirectories)
            all_files_rel = []
            for root, dirs, files in os.walk(self.history_dir):
                if "backups" in root:
                    continue
                for f in files:
                    if f.endswith(".history.jsonz") or f.endswith(".history.encz"):
                        rel_path = os.path.relpath(os.path.join(root, f), self.history_dir)
                        all_files_rel.append(rel_path)

            if not all_files_rel:
                return True, "No history files to migrate."

            print(f"[VAULT] Backing up {len(all_files_rel)} history archives to {backup_dir}...")
            for rel_f in all_files_rel:
                src = os.path.join(self.history_dir, rel_f)
                dst = os.path.join(backup_dir, rel_f)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

            # 2. Execute migration step-by-step with verification
            staged_deletions = []
            staged_creations = []

            for rel_f in all_files_rel:
                src_path = os.path.join(self.history_dir, rel_f)
                with open(src_path, "rb") as fp:
                    raw_data = fp.read()

                # A. Obtain original decompressed JSON content
                if rel_f.endswith(".encz") or raw_data.startswith(self.HEADER_MAGIC):
                    if not source_key:
                        raise ValueError(f"Cannot decrypt {f} without source master key.")
                    decrypted_raw = self.decrypt_data(raw_data, custom_key=source_key)
                    decompressed_str = zlib.decompress(decrypted_raw).decode("utf-8")
                else:
                    decompressed_str = zlib.decompress(raw_data).decode("utf-8")

                # Verify JSON structure
                parsed_json = json.loads(decompressed_str)

                # B. Prepare transformed payload
                compressed_bytes = zlib.compress(json.dumps(parsed_json).encode("utf-8"))

                if to_encrypted:
                    target_filename = rel_f.replace(".history.jsonz", ".history.encz")
                    target_path = os.path.join(self.history_dir, target_filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    transformed_data = self.encrypt_data(compressed_bytes, custom_key=target_key)
                    
                    # Verify immediately that transformed data can be decrypted and verified
                    verify_decomp = zlib.decompress(self.decrypt_data(transformed_data, custom_key=target_key)).decode("utf-8")
                    if verify_decomp != json.dumps(parsed_json):
                        raise ValueError(f"Integrity verification failed for {target_filename}")
                else:
                    target_filename = rel_f.replace(".history.encz", ".history.jsonz")
                    target_path = os.path.join(self.history_dir, target_filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    transformed_data = compressed_bytes
                    
                    verify_decomp = zlib.decompress(transformed_data).decode("utf-8")
                    if verify_decomp != json.dumps(parsed_json):
                        raise ValueError(f"Integrity verification failed for {target_filename}")

                # Write target file
                with open(target_path, "wb") as fp:
                    fp.write(transformed_data)
                staged_creations.append(target_path)

                if target_path != src_path:
                    staged_deletions.append(src_path)

            # 3. If all files verified successfully, remove old obsolete files
            for old_path in staged_deletions:
                if os.path.exists(old_path):
                    os.remove(old_path)

            print(f"[VAULT] Successfully migrated {len(all_files_rel)} archives (Backup retained at {backup_dir}).")
            return True, "Migration completed successfully."

        except Exception as e:
            print(f"[VAULT] FATAL MIGRATION ERROR: {e}. INITIATING AUTOMATIC ROLLBACK...", file=sys.stderr)
            # Automatic Rollback
            try:
                for created in staged_creations:
                    if os.path.exists(created):
                        try: os.remove(created)
                        except: pass
                for root, dirs, files in os.walk(backup_dir):
                    for f in files:
                        b_file = os.path.join(root, f)
                        rel_b = os.path.relpath(b_file, backup_dir)
                        restore_path = os.path.join(self.history_dir, rel_b)
                        os.makedirs(os.path.dirname(restore_path), exist_ok=True)
                        shutil.copy2(b_file, restore_path)
                print("[VAULT] Rollback complete. Original archives restored intact.", file=sys.stderr)
            except Exception as rollback_err:
                print(f"[VAULT] CRITICAL: Rollback encountered error: {rollback_err}", file=sys.stderr)

            return False, str(e)
