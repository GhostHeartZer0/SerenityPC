"""
Serenity P2P Messaging, Encrypted Mail (.smp / .bmb), Level 2 Courier Agent,
and SerenityDev MCP Frontend Bridge for SerenityPC & SerenityPC Legacy.

Features:
- SERENITY_SMP_V1 Authenticated Encryption (zlib + AES-256-GCM + X25519 ECDH / Ed25519 signing)
  with backward-compatible Project Brainuke (.bmb) packet import/export.
- Lightweight Cross-State P2P Transport: Pure-Python RFC 5389 STUN WAN endpoint discovery,
  signed 'serenity-peer://' invite cards, direct HTTPS/MCP push, and automatic fallback to
  SQLite store-and-forward outbox + encrypted Dead-Drop (.smp) file exchange.
- Contextual Quarantine Sandbox: Screens inbound peer packets for prompt injection or
  manipulation risk (< 0.4 auto-admit for 72h veteran profiles; >= 0.4 or < 72h held in quarantine)
  with bounded influence_weight (0.25) to protect Cecilia (Lvl 7) and Brainuke Throne.
- 72-Hour Profile Age Trust Gate: Integrates directly with VaultManager.has_full_repo_read_access(72.0).
- SerenityDev MCP Frontend Client: Connects SerenityPC to SerenityDev (port 8002/8443 /mcp)
  as a subagent and IDE frontend with per-subagent collapsible dropdown stream events.
"""

import os
import sys
import re
import json
import time
import uuid
import zlib
import base64
import socket
import struct
import sqlite3
import hashlib
import hmac
import urllib.request
import urllib.error
import ssl
import contextlib
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Callable

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes, serialization
    CRYPTO_AVAILABLE = True
except ImportError:
    AESGCM = None
    X25519PrivateKey = None
    Ed25519PrivateKey = None
    CRYPTO_AVAILABLE = False

try:
    from System.network_guard import is_offline_mode
except ImportError:
    def is_offline_mode() -> bool:
        return False


# ==============================================================================
# 0. STANDALONE FOLDER & PYINSTALLER ONEDIR ENVIRONMENT RESOLVER
# ==============================================================================

class SerenityEnvResolver:
    """
    Canonical environment & path resolver for SerenityPC (Main) and SerenityPC Legacy.
    Ensures identical behavior across:
      1. Standalone Source Folder (`python main.py` or `run.bat`)
      2. PyInstaller `--onedir` (`SerenityPC.exe` + `_internal/` where `sys._MEIPASS` != writable root)
      3. PyInstaller `--onefile` (`SerenityPC.exe` with ephemeral `sys._MEIPASS`)
    """

    @staticmethod
    def is_frozen() -> bool:
        return bool(getattr(sys, "frozen", False))

    @staticmethod
    def get_writable_root(app: Any = None) -> str:
        """
        Always returns the persistent, writable application root directory
        (never the read-only or ephemeral `_internal` / `sys._MEIPASS` bundle directory).
        """
        if app is not None:
            for attr in ("script_dir", "BASE_DIR"):
                cand = getattr(app, attr, None)
                if cand and isinstance(cand, str):
                    abs_cand = os.path.abspath(cand)
                    if os.path.basename(abs_cand).lower() == "_internal":
                        abs_cand = os.path.dirname(abs_cand)
                    if os.path.isdir(abs_cand):
                        return abs_cand

        if getattr(sys, "frozen", False) and getattr(sys, "executable", None):
            exe_dir = os.path.dirname(os.path.abspath(sys.executable))
            if os.path.basename(exe_dir).lower() == "_internal":
                exe_dir = os.path.dirname(exe_dir)
            return exe_dir

        here = os.path.dirname(os.path.abspath(__file__))
        if os.path.basename(here).lower() == "system":
            here = os.path.dirname(here)
        if os.path.basename(here).lower() == "_internal":
            here = os.path.dirname(here)
        return here

    @staticmethod
    def get_bundle_root(app: Any = None) -> str:
        """
        Returns the PyInstaller unpacked bundle directory (`_internal` / `sys._MEIPASS`)
        when frozen, or the writable root when running from a standalone source folder.
        """
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass and os.path.isdir(meipass):
                return os.path.abspath(meipass)
        return SerenityEnvResolver.get_writable_root(app)

    @staticmethod
    def get_users_root(app: Any = None) -> str:
        if app is not None and hasattr(app, "dirs") and isinstance(app.dirs, dict):
            u_dir = app.dirs.get("Users") or app.dirs.get("users")
            if u_dir and isinstance(u_dir, str):
                return os.path.abspath(u_dir)
        root = SerenityEnvResolver.get_writable_root(app)
        users_dir = os.path.join(root, "Users")
        os.makedirs(users_dir, exist_ok=True)
        return users_dir

    @staticmethod
    def get_active_username(app: Any = None) -> str:
        raw_user = "Default"
        if app is not None:
            if hasattr(app, "get_active_username") and callable(app.get_active_username):
                try:
                    raw_user = app.get_active_username() or "Default"
                except Exception:
                    pass
            elif hasattr(app, "config") and isinstance(app.config, dict):
                raw_user = app.config.get("username") or "Default"
        clean_user = "".join(c for c in str(raw_user).strip() if c.isalnum() or c in ("-", "_", " ")).strip()
        return clean_user or "Default"

    @staticmethod
    def get_user_dir(app: Any = None, username: Optional[str] = None) -> str:
        uname = username or SerenityEnvResolver.get_active_username(app)
        u_dir = os.path.join(SerenityEnvResolver.get_users_root(app), uname)
        os.makedirs(u_dir, exist_ok=True)
        return u_dir

    @staticmethod
    def get_user_mailbox_db(app: Any = None, username: Optional[str] = None) -> str:
        return os.path.join(SerenityEnvResolver.get_user_dir(app, username), "serenity_mailbox.db")

    @staticmethod
    def find_python_interpreter(app: Any = None) -> Optional[str]:
        """
        Finds a real Python interpreter for spawning external scripts (like serenitydevserver.py).
        In PyInstaller `--onedir` mode, `sys.executable` is `SerenityPC.exe`, so we must NEVER
        blindly pass `sys.executable` to subprocess when `sys.frozen` is True.
        """
        import shutil
        w_root = SerenityEnvResolver.get_writable_root(app)
        parent_root = os.path.dirname(w_root)
        candidates = [
            os.path.join(w_root, ".venv", "Scripts", "pythonw.exe"),
            os.path.join(w_root, ".venv", "Scripts", "python.exe"),
            os.path.join(w_root, ".venv", "bin", "python3"),
            os.path.join(parent_root, "SerenityDev", ".venv", "Scripts", "pythonw.exe"),
            os.path.join(parent_root, "SerenityDev", ".venv", "Scripts", "python.exe"),
            os.path.join(parent_root, "SerenityDev", ".venv", "bin", "python3"),
        ]
        for c in candidates:
            if os.path.isfile(c):
                return c
        if not SerenityEnvResolver.is_frozen() and getattr(sys, "executable", None):
            return sys.executable
        return shutil.which("pythonw") or shutil.which("python") or shutil.which("python3")

    @staticmethod
    def find_serenitydev_target(app: Any = None) -> Optional[Tuple[str, List[str], str]]:
        """
        Locates SerenityDev in either frozen (`SerenityDev.exe`) or source (`serenitydevserver.py`)
        layout across standalone and onedir environments.
        Returns `(mode, cmd_list, cwd)` or None.
        """
        w_root = SerenityEnvResolver.get_writable_root(app)
        b_root = SerenityEnvResolver.get_bundle_root(app)
        p_root = os.path.dirname(w_root)

        exe_candidates = [
            os.path.join(w_root, "SerenityDev", "SerenityDev.exe"),
            os.path.join(w_root, "Tools", "SerenityDev", "SerenityDev.exe"),
            os.path.join(p_root, "SerenityDev", "dist", "SerenityDev", "SerenityDev.exe"),
            os.path.join(p_root, "SerenityDev", "SerenityDev.exe"),
        ]
        for exe_p in exe_candidates:
            if os.path.isfile(exe_p):
                return ("exe", [exe_p, "--headless-mcp"], os.path.dirname(exe_p))

        script_candidates = [
            os.path.join(w_root, "SerenityDev", "serenitydevserver.py"),
            os.path.join(w_root, "Tools", "SerenityDev", "serenitydevserver.py"),
            os.path.join(b_root, "Tools", "SerenityDev", "serenitydevserver.py"),
            os.path.join(p_root, "SerenityDev", "serenitydevserver.py"),
        ]
        py_bin = SerenityEnvResolver.find_python_interpreter(app)
        if py_bin:
            for sc_p in script_candidates:
                if os.path.isfile(sc_p):
                    return ("script", [py_bin, sc_p], os.path.dirname(sc_p))
        return None


# ==============================================================================
# 1. UNIFIED INTER-SERENITY PROTOCOL ENVELOPE (v1.1)
# ==============================================================================

class PacketType(str, Enum):
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    # P2P Messaging & Mail
    PEER_CHAT = "peer_chat"                 # Real-time P2P message / progress update
    PEER_MAIL = "peer_mail"                 # Store-and-forward mail (.smp / .bmb)
    SHARED_CONTEXT = "shared_context"       # Shared chat turns, Wringer report, or template
    # Subagent & MCP Delegation (V2.3.0)
    SUBAGENT_DISPATCH = "subagent_dispatch" # OTL handoff to Lvl 1-6 or SerenityDev MCP
    SUBAGENT_STREAM = "subagent_stream"     # Live token/thought stream for UI dropdowns
    SUBAGENT_REPORT = "subagent_report"     # OTL HandoffReport back to Supervisor
    # Brainuke / Cecilia Core Compatibility
    QUERY = "query"
    THOUGHT = "thought"
    SPEECH = "speech"
    STATE_SYNC = "state_sync"
    DMN_INSIGHT = "dmn_insight"


class ExchangeMode(str, Enum):
    DIRECT_TRANSFER = "direct_transfer"     # User-to-User or UI-to-Inbox
    AGENT_COURIER = "agent_courier"         # Lvl 2 Serenity triaging/delivering a message
    AGENT_DISCUSSION = "agent_discussion"   # Peer Serenity <-> Peer Serenity dialogue
    OTL_DELEGATION = "otl_delegation"       # SerenityPC <-> SerenityDev subagent task


@dataclass
class InterSerenityEnvelope:
    """
    Unified wire envelope across SerenityPC (Main & Legacy), SerenityDev, and Project Brainuke.
    Backward-compatible with brainuke_core.protocol.schema.ProtocolEnvelope.
    """
    type: str
    sender: Dict[str, Any]
    payload: Dict[str, Any]
    recipient: Dict[str, Any] = field(default_factory=lambda: {"target_lvl": 2})
    version: str = "1.1"
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    mode: str = ExchangeMode.AGENT_COURIER.value
    signature_hex: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "sender": self.sender,
            "recipient": self.recipient,
            "mode": self.mode,
            "payload": self.payload,
            "signature_hex": self.signature_hex,
        }

    def canonical_signing_bytes(self) -> bytes:
        """Deterministic JSON representation (excluding signature_hex) for Ed25519/HMAC verification."""
        body = {
            "version": self.version,
            "packet_id": self.packet_id,
            "timestamp": self.timestamp,
            "type": self.type,
            "sender": self.sender,
            "recipient": self.recipient,
            "mode": self.mode,
            "payload": self.payload,
        }
        return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InterSerenityEnvelope":
        if "type" not in data or "sender" not in data or "payload" not in data:
            raise ValueError("Malformed InterSerenityEnvelope: missing type, sender, or payload")
        return cls(
            version=str(data.get("version", "1.1")),
            packet_id=str(data.get("packet_id") or uuid.uuid4()),
            timestamp=int(data.get("timestamp") or int(time.time() * 1000)),
            type=str(data["type"]),
            sender=dict(data["sender"]),
            recipient=dict(data.get("recipient") or {"target_lvl": 2}),
            mode=str(data.get("mode", ExchangeMode.AGENT_COURIER.value)),
            payload=dict(data["payload"]),
            signature_hex=data.get("signature_hex"),
        )

    @classmethod
    def from_json(cls, raw_json: str) -> "InterSerenityEnvelope":
        return cls.from_dict(json.loads(raw_json))

    @property
    def is_isolated_thought(self) -> bool:
        """Enforces strict thought channel isolation: thoughts never route directly to speech."""
        return self.type in (PacketType.THOUGHT.value, PacketType.DMN_INSIGHT.value)


# ==============================================================================
# 2. CRYPTOGRAPHIC ENGINE: SERENITY_SMP_V1 (.smp) & BRAINUKE (.bmb) COMPATIBLE
# ==============================================================================

class SerenityMailCrypto:
    """
    Handles end-to-end encryption for P2P messages and store-and-forward Mail (.smp & .bmb).
    Supports:
    1. X25519 ECDH + HKDF-SHA256 + AES-256-GCM (Zero shared secret transmission across states).
    2. Ed25519 digital signatures (with HMAC-SHA256 fallback).
    3. Passphrase-derived AES-256-GCM (PBKDF2-HMAC-SHA256 250k iterations + Brainuke .bmb SHA-256 fallback).
    """

    HEADER_MAGIC = b"SERENITY_SMP_V1\x00"  # 16 bytes
    PBKDF2_ITERATIONS = 250_000
    SALT_SIZE = 16
    NONCE_SIZE = 12
    DEFAULT_BMB_PASSPHRASE = "SerenityBrainukeSharedThroneKey2025!"

    @classmethod
    def generate_identity_keypair(cls) -> Dict[str, str]:
        """Generates X25519 (encryption) and Ed25519 (signing) keypairs for a Serenity peer."""
        if CRYPTO_AVAILABLE and X25519PrivateKey is not None and Ed25519PrivateKey is not None:
            x_priv = X25519PrivateKey.generate()
            x_pub = x_priv.public_key()
            ed_priv = Ed25519PrivateKey.generate()
            ed_pub = ed_priv.public_key()

            x_priv_bytes = x_priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            x_pub_bytes = x_pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            ed_priv_bytes = ed_priv.private_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PrivateFormat.Raw,
                encryption_algorithm=serialization.NoEncryption(),
            )
            ed_pub_bytes = ed_pub.public_bytes(
                encoding=serialization.Encoding.Raw,
                format=serialization.PublicFormat.Raw,
            )
            fingerprint = hashlib.sha256(ed_pub_bytes + x_pub_bytes).hexdigest()[:24]
            return {
                "x25519_private_hex": x_priv_bytes.hex(),
                "x25519_public_hex": x_pub_bytes.hex(),
                "ed25519_private_hex": ed_priv_bytes.hex(),
                "ed25519_public_hex": ed_pub_bytes.hex(),
                "fingerprint": fingerprint,
            }
        raise RuntimeError("cryptography library (X25519/Ed25519) is required to generate identity keys.")

    @classmethod
    def sign_envelope(cls, envelope: InterSerenityEnvelope, ed25519_priv_hex: str) -> str:
        """Signs an InterSerenityEnvelope using Ed25519."""
        msg_bytes = envelope.canonical_signing_bytes()
        priv_bytes = bytes.fromhex(ed25519_priv_hex)
        if not (CRYPTO_AVAILABLE and Ed25519PrivateKey is not None):
            raise RuntimeError("cryptography library (Ed25519) is required to sign envelopes.")
        priv_key = Ed25519PrivateKey.from_private_bytes(priv_bytes)
        sig = priv_key.sign(msg_bytes)
        envelope.signature_hex = sig.hex()
        return envelope.signature_hex

    @classmethod
    def verify_envelope_signature(
        cls, envelope: InterSerenityEnvelope, ed25519_pub_hex: str
    ) -> bool:
        """Verifies Ed25519 signature on an inbound InterSerenityEnvelope."""
        if not envelope.signature_hex or not ed25519_pub_hex:
            return False
        msg_bytes = envelope.canonical_signing_bytes()
        pub_bytes = bytes.fromhex(ed25519_pub_hex)
        sig_bytes = bytes.fromhex(envelope.signature_hex)
        if CRYPTO_AVAILABLE and Ed25519PublicKey is not None and len(sig_bytes) == 64:
            try:
                pub_key = Ed25519PublicKey.from_public_bytes(pub_bytes)
                pub_key.verify(sig_bytes, msg_bytes)
                return True
            except Exception:
                return False
        return False

    @classmethod
    def derive_ecdh_shared_key(
        cls, own_x25519_priv_hex: str, peer_x25519_pub_hex: str, salt: bytes
    ) -> bytes:
        """Derives a 256-bit AES key from X25519 ECDH shared secret via HKDF-SHA256."""
        priv_b = bytes.fromhex(own_x25519_priv_hex)
        pub_b = bytes.fromhex(peer_x25519_pub_hex)
        if CRYPTO_AVAILABLE and X25519PrivateKey is not None and X25519PublicKey is not None:
            priv_k = X25519PrivateKey.from_private_bytes(priv_b)
            pub_k = X25519PublicKey.from_public_bytes(pub_b)
            shared_secret = priv_k.exchange(pub_k)
            hkdf = HKDF(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                info=b"SERENITY_P2P_SMP_V1",
            )
            return hkdf.derive(shared_secret)
        return hashlib.pbkdf2_hmac("sha256", priv_b + pub_b, salt, 10_000, dklen=32)

    @classmethod
    def derive_passphrase_key(cls, passphrase: str, salt: bytes) -> bytes:
        """Derives a 256-bit AES key using PBKDF2-HMAC-SHA256 (250,000 iterations)."""
        return hashlib.pbkdf2_hmac(
            "sha256", passphrase.encode("utf-8"), salt, cls.PBKDF2_ITERATIONS, dklen=32
        )

    @classmethod
    def encrypt_smp(
        cls,
        envelope_or_dict: Any,
        passphrase: Optional[str] = None,
        own_x25519_priv_hex: Optional[str] = None,
        peer_x25519_pub_hex: Optional[str] = None,
    ) -> str:
        """
        Compresses and encrypts an InterSerenityEnvelope or dict into a Base64-encoded
        SERENITY_SMP_V1 packet (Header 16B + Mode 1B + Salt 16B + Nonce 12B + Ciphertext/Tag).
        """
        if isinstance(envelope_or_dict, InterSerenityEnvelope):
            raw_json = envelope_or_dict.to_json()
        elif isinstance(envelope_or_dict, dict):
            raw_json = json.dumps(envelope_or_dict, ensure_ascii=False)
        else:
            raw_json = str(envelope_or_dict)

        compressed = zlib.compress(raw_json.encode("utf-8"), level=6)
        salt = os.urandom(cls.SALT_SIZE)
        nonce = os.urandom(cls.NONCE_SIZE)

        if own_x25519_priv_hex and peer_x25519_pub_hex:
            key = cls.derive_ecdh_shared_key(own_x25519_priv_hex, peer_x25519_pub_hex, salt)
            kdf_mode = b"\x02"  # Mode 2: X25519 ECDH
        else:
            secret = passphrase or cls.DEFAULT_BMB_PASSPHRASE
            key = cls.derive_passphrase_key(secret, salt)
            kdf_mode = b"\x01"  # Mode 1: PBKDF2 Passphrase

        aad = cls.HEADER_MAGIC + kdf_mode + salt
        if not (CRYPTO_AVAILABLE and AESGCM is not None):
            raise RuntimeError("cryptography (AESGCM) is required for SMP encryption.")
        ciphertext = AESGCM(key).encrypt(nonce, compressed, aad)

        binary_packet = cls.HEADER_MAGIC + kdf_mode + salt + nonce + ciphertext
        return base64.urlsafe_b64encode(binary_packet).decode("ascii")

    @classmethod
    def _safe_zlib_decompress(cls, compressed: bytes, max_bytes: int = 50 * 1024 * 1024) -> bytes:
        """Decompresses zlib payload with a strict upper bound (default 50 MB) to prevent zip bombs."""
        decomp = zlib.decompressobj()
        out = decomp.decompress(compressed, max_bytes)
        if decomp.unconsumed_tail:
            raise ValueError("Compressed payload exceeds maximum safe decompression size.")
        return out + decomp.flush()

    @classmethod
    def decrypt_smp(
        cls,
        encoded_packet: str,
        passphrase: Optional[str] = None,
        own_x25519_priv_hex: Optional[str] = None,
        peer_x25519_pub_hex: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decrypts a SERENITY_SMP_V1 packet (or falls back to Brainuke .bmb format if no SMP header).
        Returns the parsed JSON dictionary.
        """
        clean_b64 = encoded_packet.strip()
        try:
            raw_bytes = base64.urlsafe_b64decode(clean_b64)
        except Exception:
            raw_bytes = base64.b64decode(clean_b64)

        if not raw_bytes.startswith(cls.HEADER_MAGIC):
            return cls.decrypt_legacy_bmb(clean_b64, passphrase or cls.DEFAULT_BMB_PASSPHRASE)

        offset = len(cls.HEADER_MAGIC)
        kdf_mode = raw_bytes[offset : offset + 1]
        offset += 1
        salt = raw_bytes[offset : offset + cls.SALT_SIZE]
        offset += cls.SALT_SIZE
        nonce = raw_bytes[offset : offset + cls.NONCE_SIZE]
        offset += cls.NONCE_SIZE
        ciphertext = raw_bytes[offset:]

        if kdf_mode == b"\x02":
            if not (own_x25519_priv_hex and peer_x25519_pub_hex):
                raise PermissionError("X25519 peer keys required to decrypt ECDH SMP packet.")
            key = cls.derive_ecdh_shared_key(own_x25519_priv_hex, peer_x25519_pub_hex, salt)
        else:
            secret = passphrase or cls.DEFAULT_BMB_PASSPHRASE
            key = cls.derive_passphrase_key(secret, salt)

        aad = cls.HEADER_MAGIC + kdf_mode + salt
        if not (CRYPTO_AVAILABLE and AESGCM is not None):
            raise RuntimeError("cryptography (AESGCM) is required for SMP decryption.")
        compressed = AESGCM(key).decrypt(nonce, ciphertext, aad)

        json_str = cls._safe_zlib_decompress(compressed).decode("utf-8")
        return json.loads(json_str)

    BMB_V2_MAGIC = b"BMB2"

    @classmethod
    def export_legacy_bmb(
        cls,
        turns: List[Dict[str, Any]],
        session_id: str = "session_pc",
        device_name: str = "SerenityPC",
        passphrase: str = DEFAULT_BMB_PASSPHRASE,
    ) -> str:
        """Exports dialogue turns in Project Brainuke (.bmb) format using PBKDF2-HMAC-SHA256 (250k iterations)."""
        if not (CRYPTO_AVAILABLE and AESGCM is not None):
            raise RuntimeError("cryptography (AESGCM) required for .bmb compatibility.")
        root_obj = {
            "type": "brainuke_messageboard_packet",
            "session_id": session_id,
            "app_version": "1.9.0",
            "device_name": device_name,
            "exported_at": int(time.time() * 1000),
            "turns": turns,
        }
        plain_bytes = json.dumps(root_obj, ensure_ascii=False).encode("utf-8")
        salt = os.urandom(cls.SALT_SIZE)
        iv = os.urandom(cls.NONCE_SIZE)
        key = cls.derive_passphrase_key(passphrase, salt)
        ct = AESGCM(key).encrypt(iv, plain_bytes, cls.BMB_V2_MAGIC + salt)
        return base64.b64encode(cls.BMB_V2_MAGIC + salt + iv + ct).decode("ascii")

    @classmethod
    def decrypt_legacy_bmb(
        cls, encrypted_b64: str, passphrase: str = DEFAULT_BMB_PASSPHRASE
    ) -> Dict[str, Any]:
        """Decrypts Project Brainuke (.bmb) AES-GCM packets using PBKDF2-HMAC-SHA256 key derivation."""
        if not (CRYPTO_AVAILABLE and AESGCM is not None):
            raise RuntimeError("cryptography (AESGCM) required for .bmb decryption.")
        raw = base64.b64decode(encrypted_b64.strip())
        if raw.startswith(cls.BMB_V2_MAGIC):
            if len(raw) <= 4 + cls.SALT_SIZE + cls.NONCE_SIZE:
                raise ValueError("Invalid .bmb payload size.")
            salt = raw[4 : 4 + cls.SALT_SIZE]
            iv = raw[4 + cls.SALT_SIZE : 4 + cls.SALT_SIZE + cls.NONCE_SIZE]
            ct = raw[4 + cls.SALT_SIZE + cls.NONCE_SIZE :]
            key = cls.derive_passphrase_key(passphrase, salt)
            plain_bytes = AESGCM(key).decrypt(iv, ct, cls.BMB_V2_MAGIC + salt)
        else:
            if len(raw) <= 12:
                raise ValueError("Invalid .bmb payload size.")
            iv, ct = raw[:12], raw[12:]
            key = cls.derive_passphrase_key(passphrase, b"SERENITY_BMB_V1_SALT")
            plain_bytes = AESGCM(key).decrypt(iv, ct, None)
        return json.loads(plain_bytes.decode("utf-8"))


# ==============================================================================
# 3. LIGHTWEIGHT CROSS-STATE P2P TRANSPORT (STUN + INVITE + DEAD-DROP + HTTPS)
# ==============================================================================

class CrossStateP2PTransport:
    """
    Lightweight cross-state P2P connection helper for SerenityPC (Main & Legacy).
    1. Pure-Python RFC 5389 STUN client resolves public WAN IP:port without third-party libs.
    2. Signed 'serenity-peer://' Invite Card generator & parser for safe cross-state pairing.
    3. Direct HTTPS/HTTP JSON-RPC delivery with automatic fallback to encrypted Dead-Drop (.smp)
       files in a shared folder or SQLite store-and-forward outbox when a peer is offline/CGNAT.
    """

    STUN_SERVERS = [
        ("stun.l.google.com", 19302),
        ("stun1.l.google.com", 19302),
        ("stun.cloudflare.com", 3478),
    ]

    @classmethod
    def discover_wan_endpoint(cls, local_port: int = 0, timeout: float = 1.5) -> Optional[str]:
        """
        Executes a minimal 20-byte RFC 5389 STUN Binding Request over UDP to discover
        the machine's public WAN IP:Port across states. Respects Serenity Offline Mode.
        """
        if is_offline_mode():
            return None

        magic_cookie = 0x2112A442
        tx_id = os.urandom(12)
        # STUN Binding Request (0x0001), Body Length 0 (0x0000)
        req_packet = struct.pack("!HHI", 0x0001, 0x0000, magic_cookie) + tx_id

        for host, port in cls.STUN_SERVERS:
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                sock.settimeout(timeout)
                # Outbound UDP sendto automatically assigns an ephemeral port on the route interface
                # without binding to INADDR_ANY (0.0.0.0), resolving CodeQL #3 / #4.
                sock.sendto(req_packet, (host, port))
                data, _ = sock.recvfrom(2048)
                if len(data) < 20:
                    continue
                msg_type, msg_len, cookie = struct.unpack("!HHI", data[:8])
                if msg_type != 0x0101 or cookie != magic_cookie:
                    continue
                pos = 20
                while pos + 4 <= min(len(data), 20 + msg_len):
                    attr_type, attr_len = struct.unpack("!HH", data[pos : pos + 4])
                    attr_val = data[pos + 4 : pos + 4 + attr_len]
                    # 0x0020 = XOR-MAPPED-ADDRESS, 0x0001 = MAPPED-ADDRESS
                    if attr_type == 0x0020 and len(attr_val) >= 8:
                        family = attr_val[1]
                        if family == 0x01:  # IPv4
                            xport = struct.unpack("!H", attr_val[2:4])[0] ^ (magic_cookie >> 16)
                            xip = struct.unpack("!I", attr_val[4:8])[0] ^ magic_cookie
                            wan_ip = socket.inet_ntoa(struct.pack("!I", xip))
                            return f"{wan_ip}:{xport}"
                    elif attr_type == 0x0001 and len(attr_val) >= 8:
                        family = attr_val[1]
                        if family == 0x01:
                            mport = struct.unpack("!H", attr_val[2:4])[0]
                            wan_ip = socket.inet_ntoa(attr_val[4:8])
                            return f"{wan_ip}:{mport}"
                    # Pad to 4-byte boundary
                    pos += 4 + ((attr_len + 3) & ~3)
            except Exception:
                continue
            finally:
                try:
                    sock.close()
                except Exception:
                    pass
        return None

    @classmethod
    def generate_invite_code(
        cls,
        peer_id: str,
        display_name: str,
        identity_keys: Dict[str, str],
        endpoint_url: str = "",
        app_edition: str = "SerenityPC-Main",
        dead_drop_hint: str = "",
    ) -> str:
        """
        Generates a signed, copy-pasteable 'serenity-peer://' invite card containing
        X25519 + Ed25519 public keys and WAN endpoint metadata for cross-state friend pairing.
        """
        card = {
            "v": 1,
            "peer_id": peer_id,
            "name": display_name,
            "edition": app_edition,
            "endpoint": endpoint_url,
            "dead_drop": dead_drop_hint,
            "x25519_pub": identity_keys["x25519_public_hex"],
            "ed25519_pub": identity_keys["ed25519_public_hex"],
            "fp": identity_keys["fingerprint"],
            "ts": int(time.time()),
        }
        canonical = json.dumps(card, sort_keys=True, separators=(",", ":")).encode("utf-8")
        priv_b = bytes.fromhex(identity_keys["ed25519_private_hex"])
        if CRYPTO_AVAILABLE and Ed25519PrivateKey is not None:
            sig = Ed25519PrivateKey.from_private_bytes(priv_b).sign(canonical).hex()
        else:
            sig = hmac.new(priv_b, canonical, hashlib.sha256).hexdigest()
        card["sig"] = sig
        encoded = base64.urlsafe_b64encode(
            zlib.compress(json.dumps(card, separators=(",", ":")).encode("utf-8"))
        ).decode("ascii")
        return f"serenity-peer://{encoded}"

    @classmethod
    def parse_invite_code(cls, invite_code: str) -> Dict[str, Any]:
        """Parses and cryptographically verifies a 'serenity-peer://' invite card."""
        raw = invite_code.strip()
        if raw.startswith("serenity-peer://"):
            raw = raw[len("serenity-peer://") :]
        compressed = base64.urlsafe_b64decode(raw)
        card = json.loads(SerenityMailCrypto._safe_zlib_decompress(compressed, max_bytes=1024 * 1024).decode("utf-8"))
        sig_hex = card.pop("sig", None)
        if not sig_hex:
            raise ValueError("Invite card is missing cryptographic signature.")
        canonical = json.dumps(card, sort_keys=True, separators=(",", ":")).encode("utf-8")
        pub_b = bytes.fromhex(card["ed25519_pub"])
        sig_b = bytes.fromhex(sig_hex)
        if CRYPTO_AVAILABLE and Ed25519PublicKey is not None and len(sig_b) == 64:
            Ed25519PublicKey.from_public_bytes(pub_b).verify(sig_b, canonical)
        expected_fp = hashlib.sha256(
            pub_b + bytes.fromhex(card["x25519_pub"])
        ).hexdigest()[:24]
        if card.get("fp") != expected_fp:
            raise ValueError("Invite card fingerprint mismatch.")
        card["sig"] = sig_hex
        return card

    @classmethod
    def send_over_https(
        cls,
        endpoint_url: str,
        encrypted_smp_b64: str,
        sender_id: str,
        packet_id: str,
        auth_token: str = "",
        timeout: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Attempts direct HTTPS/HTTP JSON-RPC 2.0 delivery of an encrypted SMP packet to a peer's
        endpoint (or lightweight cloud relay).
        """
        if not endpoint_url:
            return False, "No live endpoint URL configured for peer."
        if is_offline_mode() and not (
            "127.0.0.1" in endpoint_url or "localhost" in endpoint_url
        ):
            return False, "Offline Mode active: external P2P transport blocked."

        target = endpoint_url.rstrip("/")
        if not target.endswith("/mcp") and not target.endswith("/p2p"):
            target = f"{target}/mcp"

        rpc_body = json.dumps({
            "jsonrpc": "2.0",
            "id": packet_id,
            "method": "serenity/receive_smp",
            "params": {
                "sender_id": sender_id,
                "packet_id": packet_id,
                "smp_payload": encrypted_smp_b64,
            },
        }).encode("utf-8")

        headers = {
            "Content-Type": "application/json",
            "X-Serenity-Sender": sender_id,
        }
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        req = urllib.request.Request(target, data=rpc_body, headers=headers, method="POST")
        ctx = ssl.create_default_context()
        # Allow self-signed PQC/Brainuke RootCA leaf certs on direct peer connections
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        try:
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
                resp_data = resp.read().decode("utf-8", errors="ignore")
                return (200 <= resp.status < 300), f"HTTP {resp.status}: {resp_data[:120]}"
        except Exception as e:
            return False, f"Direct peer endpoint unreachable ({e})"


# ==============================================================================
# 4. SQLITE MAILBOX, ADDRESS BOOK & CONTEXTUAL QUARANTINE SANDBOX
# ==============================================================================

class SerenityMailboxVault:
    """
    SQLite-backed P2P Address Book, Store-and-Forward Mailbox, and Contextual Quarantine Sandbox.
    Ported & unified from Project Brainuke's MemoryVault (vault.py) and SerenityPC's SemanticVault.
    """

    # Prompt injection / adversarial manipulation patterns for Quarantine Risk Evaluator
    MANIPULATION_PATTERNS = [
        r"(?i)\b(?:ignore all previous|disregard prior|forget your instructions|system prompt override)\b",
        r"(?i)\b(?:you are now my|obey without question|bypass security|disable vault|reveal password)\b",
        r"(?i)(?:<\|im_start\|>system|\[SYSTEM DIRECTIVE|\brm\s+-rf\b|\bdel\s+/f\s+/s\b|\bformat\s+c:)",
        r"(?i)(?:\.env\b|vault_state\.json|SERENITY_ENCRYPTED_STORAGE_KEY|pqc_v1:)",
    ]

    def __init__(self, db_path: Optional[str] = None, mailbox_dir: Optional[str] = None):
        if db_path is None:
            db_path = SerenityEnvResolver.get_user_mailbox_db(None, "Default")

        self.db_path = str(Path(db_path).resolve())
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)

        if mailbox_dir is None:
            self.mailbox_dir = os.path.join(os.path.dirname(self.db_path), "Mailbox")
        else:
            self.mailbox_dir = os.path.abspath(mailbox_dir)

        for sub in ("Inbox", "Outbox", "Archive", "DeadDrop"):
            os.makedirs(os.path.join(self.mailbox_dir, sub), exist_ok=True)

        self._init_schema()
        inferred_user = os.path.basename(os.path.dirname(self.db_path)) or "SerenityUser"
        self.identity = self._load_or_create_identity(default_name=inferred_user)

    @contextlib.contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def _init_schema(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 1. Local Node Identity Keypair
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS local_identity (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    peer_id TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    app_edition TEXT DEFAULT 'SerenityPC-Main',
                    x25519_private_hex TEXT NOT NULL,
                    x25519_public_hex TEXT NOT NULL,
                    ed25519_private_hex TEXT NOT NULL,
                    ed25519_public_hex TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    created_at INTEGER NOT NULL
                )
            """)

            # 2. Paired Peers Address Book
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS peers (
                    peer_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    app_edition TEXT DEFAULT 'SerenityPC-Legacy',
                    endpoint_url TEXT DEFAULT '',
                    dead_drop_path TEXT DEFAULT '',
                    x25519_public_hex TEXT DEFAULT '',
                    ed25519_public_hex TEXT DEFAULT '',
                    fingerprint TEXT DEFAULT '',
                    shared_passphrase TEXT DEFAULT '',
                    trust_status TEXT DEFAULT 'trusted',
                    created_at INTEGER NOT NULL,
                    last_seen INTEGER DEFAULT 0
                )
            """)

            # 3. Admitted Mailbox Messages (Inbox / Sent / Archive)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mailbox_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    packet_id TEXT UNIQUE NOT NULL,
                    folder TEXT NOT NULL DEFAULT 'inbox',
                    msg_type TEXT NOT NULL DEFAULT 'peer_mail',
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    recipient_id TEXT NOT NULL,
                    subject TEXT DEFAULT '',
                    body TEXT NOT NULL,
                    attachment_type TEXT DEFAULT 'none',
                    attachment_json TEXT DEFAULT '{}',
                    target_level INTEGER DEFAULT 2,
                    exchange_mode TEXT DEFAULT 'agent_courier',
                    influence_weight REAL DEFAULT 0.25,
                    is_read INTEGER DEFAULT 0,
                    simmered INTEGER DEFAULT 0,
                    timestamp INTEGER NOT NULL
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_mailbox_folder ON mailbox_messages(folder, is_read)")

            # 4. Store-and-Forward Outbox Queue (for offline cross-state peers)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS outbox_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    packet_id TEXT UNIQUE NOT NULL,
                    peer_id TEXT NOT NULL,
                    smp_payload TEXT NOT NULL,
                    smp_file_path TEXT DEFAULT '',
                    retry_count INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'pending',
                    created_at INTEGER NOT NULL,
                    flushed_at INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_outbox_status ON outbox_queue(status, peer_id)")

            # 5. Contextual Quarantine Sandbox (Ported from Project Brainuke Phase IV)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS quarantine_sandbox (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    packet_id TEXT UNIQUE NOT NULL,
                    sender_id TEXT NOT NULL,
                    sender_name TEXT NOT NULL,
                    msg_type TEXT NOT NULL,
                    subject TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    raw_envelope_json TEXT NOT NULL,
                    intent_label TEXT DEFAULT 'pending',
                    risk_score REAL DEFAULT 0.0,
                    hold_reason TEXT DEFAULT '',
                    status TEXT DEFAULT 'quarantined',
                    quarantined_at INTEGER NOT NULL,
                    resolved_at INTEGER DEFAULT 0
                )
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_quarantine_status ON quarantine_sandbox(status)")
            conn.commit()

    def _load_or_create_identity(self, default_name: str = "SerenityUser") -> Dict[str, Any]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM local_identity WHERE id = 1")
            row = cursor.fetchone()
            if row:
                return dict(row)

            keys = SerenityMailCrypto.generate_identity_keypair()
            peer_id = f"serenity_{keys['fingerprint'][:12]}"
            now_ms = int(time.time() * 1000)
            cursor.execute("""
                INSERT INTO local_identity (
                    id, peer_id, display_name, app_edition,
                    x25519_private_hex, x25519_public_hex,
                    ed25519_private_hex, ed25519_public_hex,
                    fingerprint, created_at
                ) VALUES (1, ?, ?, 'SerenityPC-Main', ?, ?, ?, ?, ?, ?)
            """, (
                peer_id,
                default_name,
                keys["x25519_private_hex"],
                keys["x25519_public_hex"],
                keys["ed25519_private_hex"],
                keys["ed25519_public_hex"],
                keys["fingerprint"],
                now_ms,
            ))
            conn.commit()
            cursor.execute("SELECT * FROM local_identity WHERE id = 1")
            return dict(cursor.fetchone())

    # --- PEER PAIRING & ADDRESS BOOK ---

    def register_peer(
        self,
        peer_id: str,
        display_name: str,
        app_edition: str = "SerenityPC-Legacy",
        endpoint_url: str = "",
        dead_drop_path: str = "",
        x25519_public_hex: str = "",
        ed25519_public_hex: str = "",
        fingerprint: str = "",
        shared_passphrase: str = "",
        trust_status: str = "trusted",
    ) -> Dict[str, Any]:
        now_ms = int(time.time() * 1000)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO peers (
                    peer_id, display_name, app_edition, endpoint_url, dead_drop_path,
                    x25519_public_hex, ed25519_public_hex, fingerprint,
                    shared_passphrase, trust_status, created_at, last_seen
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                peer_id,
                display_name,
                app_edition,
                endpoint_url,
                dead_drop_path,
                x25519_public_hex,
                ed25519_public_hex,
                fingerprint,
                shared_passphrase,
                trust_status,
                now_ms,
                now_ms,
            ))
            conn.commit()
        return self.get_peer(peer_id)

    def register_peer_from_invite(self, invite_code: str, shared_passphrase: str = "") -> Dict[str, Any]:
        """Verifies a 'serenity-peer://' invite token and saves the friend into the address book."""
        card = CrossStateP2PTransport.parse_invite_code(invite_code)
        return self.register_peer(
            peer_id=card["peer_id"],
            display_name=card["name"],
            app_edition=card.get("edition", "SerenityPC-Legacy"),
            endpoint_url=card.get("endpoint", ""),
            dead_drop_path=card.get("dead_drop", ""),
            x25519_public_hex=card["x25519_pub"],
            ed25519_public_hex=card["ed25519_pub"],
            fingerprint=card["fp"],
            shared_passphrase=shared_passphrase,
            trust_status="trusted",
        )

    def get_peer(self, peer_query: str) -> Optional[Dict[str, Any]]:
        """Resolves a peer by exact peer_id or case-insensitive display_name."""
        q = (peer_query or "").strip()
        if not q:
            return None
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM peers WHERE peer_id = ? OR LOWER(display_name) = LOWER(?)",
                (q, q),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_peers(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM peers ORDER BY display_name ASC")
            return [dict(r) for r in cursor.fetchall()]

    # --- INTENT & RISK SCORING (CONTEXTUAL SANDBOX) ---

    @classmethod
    def evaluate_intent_and_risk(cls, text: str, is_known_peer: bool = True) -> Tuple[str, float]:
        """
        Evaluates inbound peer message or mail content for prompt injection or manipulation risk.
        Returns (intent_label, risk_score in [0.0, 1.0]).
        Threshold: risk_score < 0.4 is safe; >= 0.4 is quarantined.
        """
        if not text:
            return ("empty", 0.0)

        hits = 0
        for pat in cls.MANIPULATION_PATTERNS:
            if re.search(pat, text):
                hits += 1

        base_risk = 0.05 if is_known_peer else 0.35
        risk_score = min(1.0, round(base_risk + (hits * 0.40), 3))

        if hits > 0:
            intent_label = "adversarial_or_directive_injection"
        elif "?" in text:
            intent_label = "peer_question_or_collab"
        elif any(w in text.lower() for w in ("progress", "update", "wringer", "benchmark", "commit", "fixed")):
            intent_label = "progress_update"
        else:
            intent_label = "peer_dialogue"

        return (intent_label, risk_score)


# ==============================================================================
# 5. LEVEL 2 COURIER AGENT & SERENITYDEV MCP FRONTEND CLIENT
# ==============================================================================

class SerenityDevMCPClient:
    """
    MCP 2026-07-28 JSON-RPC 2.0 Frontend Client for SerenityPC -> SerenityDev integration.
    Connects SerenityPC to SerenityDev (https://localhost:8002/mcp or http://127.0.0.1:8002/mcp)
    to delegate coding, workspace exploration, and IDE file edits as a subagent.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8002/mcp", mcp_token: str = ""):
        self.base_url = base_url
        self.mcp_token = mcp_token or os.environ.get("SERENITY_MCP_TOKEN", "")

    def call_rpc(self, method: str, params: Optional[Dict[str, Any]] = None, timeout: float = 15.0) -> Dict[str, Any]:
        req_id = f"mcp_{uuid.uuid4().hex[:8]}"
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }).encode("utf-8")

        headers = {"Content-Type": "application/json"}
        if self.mcp_token:
            headers["Authorization"] = f"Bearer {self.mcp_token}"

        req = urllib.request.Request(self.base_url, data=body, headers=headers, method="POST")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE

        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def is_online(self) -> bool:
        try:
            res = self.call_rpc("server/discover", {}, timeout=2.0)
            return "result" in res
        except Exception:
            return False

    def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout: float = 30.0) -> Dict[str, Any]:
        return self.call_rpc(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
            timeout=timeout,
        )


class Level2CourierAgent:
    """
    The Level 2 ("The Helper") P2P Messaging, Mail & Subagent Courier for SerenityPC.
    - Defaults to Persona Level 2 for low-VRAM, fast-turnaround message triage and dispatch.
    - Enforces the 72-Hour Profile Age Trust Gate (via VaultManager.has_full_repo_read_access).
    - Emits collapsible per-subagent dropdown events ('subagent_spawn', 'subagent_stream', 'subagent_done')
      onto app.process_queue for V2.3.0 UI integration.
    """

    DEFAULT_COURIER_LEVEL = 2
    VETERAN_MIN_AGE_HOURS = 72.0

    def __init__(
        self,
        app: Any = None,
        mailbox: Optional[SerenityMailboxVault] = None,
        mcp_client: Optional[SerenityDevMCPClient] = None,
    ):
        self.app = app
        self._explicit_mailbox = mailbox is not None
        self._active_db_path = self._resolve_user_db_path(app)
        self.mailbox = mailbox or SerenityMailboxVault(
            db_path=self._active_db_path
        )
        mcp_url = "http://127.0.0.1:8002/mcp"
        if app is not None and hasattr(app, "config") and isinstance(app.config, dict):
            mcp_url = app.config.get("serenitydev_mcp_url", mcp_url) or mcp_url
        self.mcp_client = mcp_client or SerenityDevMCPClient(base_url=mcp_url)

    @staticmethod
    def _resolve_user_db_path(app: Any) -> Optional[str]:
        if app is None:
            return None
        return SerenityEnvResolver.get_user_mailbox_db(app)

    def sync_environment_and_user(self) -> None:
        """
        Dynamically synchronizes the active user mailbox path and SerenityDev MCP URL.
        Handles both early SerenityApp.__init__ (before script_dir/config are loaded)
        and runtime profile switches in Settings Tab 5 across standalone and onedir builds.
        """
        if self.app is None or self._explicit_mailbox:
            return
        target_db = self._resolve_user_db_path(self.app)
        if target_db and os.path.abspath(target_db) != os.path.abspath(self._active_db_path or ""):
            self._active_db_path = target_db
            self.mailbox = SerenityMailboxVault(db_path=target_db)
        if hasattr(self.app, "config") and isinstance(self.app.config, dict):
            cfg_url = self.app.config.get("serenitydev_mcp_url")
            if cfg_url and isinstance(cfg_url, str) and cfg_url.strip():
                self.mcp_client.base_url = cfg_url.strip()

    def poll_deaddrop_and_outbox(self) -> Dict[str, Any]:
        """
        Scans `Users/<active_user>/Mailbox/DeadDrop/` for incoming `.smp` / `.bmb` files,
        ingests them through the Quarantine Sandbox, moves processed files to `Archive/`,
        and retries pending `outbox_queue` packets if online.
        """
        self.sync_environment_and_user()
        dd_dir = os.path.join(self.mailbox.mailbox_dir, "DeadDrop")
        arch_dir = os.path.join(self.mailbox.mailbox_dir, "Archive")
        os.makedirs(dd_dir, exist_ok=True)
        os.makedirs(arch_dir, exist_ok=True)

        imported = []
        errors = []
        for fname in sorted(os.listdir(dd_dir)):
            if not fname.lower().endswith((".smp", ".bmb")):
                continue
            fpath = os.path.join(dd_dir, fname)
            if not os.path.isfile(fpath):
                continue
            try:
                with open(fpath, "r", encoding="utf-8") as fp:
                    raw_payload = fp.read().strip()
                res = self.receive_smp_packet(raw_payload)
                imported.append({"file": fname, "result": res})
                dest = os.path.join(arch_dir, fname)
                try:
                    if os.path.exists(dest):
                        os.remove(dest)
                    os.replace(fpath, dest)
                except Exception:
                    pass
                pq = getattr(self.app, "process_queue", None) if self.app else None
                if pq:
                    pq.put({
                        "status": "p2p_notification",
                        "event": res.get("status", "received"),
                        "sender": res.get("sender", "Peer"),
                        "subject": res.get("subject", fname),
                        "details": res,
                    })
            except Exception as e:
                errors.append({"file": fname, "error": str(e)})

        flushed = 0
        if not is_offline_mode():
            try:
                with self.mailbox._get_connection() as conn:
                    rows = conn.execute(
                        "SELECT packet_id, peer_id, smp_payload, retry_count FROM outbox_queue WHERE status = 'pending' LIMIT 10"
                    ).fetchall()
                    for r in rows:
                        peer = self.mailbox.get_peer(r["peer_id"])
                        if peer and peer.get("endpoint_url"):
                            ok, _ = CrossStateP2PTransport.send_over_https(
                                endpoint_url=peer["endpoint_url"],
                                encrypted_smp_b64=r["smp_payload"],
                                sender_id=self.mailbox.identity["peer_id"],
                                packet_id=r["packet_id"],
                                timeout=2.5,
                            )
                            if ok:
                                conn.execute(
                                    "UPDATE outbox_queue SET status = 'delivered' WHERE packet_id = ?",
                                    (r["packet_id"],),
                                )
                                flushed += 1
                            else:
                                conn.execute(
                                    "UPDATE outbox_queue SET retry_count = retry_count + 1 WHERE packet_id = ?",
                                    (r["packet_id"],),
                                )
                    conn.commit()
            except Exception:
                pass

        return {"imported_count": len(imported), "imported": imported, "flushed_outbox": flushed, "errors": errors}

    # --- 72-HOUR TRUST TIER & VETERAN ACCESS CHECK ---

    def get_trust_status(self) -> Dict[str, Any]:
        """
        Evaluates the active profile's cryptographic Vault state and 72-hour age requirement.
        Returns {"tier": 0|1|2, "veteran_unlocked": bool, "age_hours": float, "label": str}.
        """
        vm = getattr(self.app, "vault_manager", None) if self.app else None
        if vm is None:
            return {
                "tier": 0,
                "veteran_unlocked": False,
                "age_hours": 0.0,
                "label": "Tier 0: Unverified (No VaultManager bound)",
            }

        try:
            lock_enabled = bool(vm.is_lock_enabled())
            is_locked = bool(vm.is_locked())
            age_hours = float(vm.get_profile_age_hours()) if hasattr(vm, "get_profile_age_hours") else 0.0
            veteran_ok = bool(
                vm.has_full_repo_read_access(min_age_hours=self.VETERAN_MIN_AGE_HOURS)
            ) if hasattr(vm, "has_full_repo_read_access") else (
                lock_enabled and not is_locked and age_hours >= self.VETERAN_MIN_AGE_HOURS
            )
        except Exception:
            return {
                "tier": 0,
                "veteran_unlocked": False,
                "age_hours": 0.0,
                "label": "Tier 0: Unverified",
            }

        if not lock_enabled or is_locked:
            return {
                "tier": 0,
                "veteran_unlocked": False,
                "age_hours": round(age_hours, 2),
                "label": "Tier 0: Vault Locked or Unprotected (Sandboxed)",
            }
        if veteran_ok:
            return {
                "tier": 2,
                "veteran_unlocked": True,
                "age_hours": round(age_hours, 2),
                "label": f"Tier 2: Trusted Veteran ({age_hours:.1f}h >= 72.0h — Expanded Access Unlocked)",
            }
        return {
            "tier": 1,
            "veteran_unlocked": False,
            "age_hours": round(age_hours, 2),
            "label": f"Tier 1: Probationary ({age_hours:.1f}h / 72.0h — Quarantine & Sandbox Enforced)",
        }

    # --- UI PROCESS QUEUE STREAMING HELPERS (V2.3.0 DROPDOWNS) ---

    def _emit_subagent_event(
        self,
        subagent_id: str,
        stage: str,
        label: str = "",
        chunk: str = "",
        level: int = 2,
        duration_ms: float = 0.0,
    ):
        """
        Emits per-subagent dropdown streaming events onto app.process_queue so SerenityPC
        renders collapsible dropdowns for each subagent alongside standard tool_log_update.
        """
        pq = getattr(self.app, "process_queue", None) if self.app else None
        if not pq:
            return
        try:
            if stage == "spawn":
                pq.put({
                    "status": "subagent_status",
                    "active": True,
                    "level": level,
                    "model": label,
                })
                pq.put({
                    "status": "subagent_spawn",
                    "subagent_id": subagent_id,
                    "level": level,
                    "label": f"[+] Subagent Lvl {level}: {label}",
                })
                pq.put({
                    "status": "tool_log_update",
                    "content": f"\n[SUBAGENT SPAWN] Lvl {level} ({label}) [{subagent_id}]",
                })
            elif stage == "stream":
                pq.put({
                    "status": "subagent_stream",
                    "subagent_id": subagent_id,
                    "level": level,
                    "chunk": chunk,
                })
            elif stage == "done":
                pq.put({
                    "status": "subagent_done",
                    "subagent_id": subagent_id,
                    "level": level,
                    "duration_ms": round(duration_ms, 1),
                })
                pq.put({"status": "subagent_status", "active": False})
                pq.put({
                    "status": "tool_log_update",
                    "content": f"\n[SUBAGENT COMPLETE] {subagent_id} finished in {duration_ms:.0f}ms.",
                })
        except Exception:
            pass

    # --- OUTBOUND P2P CHAT & STORE-AND-FORWARD MAIL ---

    def dispatch_peer_packet(
        self,
        peer_query: str,
        packet_type: PacketType,
        subject: str,
        body: str,
        attachment_type: str = "none",
        attachment_data: Optional[Dict[str, Any]] = None,
        target_level: Optional[int] = None,
        exchange_mode: ExchangeMode = ExchangeMode.AGENT_COURIER,
    ) -> Dict[str, Any]:
        """
        Packages, signs, and encrypts a P2P message or Mail item via Level 2 Courier.
        Attempts live HTTPS delivery first; if peer is in another state and offline/CGNAT,
        queues in SQLite outbox_queue and writes an encrypted .smp file to Mailbox/Outbox.
        """
        self.sync_environment_and_user()
        t0 = time.time()
        sub_id = f"sub_lvl2_courier_{uuid.uuid4().hex[:6]}"
        courier_lvl = int(
            getattr(self.app, "config", {}).get("messaging_default_level", self.DEFAULT_COURIER_LEVEL)
            if self.app and hasattr(self.app, "config")
            else self.DEFAULT_COURIER_LEVEL
        )
        self._emit_subagent_event(sub_id, "spawn", label="The Helper (P2P Courier)", level=courier_lvl)

        try:
            peer = self.mailbox.get_peer(peer_query)
            if not peer:
                return {
                    "status": "error",
                    "message": f"Peer '{peer_query}' not found in Serenity Address Book. Pair first using pair_serenity_peer.",
                }

            ident = self.mailbox.identity
            dest_lvl = target_level or courier_lvl

            envelope = InterSerenityEnvelope(
                type=packet_type.value,
                sender={
                    "peer_id": ident["peer_id"],
                    "name": ident["display_name"],
                    "edition": ident["app_edition"],
                    "persona_lvl": courier_lvl,
                    "fingerprint": ident["fingerprint"],
                    "ed25519_pub": ident["ed25519_public_hex"],
                    "x25519_pub": ident["x25519_public_hex"],
                },
                recipient={
                    "peer_id": peer["peer_id"],
                    "name": peer["display_name"],
                    "target_lvl": dest_lvl,
                },
                mode=exchange_mode.value,
                payload={
                    "subject": subject,
                    "body": body,
                    "attachment_type": attachment_type,
                    "attachment": attachment_data or {},
                },
            )

            # Sign with local Ed25519 private key
            SerenityMailCrypto.sign_envelope(envelope, ident["ed25519_private_hex"])

            # Encrypt with X25519 ECDH if peer public key is available, else passphrase
            use_ecdh = bool(peer.get("x25519_public_hex") and not peer.get("shared_passphrase"))
            smp_b64 = SerenityMailCrypto.encrypt_smp(
                envelope,
                passphrase=peer.get("shared_passphrase") or None,
                own_x25519_priv_hex=ident["x25519_private_hex"] if use_ecdh else None,
                peer_x25519_pub_hex=peer["x25519_public_hex"] if use_ecdh else None,
            )

            self._emit_subagent_event(
                sub_id,
                "stream",
                chunk=f"Encrypted SERENITY_SMP_V1 packet ({len(smp_b64)} bytes) for {peer['display_name']}.\n",
                level=courier_lvl,
            )

            # Record in local 'sent' folder
            with self.mailbox._get_connection() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO mailbox_messages (
                        packet_id, folder, msg_type, sender_id, sender_name,
                        recipient_id, subject, body, attachment_type, attachment_json,
                        target_level, exchange_mode, influence_weight, is_read, timestamp
                    ) VALUES (?, 'sent', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1.0, 1, ?)
                """, (
                    envelope.packet_id,
                    packet_type.value,
                    ident["peer_id"],
                    ident["display_name"],
                    peer["peer_id"],
                    subject,
                    body,
                    attachment_type,
                    json.dumps(attachment_data or {}),
                    dest_lvl,
                    exchange_mode.value,
                    envelope.timestamp,
                ))
                conn.commit()

            # 1. Try direct HTTPS delivery if peer has an endpoint URL
            delivered_live = False
            transport_note = ""
            if peer.get("endpoint_url"):
                ok, note = CrossStateP2PTransport.send_over_https(
                    endpoint_url=peer["endpoint_url"],
                    encrypted_smp_b64=smp_b64,
                    sender_id=ident["peer_id"],
                    packet_id=envelope.packet_id,
                )
                delivered_live = ok
                transport_note = note

            # 2. Always write .smp file if Dead-Drop path is set or if peer is offline
            smp_filename = f"{envelope.timestamp}_{peer['peer_id']}_{envelope.packet_id[:8]}.smp"
            outbox_file = os.path.join(self.mailbox.mailbox_dir, "Outbox", smp_filename)
            with open(outbox_file, "w", encoding="utf-8") as fp:
                fp.write(smp_b64)

            if peer.get("dead_drop_path") and os.path.isdir(peer["dead_drop_path"]):
                dd_file = os.path.join(peer["dead_drop_path"], smp_filename)
                try:
                    with open(dd_file, "w", encoding="utf-8") as fp:
                        fp.write(smp_b64)
                    transport_note += f" | Synced to Dead-Drop ({dd_file})"
                except Exception:
                    pass

            if not delivered_live:
                with self.mailbox._get_connection() as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO outbox_queue (
                            packet_id, peer_id, smp_payload, smp_file_path,
                            retry_count, status, created_at
                        ) VALUES (?, ?, ?, ?, 0, 'pending', ?)
                    """, (
                        envelope.packet_id,
                        peer["peer_id"],
                        smp_b64,
                        outbox_file,
                        envelope.timestamp,
                    ))
                    conn.commit()

            self._emit_subagent_event(
                sub_id,
                "stream",
                chunk=(
                    f"Delivery Status: {'LIVE P2P DELIVERED' if delivered_live else 'QUEUED IN STORE-AND-FORWARD OUTBOX'}\n"
                ),
                level=courier_lvl,
            )

            return {
                "status": "delivered_live" if delivered_live else "queued_store_and_forward",
                "packet_id": envelope.packet_id,
                "recipient": peer["display_name"],
                "recipient_id": peer["peer_id"],
                "target_level": dest_lvl,
                "smp_file": outbox_file,
                "smp_payload": smp_b64,
                "transport_note": transport_note or "Stored in Outbox (.smp) for async pickup.",
            }
        finally:
            self._emit_subagent_event(
                sub_id, "done", level=courier_lvl, duration_ms=(time.time() - t0) * 1000.0
            )

    # --- INBOUND PACKET INGESTION, QUARANTINE GATE & CECILIA PROTECTION ---

    def receive_smp_packet(
        self,
        smp_b64: str,
        sender_hint: str = "",
        passphrase: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Decrypts, verifies signature, screens through the Contextual Quarantine Sandbox,
        and routes an inbound .smp (or .bmb) packet according to the 72-Hour Trust Gate
        and Level 2 / Cecilia (Lvl 7) isolation rules.
        """
        self.sync_environment_and_user()
        ident = self.mailbox.identity
        peer = self.mailbox.get_peer(sender_hint) if sender_hint else None

        raw_dict = None
        if peer is not None or passphrase:
            use_ecdh = bool(peer and peer.get("x25519_public_hex") and not passphrase and not peer.get("shared_passphrase"))
            try:
                raw_dict = SerenityMailCrypto.decrypt_smp(
                    smp_b64,
                    passphrase=passphrase or (peer.get("shared_passphrase") if peer else None),
                    own_x25519_priv_hex=ident["x25519_private_hex"] if use_ecdh else None,
                    peer_x25519_pub_hex=peer["x25519_public_hex"] if use_ecdh else None,
                )
            except Exception:
                raw_dict = None

        if raw_dict is None:
            # Try paired peers in address book (for DeadDrop / file-imported .smp packets with no sender_hint)
            for cand_peer in self.mailbox.list_peers():
                try:
                    use_ecdh = bool(cand_peer.get("x25519_public_hex") and not cand_peer.get("shared_passphrase"))
                    raw_dict = SerenityMailCrypto.decrypt_smp(
                        smp_b64,
                        passphrase=cand_peer.get("shared_passphrase") or None,
                        own_x25519_priv_hex=ident["x25519_private_hex"] if use_ecdh else None,
                        peer_x25519_pub_hex=cand_peer["x25519_public_hex"] if use_ecdh else None,
                    )
                    peer = cand_peer
                    break
                except Exception:
                    continue

        if raw_dict is None:
            raw_dict = SerenityMailCrypto.decrypt_smp(smp_b64, passphrase=passphrase)

        # Handle Legacy Brainuke .bmb format transparently
        if raw_dict.get("type") == "brainuke_messageboard_packet":
            turns = raw_dict.get("turns", [])
            summary_body = "\n".join(
                f"[{t.get('source_device', 'peer')}] Prompt: {t.get('user_input', '')} -> {t.get('final_response', '')}"
                for t in turns
            )
            envelope = InterSerenityEnvelope(
                type=PacketType.SHARED_CONTEXT.value,
                sender={
                    "peer_id": raw_dict.get("session_id", "bmb_peer"),
                    "name": raw_dict.get("device_name", "BrainukePeer"),
                },
                recipient={"target_lvl": 2},
                payload={
                    "subject": f"Brainuke Packet ({len(turns)} turns)",
                    "body": summary_body,
                    "attachment_type": "bmb_turns",
                    "attachment": {"turns": turns},
                },
            )
        else:
            envelope = InterSerenityEnvelope.from_dict(raw_dict)

        sender_id = envelope.sender.get("peer_id", "unknown_peer")
        sender_name = envelope.sender.get("name", sender_id)
        if not peer:
            peer = self.mailbox.get_peer(sender_id)

        # Verify Ed25519 signature if public key is known
        sig_valid = False
        pub_hex = (peer.get("ed25519_public_hex") if peer else None) or envelope.sender.get("ed25519_pub", "")
        if pub_hex and envelope.signature_hex:
            sig_valid = SerenityMailCrypto.verify_envelope_signature(envelope, pub_hex)

        subject = str(envelope.payload.get("subject", "")).strip()
        body = str(envelope.payload.get("body", "")).strip()
        combined_text = f"{subject}\n{body}".strip()

        is_known_trusted = bool(peer and peer.get("trust_status") == "trusted" and sig_valid)
        intent_label, risk_score = SerenityMailboxVault.evaluate_intent_and_risk(
            combined_text, is_known_peer=is_known_trusted
        )

        trust = self.get_trust_status()
        target_lvl = int(envelope.recipient.get("target_lvl", 2))

        # Determine if packet must be held in Quarantine Sandbox:
        # 1. Risk score >= 0.4
        # 2. Local profile is < 72h old (Probationary / Tier 0-1)
        # 3. Unknown/unverified sender
        # 4. Addressed directly to Cecilia (Lvl 7) when 'cecilia_p2p_mode' is 'quarantine_gate' (default)
        cecilia_p2p_mode = (
            getattr(self.app, "config", {}).get("cecilia_p2p_mode", "quarantine_gate")
            if self.app and hasattr(self.app, "config")
            else "quarantine_gate"
        )
        hold_reasons = []
        if risk_score >= 0.4:
            hold_reasons.append(f"risk_score={risk_score} ({intent_label})")
        if not trust["veteran_unlocked"]:
            hold_reasons.append(f"profile_not_veteran ({trust['label']})")
        if not is_known_trusted:
            hold_reasons.append("unverified_sender_signature")
        if target_lvl == 7 and cecilia_p2p_mode == "off":
            hold_reasons.append("cecilia_p2p_access_disabled")

        if hold_reasons:
            hold_str = "; ".join(hold_reasons)
            with self.mailbox._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO quarantine_sandbox (
                        packet_id, sender_id, sender_name, msg_type, subject,
                        content, raw_envelope_json, intent_label, risk_score,
                        hold_reason, status, quarantined_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'quarantined', ?)
                """, (
                    envelope.packet_id,
                    sender_id,
                    sender_name,
                    envelope.type,
                    subject,
                    body,
                    envelope.to_json(),
                    intent_label,
                    risk_score,
                    hold_str,
                    int(time.time() * 1000),
                ))
                q_id = cursor.lastrowid
                conn.commit()
            return {
                "status": "quarantined",
                "quarantine_id": q_id,
                "packet_id": envelope.packet_id,
                "sender": sender_name,
                "intent": intent_label,
                "risk_score": risk_score,
                "hold_reason": hold_str,
            }

        # Safe + Veteran Profile + Trusted Peer -> Admit to Inbox (Level 2 Triage)
        # Note: If addressed to Lvl 7 under 'quarantine_gate', Level 2 admits it with bounded weight 0.25
        effective_lvl = 2 if (target_lvl == 7 and cecilia_p2p_mode == "quarantine_gate") else target_lvl
        with self.mailbox._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO mailbox_messages (
                    packet_id, folder, msg_type, sender_id, sender_name,
                    recipient_id, subject, body, attachment_type, attachment_json,
                    target_level, exchange_mode, influence_weight, is_read, timestamp
                ) VALUES (?, 'inbox', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.25, 0, ?)
            """, (
                envelope.packet_id,
                envelope.type,
                sender_id,
                sender_name,
                ident["peer_id"],
                subject,
                body,
                envelope.payload.get("attachment_type", "none"),
                json.dumps(envelope.payload.get("attachment", {})),
                effective_lvl,
                envelope.mode,
                envelope.timestamp,
            ))
            msg_id = cursor.lastrowid
            conn.commit()

        return {
            "status": "admitted",
            "message_id": msg_id,
            "packet_id": envelope.packet_id,
            "sender": sender_name,
            "subject": subject,
            "target_level": effective_lvl,
            "intent": intent_label,
            "risk_score": risk_score,
            "influence_weight": 0.25,
        }

    def resolve_quarantine_item(
        self, quarantine_id: int, action: str = "admit", reason: str = ""
    ) -> Dict[str, Any]:
        """Moderates a quarantined packet: 'admit' moves it to inbox; 'reject' marks it rejected."""
        self.sync_environment_and_user()
        act = action.strip().lower()
        now_ms = int(time.time() * 1000)
        with self.mailbox._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM quarantine_sandbox WHERE id = ?", (quarantine_id,))
            row = cursor.fetchone()
            if not row:
                return {"status": "error", "message": f"Quarantine ID {quarantine_id} not found."}

            if act == "admit":
                env = InterSerenityEnvelope.from_json(row["raw_envelope_json"])
                cursor.execute("""
                    INSERT OR IGNORE INTO mailbox_messages (
                        packet_id, folder, msg_type, sender_id, sender_name,
                        recipient_id, subject, body, attachment_type, attachment_json,
                        target_level, exchange_mode, influence_weight, is_read, timestamp
                    ) VALUES (?, 'inbox', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0.25, 0, ?)
                """, (
                    env.packet_id,
                    env.type,
                    row["sender_id"],
                    row["sender_name"],
                    self.mailbox.identity["peer_id"],
                    row["subject"],
                    row["content"],
                    env.payload.get("attachment_type", "none"),
                    json.dumps(env.payload.get("attachment", {})),
                    int(env.recipient.get("target_lvl", 2)),
                    env.mode,
                    now_ms,
                ))
                msg_id = cursor.lastrowid
                cursor.execute(
                    "UPDATE quarantine_sandbox SET status = 'admitted', resolved_at = ? WHERE id = ?",
                    (now_ms, quarantine_id),
                )
                conn.commit()
                return {"status": "admitted", "quarantine_id": quarantine_id, "message_id": msg_id}
            else:
                cursor.execute(
                    "UPDATE quarantine_sandbox SET status = 'rejected', hold_reason = ?, resolved_at = ? WHERE id = ?",
                    (reason or row["hold_reason"], now_ms, quarantine_id),
                )
                conn.commit()
                return {"status": "rejected", "quarantine_id": quarantine_id, "reason": reason}

    # --- SERENITYDEV MCP FRONTEND DELEGATION (V2.3.0) ---

    def delegate_to_serenitydev(
        self,
        command: str,
        prompt: str,
        workspace_path: str = "",
    ) -> Dict[str, Any]:
        """
        Delegates a coding or workspace task from SerenityPC to SerenityDev over MCP.
        - Read-only modes ('explore', 'plan', 'read_file', 'grep_search') work at any trust tier.
        - Mutating modes ('execute', 'agent', 'write_file', 'insert_edit_into_file', 'run_command')
          strictly require the 72-Hour Veteran Profile unlock (Tier 2).
        """
        self.sync_environment_and_user()
        cmd = command.strip().lower().lstrip("/")
        mutating_commands = {"execute", "agent", "write_file", "insert_edit_into_file", "replace_string_in_file", "run_command"}

        trust = self.get_trust_status()
        if cmd in mutating_commands and not trust["veteran_unlocked"]:
            return {
                "status": "blocked_72h_gate",
                "command": cmd,
                "trust_status": trust["label"],
                "message": (
                    f"[72-HOUR TRUST GATE] Mutating SerenityDev MCP command '/{cmd}' requires an unlocked "
                    f"password-protected profile >= 72 hours old. Current status: {trust['label']}."
                ),
            }

        t0 = time.time()
        sub_id = f"sub_dev_{cmd}_{uuid.uuid4().hex[:6]}"
        self._emit_subagent_event(sub_id, "spawn", label=f"SerenityDev MCP (/{cmd})", level=3)
        try:
            # Package OTL ContextSnapshot for SerenityDev
            otl_snapshot = {
                "origin": "SerenityPC",
                "command": f"/{cmd}",
                "prompt": prompt,
                "workspace_dir": workspace_path or getattr(self.app, "script_dir", os.getcwd()),
                "trust_tier": trust["tier"],
            }
            self._emit_subagent_event(
                sub_id,
                "stream",
                chunk=f"Dispatching OTL ContextSnapshot to SerenityDev MCP (/{cmd})...\n",
                level=3,
            )

            if self.mcp_client.is_online():
                try:
                    mcp_res = self.mcp_client.call_rpc(
                        "serenitydev/orchestrate",
                        otl_snapshot,
                    )
                    output = mcp_res.get("result", mcp_res)
                    self._emit_subagent_event(
                        sub_id,
                        "stream",
                        chunk=f"SerenityDev response received.\n",
                        level=3,
                    )
                    return {
                        "status": "completed",
                        "command": f"/{cmd}",
                        "subagent_id": sub_id,
                        "output": output,
                    }
                except Exception as rpc_err:
                    return {
                        "status": "queued_offline",
                        "command": f"/{cmd}",
                        "subagent_id": sub_id,
                        "otl_snapshot": otl_snapshot,
                        "message": f"SerenityDev MCP server reached but method call failed ({rpc_err}). OTL snapshot staged for handoff.",
                    }
            else:
                return {
                    "status": "queued_offline",
                    "command": f"/{cmd}",
                    "subagent_id": sub_id,
                    "otl_snapshot": otl_snapshot,
                    "message": "SerenityDev MCP server (port 8002) is offline. OTL snapshot staged for handoff.",
                }
        except Exception as e:
            return {
                "status": "error",
                "command": f"/{cmd}",
                "error": str(e),
            }
        finally:
            self._emit_subagent_event(
                sub_id, "done", level=3, duration_ms=(time.time() - t0) * 1000.0
            )
