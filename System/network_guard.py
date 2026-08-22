# System/network_guard.py
# Offline Mode Network Interceptor & Traffic Guard for SerenityPC.

import socket
import sys

_OFFLINE_MODE_ACTIVE = False
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_CREATE_CONNECTION = socket.create_connection

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1", "0.0.0.0"}

def _is_loopback(address):
    try:
        if isinstance(address, tuple) and len(address) >= 1:
            host = address[0]
        elif isinstance(address, str):
            host = address
        else:
            return False
        return host in _LOOPBACK_HOSTS or host.startswith("127.")
    except Exception:
        return False

def _guarded_connect(self, address):
    global _OFFLINE_MODE_ACTIVE
    if _OFFLINE_MODE_ACTIVE and not _is_loopback(address):
        raise PermissionError(f"[OFFLINE MODE] External network connection to {address} blocked by policy.")
    return _ORIGINAL_SOCKET_CONNECT(self, address)

def _guarded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    global _OFFLINE_MODE_ACTIVE
    if _OFFLINE_MODE_ACTIVE and not _is_loopback(address):
        raise PermissionError(f"[OFFLINE MODE] External connection creation to {address} blocked by policy.")
    return _ORIGINAL_CREATE_CONNECTION(address, timeout=timeout, source_address=source_address)

def enable_offline_mode():
    global _OFFLINE_MODE_ACTIVE
    _OFFLINE_MODE_ACTIVE = True
    socket.socket.connect = _guarded_connect
    socket.create_connection = _guarded_create_connection
    print("[SECURITY] Offline Mode ENABLED: All external network calls intercepted.")

def disable_offline_mode():
    global _OFFLINE_MODE_ACTIVE
    _OFFLINE_MODE_ACTIVE = False
    socket.socket.connect = _ORIGINAL_SOCKET_CONNECT
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    print("[SECURITY] Offline Mode DISABLED: Normal network operations restored.")

def set_offline_mode(enabled: bool):
    if enabled:
        enable_offline_mode()
    else:
        disable_offline_mode()

def is_offline_mode() -> bool:
    return _OFFLINE_MODE_ACTIVE
