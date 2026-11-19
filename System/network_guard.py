# System/network_guard.py
# Offline Mode Network Interceptor & Traffic Guard for SerenityPC.

import socket
import sys

_OFFLINE_MODE_ACTIVE = False
_ORIGINAL_SOCKET_CONNECT = socket.socket.connect
_ORIGINAL_SOCKET_CONNECT_EX = socket.socket.connect_ex
_ORIGINAL_SOCKET_SENDTO = socket.socket.sendto
_ORIGINAL_CREATE_CONNECTION = socket.create_connection
_ORIGINAL_GETADDRINFO = socket.getaddrinfo

_LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

def _is_loopback(address):
    try:
        if isinstance(address, (tuple, list)) and len(address) >= 1:
            host = address[0]
        elif isinstance(address, str):
            host = address
        else:
            return False
        return isinstance(host, str) and (host.lower() in _LOOPBACK_HOSTS or host.startswith("127."))
    except Exception:
        return False

def _guarded_connect(self, address):
    if _OFFLINE_MODE_ACTIVE and not _is_loopback(address):
        raise PermissionError(f"[OFFLINE MODE] External network connection to {address} blocked by policy.")
    return _ORIGINAL_SOCKET_CONNECT(self, address)

def _guarded_connect_ex(self, address):
    if _OFFLINE_MODE_ACTIVE and not _is_loopback(address):
        raise PermissionError(f"[OFFLINE MODE] External network connect_ex to {address} blocked by policy.")
    return _ORIGINAL_SOCKET_CONNECT_EX(self, address)

def _guarded_sendto(self, data, *args):
    if _OFFLINE_MODE_ACTIVE and args:
        address = args[-1]
        if address is not None and not _is_loopback(address):
            raise PermissionError(f"[OFFLINE MODE] External UDP sendto to {address} blocked by policy.")
    return _ORIGINAL_SOCKET_SENDTO(self, data, *args)

def _guarded_create_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    if _OFFLINE_MODE_ACTIVE and not _is_loopback(address):
        raise PermissionError(f"[OFFLINE MODE] External connection creation to {address} blocked by policy.")
    return _ORIGINAL_CREATE_CONNECTION(address, timeout=timeout, source_address=source_address)

def _guarded_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if _OFFLINE_MODE_ACTIVE and host not in (None, "") and not _is_loopback(host):
        raise PermissionError(f"[OFFLINE MODE] External DNS resolution for '{host}' blocked by policy.")
    return _ORIGINAL_GETADDRINFO(host, port, family, type, proto, flags)

def enable_offline_mode():
    global _OFFLINE_MODE_ACTIVE
    _OFFLINE_MODE_ACTIVE = True
    socket.socket.connect = _guarded_connect
    socket.socket.connect_ex = _guarded_connect_ex
    socket.socket.sendto = _guarded_sendto
    socket.create_connection = _guarded_create_connection
    socket.getaddrinfo = _guarded_getaddrinfo
    print("[SECURITY] Offline Mode ENABLED: All external network calls intercepted.")

def disable_offline_mode():
    global _OFFLINE_MODE_ACTIVE
    _OFFLINE_MODE_ACTIVE = False
    socket.socket.connect = _ORIGINAL_SOCKET_CONNECT
    socket.socket.connect_ex = _ORIGINAL_SOCKET_CONNECT_EX
    socket.socket.sendto = _ORIGINAL_SOCKET_SENDTO
    socket.create_connection = _ORIGINAL_CREATE_CONNECTION
    socket.getaddrinfo = _ORIGINAL_GETADDRINFO
    print("[SECURITY] Offline Mode DISABLED: Normal network operations restored.")

def set_offline_mode(enabled: bool):
    if enabled:
        enable_offline_mode()
    else:
        disable_offline_mode()

def is_offline_mode() -> bool:
    return _OFFLINE_MODE_ACTIVE
