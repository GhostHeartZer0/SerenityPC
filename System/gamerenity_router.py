import ctypes
import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

GAME_SIGNATURES: Dict[str, Tuple[str, ...]] = {
    "genshin": ("genshin impact", "genshinimpact.exe", "yuanshen.exe"),
    "nms": ("no man's sky", "nms.exe"),
    "div2": ("tom clancy's the division 2", "the division 2", "thedivision2.exe"),
}


def get_foreground_window_title() -> str:
    """Return active Windows foreground window title via user32 (empty on non-Windows)."""
    try:
        user32 = ctypes.windll.user32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return ""
        buf = ctypes.create_unicode_buffer(512)
        user32.GetWindowTextW(hwnd, buf, 512)
        return buf.value or ""
    except Exception:
        return ""


def _running_process_names() -> str:
    try:
        res = subprocess.run(
            ["tasklist", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3,
        )
        return res.stdout.lower() if res.returncode == 0 else ""
    except Exception:
        return ""


def detect_running_game(
    window_title: Optional[str] = None,
    process_names: Optional[Iterable[str]] = None,
) -> Optional[str]:
    """Detect active supported game ('genshin', 'nms', 'div2') from window title or processes."""
    title_low = (window_title if window_title is not None else get_foreground_window_title()).lower()
    for game, sigs in GAME_SIGNATURES.items():
        if any(s in title_low for s in sigs):
            return game
    procs = " ".join(p.lower() for p in process_names) if process_names is not None else _running_process_names()
    for game, sigs in GAME_SIGNATURES.items():
        if any(s in procs for s in sigs):
            return game
    return None


def resolve_game_model(
    game_key: str,
    cfg_path: str = "benchmarks/training station/station_config.json",
    gguforge_root: str = "benchmarks/training station/GGUForge",
) -> Optional[str]:
    """Resolve forged GGUF path for a game key from station_config overrides or newest GGUForge artifact."""
    key = str(game_key or "").strip().lower()
    if not key:
        return None
    cp = Path(cfg_path)
    if cp.exists():
        try:
            cfg = json.loads(cp.read_text(encoding="utf-8"))
            mapped = ((cfg.get("models") or {}).get("gamerenity_active_models") or {}).get(key)
            if mapped and Path(mapped).exists():
                return str(Path(mapped))
        except Exception:
            pass
    gf = Path(gguforge_root)
    if not gf.exists():
        return None
    cands = [
        g
        for g in gf.glob(f"dr_zero_{key}_*/cycle_*/*.gguf")
        if not g.name.endswith("_f16.gguf")
    ]
    if not cands:
        return None
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(cands[0])


def resolve_active_game_model(
    cfg_path: str = "benchmarks/training station/station_config.json",
    gguforge_root: str = "benchmarks/training station/GGUForge",
) -> Tuple[Optional[str], Optional[str]]:
    """Return (game_key, gguf_path) if a configured game is currently running."""
    game = detect_running_game()
    if not game:
        return None, None
    return game, resolve_game_model(game, cfg_path=cfg_path, gguforge_root=gguforge_root)
