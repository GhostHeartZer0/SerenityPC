import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from System.gamerenity_router import (
    detect_running_game,
    resolve_active_game_model,
    resolve_game_model,
)

CORE_PATH = ROOT / "benchmarks" / "training station" / "cartridges" / "game_assistants" / "gamerenity_core.py"
if CORE_PATH.exists():
    _spec = importlib.util.spec_from_file_location("gamerenity_core", str(CORE_PATH))
    _core_mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_core_mod)
    GamerenityGuideManager = _core_mod.GamerenityGuideManager
    GamerenityIndicator = _core_mod.GamerenityIndicator
else:
    GamerenityGuideManager = None
    GamerenityIndicator = None


def test_detect_running_game_window_and_process():
    assert detect_running_game(window_title="Genshin Impact", process_names=[]) == "genshin"
    assert detect_running_game(window_title="No Man's Sky", process_names=[]) == "nms"
    assert detect_running_game(window_title="Tom Clancy's The Division 2", process_names=[]) == "div2"
    assert detect_running_game(window_title="Visual Studio Code", process_names=["yuanshen.exe"]) == "genshin"
    assert detect_running_game(window_title="Desktop", process_names=["NMS.exe"]) == "nms"
    assert detect_running_game(window_title="Desktop", process_names=["TheDivision2.exe"]) == "div2"
    assert detect_running_game(window_title="Notepad", process_names=["notepad.exe"]) is None


def test_resolve_game_model_override_and_fallback(tmp_path):
    forge = tmp_path / "GGUForge"
    c1 = forge / "dr_zero_genshin_test_20261115" / "cycle_1"
    c1.mkdir(parents=True)
    f16 = c1 / "model_f16.gguf"
    q6 = c1 / "model.Q6_K.gguf"
    f16.write_bytes(b"f16")
    q6.write_bytes(b"q6k")

    override = tmp_path / "override.gguf"
    override.write_bytes(b"ovr")
    cfg = tmp_path / "station_config.json"
    cfg.write_text(json.dumps({"models": {"gamerenity_active_models": {"nms": str(override)}}}), encoding="utf-8")

    # Override wins for nms; genshin falls back to GGUForge ignoring _f16.gguf
    assert resolve_game_model("nms", cfg_path=str(cfg), gguforge_root=str(forge)) == str(override)
    assert resolve_game_model("genshin", cfg_path=str(cfg), gguforge_root=str(forge)) == str(q6)
    assert resolve_game_model("unknown_game", cfg_path=str(cfg), gguforge_root=str(forge)) is None


def test_resolve_live_station_config_models():
    cfg = ROOT / "benchmarks" / "training station" / "station_config.json"
    forge = ROOT / "benchmarks" / "training station" / "GGUForge"
    if not cfg.exists():
        pytest.skip("Training Station migrated to Environment M")
    for game in ("genshin", "nms", "div2"):
        resolved = resolve_game_model(game, cfg_path=str(cfg), gguforge_root=str(forge))
        assert resolved is not None, f"Expected resolved GGUF for {game}"
        assert Path(resolved).exists(), f"Missing resolved GGUF: {resolved}"
        assert "LFM3" not in Path(resolved).name, f"LFM3 unsupported in llama_cpp: {resolved}"
        assert "270m" not in Path(resolved).name, f"Rejected 270m model selected: {resolved}"


def test_gamerenity_guide_manager_and_indicator(tmp_path):
    if GamerenityGuideManager is None or GamerenityIndicator is None:
        pytest.skip("Training Station migrated to Environment M")
    mgr = GamerenityGuideManager(guides_dir=str(tmp_path))
    gdir = tmp_path / "div2"
    gdir.mkdir()
    (gdir / "compiled_guides.json").write_text(
        json.dumps({"combat": [{"title": "Cover", "text": "Use cover."}], "builds_loadouts": []}),
        encoding="utf-8",
    )
    assert mgr.list_games() == ["div2"]
    bd = mgr.get_game_breakdown("div2")
    assert bd["total_entries"] == 1
    assert bd["categories"]["combat"] == 1
    assert mgr.get_category_snippets("div2", "combat")[0]["title"] == "Cover"

    ind = GamerenityIndicator()
    ind.set_capturing(True)
    assert ind.is_capturing is True
    ind.set_capturing(False)
    assert ind.is_capturing is False


def test_main_game_command_and_autoroute_present():
    main_src = (ROOT / "main.py").read_text(encoding="utf-8", errors="replace")
    assert 'raw_msg.lower().startswith("/game")' in main_src
    assert 'self.config.get("gamerenity_auto_route", False)' in main_src
    assert 'target_tier="gamerenity"' in main_src
