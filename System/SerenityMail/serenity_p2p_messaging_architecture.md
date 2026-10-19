# Serenity P2P Messaging, Mail & V2.3.0 Subagent/MCP Architecture Blueprint

## 1. Executive Verdict: What to Do With `Serenity Live` (`t5gemma-2`)

**Retire `Serenity Live` as a standalone runtime (`t5_server.py`), harvest its best daemon and telemetry modules into `SerenityPC`, and do not merge the Hugging Face `transformers`/`bitsandbytes` stack into your `llama.cpp` core.**

| Component in `Serenity Live` | Architectural Verdict | Destination in `SerenityPC` / Ecosystem | Rationale | 
| ----- | ----- | ----- | ----- | 
| **`Engine/t5_server.py` (`t5gemma-2` + `bitsandbytes`)** | **Retire / Archive** | Replaced by `llama.cpp` (`Gemma-4 E2B/E4B`) | Running a separate PyTorch/Transformers server on port `8001` ([t5_server.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Qv53VqmihEwtmHEPjD0hx8uD_uge7dBs%2Fview%3Fusp%3Ddrivesdk)) fragments your 6 GB+ VRAM budget and duplicates `Gemma-4` GGUF capabilities already native to [SerenityPC](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1O96TyVy0N4kExiX10VmZnEvjEbMtZjWa). | 
| **`Serenity_Tray.py` (`pystray` + UDP Singleton Lock)** | **Harvest & Merge** | `SerenityPC/System/persistence_tray.py` | The UDP socket lock (`127.0.0.1:47200` / `47201`) and `pystray` lifecycle in [Serenity_Tray.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Hnvcf0ST347GbD_g7wJLlRl2i22VAIfb%2Fview%3Fusp%3Ddrivesdk) provide the exact lightweight daemon foundation needed for background P2P message listening and Persistence Mode without keeping heavy weights in VRAM. | 
| **`GamingModeMonitor` (`psutil` + `pynvml` 90% Guard)** | **Harvest & Merge** | `SerenityPC/System/vram_guard.py` | Automatic process detection and `>90%` VRAM pressure throttling in [t5_server.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Qv53VqmihEwtmHEPjD0hx8uD_uge7dBs%2Fview%3Fusp%3Ddrivesdk) prevents background P2P courier or DMN simmer tasks from causing Windows WDDM shared memory spills while gaming. | 
| **Downtime Session Key Rotation (`/rotate_key`)** | **Harvest & Merge** | `SerenityPC/System/mcp_bridge.py` | Rotating ephemeral IPC/P2P session tokens whenever models offload ([serenity_live.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1HH8uCbC0_SZ4bLAVsCY86nJKYyvkKMYr%2Fview%3Fusp%3Ddrivesdk)) hardens long-running P2P listeners. | 
| **5-Bucket Session Compaction ("Nahida's Garden")** | **Harvest & Merge** | `SerenityPC/System/semantic_vault.py` | Compacting long sessions into structured Markdown buckets (`Directives`, `Lore`, `Hardware State`, `Decisions`, `Unresolved`) in [serenity_live.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1HH8uCbC0_SZ4bLAVsCY86nJKYyvkKMYr%2Fview%3Fusp%3Ddrivesdk) pairs naturally with your SQLite `SemanticVault` ([semantic_vault.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1EebqL_wFxXPfy7jXmt7bDyTbvqOc_UdQ%2Fview%3Fusp%3Ddrivesdk)) and V2.1.0 DMN archive cleanup. | 
| **Adaptive Cadence & Audio Interrupt Monitor** | **Keep Modular** | `SerenityPC/System/voice_cadence.py` | `SoundDeviceMicrophone` and RMS barge-in interruption (`_interrupt_monitor` in [serenity_live.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1HH8uCbC0_SZ4bLAVsCY86nJKYyvkKMYr%2Fview%3Fusp%3Ddrivesdk)) can be imported on-demand for voice sessions without needing `Serenity Live` as a separate app. | 

## 2. Unified Inter-Serenity Bus: One Protocol for P2P, Subagents, and Brainuke

Your instincts are spot-on: **P2P user-to-user messaging (`SerenityPC` ↔ `SerenityPC Legacy`), local subagent delegation (`SerenityPC` ↔ `SerenityDev`), and `Project Brainuke` (`Cecilia` Throne ↔ `SerenityDroid`) should all ride on a single unified transport and envelope layer.**

You already have 80% of the primitives built across your three codebases:

1. **Transport Layer (`SerenityDev` MCP + `Brainuke` SSE)**:

   * Use `SerenityDev`'s MCP 2026-07-28 StreamableHTTP JSON-RPC 2.0 endpoint (`/mcp`) and SSE notification stream (`/subscriptions/listen`) from [serenitydevserver.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FJ16a2vyKsXBiI2Pu97ameFZ3pLLNlE-%2Fview%3Fusp%3Ddrivesdk) and [start_native_mcp.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F15ZUxEBe4QXKPRhUX9E9lGjisupB4qfcX%2Fview%3Fusp%3Ddrivesdk).

   * Use `TLSProvisioner` ([provisioner.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Zx2AOix_aEaUJzj6uWk0NFg0Wn6b9yw0%2Fview%3Fusp%3Ddrivesdk)) for automatic Root CA + multi-SAN leaf certificate generation so peers on LAN, Tailscale/ZeroTier VPNs, or direct port-forwards connect over TLS 1.3 without SAN mismatch errors.

2. **Envelope Schema (`ProtocolEnvelope v1.1` extending `brainuke_core/protocol/schema.py`)**:

   * Extend `PacketType` and `ExchangeMode` in [schema.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1x90sYX0gJKkI50nEw_zbdWg6E9aDHluJ%2Fview%3Fusp%3Ddrivesdk) so both `SerenityPC` (Main) and `SerenityPC Legacy` ([SerenityPC Legacy v1.8.0](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Fdrive%2Ffolders%2F1MEY7GmXz5vPC7zvi-xYLf3gssGYb0GB4)) speak the exact same wire format:

```
# System/inter_serenity_protocol.py
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional, List
import time, uuid, json

class PacketType(str, Enum):
    HANDSHAKE = "handshake"
    HEARTBEAT = "heartbeat"
    # P2P Messaging & Mail
    PEER_CHAT = "peer_chat"                 # Real-time P2P chat / progress ping
    PEER_MAIL = "peer_mail"                 # Async store-and-forward mail (.smp / .bmb)
    SHARED_CONTEXT = "shared_context"       # Shared chat history / Wringer report / snippet
    # Subagent & MCP Delegation (V2.3.0)
    SUBAGENT_DISPATCH = "subagent_dispatch" # OTL handoff to Lvl 1-6 or SerenityDev
    SUBAGENT_STREAM = "subagent_stream"     # Live token/thought stream for UI dropdowns
    SUBAGENT_REPORT = "subagent_report"     # OTL HandoffReport back to Supervisor
    # Brainuke / Cecilia Core
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
    type: str
    sender: Dict[str, Any]                  # {"user": "Chris", "device": "pc_main", "app_version": "1.9.0", "persona_lvl": 2, "peer_fp": "sha256:..."}
    recipient: Dict[str, Any]               # {"user": "Jesse", "device": "pc_legacy", "target_lvl": 2}
    payload: Dict[str, Any]
    version: str = "1.1"
    packet_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    mode: str = ExchangeMode.AGENT_COURIER.value
    signature_hex: Optional[str] = None     # Ed25519 / HMAC-SHA256 peer signature

```

## 3. P2P Messaging vs. Async Mail Service (Main ↔ Legacy)

To support both instant progress updates ("Hey, check out how Gemma-4 26B scored on Wringer") and rich offline sharing (sending full `.history.encz` chat logs, prompt templates, or code patches between your `main` build and your friend's `SerenityPC Legacy` build):

| Feature | Live P2P Messaging (`PEER_CHAT` / `AGENT_DISCUSSION`) | Async Serenity Mail (`PEER_MAIL` / `.smp` Container) | 
| ----- | ----- | ----- | 
| **Delivery Mechanism** | Real-time HTTPS POST `/mcp` or SSE `/subscriptions/listen` stream between online peers. | Store-and-forward SQLite `outbox_queue` (modeled on `failed_prompt_buffer` in [vault.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F112oFrYMlm7SMN6i2hJ_eGMh6k1N3Z1ZW%2Fview%3Fusp%3Ddrivesdk)) that auto-flushes on peer heartbeat, **plus** exportable `.smp` (Serenity Mail Packet) files. | 
| **Encryption** | TLS 1.3 in transit + ephemeral session key derived via X25519 peer handshake. | End-to-end encrypted container upgrading `.bmb` ([MessageBoardCrypto.kt](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1tkFYuYdYGy1W7V8FwSbde7GhEIYpHfla%2Fview%3Fusp%3Ddrivesdk)) and `SERENITY_ENCZ_V1` ([vault_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1SdCFbD9-k39-xBLDVd9Z57_2s3xPIaQE%2Fview%3Fusp%3Ddrivesdk)): `zlib` compression + `AES-256-GCM` (`12B Nonce` + `16B Tag`). | 
| **Payload Types** | Short text, status pings, active model/VRAM telemetry, or live Serenity-to-Serenity Q&A turns. | Full chat threads, `SerializedAgentContext` snapshots ([orchestration_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1oXSfCetVyM6-pFMpDw3hlHTIRtolDVfB%2Fview%3Fusp%3Ddrivesdk)), Wringer `.md` reports, custom parameter template slots (`T1`–`T32`), or staged code diffs. | 
| **Legacy Compatibility** | Drop-in compatible with `SerenityPC Legacy` via standard Python `requests`/`http.server` or `fastapi` without requiring `Brainuke` or `SerenityDev`. | `.smp` files can be auto-synced over the P2P link or dropped into `Users/<user>/Mailbox/Inbox/` from any transport. | 

## 4. Why Level 2 Default + Cecilia/Brainuke Gatekeeping Works Best

Setting **Level 2 ("The Helper / Searcher")** as the default integration handler in `SerenityPC` Settings aligns directly with your existing codebase:

1. **Tool Registry Threshold (`level >= 2`)**:

   * In [tool_registry.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F141jdEqzEt2R5qmqUzc0tfe6iAyhTwfU-%2Fview%3Fusp%3Ddrivesdk), `GemmaToolRegistry.get_definitions(level)` enforces `if level < 2: return []`. Level 1 (`Spark` / `The Speedy`) is strictly tool-free for raw low-latency replies, making **Level 2 the lowest tier capable of executing `send_peer_message`, `read_inbox_mail`, or `share_chat_snippet`**.

2. **VRAM & Context Efficiency**:

   * In your [config.json](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1tOxVFqJ4j76T7udyuAMS8pYqFFcoTGSz%2Fview%3Fusp%3Ddrivesdk), `search` (Lvl 2) runs `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf` (\~4.89 GiB total footprint at Q4_0 KV cache per [README.md](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1AbSu6t0IFVeeEGl7eRIhtxI-mykFQeta%2Fview%3Fusp%3Ddrivesdk)), fitting 100% in 6 GB VRAM (`gpu_layer_config: 99`). It can stay resident or fast-swap in `<1.5s` to summarize incoming mail or dispatch a progress update to your friend without spinning up a 26B model.

3. **Protecting Cecilia (Lvl 7) & Brainuke via the Quarantine Sandbox**:

   * If you allow direct external P2P packets into Level 7 (`Cecilia`) or `Brainuke Throne` by default, a peer's prompt or shared chat log could pollute `semantic_vault.db` ([semantic_vault.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1EebqL_wFxXPfy7jXmt7bDyTbvqOc_UdQ%2Fview%3Fusp%3Ddrivesdk)) or trigger unwanted affect shifts.

   * Instead, **Level 2 acts as the Gatekeeper / Courier**:

     * Every inbound peer message or mail packet first lands in the **Contextual Sandbox (`quarantine_posts`)** ported from `Project Brainuke` ([vault.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F112oFrYMlm7SMN6i2hJ_eGMh6k1N3Z1ZW%2Fview%3Fusp%3Ddrivesdk), [main.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1RNCXWT3EoXEBXTX2XdJnBctg8av-HcFt%2Fview%3Fusp%3Ddrivesdk)).

     * Level 2 evaluates `intent_label` and `risk_score` (`< 0.4` auto-admits to inbox; `>= 0.4` holds in quarantine).

     * When **Cecilia / Brainuke Access** is toggled ON in Settings (or when a message is explicitly addressed to `target_lvl: 7`), Level 2 packages a clean `SerializedAgentContext` ([orchestration_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1oXSfCetVyM6-pFMpDw3hlHTIRtolDVfB%2Fview%3Fusp%3Ddrivesdk)) and hands it off to Cecilia with `source_device="peer_<name>"` and a bounded `influence_weight=0.25` (lower than PC Master `1.0` and Android Companion `0.35` in [vault.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F112oFrYMlm7SMN6i2hJ_eGMh6k1N3Z1ZW%2Fview%3Fusp%3Ddrivesdk)). Meanwhile, Cecilia's **3D PAD Affect Engine** ([affect_engine.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1ehmaQzBI7TVV-0rIQq1zR4DOY7N_JgPZ%2Fview%3Fusp%3Ddrivesdk)) retains its anti-docility dominance recoil ($D \mathrel{+}= 0.2, V \mathrel{-}= 0.15$) if a peer tries to boss her around.

## 5. Leveraging the 72-Hour Profile Age Mechanic (`has_full_repo_read_access`)

In [vault_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1SdCFbD9-k39-xBLDVd9Z57_2s3xPIaQE%2Fview%3Fusp%3Ddrivesdk) (lines 254–285) and [tool_registry.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F141jdEqzEt2R5qmqUzc0tfe6iAyhTwfU-%2Fview%3Fusp%3Ddrivesdk) (lines 274–308), you already implemented a clean cryptographic age check:

* `VaultManager.get_profile_age_hours()` checks the earliest timestamp across `vault_state.json` (`created_at`, `last_migration`, and `os.path.getctime`).

* `VaultManager.has_full_repo_read_access(min_age_hours=72.0)` requires:

  1. `lock_enabled == True` (Master password set),

  2. `not is_locked()` (Vault actively unlocked in session), and

  3. `get_profile_age_hours() >= 72.0`.

### Expanding the 72-Hour Trust Gate Across PC Access, Subagents, and P2P

We can generalize `has_full_repo_read_access(72.0)` into a unified **Trust Tier Resolver** (`VaultManager.get_trust_tier()`) that governs both local PC automation and P2P/Subagent capabilities for you (on `main`) and your friend (on `Legacy`):

| Trust Tier | Condition | Local PC & IDE Access | P2P Messaging, Mail & Subagent Permissions | 
| ----- | ----- | ----- | ----- | 
| **Tier 0: Unverified / Ephemeral** | `Default`/`Public` profile, or Vault lock disabled | Sandboxed read-only inside `Users/`, `Logs/`, `.tmp/` ([tool_registry.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F141jdEqzEt2R5qmqUzc0tfe6iAyhTwfU-%2Fview%3Fusp%3Ddrivesdk)). | Local loopback only (`127.0.0.1`). External P2P messaging disabled; `.smp` mail import is manual-only. | 
| **Tier 1: Probationary (< 72h)** | Password-protected & unlocked, `0 <= age < 72h` | Read-only inside `Users/` & `Logs/`; IDE Studio restricted to `test_syntax_sandbox` ([self_healing_ide.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FajoKV3Si0SjUaNxBIiO6JKKy0ObMl6G%2Fview%3Fusp%3Ddrivesdk)) with zero live file writes. | Can pair with a friend and send/receive text P2P chats & `.smp` mail via Level 2, but **every** inbound packet stays in Quarantine until manually approved. Remote subagent requests blocked. | 
| **Tier 2: Veteran Unlocked (>= 72h)** | Password-protected, unlocked, **`age >= 72.0h`** | **Expanded Computer Access Unlocked**: Full workspace read access + `SerenityDev` MCP write/edit tools (`insert_edit_into_file`, `replace_string_in_file`) + `run_command` (governed by `COMMAND_RULES` + `.bak` staging gate). | **Full P2P & Subagent Autonomy**: Auto-admits low-risk (`risk_score < 0.4`) mail from verified friends, unlocks **Inter-Serenity Subagent Delegation** (`SerenityPC` ↔ `SerenityDev`), and allows optional Lvl 2 Agent-to-Agent queries between paired veteran peers. | 

*(Security Invariant: Even at `>= 72h`, `_is_sandboxed_path` in [tool_registry.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F141jdEqzEt2R5qmqUzc0tfe6iAyhTwfU-%2Fview%3Fusp%3Ddrivesdk) must continue hard-blocking `.env`, `vault_state.json`, `vault.db`, `*.encz`, and `*.key`/`*.pem` from ever being read by a model tool call or transmitted over P2P.)*

## 6. V2.3.0 Subagents / MCP Overhaul & Built-In IDE Studio Design

Here is how to implement each item in your **V2.3.0: Subagents / MCP Overhaul** roadmap ([TODO.txt](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F18poI7V17gJ77Y1-8JhLCcRo_Euc-3mQO%2Fview%3Fusp%3Ddrivesdk)) so it unifies `SerenityPC`, `SerenityDev`, and P2P Messaging:

### A. Porting MCP Support & Connecting `SerenityDev` as a Subagent

* **New Module**: `SerenityPC/System/mcp_bridge.py`

  * Embeds a lightweight JSON-RPC 2.0 MCP client & server router (ported from `/mcp` and `/subscriptions/listen` in [serenitydevserver.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FJ16a2vyKsXBiI2Pu97ameFZ3pLLNlE-%2Fview%3Fusp%3Ddrivesdk)).

  * **Local Loopback Mode (`SerenityPC` → `SerenityDev`)**: When `SerenityPC` detects a coding, refactoring, or PC automation task via `IntentProfiler` ([orchestration_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1oXSfCetVyM6-pFMpDw3hlHTIRtolDVfB%2Fview%3Fusp%3Ddrivesdk)), it checks if `serenitydevserver.py` (`localhost:8002/mcp`) is online (or auto-spawns it headless using `creationflags=subprocess.CREATE_NO_WINDOW`).

  * **VRAM-Safe OTL Handshake (`Offload-then-Load`)**:

    * Both `SerenityPC` (`SerializedAgentContext` in [orchestration_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1oXSfCetVyM6-pFMpDw3hlHTIRtolDVfB%2Fview%3Fusp%3Ddrivesdk)) and `SerenityDev` (`ContextSnapshot` in [SERENITY.md](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1PxQiNxM5KIMfw1PDoeJ8nUYKuRZoQkcQ%2Fview%3Fusp%3Ddrivesdk)) already use OTL!

    * Before `SerenityPC` dispatches a heavy coding task to `SerenityDev` (`W1` Reasoning or `W2` CodeGemma-7B), `SerenityPC` calls `model_swap` / `/offload` to free VRAM (unless both share the exact same loaded `llama-server` port `8081` / `8002` instance—which you can unify by pointing both `SerenityPC` and `SerenityDev` at a shared `LlamaServerClient` runner manager from [runner_manager.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1Z4ba3MZOYE0gzYLkch8P7RQOGAbivkVn%2Fview%3Fusp%3Ddrivesdk)).

### B. Built-In IDE Studio inside `SerenityPC`

* Rather than building a bloated Electron editor inside Tkinter, combine `Project Brainuke`'s `SelfHealingIDE` ([self_healing_ide.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FajoKV3Si0SjUaNxBIiO6JKKy0ObMl6G%2Fview%3Fusp%3Ddrivesdk)) with `SerenityDev`'s surgical edit tools (`insert_edit_into_file`, `replace_string_in_file`, `multi_replace_string_in_file` from [SERENITY.md](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1PxQiNxM5KIMfw1PDoeJ8nUYKuRZoQkcQ%2Fview%3Fusp%3Ddrivesdk)):

  * **UI Surface**: Add an **`[IDE / Code Studio]`** tab/drawer alongside the backend log tabs (`Thoughts`, `Tools`, `Engine`, `Diagnostics`), or pop-out studio window matching your `cecilia_ui.py` preview box.

  * **Capabilities**:

    1. File tree & `grep_search` explorer (`workspace://files` and `workspace://git_status` MCP resources from [serenitydevserver.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FJ16a2vyKsXBiI2Pu97ameFZ3pLLNlE-%2Fview%3Fusp%3Ddrivesdk)).

    2. Side-by-side / unified diff viewer showing proposed edits from `Subagent Lvl 3/5` or `SerenityDev W2`.

    3. Automatic `py_compile` syntax validation (`test_syntax_sandbox` in [self_healing_ide.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FajoKV3Si0SjUaNxBIiO6JKKy0ObMl6G%2Fview%3Fusp%3Ddrivesdk)) + `.bak` backup staging + `[Apply Patch]` / `[Reject]` human confirmation buttons (gated by the 72-hour profile check).

### C. Coding & Tool-Calling Subagent Alignment

* Unify the tool-calling parser between `SerenityPC` ([tool_registry.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F141jdEqzEt2R5qmqUzc0tfe6iAyhTwfU-%2Fview%3Fusp%3Ddrivesdk)) and `SerenityDev` ([serenitydevserver.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1FJ16a2vyKsXBiI2Pu97ameFZ3pLLNlE-%2Fview%3Fusp%3Ddrivesdk)):

  * Port `SerenityDev` v1.6.0's `ast.parse(mode='exec')` multi-line PTC call extractor into `SerenityPC/System/tool_registry.py` so multi-line Python code edits, P2P messages, and mail bodies inside triple quotes (`"""..."""`) never fail regex extraction.

  * Register 6 new MCP/PTC tools in `GemmaToolRegistry` (enabled at `level >= 2`):

    * `send_peer_message(peer_id: str, message: str, attach_context: bool = False)`

    * `send_peer_mail(peer_id: str, subject: str, body: str, attachment_path: str = None)`

    * `check_serenity_inbox(filter_mode: str = "unread")`

    * `share_chat_turns(peer_id: str, turn_count: int = 3, include_thoughts: bool = False)`

    * `delegate_to_serenitydev(mode: str, task: str, target_files: list = None)` *(requires 72h unlock for `/execute`)*

    * `run_pc_automation(action: str, target: str, args: dict = None)` *(requires 72h unlock + human confirmation)*

### D. Streaming Subagent Output to Individual Collapsible Dropdowns

* In `SerenityPC` ([CHANGELOG.md](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F15bKLrVeADIN_F4CLGk-qkTbzOw2CLmWt%2Fview%3Fusp%3Ddrivesdk)), you already solved hierarchical accordion dropdowns for `Deep Cook` (`dc_cycle_{cnum}_{dnum}` and `toggle_cyc_tag` in `main.py`).

* Extend `check_process_queue` in `main.py` with three queue event types:

  1. `{"status": "subagent_spawn", "subagent_id": "sub_lvl2_1", "label": "Subagent Lvl 2 (Gemma-4-E4B) — P2P Courier"}` → inserts a collapsible header button `[+] Subagent Lvl 2 (Gemma-4-E4B) — P2P Courier` with its own Tkinter text tag `sa_block_sub_lvl2_1` (styled in desaturated persona secondary color via `desaturate_color()` from [serenity_resources.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1jdTFs63JgBEv7xLbhate9mIu1wEYEFQO%2Fview%3Fusp%3Ddrivesdk)).

  2. `{"status": "subagent_stream", "subagent_id": "sub_lvl2_1", "channel": "thought" | "output", "chunk": "..."}` → streams tokens live into that specific subagent's tag range (respecting your `Inline`, `Per-Step`, or `Upon Completion` formatting setting in [settings_tabs.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1wyt_HFVRKaHv9xxyEE8XznzfKz5BlWbI%2Fview%3Fusp%3Ddrivesdk)).

  3. `{"status": "subagent_done", "subagent_id": "sub_lvl2_1", "duration_ms": 1420}` → updates the button badge to `[+] ✔ Subagent Lvl 2 (1.4s)` and auto-collapses the block so the main chat stays clean.

## 7. Where to Put Integration Controls in `SerenityPC` Settings

In [settings_tabs.py](https://www.google.com/search?q=https%3A%2F%2Fdrive.google.com%2Ffile%2Fd%2F1wyt_HFVRKaHv9xxyEE8XznzfKz5BlWbI%2Fview%3Fusp%3Ddrivesdk), your 6 tabs are currently:

1. `Models & Params`

2. `Inference & Execution`

3. `Agents & Deep Cook`

4. `Additional Settings`

5. `Users & Security`

6. `Personalize & Theme`

Rather than crowding the top navigation bar with a 7th tab, split the new controls cleanly between **Tab 3 (`Agents & Deep Cook`)** and **Tab 5 (`Users & Security`)**:

| Settings Location | Control Name | Default Value | Options / Behavior | 
| ----- | ----- | ----- | ----- | 
| **Tab 3: Agents & Deep Cook** *(New Section: "Inter-Serenity, MCP & P2P Messaging")* | **Default Messaging & Mail Handler** | `LVL 2: The Helper (Search)` | `Lvl 1 (Speedy)`, **`Lvl 2 (Helper - Default)`**, `Lvl 3 (Collaborator)`, `Lvl 4 (Confidant)`, `Lvl 6 (Serenity Prime)`, `Lvl 7 (Cecilia)` | 
| **Tab 3: Agents & Deep Cook** | **Allow Cecilia / Brainuke P2P Access** | `Quarantine Gate Only` | `Off (Block Lvl 7 P2P)`, **`Quarantine Gate Only (Lvl 2 Triage -> Lvl 7)`**, `Direct Paired Throne Access` | 
| **Tab 3: Agents & Deep Cook** | **SerenityDev MCP Subagent Bridge** | `Auto-Connect (Port 8002)` | `Off`, **`Auto-Connect (Port 8002/8443)`**, `Embedded Studio Only` | 
| **Tab 3: Agents & Deep Cook** | **Subagent Stream Dropdowns** | `Per-Subagent Collapsible` | **`Per-Subagent Collapsible`**, `Combined Agentic Dropdown`, `Logs Only` | 
| **Tab 5: Users & Security** *(Inside "Secure Vault Encryption Panel")* | **72h Veteran Access Status Badge** | *Dynamic Readout* | Displays `● TRUSTED VETERAN (Age: 148.2h — Expanded PC, MCP & P2P Unlocked)` or `◐ PROBATIONARY (Age: 18.4h / 72.0h — Sandboxed)` | 
| **Tab 5: Users & Security** *(New Section: "P2P Contacts & Mail Pairing")* | **Peer Pairing & Address Book** | *Modal Launcher* | `[🔗 Pair Friend / Export RootCA]`, `[📬 Open Serenity Mailbox]`, Peer list (`Jesse [Legacy v1.8.0] — Online/Offline`), and `Background Tray Listener` toggle (when `Offline Mode` in Tab 4 is unchecked). | 
