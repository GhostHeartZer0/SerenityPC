#!/usr/bin/env python3
"""
History Persona & Prefix Repair Utility for SerenityPC
Scans, backs up, and heals .history.jsonz and .history.encz archive files.
Corrects persona labels, level grounding, and strips baked-in duplicate prefixes.
"""

import os
import sys
import re
import json
import zlib
import shutil
import argparse
import getpass
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

try:
    from System.vault_manager import VaultManager
except ImportError:
    VaultManager = None


def discover_history_files(search_dirs: List[str]) -> List[str]:
    """Finds all .history.jsonz and .history.encz archives in specified roots."""
    found = []
    for root_dir in search_dirs:
        if not os.path.exists(root_dir):
            continue
        for root, _, files in os.walk(root_dir):
            # Skip backups directories
            if "backups" in os.path.split(root):
                continue
            for f in files:
                if f.endswith(".history.jsonz") or f.endswith(".history.encz"):
                    found.append(os.path.join(root, f))
    return sorted(found)


def extract_pipeline_levels(filename: str) -> List[int]:
    """Extracts level numbers from history filename (e.g. _lvl3, _lvls1_1)."""
    m = re.search(r"_lvls?([0-9_]+)", filename, re.IGNORECASE)
    if m:
        return [int(x) for x in m.group(1).split("_") if x.isdigit()]
    return []


def clean_baked_in_prefix(content: str, role: str) -> Tuple[str, bool]:
    """Strips accidental baked-in prefixes from raw message text."""
    if not isinstance(content, str):
        return content, False

    orig = content
    cleaned = content

    if role in ("assistant", "ai", "cecilia"):
        # Strip leading Serenity: or Cecilia:
        m = re.match(r"^(?:serenity|cecilia)\s*:\s*", cleaned, re.IGNORECASE)
        if m:
            cleaned = cleaned[m.end():].lstrip()
    elif role == "user":
        # Strip leading You:
        m = re.match(r"^you\s*:\s*", cleaned, re.IGNORECASE)
        if m:
            cleaned = cleaned[m.end():].lstrip()

    return cleaned, (cleaned != orig)


def repair_file_messages(
    messages: List[Dict[str, Any]], 
    filename: str, 
    clean_prefixes: bool = True
) -> Tuple[List[Dict[str, Any]], int, List[str]]:
    """
    Validates and repairs messages for a single file according to its filename level.
    Returns (repaired_messages, changes_count, log_descriptions).
    """
    levels = extract_pipeline_levels(filename)
    is_lvl7 = any(l >= 7 for l in levels)
    target_persona = "Cecilia" if is_lvl7 else "Serenity"
    target_level = levels[0] if levels else (7 if is_lvl7 else 3)

    changes = 0
    logs = []
    repaired = []

    for idx, m in enumerate(messages):
        if not isinstance(m, dict):
            repaired.append(m)
            continue

        entry = dict(m)
        role = str(entry.get("role", "")).lower()

        if role in ("assistant", "ai", "cecilia"):
            # 1. Ground role
            if not is_lvl7 and role == "cecilia":
                entry["role"] = "assistant"
                changes += 1
                logs.append(f"Turn {idx}: role 'cecilia' -> 'assistant'")

            # 2. Ground persona
            curr_persona = entry.get("persona")
            if curr_persona != target_persona:
                entry["persona"] = target_persona
                changes += 1
                logs.append(f"Turn {idx}: persona '{curr_persona}' -> '{target_persona}'")

            # 3. Ground level
            curr_level = entry.get("level")
            level_mismatch = (
                curr_level is None or 
                (not is_lvl7 and curr_level >= 7) or 
                (is_lvl7 and curr_level < 7)
            )
            if level_mismatch:
                entry["level"] = target_level
                changes += 1
                logs.append(f"Turn {idx}: level '{curr_level}' -> '{target_level}'")

            # 4. Strip duplicate baked-in prefixes from assistant text
            if clean_prefixes:
                content = str(entry.get("content", ""))
                new_content, was_cleaned = clean_baked_in_prefix(content, "assistant")
                if was_cleaned:
                    entry["content"] = new_content
                    changes += 1
                    logs.append(f"Turn {idx}: stripped leading persona prefix from content")

        elif role == "user":
            # Strip duplicate baked-in 'You: ' from user text
            if clean_prefixes:
                content = str(entry.get("content", ""))
                new_content, was_cleaned = clean_baked_in_prefix(content, "user")
                if was_cleaned:
                    entry["content"] = new_content
                    changes += 1
                    logs.append(f"Turn {idx}: stripped leading 'You:' prefix from prompt")

        repaired.append(entry)

    return repaired, changes, logs


def run_history_repair(
    target_dirs: Optional[List[str]] = None,
    password: Optional[str] = None,
    dry_run: bool = False,
    clean_prefixes: bool = True
) -> Dict[str, Any]:
    """Runs full scan, backup, and repair process across history archives."""
    if not target_dirs:
        target_dirs = [
            os.path.join(REPO_ROOT, "History"),
            os.path.join(REPO_ROOT, "Users")
        ]

    files = discover_history_files(target_dirs)
    stats = {
        "files_scanned": len(files),
        "files_needing_repair": 0,
        "files_repaired": 0,
        "files_skipped_locked": 0,
        "total_changes": 0,
        "backup_dir": None,
        "errors": []
    }

    if not files:
        print("[REPAIR] No history files found to scan.")
        return stats

    # Initialize VaultManager if .encz files present
    has_encz = any(f.endswith(".encz") for f in files)
    vault_mgr = None
    if has_encz and VaultManager:
        state_dir = os.path.join(REPO_ROOT, "System")
        vault_mgr = VaultManager(history_dir=os.path.join(REPO_ROOT, "History"), state_dir=state_dir)
        if vault_mgr.is_lock_enabled():
            if password:
                unlocked = vault_mgr.unlock(password)
                if not unlocked:
                    print("[WARNING] Invalid vault password provided. Encrypted files will be skipped.")
            else:
                print("[NOTE] Vault is locked. Encrypted (.encz) files will be skipped unless unlocked.")

    # Create backup directory if not dry-run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = os.path.join(REPO_ROOT, "History", "backups", f"repair_backup_{timestamp}")
    if not dry_run:
        os.makedirs(backup_root, exist_ok=True)
        stats["backup_dir"] = backup_root

    for file_path in files:
        fname = os.path.basename(file_path)
        is_enc = file_path.endswith(".encz")

        # Read messages
        messages = None
        try:
            if is_enc:
                if not vault_mgr or vault_mgr.is_locked() or not vault_mgr._session_key:
                    stats["files_skipped_locked"] += 1
                    continue
                messages = vault_mgr.read_history_messages(file_path)
            else:
                with open(file_path, "rb") as fp:
                    decompressed = zlib.decompress(fp.read()).decode("utf-8")
                messages = json.loads(decompressed)
        except Exception as e:
            stats["errors"].append(f"Failed to read {file_path}: {e}")
            continue

        if not isinstance(messages, list):
            continue

        repaired_msgs, change_count, logs = repair_file_messages(messages, fname, clean_prefixes=clean_prefixes)

        if change_count > 0:
            stats["files_needing_repair"] += 1
            stats["total_changes"] += change_count

            if dry_run:
                print(f"[DRY-RUN] {fname} ({change_count} changes):")
                for log in logs[:5]:
                    print(f"   - {log}")
                if len(logs) > 5:
                    print(f"   ... and {len(logs)-5} more")
            else:
                # Backup original file
                try:
                    rel = os.path.relpath(file_path, REPO_ROOT).replace("..", "_")
                    bpath = os.path.join(backup_root, rel)
                    os.makedirs(os.path.dirname(bpath), exist_ok=True)
                    shutil.copy2(file_path, bpath)
                except Exception as b_err:
                    stats["errors"].append(f"Backup failed for {fname}: {b_err}")
                    continue

                # Write repaired file
                try:
                    if is_enc:
                        vault_mgr.write_history_messages(file_path, repaired_msgs)
                    else:
                        compressed = zlib.compress(json.dumps(repaired_msgs).encode("utf-8"))
                        with open(file_path, "wb") as out_fp:
                            out_fp.write(compressed)

                    stats["files_repaired"] += 1
                    print(f"[REPAIRED] {fname} ({change_count} fixes applied)")
                except Exception as w_err:
                    stats["errors"].append(f"Write failed for {fname}: {w_err}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="SerenityPC History File Persona & Prefix Repair Utility")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without modifying files on disk.")
    parser.add_argument("--password", "-p", type=str, default=None, help="Master vault password for .encz files.")
    parser.add_argument("--skip-clean-prefixes", action="store_true", help="Do not strip leading baked-in text prefixes.")
    parser.add_argument("--dirs", nargs="+", default=None, help="Specific directories to scan.")

    args = parser.parse_args()

    print("=" * 60)
    print(" SerenityPC History Archive Persona Repair Tool")
    print("=" * 60)

    password = args.password
    # Interactive password prompt if not provided and vault enabled
    if not password and not args.dry_run and VaultManager:
        state_dir = os.path.join(REPO_ROOT, "System")
        v = VaultManager(history_dir=os.path.join(REPO_ROOT, "History"), state_dir=state_dir)
        if v.is_lock_enabled():
            try:
                entered = getpass.getpass("Enter Vault Master Password (leave empty to skip .encz): ")
                if entered.strip():
                    password = entered.strip()
            except (EOFError, KeyboardInterrupt):
                pass

    stats = run_history_repair(
        target_dirs=args.dirs,
        password=password,
        dry_run=args.dry_run,
        clean_prefixes=not args.skip_clean_prefixes
    )

    print("\n" + "=" * 60)
    print(" Repair Summary")
    print("=" * 60)
    print(f" Files Scanned:         {stats['files_scanned']}")
    print(f" Files Needing Repair:  {stats['files_needing_repair']}")
    print(f" Files Repaired:        {stats['files_repaired']}")
    print(f" Total Changes Applied: {stats['total_changes']}")
    if stats["files_skipped_locked"] > 0:
        print(f" Encrypted (Locked):    {stats['files_skipped_locked']} files skipped (password required)")
    if stats["backup_dir"]:
        print(f" Safe Backup Directory: {stats['backup_dir']}")
    if stats["errors"]:
        print(f" Errors Encountered:    {len(stats['errors'])}")
        for e in stats["errors"][:5]:
            print(f"   [!] {e}")
    print("=" * 60)


if __name__ == "__main__":
    main()
