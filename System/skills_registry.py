"""
Shared Skill Registry, Skill Tree Info Popup, and Tooltip-Style '/' Slash Popup
for both SerenityPC (main.py) and Training Station (station_ui.py).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional
import tkinter as tk
from tkinter import ttk


@dataclass
class Skill:
    name: str
    slash: str
    category: str
    description: str
    usage: str
    scope: str = "both"  # "both", "main", "station"
    parent: Optional[str] = None
    automatable: bool = True


class SkillRegistry:
    """Central registry of SerenityPC & Training Station skills with tree hierarchy."""

    def __init__(self) -> None:
        self._skills: Dict[str, Skill] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        defaults = [
            # Skill Tree: Training Station -> Test Banks & Overlays
            Skill("testbank", "/testbank", "Training Station", "Generate custom Wringer test bank (NL or ZIM/DB)", "/testbank <prompt or name>", scope="both", automatable=True),
            Skill("overlay", "/overlay", "Training Station", "Create Wringer alt-level / 6,7 expansion overlay", "/overlay <levels e.g. lvl6,lvl7>", scope="both", parent="testbank", automatable=True),
            Skill("suggest", "/suggest", "Training Station", "Auto-suggest test bank structures from outlined DBs", "/suggest", scope="station", parent="testbank", automatable=True),
            Skill("harvest", "/harvest", "Training Station", "Harvest ZIM, Game Lore, DMN, or Antigravity data", "/harvest <source>", scope="station", automatable=True),
            # Skill Tree: Wringer & Monitoring
            Skill("wringer", "/wringer", "Wringer", "Run Wringer benchmark on active bank + optional overlay", "/wringer [lvl1,lvl2]", scope="both", automatable=True),
            Skill("monitor", "/monitor", "Automation & Monitor", "Agentic monitor status / toggle automated watches", "/monitor [status|start|stop]", scope="both", parent="wringer", automatable=True),
            # Skill Tree: Main Chat & Meta
            Skill("skills", "/skills", "System", "Open interactive Skill Tree info popup", "/skills", scope="both", automatable=True),
            Skill("tree", "/tree", "System", "Alias for /skills — open Skill Tree info popup", "/tree", scope="both", parent="skills", automatable=True),
            Skill("status", "/status", "System", "Print system / station status summary", "/status", scope="both", automatable=True),
            Skill("deepcook", "/deepcook", "Main Chat", "Toggle or run Deep Cook multi-cycle synthesis", "/deepcook <prompt>", scope="main", automatable=True),
            Skill("search", "/search", "Main Chat", "Run offline/web search lookup", "/search <query>", scope="main", automatable=True),
            Skill("image", "/image", "Main Chat", "Generate image via Media Synthesis Studio", "/image <prompt>", scope="main", automatable=True),
            Skill("video", "/video", "Main Chat", "Generate video/GIF via Media Synthesis Studio", "/video <prompt>", scope="main", parent="image", automatable=True),
            Skill("game", "/game", "Main Chat", "Hot-swap Gamerenity model (auto or genshin/nms/div2)", "/game [auto|genshin|nms|div2]", scope="main", automatable=True),
            Skill("clear", "/clear", "Main Chat", "Clear current chat history", "/clear", scope="main", automatable=True),
        ]
        for s in defaults:
            self.register(s)

    def register(self, skill: Skill) -> None:
        self._skills[skill.name] = skill

    def list_skills(self, scope: Optional[str] = None, prefix: str = "") -> List[Skill]:
        p = prefix.strip().lower()
        out = []
        for s in self._skills.values():
            if scope and s.scope not in ("both", scope):
                continue
            if p and not (s.slash.lower().startswith(p) or s.name.lower().startswith(p.lstrip("/"))):
                continue
            out.append(s)
        return out

    def get_tree(self, scope: Optional[str] = None) -> Dict[str, List[Skill]]:
        grouped: Dict[str, List[Skill]] = {}
        for s in self.list_skills(scope=scope):
            grouped.setdefault(s.category, []).append(s)
        return grouped


DEFAULT_SKILL_REGISTRY = SkillRegistry()


class SlashSkillPopup:
    """
    Tooltip-style floating popup triggered when '/' is typed in a tk.Text or tk.Entry.
    Supports Up/Down/Tab/Enter/Escape and mouse click selection.
    """

    def __init__(
        self,
        widget: tk.Widget,
        scope: str = "both",
        registry: Optional[SkillRegistry] = None,
        on_select: Optional[Callable[[Skill], None]] = None,
    ) -> None:
        self.widget = widget
        self.scope = scope
        self.registry = registry or DEFAULT_SKILL_REGISTRY
        self.on_select = on_select
        self.popup: Optional[tk.Toplevel] = None
        self.listbox: Optional[tk.Listbox] = None
        self._matches: List[Skill] = []

        self.widget.bind("<KeyRelease>", self._on_key_release, add="+")
        self.widget.bind("<FocusOut>", lambda e: self.widget.after(150, self._hide_if_unfocused), add="+")
        self.widget.bind("<Escape>", lambda e: self.hide(), add="+")

    def is_visible(self) -> bool:
        return bool(self.popup and self.popup.winfo_exists())

    def _get_text_before_cursor(self) -> str:
        if isinstance(self.widget, tk.Text):
            return self.widget.get("insert linestart", "insert")
        return self.widget.get()

    def _extract_slash_token(self) -> Optional[str]:
        line = self._get_text_before_cursor()
        stripped = line.lstrip()
        if not stripped.startswith("/"):
            return None
        if " " in stripped:
            return None
        return stripped

    def _on_key_release(self, event: tk.Event) -> None:
        if event.keysym in ("Up", "Down", "Return", "KP_Enter", "Tab", "Escape"):
            return
        token = self._extract_slash_token()
        if token is None:
            self.hide()
            return
        matches = self.registry.list_skills(scope=self.scope, prefix=token)
        if not matches:
            self.hide()
            return
        self._show(matches)

    def handle_key(self, event: tk.Event) -> bool:
        """Call from widget KeyPress handler before default Return/Up/Down logic. Returns True if consumed."""
        if not self.is_visible() or not self.listbox:
            return False
        if event.keysym == "Down":
            idx = self.listbox.curselection()
            nxt = (idx[0] + 1) % len(self._matches) if idx else 0
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(nxt)
            self.listbox.see(nxt)
            return True
        if event.keysym == "Up":
            idx = self.listbox.curselection()
            prv = (idx[0] - 1) % len(self._matches) if idx else len(self._matches) - 1
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(prv)
            self.listbox.see(prv)
            return True
        if event.keysym in ("Tab", "Return", "KP_Enter"):
            self._commit_selection()
            return True
        if event.keysym == "Escape":
            self.hide()
            return True
        return False

    def _show(self, matches: List[Skill]) -> None:
        self._matches = matches
        if not self.is_visible():
            self.popup = tk.Toplevel(self.widget)
            self.popup.wm_overrideredirect(True)
            self.popup.configure(bg="#1e293b", padx=1, pady=1)
            self.listbox = tk.Listbox(
                self.popup,
                bg="#0f172a",
                fg="#e2e8f0",
                selectbackground="#0891b2",
                selectforeground="#ffffff",
                font=("Consolas", 9),
                borderwidth=0,
                highlightthickness=0,
                width=68,
                height=min(8, len(matches)),
            )
            self.listbox.pack(fill="both", expand=True)
            self.listbox.bind("<ButtonRelease-1>", lambda e: self._commit_selection())
        else:
            self.listbox.configure(height=min(8, len(matches)))

        self.listbox.delete(0, "end")
        for s in matches:
            prefix = "  └─ " if s.parent else ""
            self.listbox.insert("end", f"{prefix}{s.slash:<12} [{s.category}] {s.description}")
        self.listbox.selection_set(0)

        # Position near cursor/widget
        x = self.widget.winfo_rootx() + 6
        y = self.widget.winfo_rooty() - min(8, len(matches)) * 20 - 12
        if y < 10:
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self.popup.wm_geometry(f"+{x}+{y}")

    def _commit_selection(self) -> None:
        if not self.listbox or not self._matches:
            self.hide()
            return
        sel = self.listbox.curselection()
        idx = sel[0] if sel else 0
        skill = self._matches[idx]
        if isinstance(self.widget, tk.Text):
            self.widget.delete("insert linestart", "insert")
            self.widget.insert("insert", f"{skill.slash} ")
        else:
            self.widget.delete(0, "end")
            self.widget.insert(0, f"{skill.slash} ")
        self.hide()
        if self.on_select:
            self.on_select(skill)

    def _hide_if_unfocused(self) -> None:
        if not self.is_visible():
            return
        try:
            focus = self.widget.focus_get()
            if focus not in (self.widget, self.listbox):
                self.hide()
        except Exception:
            self.hide()

    def hide(self) -> None:
        if self.popup and self.popup.winfo_exists():
            self.popup.destroy()
        self.popup = None
        self.listbox = None


def show_skill_tree_popup(
    parent: tk.Widget,
    scope: Optional[str] = None,
    registry: Optional[SkillRegistry] = None,
    on_insert: Optional[Callable[[str], None]] = None,
) -> tk.Toplevel:
    """Displays a compact dark-themed Skill Tree info popup."""
    reg = registry or DEFAULT_SKILL_REGISTRY
    win = tk.Toplevel(parent)
    win.title("SerenityPC — Skill Tree")
    win.geometry("700x430")
    win.configure(bg="#12141a")
    win.transient(parent.winfo_toplevel())

    hdr = tk.Frame(win, bg="#1b1f29", padx=10, pady=8)
    hdr.pack(fill="x")
    tk.Label(hdr, text="SKILL TREE & SLASH COMMANDS", bg="#1b1f29", fg="#00e5ff", font=("Segoe UI", 11, "bold")).pack(side="left")
    tk.Label(hdr, text="Type '/' in prompt box for live popup", bg="#1b1f29", fg="#94a3b8", font=("Segoe UI", 9)).pack(side="right")

    cols = ("slash", "usage", "description")
    tree_wrap = tk.Frame(win, bg="#12141a")
    tree_wrap.pack(fill="both", expand=True, padx=8, pady=8)
    tree = ttk.Treeview(tree_wrap, columns=cols, show="tree headings", height=14)
    tree_vbar = ttk.Scrollbar(tree_wrap, orient="vertical", command=tree.yview)
    tree.configure(yscrollcommand=tree_vbar.set)
    tree.heading("#0", text="Category / Skill")
    tree.heading("slash", text="Command")
    tree.heading("usage", text="Usage")
    tree.heading("description", text="Description")
    tree.column("#0", width=165)
    tree.column("slash", width=95)
    tree.column("usage", width=175)
    tree.column("description", width=240)
    tree.pack(side="left", fill="both", expand=True)
    tree_vbar.pack(side="right", fill="y")

    grouped = reg.get_tree(scope=scope)
    skill_nodes: Dict[str, str] = {}
    for cat, skills in grouped.items():
        cat_id = tree.insert("", "end", text=cat, open=True)
        # Insert roots first, then children
        roots = [s for s in skills if not s.parent]
        children = [s for s in skills if s.parent]
        for s in roots:
            nid = tree.insert(cat_id, "end", text=s.name, values=(s.slash, s.usage, s.description), open=True)
            skill_nodes[s.name] = nid
        for s in children:
            parent_nid = skill_nodes.get(s.parent, cat_id)
            nid = tree.insert(parent_nid, "end", text=f"└─ {s.name}", values=(s.slash, s.usage, s.description), open=True)
            skill_nodes[s.name] = nid

    def _insert_selected(_evt=None):
        sel = tree.selection()
        if not sel:
            return
        vals = tree.item(sel[0], "values")
        if vals and on_insert:
            on_insert(f"{vals[0]} ")
            win.destroy()

    tree.bind("<Double-1>", _insert_selected)

    bar = tk.Frame(win, bg="#12141a", padx=8, pady=6)
    bar.pack(fill="x", pady=(0, 6))
    if on_insert:
        ttk.Button(bar, text="Insert Selected Skill", command=_insert_selected).pack(side="left")
    ttk.Button(bar, text="Close", command=win.destroy).pack(side="right")
    return win
