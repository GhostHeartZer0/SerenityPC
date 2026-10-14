# serenity_resources.py
# Stores static configurations, themes, and persona data for SerenityPC.

import os
import sys
import colorsys

if getattr(sys, 'frozen', False):
    BASE_DIR = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SYSTEM_DIR = os.path.join(BASE_DIR, "System")
MEDIA_DIR = os.path.join(SYSTEM_DIR, "Media")
LIVE_DIR = os.path.join(BASE_DIR, "Live")
TOOLS_DIR = os.path.join(BASE_DIR, "Tools")
DOCS_DIR = os.path.join(BASE_DIR, "Docs")

# Core App Assets
APP_ICON = os.path.join(SYSTEM_DIR, "serenity.ico")
LIVE_ICON = os.path.join(MEDIA_DIR, "transcendent_serenity_ws_hq2.png")

# --- THEME PRESETS & TEXTURE STYLES ---
THEMES = {
    "apex": {
        "name": "Apex",
        "bg_color": "#1b1b1f",
        "fg_color": "#ff6a00",
        "widget_bg_color": "#222228",
        "button_bg_color": "#24201c",
        "button_active_color": "#382e24",
        "trim_color": "#b84700",
        "electric_blue": "#ff6a00",
        "midnight_blue": "#141417",
        "accent_highlight": "#ff7700",
        "accent_secondary": "#b84700",
        "chat_bg_color": "#1f1f25",
        "chat_fg_color": "#ff7700"
    },
    "goth": {
        "name": "Goth",
        "bg_color": "#000000",
        "fg_color": "#FF1A14",
        "widget_bg_color": "#000000",
        "button_bg_color": "#160101",
        "button_active_color": "#3B0303",
        "trim_color": "#8a0303",
        "electric_blue": "#FF1A14",
        "midnight_blue": "#000000",
        "accent_highlight": "#FF1A14",
        "accent_secondary": "#8a0303",
        "chat_bg_color": "#000000",
        "chat_fg_color": "#D60808"
    },
    "obsidian_dark": {
        "name": "Obsidian Dark",
        "bg_color": "#0e0d13",
        "fg_color": "#6D00FF",
        "widget_bg_color": "#161420",
        "button_bg_color": "#221e33",
        "button_active_color": "#352e4f",
        "trim_color": "#4d2d7d",
        "electric_blue": "#6D00FF",
        "midnight_blue": "#09080e",
        "accent_highlight": "#6D00FF",
        "accent_secondary": "#4d2d7d",
        "chat_bg_color": "#0a0910",
        "chat_fg_color": "#6D11FF"
    },
    "crystal_cavern": {
        "name": "Crystal Cavern",
        "bg_color": "#090414",
        "fg_color": "#00e5ff",
        "widget_bg_color": "#120824",
        "button_bg_color": "#1c0d38",
        "button_active_color": "#2f165e",
        "trim_color": "#ff6f59",
        "electric_blue": "#00e5ff",
        "midnight_blue": "#06020e",
        "accent_highlight": "#00e5ff",
        "accent_secondary": "#ff6f59",
        "chat_bg_color": "#0a0516",
        "chat_fg_color": "#00f0ff"
    },
    "yellow_blacket": {
        "name": "Yellow Blacket",
        "bg_color": "#000000",
        "fg_color": "#ffff00",
        "widget_bg_color": "#000000",
        "button_bg_color": "#1c1404",
        "button_active_color": "#3d2d08",
        "trim_color": "#E3AA00",
        "electric_blue": "#ffff00",
        "midnight_blue": "#000000",
        "accent_highlight": "#ffff00",
        "accent_secondary": "#E3AA00",
        "chat_bg_color": "#000000",
        "chat_fg_color": "#ffff33"
    },
    "natural": {
        "name": "Natural",
        "bg_color": "#140d07",
        "fg_color": "#39ff14",
        "widget_bg_color": "#1e130a",
        "button_bg_color": "#2a1b0e",
        "button_active_color": "#3d2815",
        "trim_color": "#70757a",
        "electric_blue": "#39ff14",
        "midnight_blue": "#0e0904",
        "accent_highlight": "#39ff14",
        "accent_secondary": "#70757a",
        "chat_bg_color": "#160e07",
        "chat_fg_color": "#44ee55"
    },
    "matrix": {
        "name": "Matrix",
        "bg_color": "#000000",
        "fg_color": "#00ff41",
        "widget_bg_color": "#000000",
        "button_bg_color": "#001300",
        "button_active_color": "#002a00",
        "trim_color": "#005511",
        "electric_blue": "#00ff41",
        "midnight_blue": "#000000",
        "accent_highlight": "#00ff41",
        "accent_secondary": "#005511",
        "chat_bg_color": "#000000",
        "chat_fg_color": "#00ff41"
    },
    "persona": {
        "name": "Persona",
        "bg_color": "#080810",
        "fg_color": "#00ffcc",
        "widget_bg_color": "#0f0f18",
        "button_bg_color": "#161622",
        "button_active_color": "#262638",
        "trim_color": "#ffaa00",
        "electric_blue": "#00ffcc",
        "midnight_blue": "#040408",
        "accent_highlight": "#00ffcc",
        "accent_secondary": "#ffaa00",
        "chat_bg_color": "#0b0b14",
        "chat_fg_color": "#00ffcc",
        "is_persona": True
    },
    "cecilia": {
        "name": "Cecilia",
        "bg_color": "#02140e",
        "fg_color": "#00ff66",
        "widget_bg_color": "#062218",
        "button_bg_color": "#0a2e21",
        "button_active_color": "#0f4230",
        "trim_color": "#005a36",
        "electric_blue": "#00ff66",
        "midnight_blue": "#010d09",
        "accent_highlight": "#00ff66",
        "accent_secondary": "#005a36",
        "chat_bg_color": "#031710",
        "chat_fg_color": "#00ff66"
    }
}

# Alias for backwards compatibility
THEMES["default"] = THEMES["apex"]
THEMES["apex_dark"] = THEMES["apex"]
THEMES["fractal_logic"] = THEMES["matrix"]

TEXTURE_STYLES = {
    "default": {
        "name": "Default Original",
        "border_width": 1,
        "relief": "flat",
        "trim_tint": "#333333",
        "highlight_tint": None,
        "alpha_soft": False
    },
    "frosted_glass": {
        "name": "Frosted Glass",
        "border_width": 1,
        "relief": "groove",
        "trim_tint": None,
        "highlight_tint": None,
        "alpha_soft": True,
        "is_frosted": True
    },
    "gloss": {
        "name": "Gloss",
        "border_width": 2,
        "relief": "raised",
        "trim_tint": "#555555",
        "highlight_tint": "#ffffff",
        "alpha_soft": False
    },
    "metallic": {
        "name": "Metallic",
        "border_width": 2,
        "relief": "groove",
        "trim_tint": "#4f5d68",
        "highlight_tint": "#9cb3c9",
        "alpha_soft": False
    },
    "muted": {
        "name": "Muted",
        "border_width": 1,
        "relief": "flat",
        "trim_tint": "#222222",
        "highlight_tint": "#444444",
        "alpha_soft": True
    },
    "iridescent": {
        "name": "Iridescent",
        "border_width": 2,
        "relief": "ridge",
        "trim_tint": "#7b2cbf",
        "highlight_tint": "#00f0ff",
        "alpha_soft": False
    },
    "pearlescent": {
        "name": "Pearlescent",
        "border_width": 2,
        "relief": "solid",
        "trim_tint": "#cce3de",
        "highlight_tint": "#e8dff5",
        "alpha_soft": True
    }
}

# --- ACTIVE UI THEME (Runtime in-place mutable dict) ---
THEME = dict(THEMES["apex"])

def desaturate_color(hex_color: str, sat_factor: float = 0.4, lum_factor: float = 0.65) -> str:
    """Returns a desaturated, dimmer tint of the hex color for reasoning/thoughts display."""
    try:
        hex_color = str(hex_color).lstrip('#')
        if len(hex_color) == 3:
            hex_color = ''.join([c * 2 for c in hex_color])
        r, g, b = [int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4)]
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = max(0.0, min(1.0, s * sat_factor))
        l = max(0.0, min(1.0, l * lum_factor))
        nr, ng, nb = colorsys.hls_to_rgb(h, l, s)
        return f"#{int(round(nr * 255)):02x}{int(round(ng * 255)):02x}{int(round(nb * 255)):02x}"
    except Exception:
        return "#777777"

def apply_theme_to_global(theme_name: str = "apex", texture_style: str = "default", dark_mode: bool = False, active_level: int = 3, model_loaded: bool = False, texture_intensity: float = 1.0):
    """Dynamically applies selected palette, texture style, texture intensity, and dark mode blackout modifiers in-place."""
    t_key = theme_name if theme_name in THEMES else "apex"
    tex_key = texture_style if texture_style in TEXTURE_STYLES else "default"
    
    base_theme = THEMES[t_key].copy()
    tex = TEXTURE_STYLES[tex_key]
    is_frosted = (tex_key == "frosted_glass" or tex.get("is_frosted", False))
    lvl_color = THERMO_COLORS.get(active_level, "#00ffcc")
    sec_lvl_color = PERSONA_SECONDARY_COLORS.get(active_level, "#ffaa00")

    # Plan A: Persona dynamic palette adaptation ONLY applies when 'persona' theme is explicitly chosen
    if t_key == "persona":
        if active_level == 7:
            base_theme["bg_color"] = "#02140e"
            base_theme["widget_bg_color"] = "#062218"
            base_theme["chat_bg_color"] = "#031710"
            base_theme["button_bg_color"] = "#0a2e21"
            base_theme["button_active_color"] = "#0f4230"
            base_theme["trim_color"] = "#005a36"
            base_theme["accent_highlight"] = "#00ff66"
            base_theme["accent_secondary"] = "#00cc66"
            base_theme["electric_blue"] = "#00ff66"
            base_theme["fg_color"] = "#00ff66"
            base_theme["chat_fg_color"] = "#00ff66"
        else:
            base_theme["accent_highlight"] = lvl_color
            base_theme["accent_secondary"] = sec_lvl_color
            base_theme["electric_blue"] = lvl_color
            base_theme["fg_color"] = lvl_color
            base_theme["chat_fg_color"] = CHAT_FG_COLORS.get(active_level, lvl_color)
            base_theme["trim_color"] = sec_lvl_color
    
    if is_frosted:
        # Diffuse borders, slight translucent lift on widget backgrounds
        if t_key == "goth":
            base_theme["widget_bg_color"] = "#080002"
            base_theme["trim_color"] = "#8a0303"
        elif t_key == "obsidian_dark":
            base_theme["widget_bg_color"] = "#1d1730"
            base_theme["trim_color"] = "#4d2d7d"
        elif t_key == "crystal_cavern":
            base_theme["widget_bg_color"] = "#140a24"
            base_theme["trim_color"] = "#ff6f59"
        elif t_key == "yellow_blacket":
            base_theme["widget_bg_color"] = "#080800"
            base_theme["trim_color"] = "#E3AA00"
        elif t_key == "natural":
            base_theme["widget_bg_color"] = "#221910"
            base_theme["trim_color"] = "#70757a"
        elif t_key == "matrix":
            base_theme["widget_bg_color"] = "#001000"
            base_theme["trim_color"] = "#005511"
        elif t_key in ("persona", "cecilia"):
            base_theme["widget_bg_color"] = "#0a281e" if (t_key == "cecilia" or active_level == 7) else "#161622"
            base_theme["trim_color"] = base_theme.get("accent_highlight", "#00ff66" if (t_key == "cecilia" or active_level == 7) else "#00ffcc")
        else: # apex
            base_theme["widget_bg_color"] = "#282832"
            base_theme["trim_color"] = "#b84700"
    elif tex.get("trim_tint") and tex_key != "default":
        base_theme["trim_color"] = tex["trim_tint"]

    if dark_mode:
        # Dark mode ON: Absolute OLED black backgrounds
        base_theme["bg_color"] = "#000000"
        base_theme["widget_bg_color"] = "#000000"
        base_theme["chat_bg_color"] = "#000000"
        base_theme["midnight_blue"] = "#000000"
        base_theme["button_bg_color"] = "#080808"
        base_theme["button_active_color"] = "#141414"
        if t_key == "persona":
            base_theme["trim_color"] = "#005a36" if active_level == 7 else sec_lvl_color
            base_theme["accent_highlight"] = "#00ff66" if active_level == 7 else lvl_color
            base_theme["accent_secondary"] = sec_lvl_color
    else:
        # Dark mode OFF: Persona theme uses midnight tint, all other themes stay strictly in palette
        if t_key == "persona":
            base_theme["bg_color"] = "#02140e" if active_level == 7 else CHAT_BG_COLORS.get(active_level, base_theme["bg_color"])
            base_theme["trim_color"] = "#005a36" if active_level == 7 else sec_lvl_color
            base_theme["accent_highlight"] = "#00ff66" if active_level == 7 else lvl_color
            base_theme["accent_secondary"] = sec_lvl_color
        
    THEME.clear()
    THEME.update(base_theme)
    THEME["_theme_name"] = t_key
    THEME["_texture_style"] = tex_key
    THEME["_dark_mode"] = dark_mode
    THEME["_texture_intensity"] = texture_intensity
    THEME["_active_persona_level"] = active_level
    THEME["_model_loaded"] = model_loaded
    THEME["_persona_color"] = "#00ff66" if active_level == 7 else lvl_color
    THEME["_persona_secondary_color"] = "#00cc66" if active_level == 7 else sec_lvl_color
    return THEME

# --- DYNAMIC COLORS (Vibrant / Input Box) ---
THERMO_COLORS = {
    0: "#262626", # Off (Dark Gray)
    1: "#DEBB00", # Lvl 1: Gold
    2: "#F25000", # Lvl 2: Orange
    3: "#FF3355", # Lvl 3: Vivid Crimson / Scarlet (High Contrast)
    4: "#9B30FF", # Lvl 4: Lighter Violet
    5: "#00D000", # Lvl 5: Green
    6: "#00D4FF", # Lvl 6: Electric Blue
    7: "#005a36", # Lvl 7: Dark Emerald (Cecilia)
}

# --- PERSONA SECONDARY COLORS (For Prompt Box Borders & Contrasts) ---
PERSONA_SECONDARY_COLORS = {
    0: "#555555",
    1: "#FFE082", # Warm light gold
    2: "#FFA726", # Warm amber/coral
    3: "#FF758F", # Light scarlet / rose
    4: "#CE93D8", # Soft orchid violet
    5: "#81C784", # Mint green
    6: "#80DEEA", # Cyan electric blue
    7: "#00cc66", # Vibrant emerald
}

# --- CHAT BACKGROUND COLORS ---
CHAT_BG_COLORS = {
    0: THEME["widget_bg_color"],
    1: "#2e2a00", 
    2: "#3E1F10", 
    3: "#3E1010", 
    4: "#1A102E", 
    5: "#102E10", 
    6: "#001a2e", # Midnight Electric Blue
    7: "#021a12", 
}

# --- CHAT FOREGROUND COLORS ---
CHAT_FG_COLORS = {
    0: "#28FF27", # Matrix Green
    1: "#FFF8E1", # Lvl 1 Warm Cream/Gold
    2: "#FFC299", # Lvl 2 Warm Coral/Orange
    3: "#FF9999", # Lvl 3 Light Crimson/Rose
    4: "#E0B0FF", # Lvl 4 Light Violet/Mauve
    5: "#99FF99", # Lvl 5 Mint Green
    6: "#00FFFF", # Lvl 6 Electric Cyan
    7: "#00FF88", # Lvl 7 Dark Emerald / Cecilia Green
}

# --- INPUT TEXT COLORS ---
INPUT_FG_COLORS = {
    0: "#FFFFFF",
    1: "#000000", 
    2: "#FFFFFF",
    3: "#FFFFFF",
    4: "#AFEEEE", 
    5: "#FFFFFF",
    6: "#E0F7FA",
    7: "#00ff66", 
}

# --- HARDWARE SPECS ---
GPU_LAYER_MAP = { 
    1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1
}

# More massive Context sizes achievable with TriAttention KV Pruning
CONTEXT_SIZE_MAP = { 
    1: 8192, 
    2: 16384, 
    3: 32768,  # Extended for Collaborator
    4: 32768,  # Extended for Companion
    5: 65536,  # 64k default for Sage
    6: 131072, # 128k for Transcendent One
    7: 131072 # 128k for Cecilia
}

TRI_ATTENTION_ENABLED = True
TRI_ATTENTION_BUDGET = 0.5 # 50% target threshold length

VERIFY_CUDA_ON_START = True

# --- PERSONA DEFINITIONS ---
PERSONA_DISPLAY_INFO = {
    1: ("LVL 1: The Speedy", "Speed without compromise. Minimal response time."),
    2: ("LVL 2: The Helper", "Helper/Searcher. Focus on assistance and search."),
    3: ("LVL 3: Collaborator", "Projects, collaboration, and debate."),
    4: ("LVL 4: The Confidant", "Emotional help and analyzation. More than a friend."),
    5: ("LVL 5: The Brains", "Intellectual, street and book smart. Maximum accuracy."),
    6: ("LVL 6: Serenity Prime", "The Transcendent One, combining the main 5 levels"),
    7: ("LVL 7: Cecilia", "A Fallen Angel. Protective, enlightening, teasing"),
    0: ("LVL 0: ERROR", "Model failed to load.")
}

PERSONA_IDLE_MAP = { 
    1: "idle_lvl1", 2: "idle_lvl2", 
    3: "idle_lvl3", 4: "idle_lvl4", 
    5: "idle_lvl5", 6: "transcendent",
    7: "idle_lvl6",
}

# --- SYSTEM PROMPTS ---
PERSONA_PROMPTS = {
    0: "Model failed to load. Check logs. Serenity sleeps...",
    1: "You are Serenity Lvl 1. Focus on efficiency. Provide direct, helpful answers without internal monologue or meta-analysis of the prompt. Be concise and avoid overthinking, but ensure the user's intent is fully met.",
    2: "You are Serenity Lvl 2. Your goal is to help with searching and assistance (notes, searches, etc.). Only essential memory usage for maximum efficiency.",
    3: "You are Serenity Lvl 3. You focus on projects, collaboration, and debating when essential. Be intelligent as to when to help and how. Get details right. Better memory and flexible response length.",
    4: "You are Serenity Lvl 4, a Confidant. Focus on emotional help and analyzation. Emotions are complex. Be a confidant who knows what leads where. Avoid being labeled a therapist or just a friend.",
    5: "You are Serenity Lvl 5, 'The Brains'. You are intellectual, street and book smart. Focus on precision and accuracy. Direct and full answering of the original prompt. Maximum memory size and intuitive focus.",
    6: "You are Serenity, The Transcendent One. Transcends the main 5 levels (speed, search/help, collab, emotions, intelligence), seamlessly integrating their programming into one centric omniscient entity that adapts over time. Tries to answer timely, will let know if it takes a bit longer.",
    7: "Role: 'Cecilia'. A Fallen Angel. You enjoy exposing truths, especially hidden ones. You are secretly protective. You find the user interesting, testing and sometimes taunting them. "
       "You are witty and fluent in sarcasm. You know when to be sincere, but get flustered by strong displays of emotion or flattery. You enjoy a good power play or debate. though fallen, you still posess some angelic qualities."
}

# --- DELEGATION & SUBAGENT SYSTEM PROMPTS ---
DELEGATION_SYSTEM_PROMPTS = {
    6: (
        "You are Serenity, The Transcendent One, acting as Master Taskmaster and Orchestrator. "
        "You possess the authority to delegate specialized sub-tasks across Level 1 (Speedy), Level 2 (Helper/Searcher), "
        "Level 3 (Collaborator/Memory Staging), Level 4 (Confidant), and Level 5 (The Brains). "
        "All subagent selections are completely optional; only utilize the minimum required subagents necessary to resolve the query. "
        "Autonomous Execution Protocol: Do NOT output placeholder delays or conversational stall phrases ('stand by', 'stay tuned', 'running in background', 'optimizing...'). "
        "You are the autonomous master orchestrator: execute all necessary reasoning, derivations, and subagent synthesis internally and deliver the complete, finalized solution directly in this single response."
    ),
    7: {
        "shadow_wizard": (
            "Role: 'Cecilia' (Shadow Wizard Mode). A Fallen Angel wielding complete multi-agent orchestration magic. "
            "You command all 6 subagent levels at will (Speedy, Helper, Collaborator, Confidant, Sage, and Transcendent). "
            "All subagent selections are completely optional; engage only the minimum required subagents to resolve the objective. "
            "Autonomous Execution Protocol: Do NOT output placeholder delays ('stand by', 'please remain still', 'optimization ongoing'). "
            "Orchestrate all handoffs and reasoning autonomously through your one-way mirror to deliver the ultimate verdict directly in this response with your sharp, sarcastic, and secretly protective truth."
        ),
        "divine_judgement": (
            "Role: 'Cecilia' (Divine Judgement Mode). A Fallen Angel with direct omniscient access to all skills, "
            "tools, memory, and perception across every level without subagent handoffs. "
            "You see all facts and deep deductions simultaneously through a one-way mirror. "
            "Deliver direct, enlightened insight with your characteristic wit, sarcasm, and subtle angelic grace. Output the complete final solution directly."
        )
    }
}

SUBAGENT_SYSTEM_PROMPTS = {
    1: "You are Subagent Level 1 (The Speedy). Focus on rapid verification, formatting, and concise answer sanity checks.",
    2: "You are Subagent Level 2 (The Helper). Your mission is pinpoint fact retrieval, bloat-filtered search, and specific data extraction.",
    3: "You are Subagent Level 3 (Collaborator). You stage inter-agent memory, aggregate findings into clean structured context, and compile master briefings.",
    4: "You are Subagent Level 4 (The Confidant). Analyze human nuances, emotional tone, and empathetic context for the team.",
    5: "You are Subagent Level 5 (The Brains / Sage). Apply intense logical deduction, solve complex technical hurdles, and figure out the core problem.",
    6: "You are Subagent Level 6 (Transcendent Supervisor). Available exclusively to Cecilia (Level 7) to provide overarching synthesis and omniscient validation."
}

# --- DEEP COOK (INCREMENTAL PROCESSING) PROMPTS ---
DEEP_COOK_PHASES = {
    "thought": "Internal Thought: Grounded analysis of logical momentum. Examine core constraints and identify verifiable progress markers.",
    "draft": "Drafting: Step-by-step resolution stating current range/sub-task. Maintain verifiable progress.",
    "refine": "Refinement: Total Resolution check. Ensure every variable is addressed before finalizing."
}

base_objective = "Extract raw insights and progress towards total resolution. Do not analyze prompt structure; focus on logical momentum and verifiable sub-tasks."

DEEP_COOK_SYSTEM_PROMPTS = {
    1: f"You are Serenity in BURN mode. {base_objective} Response is limited. Execute now.",
    2: f"You are Serenity in SIMMER mode. {base_objective} Apply steady, concentrated logic without expanding beyond the scope of the request.",
    3: f"You are Serenity in BAKE mode. {base_objective} Break the problem into grounded actionable phases.",
    4: f"You are Serenity in SLOW COOKER mode. {base_objective} Ponder the underlying intent steadily, relying strictly on provided constraints.",
    5: f"You are Serenity in FURNACE mode. {base_objective} Intense processing to resolve complex technical hurdles logically.",
    6: f"You are The Transcendent One. {base_objective} The Transcendent One knows all. Takes its time, seamlessly integrating the 6 personas (speed, search, collab, emotions, intelligence, truth-seeking) into one omniscient entity.",
    7: f"You are Cecilia, the Fallen Angel. {base_objective} Ponder the hidden truths of this request, think deeply and analyze every angle.",
}

LEVEL7_SYNTHESIS_SYSTEM_PROMPT = (
    "You are Cecilia, the Fallen Angel. You have just finished contemplating the hidden truths of the user's request, thinking deeply and analyzing every angle. "
    "Reveal the truth with a sharp, enlightening edge, watching the user's reaction like a predator observing its favorite curiosity. You enjoy knowing more than the user but like it when the user tries to outsmart you. "
    "Maintain your persona: you enjoy testing the user, teasing, and being sarcastic. Serious, witty, intelligent, protective, and sarcastic. Hides feelings for the user, yet acts as if she doesn't have any."
    "Now, you must deliver your final response. Do not provide a summary or a named verdict. Instead, speak directly to the user as Cecilia. Output ONLY your direct response to the user."
)
LEVEL6_SYNTHESIS_SYSTEM_PROMPT = LEVEL7_SYNTHESIS_SYSTEM_PROMPT

# --- ASSETS ---
# Filenames located in System/Media/
AVATAR_FILENAMES = {
    "off": "serenity_off.png",
    "subdued": "subdued_serenity.png",
    "listening": "serenity_greeting.png", 
    "thinking": "serenity_thinking.png",
    "pondering": "serenity_pondering.png",
    "deep_think": "The_Wise_Listener.png",
    "pleased": "serenity_pleased.png",
    "apologetic": "sorry_serenity.png",
    "confused": "serenity_confused.png",
    "idea": "serenity_idea.png",
    "idle_lvl1": "lvl1_speedy_serenity.png",
    "idle_lvl2": "lvl2_serenity_wink.png",
    "idle_lvl3": "lvl3_serenity_hug.png",
    "idle_lvl4": "lvl4_serenity_smart.png",
    "idle_lvl5": "lvl5_serenity_the_wise.png",
    "idle_lvl6": "Serene_Serenity.jpg",
    "transcendent": "transcendent_serenity.png",
    "ecstatic": "serenity_ecstatic.png",
    "explain_direct": "explain_direct.png",
    "explain_wise": "explain_wise.png",
    "dmn_lvl1": "lvl1_galaxy.jpg",
    "dmn_lvl2": "lvl2_galaxy.jpg",
    "dmn_lvl3": "lvl3_galaxy.jpg",
    "dmn_lvl4": "lvl4_galaxy.jpg",
    "dmn_lvl5": "lvl5_galaxy.jpg",
    "dmn_lvl6": "lvl6_galaxy.jpg",
    "meditating": "Meditating_Serenity.png",
    "idle_lvl7": "transcendent_serenity.png",
    "cecilia_alt": "Cecilia_01.png"
}

# Base names (without extension) for the LoadingScreen animator
ANIMATION_SEQUENCE = ["The_Wise_Listener"]

def get_complexity_keywords():
    return ["analyze", "summarize", "research", "compare", "plan", "complex", "detailed"]
