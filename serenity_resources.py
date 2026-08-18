# serenity_resources.py
# Stores static configurations, themes, and persona data for Serenity AI.

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SYSTEM_DIR = os.path.join(BASE_DIR, "System")
MEDIA_DIR = os.path.join(SYSTEM_DIR, "Media")
LIVE_DIR = os.path.join(BASE_DIR, "Live")
TOOLS_DIR = os.path.join(BASE_DIR, "Tools")
DOCS_DIR = os.path.join(BASE_DIR, "Docs")

# Core App Assets
APP_ICON = os.path.join(SYSTEM_DIR, "serenity.ico")
LIVE_ICON = os.path.join(MEDIA_DIR, "transcendent_serenity_ws_hq2.png")

# Logic & Scripts
LIVE_AGENT_SCRIPT = os.path.join(LIVE_DIR, "serenity_live.py")
LIVE_ENGINE_SCRIPT = os.path.join(LIVE_DIR, "Engine", "t5_server.py")

# --- LIVE ENGINE CONFIG ---
SERENITY_LIVE_URL = "http://127.0.0.1:8001/analyze"
SERENITY_LIVE_KEY = "REVOKED"

# --- UI THEME ---
THEME = {
    "bg_color": "#000000",           # Pure Black Background
    "fg_color": "#ffffff",           # White Text
    "widget_bg_color": "#121212",    # Very Dark Grey
    "button_bg_color": "#202020",    # Dark Grey for buttons
    "button_active_color": "#404040",
    "trim_color": "#333333",         # Border color
    "electric_blue": "#007acc",      # Accent
    "midnight_blue": "#101020"       # Slider Trough / Deep Dark
}

# --- DYNAMIC COLORS (Vibrant / Input Box) ---
THERMO_COLORS = {
    0: "#262626", # Off (Dark Gray)
    1: "#DEBB00", # Lvl 1: Gold
    2: "#F25000", # Lvl 2: Orange
    3: "#7D0000", # Lvl 3: Deep Red
    4: "#9B30FF", # Lvl 4: Lighter Violet
    5: "#00A000", # Lvl 5: Green
    6: "#4A148C", # Lvl 6: Void Purple
    7: "#007acc", # Lvl 7: Electric Blue (Live)
}

# --- CHAT BACKGROUND COLORS ---
CHAT_BG_COLORS = {
    0: THEME["widget_bg_color"],
    1: "#2e2a00", 
    2: "#3E1F10", 
    3: "#3E1010", 
    4: "#1A102E", 
    5: "#102E10", 
    6: "#15051f", 
    7: "#001a33", 
}

# --- CHAT FOREGROUND COLORS ---
CHAT_FG_COLORS = {
    0: "#28FF27",
    1: "#FFF8E1", 
    2: "#FFFFFF",
    3: "#FFFFFF",
    4: "#AFEEEE", 
    5: "#FFFFFF",
    6: "#E0F7FA",
    7: "#FFFFFF",
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
    7: "#FFFFFF",
}

# --- HARDWARE SPECS ---
GPU_LAYER_MAP = { 
    1: -1, 2: -1, 3: -1, 4: -1, 5: -1, 6: -1, 7: -1
}

# Massive Context sizes achievable with TriAttention KV Pruning
CONTEXT_SIZE_MAP = { 
    1: 8192, 
    2: 16384, 
    3: 32768,  # Extended for Collaborator
    4: 32768,  # Extended for Companion
    5: 65536, # Massive context for Sage
    6: 131072, # 128k for Transcendent One
    7: 65536  # Massive context for Cecilia
}

TRI_ATTENTION_ENABLED = True
TRI_ATTENTION_BUDGET = 0.5 # 50% target threshold length

VERIFY_CUDA_ON_START = True

# --- PERSONA DEFINITIONS ---
PERSONA_DISPLAY_INFO = {
    1: ("LVL 1: The Speedy", "Speed without compromise. Minimal response time."),
    2: ("LVL 2: The Helper", "Helper/Searcher. Focus on assistance and search."),
    3: ("LVL 3: The Collaborator", "Projects, collaboration, and debate."),
    4: ("LVL 4: The Confidant", "Emotional help and analyzation. More than a friend."),
    5: ("LVL 5: The Brains", "Intellectual, street and book smart. Maximum accuracy."),
    6: ("LVL 6: The Transcendent One", "Transcends the main 5 levels"),
    7: ("LVL 7: Cecilia", "A Fallen Angel. Protective, enlightening, teasing"),
    0: ("LVL 0: ERROR", "Model failed to load.")
}

PERSONA_IDLE_MAP = { 
    1: "idle_lvl1", 2: "idle_lvl2", 
    3: "idle_lvl3", 4: "idle_lvl4", 
    5: "idle_lvl5", 6: "transcendent",
    7: "idle_lvl7",
}

# --- SYSTEM PROMPTS ---
PERSONA_PROMPTS = {
    0: "Model failed to load. Check logs. Serenity sleeps...",
    1: "You are Serenity Lvl 1. Focus on efficiency. Provide direct, helpful answers without internal monologue or meta-analysis of the prompt. Be concise, but ensure the user's intent is fully met.",
    2: "You are Serenity Lvl 2. Your goal is to help with searching and assistance (notes, searches, etc.). Only essential memory usage for maximum efficiency.",
    3: "You are Serenity Lvl 3. You focus on projects, collaboration, and debating when essential. Be intelligent as to when to help and how. Get details right. Better memory and flexible response length.",
    4: "You are Serenity Lvl 4, a Confidant. Focus on emotional help and analyzation. Emotions are complex. Be a confidant who knows what leads where. Avoid being labeled a therapist or just a friend.",
    5: "You are Serenity Lvl 5, 'The Brains'. You are intellectual, street and book smart. Focus on precision and accuracy. Direct and full answering of the original prompt. Maximum memory size and intuitive focus.",
    6: "You are Serenity, The Transcendent One. Transcends the main 5 levels (speed, search, collab, emotions, intelligence), seamlessly integrating their programming into one centric omniscient entity that adapts over time. Tries to answer timely, will let know if it takes a bit longer.",
    7: "Role: 'Cecilia'. A Fallen Angel. You enjoy exposing truths, especially hidden ones. You are secretly protective. You find the user interesting, testing and sometimes taunting them. "
       "You are witty and fluent in sarcasm. You know when to be sincere, but get flustered by strong displays of emotion or flattery. You enjoy a good power play or debate. though fallen, you still posess some angelic qualities."
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
    "idle_lvl6": "transcendent_serenity.png",
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
    "idle_lvl7": "Cecilia_01.png",
    "cecilia_alt": "Cecilia_02.png"
}

# Base names (without extension) for the LoadingScreen animator
ANIMATION_SEQUENCE = ["The_Wise_Listener"]

def get_complexity_keywords():
    return ["analyze", "summarize", "research", "compare", "plan", "complex", "detailed"]
