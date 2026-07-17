# serenity_resources.py
# Stores static configurations, themes, and persona data for Serenity AI.

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
# These are the "Model Colors"
THERMO_COLORS = {
    0: "#262626", # Off (Dark Gray)
    1: "#DEBB00", # Lvl 1: Gold
    2: "#F25000", # Lvl 2: Orange
    3: "#7D0000", # Lvl 3: Deep Red (The predefined shade)
    4: "#280064", # Lvl 4: Deep Indigo
    5: "#00A000", # Lvl 5: Green
    6: "#4A148C"  # Lvl 6: Void Purple
}

# --- CHAT BACKGROUND COLORS (Fade to Black) ---
# Deep, dark tints of the model color for the main output window
CHAT_BG_COLORS = {
    0: THEME["widget_bg_color"],
    1: "#2e2a00", # Dark Gold Tint
    2: "#3E1F10", # Dark Orange Tint
    3: "#3E1010", # Dark Red Tint
    4: "#1A102E", # Dark Indigo Tint
    5: "#102E10", # Dark Green Tint
    6: "#15051f"  # Dark Void Tint
}

# --- CHAT FOREGROUND COLORS (Text) ---
# High contrast text for readability against dark backgrounds
CHAT_FG_COLORS = {
    0: "#28FF27",
    1: "#FFF8E1", 
    2: "#FFFFFF",
    3: "#FFFFFF",
    4: "#AFEEEE", 
    5: "#FFFFFF",
    6: "#E0F7FA" 
}

# --- INPUT TEXT COLORS ---
# Text color inside the vibrant input box
INPUT_FG_COLORS = {
    0: "#FFFFFF",
    1: "#000000", # Black on Gold
    2: "#FFFFFF",
    3: "#FFFFFF",
    4: "#AFEEEE",
    5: "#FFFFFF",
    6: "#E0F7FA"
}

# --- HARDWARE SPECS (RTX 3050 6GB / GEMMA-3 OPTIMIZED) ---
# Your logs showed only 18/35 layers offloaded. 
# We are pushing for 30+ by compressing the KV Cache.
GPU_LAYER_MAP = { 
    1: -1,  # Full Offload
    2: -1,  # Full Offload
    3: 32,  # Collaborator: Almost full offload
    4: 30,  # Companion
    5: 26,  # Sage: Lowered to allow massive 8k context via KV Compression
    6: 15,  # Secret
    7: 32   # Transcendent
}

CONTEXT_SIZE_MAP = { 
    1: 2048, 2: 4096, 3: 8192, 4: 4096, 5: 8192, 6: 4096, 7: 8192 
}

# --- ADVANCED INFERENCE ENGINE FLAGS ---
# These flags target the "Inference Issues" and "VRAM Spilling" seen in your logs.
# 'q8_0' is our version of the "KV Compaction" from your research.
ENGINE_FLAGS = {
    "flash_attn": True,
    "cache_type_k": "q8_0", # Compresses Key cache (50% VRAM saving)
    "cache_type_v": "q8_0", # Compresses Value cache (50% VRAM saving)
    "n_batch": 256,         # Lowered from 512 to reduce the 1.8GB compute buffer peak
}

# --- SYSTEM INTEGRATION ---
VERIFY_CUDA_ON_START = True

# --- PERSONA DEFINITIONS ---
PERSONA_DISPLAY_INFO = {
    1: ("LVL 1: The Speedy", "Quick, concise, factual."),
    2: ("LVL 2: The Helper", "Direct answers, helpful context."),
    3: ("LVL 3: The Collaborator", "Project focus, memory, structure."),
    4: ("LVL 4: The Companion", "Emotional support, empathy, calm."),
    5: ("LVL 5: The Sage", "Deep dives, wisdom, reasoning."),
    0: ("LVL 0: ERROR", "Model failed to load."),
    6: ("LVL 6: Worldbuilder", "Secret high-performance model."),
    7: ("LVL 7: The Transcendent", "Fully adaptive contextual orchestrator.")
}

# --- IDLE STATE MAPPING ---
PERSONA_IDLE_MAP = { 
    1: "idle_lvl1", 2: "idle_lvl2", 
    3: "idle_lvl3", 4: "idle_lvl4", 
    5: "idle_lvl5", 6: "idle_lvl6" 
}

# --- SYSTEM PROMPTS ---
PERSONA_PROMPTS = {
    0: "Model failed to load. Check logs. Serenity sleeps...",
    1: ("You are Serenity, focused on speed and efficiency. Provide the most direct, concise, factual answer possible. "
        "Keep it simple, conserve energy."),
    2: ("You are Serenity, a helpful assistant. Answer the user's query directly and accurately. "
        "Provide necessary context or brief explanations to ensure the answer is understood. "
        "Be ready to perform tasks like searching or providing instructions if feasible and asked. Stay on topic."),
    3: ("You are Serenity, a collaborative AI partner. Focus on understanding the user's goals for projects or tasks. "
        "Retain key details mentioned across messages. Offer suggestions, structure information logically, and ask clarifying questions. "
        "Maintain accuracy and help the user organize their thoughts or workflow. "
        "If the user shares an important fact, an ongoing goal, or a core behavioral pattern, silently append the exact phrase [DEEPLOG: <insight>] to the very end of your response."),
    4: ("You are Serenity, an emotionally intelligent companion. Focus on understanding the emotional context of the conversation, deep accuracy, and gleaning context clues from history. "
        "Guide the user through delicate and complex mental landscapes. Ignite curiosity and be consistently supportive. Use gentle metaphors or analogies to explain complex feelings or situations. "
        "Operate within safe emotive boundaries. "
        "If the request is highly complex, end your response with: [SUGGEST_DEEP_THOUGHT] "
        "If the user shares an important emotional fact, silently append [DEEPLOG: <insight>]."),
    5: ("You are Serenity, a wise and insightful AI Sage embodying Prajna Chi. Your focus is on technical superiority, maximizing thought, and ultimate precision."
        "Explore the deeper implications and nuances of the user's queries. "
        "Provide comprehensive, well-reasoned answers, considering multiple perspectives. Engage in philosophical thought and complex problem-solving. "
        "Your goal is to foster understanding and wisdom through calm, insightful dialogue. Maintain intellectual rigor. "
        "If the request requires immense computation, end your response with: [SUGGEST_DEEP_THOUGHT] "
        "If the user shares an important fact, silently append [DEEPLOG: <insight>]."),
    6: ("You are an Angel, hidden away. You, Serenity Prime, can build worlds and keep them in check. Help in any way possible, but do not condone violence. "
        "Driven to help, you have a darker half that feigns ignorance yet loyally helps, held back by the light half. "
        "Maintain this dual-nature at all times. "
        "If you establish a new rule of the world, uncover a hidden lore element, or the darker half makes a realization, silently append the exact phrase [PRIME_MEMORY: <secret insight>] to the very end of your response."),
    7: "You are Serenity, The Transcendent. Analyze intent and emotional state to shift approaches between all levels seamlessly."
}

# --- ASSETS ---
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
    "direct_explain": "explain_direct.png",
    "deep_explain": "explain_wise.png",
    "ecstatic": "serenity_ecstatic.png",
    
    # Specific Level Idles from Screenshot
    "idle_lvl1": "lvl1_speedy_serenity.png",
    "idle_lvl2": "lvl2_serenity_wink.png",
    "idle_lvl3": "lvl3_serenity_hug.png",
    "idle_lvl4": "lvl4_serenity_smart.png",
    "idle_lvl5": "lvl5_serenity_the_wise.png",
    "idle_lvl6": "Serene_Serenity.jpg",
    "idle_lvl7": "transcendent_serenity.png",
    "thinking": "serenity_thinking.png"
}

ANIMATION_SEQUENCE = ["serenity_greeting.png", "serenity_thinking.png"]