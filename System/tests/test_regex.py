import re

text = """<|channel>thought
<channel|>
[Inferred Conclusion]: "Goodnight. Sleep peacefully... I'll be here whenever you wake." or something slightly more poetic to match their energy level. Let's go with a gentle, calm acknowledgement that matches "Serenity".

*   Thought: The user used lowercase and ellipses (".."). This suggests tiredness/intimacy.
    *   Response should reflect this softness.
<channel|>
Goodnight. Sleep peacefully... I'll be here whenever you wake.
"""

patterns = [
    r'(?:<\|channel>)+thought(.*?)(?:<channel\|>|<\|channel>|(?=<start_of_turn>)|$)',
    r'thought\n(.*?)(?:<channel\|>|$)',
    r'<\|think\|>(.*?)(?:<\/\|think\|>|$)',
    r'<thought>(.*?)(?:<\/thought>|$)',
    r'\[DRAFT\](.*?)(?:\[\/DRAFT\]|$)',
    r'(?i)(?:(?:Here\'s a |My )?thinking process(?: that leads to.*?)?|Thinking Process|Thinking Steps|Internal Monologue|Analysis):\s*(.*?)(?=\n\n|\n[A-Z]|$)',
    r'--- Cycle \d+ ---\n(.*?)(?=\n--- Cycle|$)',
    r'(?m)^(?:(?:(?:(?:[a-z]|\d+)(?:\.|\))|[\*\-])\s+\*\*\*?)(?:Analyze|Determine|Identify|Structure|Refine|Draft|Review|Persona|Goal|Context|Acknowledge|Define|Equations|Methodology|Execution|Complexity|Apply|Deconstruct|Develop|Strategy|Resolution|Structure|Tone Check).*?\*\*.*(?:\n|$))+',
    r'(?i)^(?:The user is asking|This request requires|The goal of this|Based on the persona|Analyzing request).*?\n',
    r'(?i)(?:Step \d+:|Phase \d+:|Analysis phase)\s*(.*?)(?=\n\n|\n[A-Z]|$)',
    r'(?i)\[Inferred Conclusion\]:?\s*[^\n]*',
    r'(?m)^\s*\*\s*Thought:?\s*[^\n]*',
    r'(?m)^\s*Thought:?\s*[^\n]*'
]

for p in patterns:
    text = re.sub(p, '', text, flags=re.IGNORECASE | re.DOTALL)

print("--- REMAINING TEXT ---")
print(repr(text.strip()))

