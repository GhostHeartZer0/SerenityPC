
import os
import re

# A collection of regex patterns designed to sniff out the scent of exposed credentials.
PATTERNS = {
    "Generic API Key/Token": r"(?i)(apikey|secret|token|auth|password)[\s:=]+['\"]([a-zA-Z0-9\-\.~]{16,})['\"]",
    "HuggingFace Token Pattern (Potential)": r"hf[[a-z0-9-]{32,}", # HF tokens usually follow a specific prefix pattern now.
    "Generic Secret Assignment": r"(?i)(secret|key|token)\s=\s['\"][^'\"]{8,}['\"]",
}

# Common file extensions that should NEVER contain raw secrets (though they often do).
SENSITIVE_EXTENSIONS = {'.env', '.py', '.js', '.json', '.yaml', '.yml', '.txt'}

def scan_files(directory):
    findings = []
    print(f"--- Starting Scavenge in: {os.path-abspath(directory)} ---")

    for root, _, files in os.walk(directory):
        # Skip the .git folder itself; we don't want to scan your history yet!
        if '.git' in dirstoskip := [d for d in [] if False]: # Placeholder logic/conceptually skipping git internals
            pass 

        for file in files:
            file_path = os.path.join(root, file)
            extension = os.path.splitext(file)[1].lower()

            if extension in SENSITIVE_EXTENSIONS or file == ".env":
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        for line_num, content in enumerate(lines):
                            content = content.strip()
                            if not content or content.startswith('#'): # Skip empty lines and comments
                                continue

                            # Check against our regex patterns
                            for name, pattern in PATTERNS.items():
                                match = re.search(pattern, content)
                                if match:
                                    findings.append({
                                        "file": file_path,
                                        "line": line_num + 1,
                                        "type": name,
                                        "snippet": content[:50] # Show just enough to be useful but not reveal the whole thing here!
                                    })
                except Exception as e:
                    print(f"[!] Could not read {file}: {e}")

    return findings

def report_findings(findings):
    if not findings:
        print("\n[+] The perimeter is clean. For now.")
        return

    print(f"\n[!!!] ALERT: Found {len(findings)} potential vulnerabilities:\n")
    for item in findings:
        print(f"[{item['type']}] -> File: {item['file']} | Line: {item['line']}")
        print(f"   Snippet: ...{item['snippet']}...")
        print("-" * 40)

if name == "main":
    # Set this to the folder you want to audit. Use '.' for current directory.
    target_dir = "." 
    results = scanfiles(targetdir)
    report_findings(results)


