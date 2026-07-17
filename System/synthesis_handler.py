import os

def generate_master_summary(file_list, llm, persona_prompt=""):
    """
    Takes a list of segment text files and compiles them into a Master Overview.
    Designed for the 27B model (The Sage) or Qwen 9B in Deep Cook mode.
    """
    full_context = ""
    for file_path in file_list:
        with open(file_path, 'r', encoding='utf-8') as f:
            # We add a header for the model to know which part it's reading
            full_context += f"\n--- DATA SEGMENT: {os.path.basename(file_path)} ---\n"
            full_context += f.read()

    system_prompt = (
        f"{persona_prompt}\n\n"
        "You are a Grandmaster Theorycrafter reviewing chronological combat logs. "
        "Review these chronological combat segments. Identify DPR trends, recurring mechanical flaws (failed dodges/rotations), and provide a final Roster Verdict."
    )

    # Use create_chat_completion for llama-cpp-python compatibility
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the analysis segments:\n{full_context}"}
        ],
        temperature=0.7,
        max_tokens=4096
    )
    
    master_summary = response["choices"][0]["message"]["content"]
    
    # Save the final boss report
    # Use the first file to get the original base name
    original_base = file_list[0].split('_Part')[0]
    output_name = f"{original_base}_MASTER_OVERVIEW.txt"
    
    with open(output_name, 'w', encoding='utf-8') as f:
        f.write(master_summary)
    
    return master_summary