import sys

with open('C:/Users/ccrg6/SerenityPC/Live/Engine/t5_server.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_content = """@app.post("/stream", dependencies=[Depends(verify_local_key)])
async def stream_log(request: LogRequest):
    signal_ping()
    device = next(model.parameters()).device
    
    if ACTIVE_CORE == "light":
        system_prompt = (
            "You are Serenity. Direct and friendly. "
            "Respond ONLY with a JSON object: {\\"speech\\": \\"...\\"}\\n\\n"
            f"User: {request.text}\\nResponse:"
        )
    elif ACTIVE_CORE == "med":
        system_prompt = (
            "Serenity is a fast, witty AI. Serenity always answers in this JSON format:\\n"
            "{\\"thought\\": \\"I am thinking about x\\", \\"action\\": \\"none\\", \\"speech\\": \\"Hello!\\"}\\n\\n"
            "Rules: Be brief. Do not use brackets in speech. Use valid JSON.\\n\\n"
            f"User: {request.text}\\n"
            "Serenity JSON:"
        )
    else:
        system_prompt = (
            "You are Serenity, an AI assistant. You MUST respond with a valid JSON dictionary.\\n"
            "Format: {\\"thought\\": \\"[Your internal reasoning]\\", \\"action\\": \\"none\\", \\"speech\\": \\"[Your verbal response]\\"}\\n\\n"
            "Task: Respond naturally, but do NOT use the literal bracketed words from the format above.\\n"
            f"User: {request.text}\\nSerenity JSON Response:"
        )
    
    if request.image_b64:
        b64_str: str = str(request.image_b64)
        image_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        multimodal_prompt = f"<img> {system_prompt}"
        inputs = processor(images=image, text=multimodal_prompt, return_tensors="pt").to(device)
        logging.info("[VISION] Processing multimodal payload.")
    else:
        inputs = processor(text=system_prompt, return_tensors="pt").to(device)

    gen_kwargs = {}
    if ACTIVE_CORE == "light":
        gen_kwargs = {"temperature": 0.2, "top_k": 40, "repetition_penalty": 1.1}
    elif ACTIVE_CORE == "heavy":
        gen_kwargs = {"temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.2}
    else: # med
        gen_kwargs = {"temperature": 0.35, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15}
    
    request_dict = request.dict(exclude={"text", "max_tokens", "image_b64"})
    actual_gen_kwargs = {**gen_kwargs}
    for k, v in request_dict.items():
        if v is not None:
            actual_gen_kwargs[k] = v

    streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
    
    generation_kwargs = dict(
        **inputs,
        max_new_tokens=request.max_tokens,
        do_sample=True,
        **actual_gen_kwargs,
        streamer=streamer
    )

    def generate_and_stream():
        with torch.no_grad():
            model.generate(**generation_kwargs)

    threading.Thread(target=generate_and_stream).start()

    def generate():
        for new_text in streamer:
            if new_text:
                yield new_text + "\\n"
    return StreamingResponse(generate(), media_type="text/plain")

@app.post("/analyze", dependencies=[Depends(verify_local_key)])
async def analyze_log(request: LogRequest):
    signal_ping()
    device = next(model.parameters()).device
    
    system_prompt = (
        "You are Serenity, an AI assistant. You MUST respond with a valid JSON dictionary.\\n"
        "Format: {\\"thought\\": \\"...\\", \\"action\\": \\"none\\", \\"speech\\": \\"...\\"}\\n\\n"
        f"User: {request.text}\\nSerenity JSON Response:"
    )
    
    if request.image_b64:
        b64_str: str = str(request.image_b64)
        image_bytes = base64.b64decode(b64_str)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        multimodal_prompt = f"<img> {system_prompt}"
        inputs = processor(images=image, text=multimodal_prompt, return_tensors="pt").to(device)
    else:
        inputs = processor(text=system_prompt, return_tensors="pt").to(device)

    actual_gen_kwargs = {"temperature": 0.35, "top_p": 0.9, "top_k": 50, "repetition_penalty": 1.15}
    request_dict = request.dict(exclude={"text", "max_tokens", "image_b64"})
    for k, v in request_dict.items():
        if v is not None:
            actual_gen_kwargs[k] = v

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=request.max_tokens,
            do_sample=True,
            **actual_gen_kwargs
        )
    
    raw = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    import json as _json, re as _re
    result = raw
    if "Serenity JSON Response:" in result:
        result = result.split("Serenity JSON Response:")[-1].strip()
    
    match = _re.search(r'\\{.*\\}', result, _re.DOTALL)
    if match:
        json_str = match.group(0)
        try:
            parsed = _json.loads(json_str)
            return {"thought": parsed.get("thought", "None"), "action": "none", "speech": parsed.get("speech", result)}
        except _json.JSONDecodeError:
            pass
            
    return {"thought": "No internal thought generated.", "action": "none", "speech": result}
"""

with open('C:/Users/ccrg6/SerenityPC/Live/Engine/t5_server.py', 'w', encoding='utf-8') as f:
    f.writelines(lines[:164])
    f.write(new_content)
    f.writelines(lines[360:])
