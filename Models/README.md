# SerenityPC Model Directory Disclaimer

This folder contains the models used by SerenityPC. Model weight files are excluded from this Git repository to avoid large file size limits.

## Models Layout

To run SerenityPC, you must download the respective model files and place them in the following subdirectories:

### Compact Model (E2B)
Place in `Models/E2B/`:
- `gemma-4-E2B-it-Q8_0-MTP.gguf`
- `gemma-4-E2B-it-qat-UD-Q4_K_XL.gguf`
- `gemma-4-E2B-it-qat-assistant-MTP-Q8_0.gguf`
- `mmproj-BF16.gguf` (Vision Projector)

### Small Model (E4B)
Place in `Models/E4B/`:
- `gemma-4-E4B-it-Q8_0-MTP.gguf`
- `gemma-4-E4B-it-qat-UD-Q4_K_XL.gguf`
- `gemma-4-E4B-it-qat-assistant-MTP-Q8_0.gguf`
- `mmproj-BF16.gguf` (Vision Projector)

Place in `Models/E4B/mobile/`:
- `gemma-4-E4B-it-qat-mobile-UD-Q2_K_XL.gguf`
- `mmproj-BF16.gguf`
- `mmproj-F16.gguf`

### Medium Model (12B)
Place in `Models/12B/`:
- `gemma-4-12B-it-Q8_0-MTP.gguf`
- `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`
- `gemma-4-12B-it-qat-q4_0-unquantized-assistant-Q8_0.gguf`

### MoE Model (26B-A4B)
Place in `Models/26B-A4B/`:
- `gemma-4-26B-A4B-it-Q8_0-MTP.gguf`
- `gemma-4-26B-A4B-it-qat-UD-Q4_K_XL.gguf`
- `mmproj-BF16.gguf` (Vision Projector)

---

## Live Interface Models (T5Gemma)

The Live interface uses safetensors models, which must be placed in their respective `Live/` subdirectories:

- `Live/t5gemma-2-270m-270m/model.safetensors`
- `Live/t5gemma-2-1b-1b/model.safetensors`
- `Live/t5gemma-2-4b-4b/` (safetensors shards: `model-00001-of-00004.safetensors` to `model-00004-of-00004.safetensors`, `model.safetensors.index.json`, etc.)

---

## Download Instructions

1. Obtain model weights from the Hugging Face hub (e.g., from the Google Gemma collections or custom quantized repositories).
2. Download the `.gguf` and `.safetensors` files matching the names above.
3. Move the downloaded files to the designated folders under `Models/` or `Live/` as described.
