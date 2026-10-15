from transformers import pipeline

pipe = pipeline("image-text-to-text", model="unsloth/gemma-3n-E4B-it-GGUF")
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is on the candy?"}
        ]
    },
]
pipe(text=messages)

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("unsloth/gemma-3n-E4B-it-GGUF", torch_dtype="auto")