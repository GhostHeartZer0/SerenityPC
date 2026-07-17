### Welcome to SerenityAI, a locally independent smart chat interface! ###
This is SerenityPC, with an android version still in development.
Although most models can call web search, they function perfectly well offline once the setup completes.

#### Quick Start ####
Select the models in settings at the top left.
Move the Persona slider to desired level.
To get started quickly, type a prompt and press enter.
The Begin and Persona buttons Load a model.
Once loaded, the Begin button turns to Offload.
you can either offload and reload to swap personas, or click the Persona button.
Deep cook starts a cycle that may take awhile, but gives a more thought out answer.  
Check settings ; One-shot loads on click, toggle loads the selected Deep Cook model before it starts.

#### System Requirements ####
Required:  
4GB VRAM  
8GB RAM  
a CPU with at least 4 cores.  
At least GB storage.  

Reccomended:  
At least 6GB of VRAM, 12-24GB preferred.  
16GB of RAM, 32 preferred.  
A CPU with at least 8 logical cores  
At Least GB storage  
To decide which models to run, take the total size of the model with estimated cache size against your VRAM for speed and VRAM+RAM for possibility.
Give 1GB VRAM to ensure system stability.
Cache size calculation:
KV cache chosen in settings goes like this:
Q4_0 ia a quarter the full, uncompressed size.
Q8_0 divides it by 8. FP16 is full size.
This chart measures the amount of memory a few various models require, assuming Q4_0:

#### Model Footprints ####
| Model Tier | Variant | Weight Size (FS) | KV Cache (KV) | Total Footprint (FT) |
| :--- | :--- | :--- | :---: | ---: |
| Gemma-4 E2B | Recommended (IQ4) | 2.76 GiB | 56.54 MiB | 2.82 GiB |

| Gemma-4 E4B | Recommended (Q4_K) | 4.74 GiB | 152.44 MiB | 4.89 GiB |

| Gemma-4 26B | Recommended 1 (MXFP4) | 15.47 GiB | 250.31 MiB | 15.72 GiB |

| Gemma-4 26B | Recommended 2 (Q5_K) | 19.75 GiB | 250.31 MiB | 20.00 GiB |

| Gemma-4 31B | Required (Q4_K) | 17.46 GiB | 1,001.25 MiB | 18.46 GiB |

#### Tiers of SerenityPC ####
| <h6> Tier |Required VRAM, RAM | reccomended<br>VRAM, RAM | optimal<br>VRAM, RAM | Suggested<br>Models <h6> |
| :--- | :--- | :--- | :--- | :--- |
| Compact | 4 | 8 | 4 | 12 | 6 | 16 | E2B, sm E4B |
| Small | 4 | 16 | 6 | 24 | 6 | 32 | E2B, E4B, sm 26B  |
| Medium | 4 | 20 | 6 | 32 | 8 | 48 | E2B, E4B, 26B, sm 31B |
| Performance | 6 | 32 | 8 | 48 | 12+ | 48+ | E2B, E4B, 26B, 31B |
^memory (VRAM, RAM) is measured in GB.   

##### Notes #####
If you only get thoughts. Just prompt it to keep going.  
For best results, offload just enough layers to fit in VRAM, aiming for no Shared VRAM usage. 
E2B can run on almost anything.  
E4B models may vary, with 6GB of VRAM reccomended.  
The smallest 26B model just barely runs on 16GB of RAM with a 6GB GPU, I'd steer clear of doing that.  
The 26B models run faster cause only 4B is active at once.  
31B models have higer requirements and run slower, as being dense models, the whole thing is active at once.  

##### LAYER COUNTS #####
Most smaller models can fit in 6GB of VRAM.  
for 26B, 7-9 layers is a good start point. Reduce if you notice a spike in Shared VRAM usage. For 31B, YMMV.  
The higher the B-count (parameters), the smarter the model. 



  How to get started:
First, run setup.py, and for the Live interface, setup_engine.py. you'll find desktop shortcuts for each, and when they finish, you'll get to see if you have GPU support.
Clicking the desktop icons will run their respective interface.
Live is also runnable from SerenityPC herself. If for some reason it doesn't run, click in main.py to launch SerenityPC, and t5_engine for Serenity Live. 

Explaining the Interface:

There are 5 main levels, 7 total.
Lvl 1 is for fast response time and speed.
Lvl 2 focuses on web searching and general help.
Lvl 3 is collaborative, project focused.
Lvl 4 unravels emotional complexity.
Lvl 5 is the smart one.
As for the others...
Lvl 6 is a secret, hidden level. 
Deep Cook uses iterative processing and cycles to think more deeply about a prompt, resulting in a much more accurate response. It takes the previous cycle's findings and periodically weighs it against the main prompt to ensure it stays on track. It is meant as an extension of each of these 6 levels, designed for more deep thought and reasoning at the cost of speed.
Lvl 7 is technically Live mode, visible when launching the Live interface from the main UI. Its meant to learn from each level, creating a smarter, combined interface, as Live otherwise has just the one level. 
Live, instead, has Cores, tied to different models.
Those models are as follows:
Light: t5gemma-2-270m-270m, requiring GB of VRAM.
Med: t5gemma-2-1b-1b, requiring GB of VRAM (or RAM, but that's slower)
Heavy: t5gemma-2-4b-4b, needing GB of VRAM/DRAM to function.
Context size also matters. You can adjust this in settings. With the advancements of TurboQuant (Google DeepMind) and TriAttention (Researchers from MIT, NVIDIA, and Zhejiang University), context takes up far less memory than it used to. 
It is the combined size limit of the prompt, history, and model thoughts that go into the final response. 
In the main SerenityPC UI,
  Settings:
Deep Cook Behavior has two functions:
One-Shot loads the deep cook model for a single prompt, activated by clicking its respective button right next to send.
Toggle Mode changes thaycbutton to turn the Deep Cook model and mode on or off.
Manual VRAM override adjusts Auto-detect, do you don't ha e to mess with layers and context size.
Rich Media Rendering details how images and video are displayed.
Global Overrides:
Global KV Cache relates to the KV Translator Map up top. It details the level of compression that the Key-Value cache goes thru, which doesn't affect the result much. Its mainly used to prevent context bloating your memory and for compatibility purposes.
Video Processing Sub-Chunk Size details how many frames the AI can gobble at once, which ultimately results in the compiled descriptions (temporarily stored in memory), which then get considered for the final response.
Video inferencing is real slow, limited mostly by memory and model size.
A Token can vary in size, from a symbol or syllable to a word.
A parameter (B in A4B) is wssentially a neuron connecting dots in thought during training.the A stands for active. 
E in E4B stands for Effective, meaning it acts larger than it really is.
As for the parameters:
Layers dictates how many llm layers are offloaded to the GPU, to be stored in VRAM. this dictates how fast the model can understand and respond, as VRAM on the GPU is much faster and reccomended for fastest response time.
The larger models (26B, 31B) are slower, as they usually can't fit entirely in VRAM, as they require expensive hardware with up to 24GB VRAM. instead, this is where layer offloading is crucial. Offloading just enough (-1 if all can fit), allows the model to split the layers better, resulting in faster loading times. However, the intelligence and accuracy of these models is much higher, at the cost of speed.
Ctx is context size. 1024 x the number next to k (thousand) provides the value.
4k is 4096, 8k 8192, 16k 16384, 32k 32768, 64k 65536, and so on.
Response tuning:
Batch is the chunk size in tokens sent to the model at once for evaluation. This effects memory during processing.
256, 512, 1024, 1280, 1536, 1792, 2048.
Temp:  increases or decreases the randomness. Low  Factual, precise, consistent (e.g., coding, summaries).High Creative, varied, unpredictable (e.g., brainstorming, poetry). 0, Deterministic (always chooses the highest probability).
Top P: a value between 0-1 meaning percent (0%-100%).
Adjusts the cumulative probability of the batch of tokens, so only the top that percent probability get considered.
Min-P: controls hallucinations, cutting off tokens below a certsin threshold. The higher the number, the wackier it can get. It takes the top token probability and multiplies it by the value, cutting off probability of the tokens below that being used. If the top token is 99% probable and this value is set to 0.1, it curs off tokens below a 9.9% probability. 
Repetition Penalty: stops repeats. -2.0 to 2.0. Higher the number is more restricted.

Frequency Penalty: restricts tokens that appear frequently. -2.0 to 2.0. Higher number is more restriction.
Top-K: restricts the next most likely tokens so only that many are considered.
Begin! Starts the model. (Clicking on the level does the same)
It turns to Offload, which frees up memory.
Live launches the Live interface (use with caution, running two llms at once can clog your memory easily)
Video allows you to load a video, activating the video engine selected in settings. 
Set Vision in settings selects the vision model (not all support this)
Projector: make sure to select the respective .mmproj proector file associated with the model so it can actually process the video. This feeds the llm the right data.
BF16 is reccomended for NVIDIA graphics cards, 
F16 is more universally compatible. If you have the wrong version, contact me.
Pulse simulates a DMN (Default Mode Network), allowing Serenity to ponder recent histories, enhancing her understandings. This feeds prime.chronicles.txt.
RGB, if supported, os a miche feature still under development, allowing light sync with its own little UI. It can allow automatic light switching or change the RGBs to the model color. Please not this is an experimental feature that probably doesn't work yet.
Clean clears the selected media from memory (Video)
The plus sign next to the level slider adds inages, audio, or documents to the prompt.
Halt attempts to stop a running model, useful if it gets caught.
The Backend Logs:
Left was originally for thoughts, which became a dropdown right in the UI. it now mainly shows model loading statuses.
The tool icon tab shows tool calls, such as web search.
The warning triangle tab shows the entire model loading process and generation info, minus the super redundant parts.
The right tab is a condensed version of the previous tab, showing choice model loading info and generation info.
t/s is tokens per second. You can see how many layers the model has, as well as the current cache sizes and more.
I believe that prerty much covers it. If you have any ideas, complaints, or notice something glaringly obvious that I should have caught, please reach out to me at ccrg69@gmail.com . I value input!
Thanks for reading, 
-GhostHeartZer0

#### Licenses & Credits ####
- **SerenityPC**: Released under the MIT License ([LICENSE](LICENSE)). Copyright (c) 2026 GhostHeartZer0 / SerenityAI Team.
- **llama.cpp / llama-cpp-python**: MIT License (Georgi Gerganov, Andrei Betlen & contributors).
- **Google Gemma Models (Gemma-4, Gemma 2, T5Gemma)**: Subject to Google Gemma Terms of Use (ai.google.dev/gemma/terms).
- **Qwen Models**: Subject to Qwen License Agreement (Alibaba Cloud).
- **TurboQuant Architecture**: Google DeepMind research team.
- **TriAttention KV Pruning**: MIT, NVIDIA, & Zhejiang University research team.
- **Sentence-Transformers & TurboVec**: Apache 2.0 / MIT.
- **PyTorch, OpenCV & Pillow**: BSD 3-Clause / Apache 2.0 / HPND.
- **MSI Mystic Light SDK**: Copyright (c) Micro-Star International Co., Ltd.