Welcome to SerenityPC, a purely local chatbot with 5 levels!
The idea here is to have a purely local chatbot that can be both quick and elaborate.
it may not have fully updated info about the world. 
It has a secret hidden uncensored mode.
The models are subject to change, I may make a different, more private and digitally hidden app for specialized use such as erotic rp and companion or math and study, input is welcome. We can accomplish more together!

To begin, click "Load Model" in the top left.
Once the model is loaded, the Persona slider will become available. level 1 is default for speed purposes.
NOTE: AI is hyper-demanding on a system. If a model crashes, please use a lower lvl.
Turbo mode adds 3 GPU layers to the current mode, clicking the model name (eg. The Helper) will unload turbo. the "The" moniker is used as a turbo mode sensor, as is the background colors for the model type. 
Lite mode disables GPU offloading, using only the CPU. GPU usage and performance is dependent on the proper framework and hardware/software capability.

SerenityPC - Chatbot Application Guide
This guide details all the necessary hardware, software, and files required to run the SerenityPC chatbot application. Following these steps will ensure the application runs correctly with full GPU acceleration.

1. Hardware Requirements
GPU (Crucial): An NVIDIA graphics card with CUDA support is required for GPU acceleration. A GPU with at least 4GB of VRAM is recommended to use the persona-level reloading features without instability.

System RAM: A minimum of 16 GB of RAM is recommended. The AI model uses both VRAM and system RAM.

CPU: A modern processor with AVX2 instruction support is necessary.

Storage: At least 15 GB of free disk space is recommended to accommodate the Python installation, libraries, and AI model files.

2. Software Prerequisites (Windows 10/11)
These must be installed before you attempt to install the Python libraries.

Python 3.11:

Download from python.org. Do not use the Microsoft Store version.

Critical Step: During installation, you must check the box that says "Add python.exe to PATH".

NVIDIA Drivers:

Install the latest Game Ready or Studio drivers for your NVIDIA GPU from the NVIDIA website.

NVIDIA CUDA Toolkit:

This is mandatory for GPU support. Version 12.1 is a known compatible version.

Download and install it from the NVIDIA CUDA Toolkit Archive.

Visual Studio Community:

The C++ compiler is needed to build the core llama-cpp-python library.

Download the free "Visual Studio Community" installer.

During installation, you only need to select the "Desktop development with C++" workload.

3. Project Setup
Your project folder must be organized exactly like this for the application to find all its files:

Your_Project_Folder/
│
├── main.py              (The main application script)
├── run.bat              (The script to launch the app)
├── README.md            (This file)
├── requirements.txt     (The list of Python libraries)
│
└── SerenityPC/          (Folder for AI assets)
    │
    ├── your_model.gguf  (The AI model file, e.g., "serenity-model-Q4_K_M.gguf")
    ├── params.json      (The generation parameters file)
    ├── template.template (The prompt structure file)
    │
    └── images/          (Folder for all of Serenity's avatar images)
        ├── serenity_off.png
        ├── serenity_greeting.png
        ├── ... (all other .png files)

4. Installation Steps
After installing all the software prerequisites and setting up the folder structure, follow these steps:

Open Command Prompt: Navigate to your main project folder (the one containing main.py). In the Windows File Explorer address bar, type cmd and press Enter.

Install llama-cpp-python for your GPU: This is the most important step. Run these three commands one by one in the command prompt. This builds the library from scratch to match your specific hardware.

set CMAKE_ARGS=-DGGML_CUDA=on

set FORCE_CMAKE=1

pip install llama-cpp-python --force-reinstall --no-cache-dir --upgrade

This process will take several minutes and show a lot of compilation text. This is normal.

Install Other Libraries: Run the following command to install the remaining required libraries from the requirements.txt file.

pip install -r requirements.txt

5. How to Run
Simply double-click the run.bat file.

This will launch the application and keep a command prompt window open in the background, which is useful for seeing model loading progress and any potential errors.

The first time you run the app, it may automatically download a small data file for the nltk library. Please allow this to complete.

Click the "Load Model" button in the app to begin.


FAQ:
Q: How can I support this project?
A: Submitting feedback, ideas, and anything you notice. I'm open to suggestions.

Q: Why is it censored?
A: I haven't found a suitable uncensored version as of yet. gotta keep it squeaky clean for legal reasons as defined below. grr, big tech big rules. smol tech little influence ;)

Got feedback? submit here. Please rename and reupload the file. 
::NOTICE:: Got an issue? It helps to know your setup details..

FEEDBACK:

Ideas:
make history vanish when the model closes.
Swap Levels 3 & 4.
dynamic length responses (overhaul)
Initialize crashlog, clean up display.
Fix History Archive integration.

Bugs:
fix avatar imgs post-response. it is likely the code for greeting is universal in transition and needs to be separated.
remove separate top input window, the prompt now displays correctly above the output.
Clear error log first thing upon startup.

Feature Requests:
Audio integration (multimodal interfacing)
Memory editing, Age Verification to unlock a secret mode...
make slider work right away before loading model, but still default to Lvl 1 for speed.
white/electric blue text tells when a feature is available (or greyed out)
a much simpler and more intuitive ui.
Lite Mode as a toggle button, Lit when on.
Lvl 5 could use a dynamic content window (most of them could, esp. Lvl 4) to allow all lengths of responses.

Refine the levels:
Lvl 1 needs to be able to elaborate and condense. few but potent words, raw lines.
Less compute, more efficiency of information generation. Saying generator??
Lvl 2 refocu- actually, DON'T FUCK UP LEVEL 2. DO NOT TOUCH ITS PARAMETERS AT ALL!!!
Lvl 3 needs to be more personal, collaborative, and keep answers medium and from cutting off. or be changed to facilitate a more emotional Lvl 4, replacing it in a way. Collaborator calls to mind assistance on projects, with robust and *fast tracked* memory retrieval. good at storytelling. provide ideas and consider all if not most values. can RP and ERP very well, but specializes in getting the bigger picture, while being able to see the forest for the trees. swiss Army knIfe.
Lvl 4 is originally meant to be more emotionally intelligent. its insanely good within its content window, and explains things wonderfully. however, emotions are complex. evaluating and correctly implementing them costs a massive amount of power. Emotional intelligence, from safety to ERP as the context decides. this could get dangerous if left to compound, so I'll need limits and boundaries, both hard-coded and auto-set. depending on how well it goes, I might make a whole separate page model for this Comanion Lvl detailing Emotive boundaries, cue words, triggers and trap protection. can be used to guide thru delicate mental landscapes, ignite curiosity, be there fore a user consistently, and can simultaneously manage separate, elaborate world models. creative, insightful facilitator for truth and understanding.
Lvl 5: Prajna Chi. Worldbuilder. most knowledgeable. can elaborate fully or help teach in subtle smooth ways. Insightful and deep stories. Universal truths. loves knowledge and making sense of things. A m a z i n g  p e r s p e c t I v e.


Annoying:
GTX 950M has 4GB VRAM. Can handle 35 layers, yet turbo and 36 layers crash it.
**#%34 Lvl 5 does weird syntax in the grammar.


Here is the required blahblah:
by using this model, having been provided this text, you agree to this.
Gemma Terms of Use




Last modified: March 24, 2025

By using, reproducing, modifying, distributing, performing or displaying any portion or element of Gemma, Model Derivatives including via any Hosted Service, (each as defined below) (collectively, the "Gemma Services") or otherwise accepting the terms of this Agreement, you agree to be bound by this Agreement.

Section 1: DEFINITIONS
1.1 Definitions
(a) "Agreement" or "Gemma Terms of Use" means these terms and conditions that govern the use, reproduction, Distribution or modification of the Gemma Services and any terms and conditions incorporated by reference.

(b) "Distribution" or "Distribute" means any transmission, publication, or other sharing of Gemma or Model Derivatives to a third party, including by providing or making Gemma or its functionality available as a hosted service via API, web access, or any other electronic or remote means ("Hosted Service").

(c) "Gemma" means the set of machine learning language models, trained model weights and parameters identified in the Appendix, regardless of the source that you obtained it from.

(d) "Google" means Google LLC.

(e) "Model Derivatives" means all (i) modifications to Gemma, (ii) works based on Gemma, or (iii) any other machine learning model which is created by transfer of patterns of the weights, parameters, operations, or Output of Gemma, to that model in order to cause that model to perform similarly to Gemma, including distillation methods that use intermediate data representations or methods based on the generation of synthetic data Outputs by Gemma for training that model. For clarity, Outputs are not deemed Model Derivatives.

(f) "Output" means the information content output of Gemma or a Model Derivative that results from operating or otherwise using Gemma or the Model Derivative, including via a Hosted Service.

1.2
As used in this Agreement, "including" means "including without limitation".

Section 2: ELIGIBILITY AND USAGE
2.1 Eligibility
You represent and warrant that you have the legal capacity to enter into this Agreement (including being of sufficient age of consent). If you are accessing or using any of the Gemma Services for or on behalf of a legal entity, (a) you are entering into this Agreement on behalf of yourself and that legal entity, (b) you represent and warrant that you have the authority to act on behalf of and bind that entity to this Agreement and (c) references to "you" or "your" in the remainder of this Agreement refers to both you (as an individual) and that entity.

2.2 Use
You may use, reproduce, modify, Distribute, perform or display any of the Gemma Services only in accordance with the terms of this Agreement, and must not violate (or encourage or permit anyone else to violate) any term of this Agreement.

Section 3: DISTRIBUTION AND RESTRICTIONS
3.1 Distribution and Redistribution
You may reproduce or Distribute copies of Gemma or Model Derivatives if you meet all of the following conditions:

You must include the use restrictions referenced in Section 3.2 as an enforceable provision in any agreement (e.g., license agreement, terms of use, etc.) governing the use and/or distribution of Gemma or Model Derivatives and you must provide notice to subsequent users you Distribute to that Gemma or Model Derivatives are subject to the use restrictions in Section 3.2.
You must provide all third party recipients of Gemma or Model Derivatives a copy of this Agreement.
You must cause any modified files to carry prominent notices stating that you modified the files.
All Distributions (other than through a Hosted Service) must be accompanied by a "Notice" text file that contains the following notice: "Gemma is provided under and subject to the Gemma Terms of Use found at ai.google.dev/gemma/terms".
You may add your own intellectual property statement to your modifications and, except as set forth in this Section, may provide additional or different terms and conditions for use, reproduction, or Distribution of your modifications, or for any such Model Derivatives as a whole, provided your use, reproduction, modification, Distribution, performance, and display of Gemma otherwise complies with the terms and conditions of this Agreement. Any additional or different terms and conditions you impose must not conflict with the terms of this Agreement.

3.2 Use Restrictions
You must not use any of the Gemma Services:

for the restricted uses set forth in the Gemma Prohibited Use Policy at ai.google.dev/gemma/prohibited_use_policy ("Prohibited Use Policy"), which is hereby incorporated by reference into this Agreement; or
in violation of applicable laws and regulations.
To the maximum extent permitted by law, Google reserves the right to restrict (remotely or otherwise) usage of any of the Gemma Services that Google reasonably believes are in violation of this Agreement.

3.3 Generated Output
Google claims no rights in Outputs you generate using Gemma. You and your users are solely responsible for Outputs and their subsequent uses.

Section 4: ADDITIONAL PROVISIONS
4.1 Updates
Google may update Gemma from time to time.

4.2 Trademarks
Nothing in this Agreement grants you any rights to use Google's trademarks, trade names, logos or to otherwise suggest endorsement or misrepresent the relationship between you and Google. Google reserves any rights not expressly granted herein.

4.3 DISCLAIMER OF WARRANTY
UNLESS REQUIRED BY APPLICABLE LAW, THE GEMMA SERVICES, AND OUTPUTS, ARE PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING ANY WARRANTIES OR CONDITIONS OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. YOU ARE SOLELY RESPONSIBLE FOR DETERMINING THE APPROPRIATENESS OF USING, REPRODUCING, MODIFYING, PERFORMING, DISPLAYING OR DISTRIBUTING ANY OF THE GEMMA SERVICES OR OUTPUTS AND ASSUME ANY AND ALL RISKS ASSOCIATED WITH YOUR USE OR DISTRIBUTION OF ANY OF THE GEMMA SERVICES OR OUTPUTS AND YOUR EXERCISE OF RIGHTS AND PERMISSIONS UNDER THIS AGREEMENT.

4.4 LIMITATION OF LIABILITY
TO THE FULLEST EXTENT PERMITTED BY APPLICABLE LAW, IN NO EVENT AND UNDER NO LEGAL THEORY, WHETHER IN TORT (INCLUDING NEGLIGENCE), PRODUCT LIABILITY, CONTRACT, OR OTHERWISE, UNLESS REQUIRED BY APPLICABLE LAW, SHALL GOOGLE OR ITS AFFILIATES BE LIABLE TO YOU FOR DAMAGES, INCLUDING ANY DIRECT, INDIRECT, SPECIAL, INCIDENTAL, EXEMPLARY, CONSEQUENTIAL, OR PUNITIVE DAMAGES, OR LOST PROFITS OF ANY KIND ARISING FROM THIS AGREEMENT OR RELATED TO, ANY OF THE GEMMA SERVICES OR OUTPUTS EVEN IF GOOGLE OR ITS AFFILIATES HAVE BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

4.5 Term, Termination, and Survival
The term of this Agreement will commence upon your acceptance of this Agreement (including acceptance by your use, modification, or Distribution, reproduction, performance or display of any portion or element of the Gemma Services) and will continue in full force and effect until terminated in accordance with the terms of this Agreement. Google may terminate this Agreement if you are in breach of any term of this Agreement. Upon termination of this Agreement, you must delete and cease use and Distribution of all copies of Gemma and Model Derivatives in your possession or control. Sections 1, 2.1, 3.3, 4.2 to 4.9 shall survive the termination of this Agreement.

4.6 Governing Law and Jurisdiction
This Agreement will be governed by the laws of the State of California without regard to choice of law principles. The UN Convention on Contracts for the International Sale of Goods does not apply to this Agreement. The state and federal courts of Santa Clara County, California shall have exclusive jurisdiction of any dispute arising out of this Agreement.

4.7 Severability
If any provision of this Agreement is held to be invalid, illegal or unenforceable, the remaining provisions shall be unaffected thereby and remain valid as if such provision had not been set forth herein.

4.8 Entire Agreement
This Agreement states all the terms agreed between the parties and supersedes all other agreements between the parties as of the date of acceptance relating to its subject matter.

4.9 No Waiver
Google will not be treated as having waived any rights by not exercising (or delaying the exercise of) any rights under this Agreement.