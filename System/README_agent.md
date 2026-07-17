Serenity AI - Live Agent Setup Guide
This guide explains how to set up and run the live_agent.py, a background assistant that responds to voice commands.
1. Installation
2. This agent requires its own set of Python libraries.
3. Open Command Prompt: Navigate to the folder containing live_agent.py and agent_requirements.txt.
4. Install Libraries: Run the following command to install all necessary packages:pip install -r agent_requirements.txt
Note: pyaudio can sometimes be tricky to install on Windows. If you encounter errors, you may need to find a pre-compiled "wheel" (.whl) file for your specific Python version.
2. How to Run
3. Simply double-click the live_agent.py file or run python live_agent.py from a command prompt.
4. A new "S" icon will appear in your system tray (you may need to click the ^ icon to see it).
3. How to Use
Right-click the system tray icon to open the menu.
Enable Listening: Click "Enable Listening" to have the agent actively listen for your commands through your microphone.
Give a Command: Say "Serenity" followed by a command, for example:"Serenity, where are you?""Serenity, open my browser.""Serenity, search for the weather."
Change Voice: Right-click the icon, go to the "Voices" submenu, and select a different voice.
4. How to Add Your Own Voice Library (Advanced)
The agent is designed to be modular. 
To add a new TTS engine (e.g., from a cloud service like ElevenLabs):Open live_agent.py in a code editor. 
Create a New Class: Create a new class that inherits from TTSEngine.class MyCustomEngine(TTSEngine):def __init__(self):
# Your API key setup and initialization code here
self.voices = ["Voice One", "Voice Two"] # List of voices from your service

    def list_voices(self):
        return self.voices

    def set_voice(self, voice_name):
        # Code to tell your service which voice to use
        self.current_voice = voice_name

    def say(self, text):
        # Your code to send the text to the service and play the audio
        pass
Register Your Engine: In the LiveAgent's __init__ method, add your new engine to the self.tts_engines dictionary:self.tts_engines = {
"Default (pyttsx3)": Pyttsx3Engine(),
"My Custom Service": MyCustomEngine() # Add your engine here
}
Run the Agent: The new voices from your custom service will now automatically appear in the "Voices" submenu in the system tray.