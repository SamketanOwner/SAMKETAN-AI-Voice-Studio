from kokoro_onnx import Kokoro
import soundfile as sf
import os

# 1. Initialize the model
# We use the exact filenames you downloaded
model_file = "kokoro-v0_19.onnx"
voice_file = "voices-v1.0.bin" # Change this to match the .bin file

print("Loading Samketan AI Engine...")
kokoro = Kokoro(model_file, voice_file)

# 2. Your Business Script
# Tip: Use commas for natural pauses
text = "Welcome to Bhuvi Warehouse, located in the heart of Kalyana Karnataka. We provide professional storage solutions for your business. Contact Sanjay Kumar for more details."

print("Generating high-quality human voice...")

# 3. Create the audio
# 'af_heart' is a clear, human-like professional voice.
samples, sample_rate = kokoro.create(
    text, 
    voice="af_heart", 
    speed=1.0, 
    lang="en-us"
)

# 4. Save the file
output_path = "bhoodevi_ad.wav"
sf.write(output_path, samples, sample_rate)

print(f"--- SUCCESS! ---")
print(f"Your ad has been saved as: {output_path}")
print(f"Location: {os.getcwd()}")