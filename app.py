import streamlit as st
import os
import subprocess
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from kokoro_onnx import Kokoro

MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices-v1.0.bin"
# This is a direct link to the raw binary file
MODEL_URL = "https://media.githubusercontent.com/media/SamketanOwner/SAMKETAN-AI-Voice-Studio/main/kokoro-v0_19.onnx"

def ensure_models_exist():
    # If file is missing or is just a tiny LFS pointer (< 1MB)
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("Downloading High-Quality AI Brain (374MB)... This only happens once."):
            # Force download using curl
            subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_FILE])
            
            # If curl fails, try the git lfs pull command
            if os.path.getsize(MODEL_FILE) < 1000000:
                subprocess.run(["git", "lfs", "pull"])
        st.rerun()

ensure_models_exist()

# --- 2. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")
# ... (rest of your sidebar and title code)

# Paths to your high-quality brain files (uploaded via LFS)
MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices-v1.0.bin"

# Sidebar Branding
st.sidebar.title("SAMKETAN AI")
st.sidebar.info("Proprietor: Sanjay Kumar")
st.sidebar.write("📍 Kalyana Karnataka Region")

# Main Header
st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("High-Quality Human Voice Production")

# --- 2. THE 3-TAB INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Library", "✍️ Text-to-Voice", "🎼 Studio Master"])

with tab1:
    st.header("Human Voice Selection")
    # 'af_heart' and 'af_sky' are excellent female voices; 'am_michael' is a great male voice
    voice_choice = st.selectbox("Select Voice Personality", ["af_heart", "am_michael", "af_sky", "bf_isabella"])
    st.success(f"Selected Voice: {voice_choice}")

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script...")
    
    if st.button("Generate High-Quality Voice"):
        if script:
            if os.path.exists(MODEL_FILE) and os.path.exists(VOICE_FILE):
                # Check the size of the model file
                model_size = os.path.getsize(MODEL_FILE)
                
                if model_size < 1000: # It's a tiny LFS pointer
                    st.error(f"File Error: The model file is only {model_size} bytes. This is just a 'pointer' text file. Your 374MB model didn't download from GitHub LFS correctly.")
                    st.info("Check your 'setup.sh' and 'packages.txt' files on GitHub.")
                else:
                    with st.spinner("Samketan AI is synthesizing human speech..."):
                        try:
                            # Pass the filenames directly
                            kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
                            samples, sample_rate = kokoro.create(script, voice=voice_choice, speed=1.0)
                            
                            out_buffer = io.BytesIO()
                            sf.write(out_buffer, samples, sample_rate, format='WAV')
                            st.session_state['gen_audio'] = out_buffer.getvalue()
                            
                            st.success("Human speech generated!")
                            st.audio(st.session_state['gen_audio'])
                        except Exception as e:
                            st.error(f"Detailed Engine Error: {e}")
            else:
                st.error("AI Brain files not found.")
        else:
            st.warning("Please enter text first.")

with tab3:
    st.header("Background Music & Mixing")
    bg_music = st.file_uploader("Upload Music (MP3/WAV)", type=['wav', 'mp3'])
    music_reduction = st.slider("Music Volume Reduction (dB)", 0, 20, 12)

    if st.button("Master Final Output"):
        if 'gen_audio' in st.session_state and bg_music:
            with st.spinner("Mixing studio quality audio..."):
                try:
                    # Load synthesized voice from memory as WAV
                    v_audio = AudioSegment.from_file(io.BytesIO(st.session_state['gen_audio']), format="wav")
                    
                    # Load background music
                    m_audio = AudioSegment.from_file(bg_music)
                    
                    # Apply volume reduction to music
                    m_audio = m_audio - music_reduction
                    
                    # Mix voice over music
                    final_mix = m_audio.overlay(v_audio)
                    
                    # Export final result
                    final_out = io.BytesIO()
                    final_mix.export(final_out, format="mp3")
                    
                    st.audio(final_out)
                    st.download_button("Download Mastered Mix", final_out, "samketan_master.mp3")
                except Exception as e:
                    st.error(f"Mixing Error: {e}")
        else:
            st.warning("Generate voice in Tab 2 and upload music here first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
