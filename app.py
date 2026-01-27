import streamlit as st
import os
import subprocess
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from kokoro_onnx import Kokoro

# --- THE BULLETPROOF DOWNLOAD FIX ---
MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices-v1.0.bin"
# Using the special 'media' link which often bypasses LFS pointer issues
MODEL_URL = "https://media.githubusercontent.com/media/SamketanOwner/SAMKETAN-AI-Voice-Studio/main/kokoro-v0_19.onnx"

def download_model():
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000: # Less than 1MB means it's a pointer
        with st.spinner("Downloading High-Quality AI Brain (374MB). This may take 3-5 minutes..."):
            # Method 1: Direct media download
            subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_FILE])
            
            # Re-check: If still small, try Method 2: Git LFS Force
            if os.path.getsize(MODEL_FILE) < 1000000:
                subprocess.run(["git", "lfs", "pull"])
        st.rerun()

download_model()

# --- 1. SETTINGS & BRANDING ---
# (Rest of your code follows below...)
# --- 1. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

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
            # Check if files exist
            if os.path.exists(MODEL_FILE) and os.path.exists(VOICE_FILE):
                model_size = os.path.getsize(MODEL_FILE)
                
                # SENSOR: If file is too small, it's an LFS pointer (the cause of the UTF-8 error)
                if model_size < 10000: 
                    st.error(f"LFS Error: The model file is only {model_size} bytes. GitHub is sending a text 'pointer' instead of the 374MB AI brain. Please add a 'setup.sh' file to your repo with 'git lfs pull'.")
                else:
                    with st.spinner("Samketan AI is synthesizing human speech..."):
                        try:
                            # Initialize Engine
                            kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
                            samples, sample_rate = kokoro.create(script, voice=voice_choice, speed=1.0)
                            
                            # Handle binary data to avoid the 'utf-8' error
                            out_buffer = io.BytesIO()
                            sf.write(out_buffer, samples, sample_rate, format='WAV')
                            
                            st.session_state['gen_audio'] = out_buffer.getvalue()
                            st.success("Human speech generated!")
                            st.audio(st.session_state['gen_audio'])
                        except Exception as e:
                            st.error(f"Detailed Engine Error: {e}")
            else:
                st.error("AI Brain files not found in the directory.")
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
