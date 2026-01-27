import streamlit as st
import os
import requests
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from kokoro_onnx import Kokoro

# --- 1. SETTINGS & PATHS (DEFINED ONCE) ---
MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices-v1.0.bin"

# These links bypass GitHub LFS and pull the raw binary data directly
MODEL_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/kokoro-v0_19.onnx"
VOICE_URL = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices-v1.0.bin"

def download_file(url, path):
    """Downloads a file in binary chunks to prevent UTF-8 corruption."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def ensure_models_exist():
    """Checks if the files are real binary files or tiny LFS pointers."""
    # If file doesn't exist or is smaller than 1MB (a pointer), download it
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("🚀 Samketan AI is fetching high-quality voices... (374MB). This takes 3-5 minutes."):
            try:
                download_file(MODEL_URL, MODEL_FILE)
                download_file(VOICE_URL, VOICE_FILE)
            except Exception as e:
                st.error(f"Download Failed: {e}")
                return
        st.rerun()

# Run the download check on app startup
ensure_models_exist()

# --- 2. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

st.sidebar.title("SAMKETAN AI")
st.sidebar.info(f"Proprietor: Sanjay Kumar\n📍 Kalyana Karnataka Region")
st.sidebar.markdown("---")
st.sidebar.write("Project: AI development for Bhuvi")

st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("High-Quality Human Voice Production")

# --- 3. THE INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Library", "✍️ Text-to-Voice", "🎼 Studio Master"])

with tab1:
    st.header("Human Voice Selection")
    # Voice choices based on Kokoro standard models
    voice_choice = st.selectbox("Select Voice Personality", ["af_heart", "am_michael", "af_sky", "bf_isabella"])
    st.success(f"Selected Voice: {voice_choice}")
    st.info("Tip: 'am_michael' is excellent for warehouse announcements.")

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script here...")
    
    if st.button("Generate High-Quality Voice"):
        if script:
            with st.spinner("Synthesizing human speech..."):
                try:
                    # Pass the file paths to the engine
                    kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
                    samples, sample_rate = kokoro.create(script, voice=voice_choice, speed=1.0)
                    
                    # Convert samples to WAV bytes
                    out_buffer = io.BytesIO()
                    sf.write(out_buffer, samples, sample_rate, format='WAV')
                    st.session_state['gen_audio'] = out_buffer.getvalue()
                    
                    st.success("Human speech generated successfully!")
                    st.audio(st.session_state['gen_audio'])
                except Exception as e:
                    st.error(f"Detailed Engine Error: {e}")
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
                    # Load synthesized voice
                    v_audio = AudioSegment.from_file(io.BytesIO(st.session_state['gen_audio']), format="wav")
                    
                    # Load background music
                    m_audio = AudioSegment.from_file(bg_music)
                    
                    # Reduce music volume so voice is clear
                    m_audio = m_audio - music_reduction
                    
                    # Mix voice over music
                    final_mix = m_audio.overlay(v_audio)
                    
                    # Export final result
                    final_out = io.BytesIO()
                    final_mix.export(final_out, format="mp3")
                    
                    st.audio(final_out)
                    st.download_button("Download Mastered Mix", final_out, "samketan_master.mp3")
                    st.success("Your studio-quality ad is ready!")
                except Exception as e:
                    st.error(f"Mixing Error: {e}")
        else:
            st.warning("Please generate the voice in Tab 2 and upload background music here first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
