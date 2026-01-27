import streamlit as st
import os
import requests
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from kokoro_onnx import Kokoro

# --- 1. SETTINGS & PATHS (STABLE LINKS) ---
MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices.bin" # The library specifically looks for this name

# Using the official stable release links from the developer
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
VOICE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"

def download_file(url, path):
    """Downloads a file in binary chunks to prevent UTF-8 corruption."""
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

def ensure_models_exist():
    """Checks if the files are real binary files and not 404/pointers."""
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("🚀 Fetching High-Quality AI Brain (374MB)... Please wait 3-5 mins."):
            try:
                download_file(MODEL_URL, MODEL_FILE)
                download_file(VOICE_URL, VOICE_FILE)
            except Exception as e:
                st.error(f"Download Failed: {e}. Please check your internet connection on the server.")
                return
        st.rerun()

# Run the download check on app startup
ensure_models_exist()

# --- 2. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

st.sidebar.title("SAMKETAN AI")
st.sidebar.info(f"Proprietor: Sanjay Kumar\n📍 Kalyana Karnataka Region")
st.sidebar.write("Project: AI development for Bhuvi")

st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("High-Quality Human Voice Production")

# --- 3. THE INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Library", "✍️ Text-to-Voice", "🎼 Studio Master"])

with tab1:
    st.header("Human Voice Selection")
    # Voice choices 
    voice_choice = st.selectbox("Select Voice Personality", ["af_heart", "am_michael", "af_sky", "bf_isabella"])
    st.success(f"Selected Voice: {voice_choice}")

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150)
    
    if st.button("Generate High-Quality Voice"):
        if script:
            # Check the actual size before even trying to open it
            if os.path.exists(MODEL_FILE):
                size = os.path.getsize(MODEL_FILE)
                if size < 1000000: # It's a pointer or a corrupted file
                    st.error(f"Corrupted file detected ({size} bytes). Deleting and re-downloading...")
                    os.remove(MODEL_FILE)
                    st.rerun()
            
            with st.spinner("Samketan AI is synthesizing..."):
                try:
                    # THE FIX: Explicitly telling the engine to ignore encoding
                    kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
                    samples, sample_rate = kokoro.create(script, voice=voice_choice, speed=1.0)
                    
                    out_buffer = io.BytesIO()
                    sf.write(out_buffer, samples, sample_rate, format='WAV')
                    st.session_state['gen_audio'] = out_buffer.getvalue()
                    st.audio(st.session_state['gen_audio'])
                except UnicodeDecodeError:
                    st.error("The system tried to read the model as text. Forcing a hard reset...")
                    if os.path.exists(MODEL_FILE): os.remove(MODEL_FILE)
                    st.rerun()
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
                    v_audio = AudioSegment.from_file(io.BytesIO(st.session_state['gen_audio']), format="wav")
                    m_audio = AudioSegment.from_file(bg_music)
                    m_audio = m_audio - music_reduction
                    final_mix = m_audio.overlay(v_audio)
                    
                    final_out = io.BytesIO()
                    final_mix.export(final_out, format="mp3")
                    
                    st.audio(final_out)
                    st.download_button("Download Mastered Mix", final_out, "samketan_master.mp3")
                except Exception as e:
                    st.error(f"Mixing Error: {e}")
        else:
            st.warning("Please generate the voice in Tab 2 first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
