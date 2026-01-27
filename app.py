import streamlit as st
import os
import requests
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from kokoro_onnx import Kokoro

# --- 1. SETTINGS & PATHS ---
MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices.bin"

# Official stable release links
MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx"
VOICE_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin"

def download_file(url, path):
    """Downloads a file in 1MB binary chunks for speed and safety."""
    with requests.get(url, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024*1024): 
                if chunk:
                    f.write(chunk)

def ensure_models_exist():
    """Validates binary integrity. If it's a 'fake' text pointer, it deletes and pulls the real file."""
    # Check if file is missing or suspiciously small (< 1MB means it's a text pointer)
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        status_placeholder = st.empty()
        status_placeholder.info("🚀 Samketan AI is fetching the 374MB Brain. This takes 3-5 minutes...")
        try:
            download_file(MODEL_URL, MODEL_FILE)
            download_file(VOICE_URL, VOICE_FILE)
            status_placeholder.success("Download Complete! Initializing...")
            st.rerun()
        except Exception as e:
            status_placeholder.error(f"Download Failed: {e}")
            st.stop()

# Auto-verify on startup
ensure_models_exist()

# --- 2. BRANDING & UI ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

st.sidebar.title("SAMKETAN AI")
st.sidebar.info(f"Proprietor: Sanjay Kumar\n📍 Kalyana Karnataka Region")
st.sidebar.write("Project: AI development for Bhuvi")

st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("High-Quality Human Voice Production")

# --- 3. THE STUDIO INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Library", "✍️ Text-to-Voice", "🎼 Studio Master"])

with tab1:
    st.header("Human Voice Selection")
    voice_choice = st.selectbox("Select Voice Personality", ["af_heart", "am_michael", "af_sky", "bf_isabella"])
    st.success(f"Selected Voice: {voice_choice}")

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script here...")
    
    if st.button("Generate High-Quality Voice"):
        if script:
            # Final binary check before synthesis
            if os.path.getsize(MODEL_FILE) < 1000000:
                st.error("Detected corrupted model file. Re-syncing...")
                os.remove(MODEL_FILE)
                st.rerun()

            with st.spinner("Samketan AI is synthesizing human speech..."):
                try:
                    # Initialize engine using the verified binary path
                    kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
                    samples, sample_rate = kokoro.create(script, voice=voice_choice, speed=1.0)
                    
                    # Convert to WAV
                    out_buffer = io.BytesIO()
                    sf.write(out_buffer, samples, sample_rate, format='WAV')
                    st.session_state['gen_audio'] = out_buffer.getvalue()
                    
                    st.success("Human speech generated!")
                    st.audio(st.session_state['gen_audio'])
                except UnicodeDecodeError:
                    st.error("System tried to read binary as text. Deleting cache and restarting...")
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
            st.warning("Please generate the voice in Tab 2 and upload music here first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
