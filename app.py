import streamlit as st
import os
import subprocess
import io
import soundfile as sf
import numpy as np
from pydub import AudioSegment
from kokoro_onnx import Kokoro

# --- 1. THE RECOVERY ENGINE ---
MODEL_FILE = "kokoro-v0_19.onnx"
VOICE_FILE = "voices-v1.0.bin"
MODEL_URL = "https://media.githubusercontent.com/media/SamketanOwner/SAMKETAN-AI-Voice-Studio/main/kokoro-v0_19.onnx"

def ensure_models_exist():
    # If file is a tiny pointer (< 1MB), force the 374MB download
    if not os.path.exists(MODEL_FILE) or os.path.getsize(MODEL_FILE) < 1000000:
        with st.spinner("Downloading High-Quality AI Brain (374MB)... Please wait 5 minutes."):
            subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_FILE])
        st.rerun()

ensure_models_exist()

# --- 2. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

st.sidebar.title("SAMKETAN AI")
st.sidebar.info("Proprietor: Sanjay Kumar")
st.sidebar.write("📍 Kalyana Karnataka Region")

st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("High-Quality Human Voice Production")

# --- 3. THE INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Library", "✍️ Text-to-Voice", "🎼 Studio Master"])

with tab1:
    st.header("Human Voice Selection")
    voice_choice = st.selectbox("Select Voice Personality", ["af_heart", "am_michael", "af_sky", "bf_isabella"])
    st.success(f"Selected Voice: {voice_choice}")

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script...")
    
    if st.button("Generate High-Quality Voice"):
        if script:
            if os.path.exists(MODEL_FILE) and os.path.exists(VOICE_FILE):
                # SENSOR: Check if the file is a fake 'pointer'
                with open(MODEL_FILE, 'rb') as f:
                    header = f.read(100)
                
                if b"version https://git-lfs" in header:
                    st.error("Detected a 'Pointer' file instead of the AI Brain. Attempting to force download...")
                    subprocess.run(["curl", "-L", MODEL_URL, "-o", MODEL_FILE])
                    st.rerun()
                else:
                    with st.spinner("Samketan AI is synthesizing human speech..."):
                        try:
                            # PASSING THE FILENAME (STRING) AS REQUIRED
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
            st.warning("Generate voice in Tab 2 and upload music here first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
