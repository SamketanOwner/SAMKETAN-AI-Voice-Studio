import streamlit as st
import os
from pydub import AudioSegment
from kokoro_onnx import Kokoro
import io
import soundfile as sf
import numpy as np

# --- 1. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

# Paths to your high-quality brain files (the ones you pushed with LFS)
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
    # 'af_heart' is the best female voice; 'am_michael' is a great male voice
    voice_choice = st.selectbox("Select Voice Personality", ["af_heart", "am_michael", "bf_isabella", "af_sky"])
    st.success(f"Selected Voice: {voice_choice}")

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script for Bhoodevi Warehouse...")
    
    # --- Inside Tab 2: Generate Speech ---
if st.button("Generate High-Quality Voice"):
    if script:
        with st.spinner("Synthesizing human speech..."):
            try:
                kokoro = Kokoro(MODEL_FILE, VOICE_FILE)
                samples, sample_rate = kokoro.create(script, voice=voice_choice, speed=1.0)
                
                # FIX: Create a "Binary" buffer to hold the audio data
                out_buffer = io.BytesIO() 
                
                # Use soundfile to write the raw samples into that buffer as a WAV
                sf.write(out_buffer, samples, sample_rate, format='WAV')
                
                # Save the raw BYTES to the session state (this avoids the UTF-8 error)
                st.session_state['gen_audio'] = out_buffer.getvalue()
                
                st.success("Human speech generated!")
                st.audio(st.session_state['gen_audio'])
            except Exception as e:
                st.error(f"Error generating audio: {e}")
            else:
                st.error("AI Brain files not found. Please ensure the .onnx and .bin files are uploaded.")
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
                    # IMPORTANT: We tell Pydub this is definitely a WAV file from memory
                    v_audio = AudioSegment.from_file(io.BytesIO(st.session_state['gen_audio']), format="wav")
                    
                    # Load background music
                    m_audio = AudioSegment.from_file(bg_music)
                    
                    # Apply volume reduction
                    m_audio = m_audio - music_reduction
                    
                    # Mix
                    final_mix = m_audio.overlay(v_audio)
                    
                    # Export to MP3
                    final_out = io.BytesIO()
                    final_mix.export(final_out, format="mp3")
                    
                    st.audio(final_out)
                    st.download_button("Download Mastered Mix", final_out, "samketan_master.mp3")
                except Exception as e:
                    st.error(f"Mixing Error: {e}")
                    st.write("Tip: Try generating the voice again in Tab 2 before mixing.")
        else:
            st.warning("Generate voice in Tab 2 and upload music here first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
