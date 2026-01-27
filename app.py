import streamlit as st
import os
from pydub import AudioSegment
from gtts import gTTS
from io import BytesIO
from PIL import Image

# --- 1. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

# Sidebar Logo Logic
if os.path.exists("SAMKETAN_LOGO.png"):
    st.sidebar.image("SAMKETAN_LOGO.png", use_container_width=True)
    st.sidebar.markdown("---")

st.sidebar.title("SAMKETAN AI")
st.sidebar.info("Proprietor: Sanjay Kumar")
st.sidebar.write("📍 Kalyana Karnataka Region")

# Main Header
st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("Free Business Voice Generator & Mixer")

# --- 2. THE 3-TAB INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Library", "✍️ Text-to-Voice", "🎼 Studio Master"])

with tab1:
    st.header("Voice Settings")
    st.info("Using Google Free Voice Engine. No API Key required!")
    language = st.selectbox("Select Language", ["en", "hi", "kn"], format_func=lambda x: {"en":"English", "hi":"Hindi", "kn":"Kannada"}[x])

with tab2:
    st.header("Generate Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script...")
    
    if st.button("Generate Voice"):
        if script:
            with st.spinner("Generating audio..."):
                try:
                    # Google TTS Logic
                    tts = gTTS(text=script, lang=language)
                    audio_fp = BytesIO()
                    tts.write_to_fp(audio_fp)
                    
                    st.session_state['gen_audio'] = audio_fp.getvalue()
                    st.success("Speech generated! Go to Tab 3 to mix with music.")
                    st.audio(st.session_state['gen_audio'])
                except Exception as e:
                    st.error(f"Error: {e}")
        else:
            st.warning("Please enter some text.")

with tab3:
    st.header("Background Music & Mixing")
    bg_music = st.file_uploader("Upload Music (MP3/WAV)", type=['wav', 'mp3'])
    music_reduction = st.slider("Music Volume Reduction (dB)", 0, 20, 12)

    if st.button("Master Final Output"):
        if 'gen_audio' in st.session_state and bg_music:
            with st.spinner("Mixing studio quality audio..."):
                # Load synthesized voice
                v_audio = AudioSegment.from_file(BytesIO(st.session_state['gen_audio']), format="mp3")
                # Load music
                m_audio = AudioSegment.from_file(bg_music)
                
                # Apply Ducking
                m_audio = m_audio - music_reduction
                
                # Mix
                final_mix = m_audio.overlay(v_audio)
                
                out_buffer = BytesIO()
                final_mix.export(out_buffer, format="mp3")
                st.audio(out_buffer)
                st.download_button("Download Mastered Mix", out_buffer, "samketan_master.mp3")
        else:
            st.warning("Generate voice in Tab 2 and upload music here first.")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
