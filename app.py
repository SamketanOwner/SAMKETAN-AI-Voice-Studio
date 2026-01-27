import streamlit as st
import os
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from io import BytesIO

# --- 1. SETTINGS & BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")
st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("Professional Voice Cloning & Studio Mixing")

# --- 2. AUTHENTICATION (Revenue Security) ---
# This pulls your API key safely from Streamlit Secrets
API_KEY = st.secrets.get("ELEVENLABS_API_KEY", "")
client = ElevenLabs(api_key=API_KEY)

# --- 3. THE 3-TAB INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Sample", "✍️ Text-to-Voice", "🎼 Studio Master"])

# TAB 1: Uploading the source voice
with tab1:
    st.header("Upload Voice Sample")
    st.info("Upload 30-60 seconds of clear audio to clone the voice.")
    up_voice = st.file_uploader("Target Voice (WAV/MP3)", type=['wav', 'mp3'])
    if up_voice:
        st.success("Voice sample loaded!")

# TAB 2: Text input and Generation
with tab2:
    st.header("Generate AI Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script here...")
    
    if st.button("Generate Voice"):
        if not API_KEY:
            st.error("API Key missing. Please add it to Streamlit Secrets.")
        elif up_voice and script:
            with st.spinner("AI is cloning and speaking..."):
                # CLONING LOGIC
                voice = client.clone(
                    name="TempVoice",
                    files=[up_voice]
                )
                # GENERATION LOGIC
                audio_gen = client.generate(text=script, voice=voice.voice_id)
                
                # Store in session so Tab 3 can see it
                st.session_state['gen_audio'] = audio_gen
                st.success("Speech generated! Go to Tab 3 to add music.")
        else:
            st.warning("Please upload a voice in Tab 1 and enter text here.")

# TAB 3: Mixing and Ducking
with tab3:
    st.header("Background Music & Mixing")
    bg_music = st.file_uploader("Upload Music", type=['wav', 'mp3'])
    music_reduction = st.slider("Music Volume Reduction (dB)", 0, 20, 12)

    if st.button("Master Final Output"):
        if 'gen_audio' in st.session_state and bg_music:
            with st.spinner("Mastering audio for clarity..."):
                # Load synthesized voice
                v_audio = AudioSegment.from_file(BytesIO(b"".join(st.session_state['gen_audio'])))
                # Load music
                m_audio = AudioSegment.from_file(bg_music)
                
                # Apply "Ducking" (Lower music volume)
                m_audio = m_audio - music_reduction
                
                # Mix them
                final_mix = m_audio.overlay(v_audio)
                
                # Export and Display
                out_buffer = BytesIO()
                final_mix.export(out_buffer, format="mp3")
                st.audio(out_buffer)
                st.download_button("Download Mastered Mix", out_buffer, "samketan_master.mp3")
        else:
            st.warning("Ensure you have generated voice in Tab 2 and uploaded music here.")

st.markdown("---")
st.caption("Powered by SAMKETAN AI | Proprietor: Sanjay Kumar")
