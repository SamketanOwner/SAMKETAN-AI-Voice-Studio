import streamlit as st
import os
from pydub import AudioSegment
from elevenlabs.client import ElevenLabs
from io import BytesIO
from PIL import Image

# --- 1. SETTINGS & BRANDING (MUST BE FIRST) ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")

# Sidebar Branding Logic
if os.path.exists("SAMKETAN_LOGO.png"):
    st.sidebar.image("SAMKETAN_LOGO.png", use_container_width=True)
    st.sidebar.markdown("---")
else:
    st.sidebar.warning("Logo 'SAMKETAN_LOGO.png' not found. Please upload it to GitHub.")

st.sidebar.title("SAMKETAN AI")
st.sidebar.info("Proprietor: Sanjay Kumar")
st.sidebar.write("📍 Kalyana Karnataka Region")
st.sidebar.write("📧 Contact for Business Inquiries")

# Main Page Header
st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("Professional Voice Cloning & Studio Mixing")

# --- 2. AUTHENTICATION ---
API_KEY = st.secrets.get("ELEVENLABS_API_KEY", "")
client = ElevenLabs(api_key=API_KEY)

# --- 3. THE 3-TAB INTERFACE ---
tab1, tab2, tab3 = st.tabs(["📂 Voice Sample", "✍️ Text-to-Voice", "🎼 Studio Master"])

# TAB 1: Uploading the source voice
with tab1:
    st.header("Upload Voice Sample")
    st.info("Upload 30-60 seconds of clear audio to clone the voice.")
    # This line must be indented with 4 spaces to stay inside 'with tab1:'
    up_voice = st.file_uploader("Target Voice (WAV/MP3/M4A)", type=['wav', 'mp3', 'm4a'], key="voice_upload")
    if up_voice:
        st.success("Voice sample loaded successfully!")
# TAB 2: Text input and Generation
with tab2:
    st.header("Generate AI Speech")
    script = st.text_area("What should the voice say?", height=150, placeholder="Enter your business script here...")
    
    if st.button("Generate Voice"):
        if not API_KEY:
            st.error("API Key missing. Please add it to Streamlit Secrets.")
        elif up_voice and script:
            with st.spinner("AI is cloning and speaking..."):
                try:
                   # --- UPDATED CLONING LOGIC FOR 2026 ---
try:
    # We now use voices.ivc.create instead of just clone
    voice = client.voices.ivc.create(
        name="TempVoice",
        files=[up_voice]
    )
    
    # Generate the speech using the new voice ID
    audio_gen = client.text_to_speech.convert(
        text=script, 
        voice_id=voice.voice_id,
        model_id="eleven_turbo_v2_5" # Use Turbo for faster results!
    )
    
    # Store in session so Tab 3 can see it
    st.session_state['gen_audio'] = b"".join(audio_gen)
    st.success("Speech generated successfully!")
    st.audio(st.session_state['gen_audio'])
    
except Exception as e:
    st.error(f"AI Generation Error: {e}")
        else:
            st.warning("Please upload a voice in Tab 1 and enter text here.")

# TAB 3: Mixing and Ducking
with tab3:
    st.header("Background Music & Mixing")
    bg_music = st.file_uploader("Upload Music", type=['wav', 'mp3'], key="music_upload")
    music_reduction = st.slider("Music Volume Reduction (dB)", 0, 20, 12)

    if st.button("Master Final Output"):
        if 'gen_audio' in st.session_state and bg_music:
            with st.spinner("Mastering audio for clarity..."):
                # Load synthesized voice
                v_audio = AudioSegment.from_file(BytesIO(st.session_state['gen_audio']))
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
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
