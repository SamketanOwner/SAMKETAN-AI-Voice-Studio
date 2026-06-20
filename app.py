import streamlit as st
import io, os, numpy as np, soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts, XttsAudioConfig
from TTS.utils.manage import ModelManager
import torch

# ===== HF_TOKEN SETUP =====
hf_token = st.secrets.get("HF_TOKEN")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token
    from huggingface_hub import login
    login(token=hf_token)

os.environ.setdefault("COQUI_TOS_AGREED", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def patch_torch_load_for_coqui():
    if getattr(torch.load, "_coqui_legacy_patch", False):
        return
    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    patched_load._coqui_legacy_patch = True
    torch.load = patched_load


def allow_coqui_checkpoint_classes():
    safe_classes = [BaseDatasetConfig, XttsConfig, XttsAudioConfig]
    try:
        from TTS.tts.models.xtts import XttsArgs
        safe_classes.append(XttsArgs)
    except Exception:
        pass
    try:
        torch.serialization.add_safe_globals(safe_classes)
    except AttributeError:
        pass


# --- AUDIO QUALITY VALIDATOR ---
def validate_voice_sample(audio_path, min_duration=15, max_duration=60):
    """Validate voice sample quality for Indian English accent preservation."""
    audio = AudioSegment.from_file(audio_path)
    duration_sec = len(audio) / 1000.0
    
    if duration_sec < min_duration:
        return False, f"Too short ({duration_sec:.1f}s). Need at least {min_duration}s of EXPRESSIVE Indian English speech."
    if duration_sec > max_duration:
        return False, f"Too long ({duration_sec:.1f}s). Trim to {max_duration}s max."
    
    # Check approximate loudness
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 0.02:
        return False, "Audio too quiet. Record in normal speaking voice."
    
    return True, "✅ Voice sample validated."


# --- MODEL LOADER ---
@st.cache_resource
def load_xtts():
    allow_coqui_checkpoint_classes()
    patch_torch_load_for_coqui()
    name = "tts_models/multilingual/multi-dataset/xtts_v2"
    path = ModelManager().download_model(name)[0]
    cfg = XttsConfig()
    cfg.load_json(os.path.join(path, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=path, eval=True)
    return model


# --- UI BRANDING ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")
st.sidebar.title("SAMKETAN AI")
st.sidebar.info("Proprietor: Sanjay Kumar\n- Kalyana Karnataka Region")
st.sidebar.write("Project: AI development for Indian English Voice Cloning")
st.sidebar.markdown("---")
st.sidebar.markdown("**Voice Cloning Tips:**\n- Record 20-40s of CLEAR Indian English\n- Speak naturally with emotion\n- Quiet room, good microphone\n- Avoid robotic/monotone speech")

st.title("SAMKETAN AI-Voice-Studio")
st.subheader("🎙️ Indian English Voice Cloning & Production")

tab1, tab2, tab3, tab4 = st.tabs(["Voice Sample", "Generate Voice", "Studio Master", "Settings"])

# --- TAB 1: VOICE SAMPLE ---
with tab1:
    st.header("📤 Upload Your Voice Sample")
    st.warning("**IMPORTANT:** For natural Indian English output, record yourself speaking expressively with Indian English accent/intonation")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Recording Requirements:
        - **Duration:** 20-40 seconds (optimal: 30s)
        - **Language:** Indian English (use natural accent)
        - **Tone:** Speak EXPRESSIVELY — vary pitch, speed, emotion
        - **Quality:** Quiet room, clear voice, 16kHz+ sample rate
        - **Format:** WAV, MP3, or FLAC
        
        ❌ **Avoid:**
        - Robotic/monotone speech
        - Background noise
        - Whispering or very soft voice
        - Reading too quickly
        """)
    
    with col2:
        st.info("💡 Tip: Read a paragraph with different emotions (angry, happy, sad) to help the AI capture natural variation in your voice.")
    
    consent = st.checkbox("✅ I confirm I am authorized to clone this voice and own the IP rights.", value=False)
    
    sample = st.file_uploader("Upload Voice Sample (WAV/MP3/FLAC)", type=["wav", "mp3", "flac"])
    
    if sample and consent:
        with st.spinner("🔄 Processing and optimizing your voice profile..."):
            try:
                # CRITICAL FIX: Save uploaded file to disk first
                # Streaming from Streamlit's file object causes ffmpeg errors
                file_ext = sample.name.split('.')[-1].lower()
                temp_file = f"temp_upload.{file_ext}"
                
                # Save uploaded file to disk
                with open(temp_file, "wb") as f:
                    f.write(sample.getbuffer())
                
                # Now read from disk using the saved file
                audio_segment = AudioSegment.from_file(temp_file)
                
                duration_sec = len(audio_segment) / 1000.0
                st.info(f"Audio duration: {duration_sec:.1f}s")
                
                # Validate before processing
                is_valid, msg = validate_voice_sample(sample, min_duration=15, max_duration=60)
                if not is_valid:
                    st.error(msg)
                else:
                    # Enhance audio for better voice conditioning
                    # Normalize loudness to -20 LUFS (standard for voice)
                    samples = np.array(audio_segment.get_array_of_samples()).astype(np.float32) / 32768.0
                    if audio_segment.channels == 2:
                        samples = np.mean(samples.reshape((-1, 2)), axis=1)
                    
                    meter = pyln.Meter(audio_segment.frame_rate)
                    try:
                        loudness = meter.integrated_loudness(samples)
                        normalized = pyln.normalize.loudness(samples, loudness, -20)
                    except:
                        normalized = samples
                    
                    # Convert to 24kHz Mono WAV (XTTS requirement)
                    audio_segment = AudioSegment(
                        (normalized * 32767).astype(np.int16).tobytes(),
                        frame_rate=audio_segment.frame_rate,
                        sample_width=2,
                        channels=1
                    )
                    audio_segment = audio_segment.set_frame_rate(24000).set_channels(1)
                    audio_segment.export("ref.wav", format="wav")
                    
                    st.session_state["ref"] = "ref.wav"
                    st.success("✅ Voice profile processed successfully!")
                    st.audio("ref.wav", format="audio/wav")
                    st.info("Proceed to Tab 2: Generate Voice")
                    
            except Exception as e:
                st.error(f"❌ Audio Processing Error: {str(e)[:150]}")
                st.info("**Quick fix:** Try uploading a different WAV file")
                # Cleanup temp file if it exists
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except:
                    pass
    
    elif sample and not consent:
        st.warning("Please check the consent box before uploading.")

# --- TAB 2: GENERATE VOICE ---
with tab2:
    st.header("🎤 Generate Speech in Your Cloned Voice")
    
    if not st.session_state.get("ref"):
        st.warning("⚠️ Please upload and process a voice sample in Tab 1 first.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            script = st.text_area(
                "What should the voice say?",
                height=150,
                placeholder="Type text here in Indian English or English...",
                help="Use natural Indian English phrasing for best results"
            )
            
            lang = st.selectbox(
                "Language/Accent",
                ["en-IN", "en", "hi"],
                index=0,
                help="en-IN = Indian English accent, en = General English"
            )
        
        with col2:
            st.markdown("### Voice Parameters")
            temp = st.slider(
                "🎚️ Expressiveness",
                0.5, 1.0, 0.70, 0.02,
                help="Lower (0.5-0.65) = steady, Higher (0.75-1.0) = emotional"
            )
            rep_pen = st.slider(
                "🔁 Repetition Penalty",
                1.0, 15.0, 5.0, 0.5,
                help="Avoid repeated words/phrases"
            )
            speed = st.slider(
                "⏱️ Speaking Rate",
                0.7, 1.3, 1.0, 0.05,
                help="0.7-0.85 = slower (clearer), 1.0 = normal, 1.1-1.3 = faster"
            )
            top_p = st.slider(
                "🎯 Voice Consistency",
                0.75, 1.0, 0.85, 0.02,
                help="Lower (0.75-0.80) = more consistent to your voice, Higher = more variation"
            )
        
        if st.button("🚀 Generate High-Quality Voice", use_container_width=True):
            if not script.strip():
                st.warning("Please enter text first.")
            else:
                with st.spinner("🎙️ Samketan AI is cloning your voice in Indian English..."):
                    try:
                        model = load_xtts()
                        
                        # Get speaker embedding from reference audio
                        gpt_lat, spk_emb = model.get_conditioning_latents(
                            audio_path=[st.session_state["ref"]],
                            use_gpt_latents=True  # Preserve speaker characteristics
                        )
                        
                        # Split script into sentences for better prosody
                        parts = [s.strip() for s in 
                                script.replace("!", ".").replace("?", ".").split(".") 
                                if s.strip()]
                        
                        chunks = []
                        progress_bar = st.progress(0)
                        
                        for i, s in enumerate(parts):
                            # Inference with optimized parameters for Indian English
                            out = model.inference(
                                s + ".",
                                lang if lang != "en-IN" else "en",  # XTTS uses "en" for all English variants
                                gpt_lat,
                                spk_emb,
                                temperature=temp,
                                repetition_penalty=rep_pen,
                                speed=speed,
                                top_p=top_p,
                                length_penalty=1.0
                            )
                            chunks.append(np.array(out["wav"]))
                            chunks.append(np.zeros(int(24000 * 0.25)))  # Silence between sentences
                            
                            progress_bar.progress((i + 1) / len(parts))
                        
                        # Concatenate all chunks
                        audio = np.concatenate(chunks)
                        
                        # Normalize output audio
                        audio = audio.astype(np.float32) / np.max(np.abs(audio)) * 0.95
                        
                        # Save to buffer
                        buf = io.BytesIO()
                        sf.write(buf, audio, 24000, format="WAV")
                        st.session_state["gen_audio"] = buf.getvalue()
                        
                        st.success("✅ Humanoid speech generated in Indian English!")
                        st.audio(st.session_state["gen_audio"], format="audio/wav")
                        
                        # Download button
                        st.download_button(
                            label="📥 Download Generated Audio",
                            data=st.session_state["gen_audio"],
                            file_name="samketan_cloned_voice.wav",
                            mime="audio/wav"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Voice Generation Error: {str(e)[:200]}")
                        st.info("**Troubleshooting:**\n- Ensure voice sample is 15-60 seconds\n- Try shorter text (1-2 sentences)\n- Check HF_TOKEN in secrets")

# --- TAB 3: STUDIO MASTER ---
with tab3:
    st.header("🎛️ Background Music & Audio Mixing")
    
    if "gen_audio" not in st.session_state:
        st.warning("Generate voice first in Tab 2.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            bg_music = st.file_uploader("Upload Background Music (WAV/MP3)", type=["wav", "mp3"])
            music_db = st.slider("🔊 Music Level", -30, -6, -16, 1)
        
        with col2:
            duck_db = st.slider("🤫 Voice Ducking", 0, 12, 8, 1,
                              help="Reduce music volume when voice speaks")
            target_lufs = st.slider("🎯 Final Loudness (LUFS)", -24, -12, -16, 1)
        
        if st.button("🎚️ Master Final Output", use_container_width=True):
            if not bg_music:
                st.warning("Upload background music first.")
            else:
                with st.spinner("🎵 Mixing studio-quality audio..."):
                    try:
                        voice = AudioSegment.from_file(
                            io.BytesIO(st.session_state["gen_audio"]), format="wav"
                        )
                        music = AudioSegment.from_file(bg_music)
                        
                        # Frequency separation for better ducking
                        masked = music.low_pass_filter(300).overlay(
                            music.high_pass_filter(4000)
                        ) - 4
                        masked = masked.set_frame_rate(voice.frame_rate)
                        
                        # Match length
                        while len(masked) < len(voice):
                            masked += masked
                        masked = masked[:len(voice)]
                        
                        # Apply ducking and gain
                        masked = masked.apply_gain(music_db - masked.dBFS) - duck_db
                        mix = masked.overlay(voice)
                        
                        # Normalize
                        samples = np.array(mix.get_array_of_samples()).astype(np.float32) / 32768.0
                        if mix.channels == 2:
                            samples = samples.reshape((-1, 2))
                        
                        meter = pyln.Meter(mix.frame_rate)
                        loud = meter.integrated_loudness(samples)
                        norm = pyln.normalize.loudness(samples, loud, target_lufs)
                        
                        out = io.BytesIO()
                        sf.write(out, norm, mix.frame_rate, format="WAV")
                        mixed_audio = out.getvalue()
                        
                        st.success("✅ Audio mixed successfully!")
                        st.audio(mixed_audio, format="audio/wav")
                        
                        st.download_button(
                            label="📥 Download Mixed Audio",
                            data=mixed_audio,
                            file_name="samketan_mixed_output.wav",
                            mime="audio/wav"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Mixing Error: {e}")

# --- TAB 4: ADVANCED SETTINGS ---
with tab4:
    st.header("⚙️ Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Secrets Status")
        if st.secrets.get("HF_TOKEN"):
            st.success("✅ HF_TOKEN configured")
        else:
            st.error("❌ HF_TOKEN not found in secrets")
            st.markdown("""
            **To fix:**
            1. Go to **Streamlit Cloud** > App settings
            2. Click **Secrets**
            3. Add: `HF_TOKEN = "hf_YOUR_KEY_HERE"`
            4. Get token from https://huggingface.co/settings/tokens
            """)
    
    with col2:
        st.subheader("Model Info")
        st.info("""
        **XTTS v2 Details:**
        - Multilingual voice cloning
        - Fast inference (~10s/sentence)
        - Preserves speaker characteristics
        - Works best with 15-60s reference
        """)
    
    st.markdown("---")
    st.markdown("**🎯 Tips for Best Indian English Results:**")
    st.markdown("""
    1. **Reference Audio**: Record natural Indian English with accent intact
    2. **Expressive Speech**: Vary tone, speed, emotion in reference sample
    3. **Text Input**: Use natural phrasing that matches Indian English patterns
    4. **Parameters**: Keep temperature 0.65-0.75 for consistency
    5. **Voice Sample**: 20-40 seconds is optimal (not shorter, not longer)
    """)

st.markdown("---")
st.caption("(c) 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar | v2.0 Indian English Edition")
