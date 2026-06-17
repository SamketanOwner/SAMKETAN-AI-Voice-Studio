import streamlit as st
import io, os, numpy as np, soundfile as sf
import pyloudnorm as pyln
from pydub import AudioSegment
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.manage import ModelManager

# --- MODEL LOADER (cached so it loads once) ---
@st.cache_resource
def load_xtts():
    name = "tts_models/multilingual/multi-dataset/xtts_v2"
    path = ModelManager().download_model(name)[0]
    cfg = XttsConfig(); cfg.load_json(os.path.join(path, "config.json"))
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=path, eval=True)
    return model

# --- BRANDING & UI ---
st.set_page_config(page_title="SAMKETAN AI-Voice-Studio", layout="wide")
st.sidebar.title("SAMKETAN AI")
st.sidebar.info("Proprietor: Sanjay Kumar\n📍 Kalyana Karnataka Region")
st.sidebar.write("Project: AI development for Bhuvi")

st.title("🎙️ SAMKETAN AI-Voice-Studio")
st.subheader("Humanoid Voice Cloning & Production")

tab1, tab2, tab3 = st.tabs(["🎤 Clone Your Voice", "✍️ Text-to-Voice", "🎼 Studio Master"])

# --- TAB 1: VOICE SAMPLE ---
with tab1:
    st.header("Upload Your Voice Sample")
    st.caption("Record 20–40s speaking EXPRESSIVELY (varied pitch/emotion). "
               "A flat sample produces a flat voice. Quiet room, WAV/FLAC, 44.1–48kHz.")
    consent = st.checkbox("I confirm I am authorized to clone this voice (consent_flag).")
    sample = st.file_uploader("Voice sample", type=["wav", "flac", "mp3"])
    if sample:
        with open("ref.wav", "wb") as f:
            f.write(sample.read())
        st.session_state["ref"] = "ref.wav"
        st.audio("ref.wav")
        st.success("Sample stored. Go to Tab 2.")

# --- TAB 2: GENERATE ---
with tab2:
    st.header("Generate Speech in Your Cloned Voice")
    script = st.text_area("What should the voice say?", height=150,
                          placeholder="Enter your business script here...")
    lang = st.selectbox("Language", ["en", "hi", "es", "fr", "de", "it"])

    st.subheader("Modulation / Variation Controls")
    temp = st.slider("Expressiveness (temperature)", 0.5, 0.95, 0.75, 0.05,
                     help="Higher = more natural human variation, less robotic.")
    rep_pen = st.slider("Repetition penalty", 1.0, 10.0, 5.0, 0.5,
                        help="Keeps prosody from going monotone.")
    speed = st.slider("Speaking rate", 0.8, 1.2, 1.0, 0.05)

    if st.button("Generate High-Quality Voice"):
        if not consent:
            st.warning("Please confirm consent in Tab 1.")
        elif not st.session_state.get("ref"):
            st.warning("Please upload a voice sample in Tab 1 first.")
        elif not script.strip():
            st.warning("Please enter text first.")
        else:
            with st.spinner("Samketan AI is cloning your voice with natural variation..."):
                try:
                    model = load_xtts()
                    gpt_lat, spk_emb = model.get_conditioning_latents(
                        audio_path=[st.session_state["ref"]])
                    # sentence-by-sentence for natural prosody resets
                    parts = [s.strip() for s in script.replace("!", ".").replace("?", ".").split(".") if s.strip()]
                    chunks = []
                    for s in parts:
                        out = model.inference(
                            s + ".", lang, gpt_lat, spk_emb,
                            temperature=temp,
                            repetition_penalty=rep_pen,
                            speed=speed,
                            enable_text_splitting=True)
                        chunks.append(np.array(out["wav"]))
                        chunks.append(np.zeros(int(24000 * 0.25)))  # natural pause
                    audio = np.concatenate(chunks)
                    buf = io.BytesIO()
                    sf.write(buf, audio, 24000, format="WAV")
                    st.session_state["gen_audio"] = buf.getvalue()
                    st.success("Humanoid speech generated!")
                    st.audio(st.session_state["gen_audio"])
                    st.download_button("Download Voice", st.session_state["gen_audio"], "samketan_voice.wav")
                except Exception as e:
                    st.error(f"Detailed Engine Error: {e}")

# --- TAB 3: MIXING & MASTERING ---
with tab3:
    st.header("Background Music & Mixing")
    bg_music = st.file_uploader("Upload Music (MP3/WAV)", type=["wav", "mp3"])
    music_db = st.slider("Music level (dB, mild = -16)", -30, -6, -16)
    duck_db = st.slider("Ducking amount (dB)", 0, 12, 8,
                        help="How much to lower music while voice is speaking.")
    target_lufs = st.slider("Final loudness (LUFS)", -24, -12, -16)

    if st.button("Master Final Output"):
        if "gen_audio" not in st.session_state or not bg_music:
            st.warning("Generate the voice in Tab 2 and upload music here first.")
        else:
            with st.spinner("Mixing studio quality audio..."):
                try:
                    voice = AudioSegment.from_file(io.BytesIO(st.session_state["gen_audio"]), format="wav")
                    music = AudioSegment.from_file(bg_music)
                    # EQ mask: carve 300-4000 Hz band where voice sits, then attenuate
                    masked = music.low_pass_filter(300).overlay(music.high_pass_filter(4000)) - 4
                    masked = masked.set_frame_rate(voice.frame_rate)
                    # loop/trim music to match voice length
                    while len(masked) < len(voice):
                        masked += masked
                    masked = masked[:len(voice)]
                    # normalize music to target level, then duck under voice
                    masked = masked.apply_gain(music_db - masked.dBFS) - duck_db
                    mix = masked.overlay(voice)
                    # LUFS loudness normalization
                    samples = np.array(mix.get_array_of_samples()).astype(np.float32) / 32768.0
                    if mix.channels == 2:
                        samples = samples.reshape((-1, 2))
                    meter = pyln.Meter(mix.frame_rate)
                    loud = meter.integrated_loudness(samples)
                    norm = pyln.normalize.loudness(samples, loud, target_lufs)
                    out = io.BytesIO()
                    sf.write(out, norm, mix.frame_rate, format="WAV")
                    st.audio(out.getvalue())
                    st.download_button("Download Mastered Mix", out.getvalue(), "samketan_master.wav")
                except Exception as e:
                    st.error(f"Mixing Error: {e}")

st.markdown("---")
st.caption("© 2026 SAMKETAN AI | Proprietary Business Solution by Sanjay Kumar")
