import os
import numpy as np
import librosa
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import tempfile
from pydub import AudioSegment
from audiorecorder import audiorecorder

AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe   = r"C:\ffmpeg\bin\ffprobe.exe"
DATASET_PATH = "dataset"
DURATION     = 3

# ─────────────────────────────────────────────────────────
st.set_page_config(page_title="CryNova", page_icon="🍼", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── DESIGN TOKENS ── */
:root {
    --maroon:        #6B1A2A;
    --maroon-deep:   #4A1019;
    --maroon-mid:    #8C2237;
    --maroon-light:  #B85068;
    --maroon-muted:  #D4A0AA;
    --beige:         #F5EFE6;
    --beige-dark:    #EAE0D0;
    --beige-mid:     #D6C9B5;
    --beige-deep:    #C4B49A;
    --cream:         #FBF8F3;
    --text-primary:  #2C1010;
    --text-secondary:#6B4F52;
    --text-muted:    #9E8082;
    --border:        #D6C9B5;
    --border-strong: #B9A898;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text-primary);
}

.stApp {
    background-color: var(--beige);
    min-height: 100vh;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; visibility: hidden; }

.block-container {
    padding: 1.6rem 2.4rem 2rem !important;
    max-width: 100% !important;
}

[data-testid="stVerticalBlock"] { gap: 1.1rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }

/* ────────────── HEADER ────────────── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.6rem;
    background-color: var(--maroon-deep);
    border-radius: 8px;
    margin-bottom: 0.25rem;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 1rem;
}
.header-logo {
    width: 40px; height: 40px;
    background-color: var(--maroon-mid);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.header-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--cream);
    letter-spacing: 0.03em;
    line-height: 1;
}
.header-sub {
    font-size: 0.66rem;
    color: var(--maroon-muted);
    font-weight: 400;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-top: 4px;
}
.header-pills {
    display: flex; gap: 0.4rem;
    flex-wrap: wrap;
    justify-content: flex-end;
    align-items: center;
}
.pill {
    font-size: 0.62rem; font-weight: 600;
    padding: 0.22rem 0.72rem;
    border-radius: 2px;
    letter-spacing: 0.07em;
    text-transform: uppercase;
    white-space: nowrap;
    border: 1px solid rgba(255,255,255,0.15);
    color: var(--beige-mid);
    background-color: rgba(255,255,255,0.07);
}

/* ────────────── STATUS BADGE ── */
.badge-wrap { margin-bottom: 0.1rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    background-color: #EAF3E8;
    border: 1px solid #9EC99B;
    color: #2D5E2A;
    font-weight: 600; font-size: 0.76rem;
    padding: 0.28rem 0.9rem;
    border-radius: 3px;
    letter-spacing: 0.03em;
}
.badge-err {
    background-color: #F5E8E8;
    border-color: #C98989;
    color: #6B2020;
}

/* ────────────── PANEL CARDS ── */
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
    background-color: var(--cream) !important;
    border-radius: 6px !important;
    padding: 1.3rem 1.5rem !important;
    gap: 0.9rem !important;
    border: 1px solid var(--border) !important;
}

/* ────────────── PANEL TITLE ── */
.panel-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--maroon-deep);
    padding-bottom: 0.5rem;
    border-bottom: 1.5px solid var(--maroon-deep);
    margin-bottom: 0.05rem;
    letter-spacing: 0.02em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.panel-icon {
    width: 20px; height: 20px;
    background-color: var(--maroon-deep);
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.6rem; color: var(--cream);
    flex-shrink: 0;
    font-style: normal;
}

/* ────────────── RADIO ── */
.stRadio { margin: 0 !important; }
div[role="radiogroup"] { gap: 1rem !important; margin-top: 0 !important; }
div[role="radiogroup"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}
.stRadio [data-testid="stWidgetLabel"] { display: none !important; }

/* ────────────── FILE UPLOADER ── */
[data-testid="stFileUploader"] > section > div,
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--beige) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: 5px !important;
    padding: 0.65rem 0.9rem !important;
}
[data-testid="stFileUploader"] * {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.77rem !important;
    color: var(--text-secondary) !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background-color: var(--maroon) !important;
    color: var(--cream) !important;
    border: none !important;
    border-radius: 3px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.74rem !important;
    font-weight: 600 !important;
    padding: 0.3rem 0.85rem !important;
}

/* ────────────── AUDIO PLAYER ── */
audio { width: 100%; border-radius: 4px; height: 34px; }

/* ────────────── RESULT CARD ── */
.result-card {
    background-color: var(--maroon-deep);
    border: 1px solid var(--maroon-mid);
    border-radius: 6px;
    padding: 1.3rem 1.5rem;
}
.result-sub {
    font-size: 0.6rem;
    color: var(--maroon-muted);
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-bottom: 0.25rem;
}
.result-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.2rem;
    font-weight: 700;
    color: var(--cream);
    line-height: 1;
    margin-bottom: 0.3rem;
    letter-spacing: 0.05em;
}
.result-meaning {
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--maroon-light);
    margin-bottom: 1rem;
}
.conf-label {
    font-size: 0.65rem;
    font-weight: 500;
    color: var(--beige-mid);
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
    text-transform: uppercase;
}
.conf-track {
    background-color: rgba(255,255,255,0.1);
    border-radius: 2px;
    height: 5px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 2px;
    background-color: var(--maroon-light);
}

/* ────────────── SUGGESTION ── */
.sug-card {
    background-color: var(--beige-dark);
    border: 1px solid var(--border-strong);
    border-left: 3px solid var(--maroon);
    border-radius: 4px;
    padding: 0.9rem 1rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-top: 0.6rem;
}
.sug-icon { font-size: 1.25rem; flex-shrink: 0; margin-top: 2px; }
.sug-title {
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--maroon);
    margin-bottom: 0.2rem;
}
.sug-text {
    font-size: 0.83rem;
    font-weight: 400;
    color: var(--text-primary);
    line-height: 1.55;
}

/* ────────────── EMPTY STATE ── */
.empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 2.6rem 1rem;
    gap: 0.45rem;
}
.empty-icon {
    width: 46px; height: 46px;
    background-color: var(--beige-dark);
    border: 1px solid var(--border);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem;
    margin-bottom: 0.4rem;
}
.empty-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: var(--text-secondary);
}
.empty-text {
    font-size: 0.77rem;
    font-weight: 400;
    color: var(--text-muted);
    line-height: 1.65;
}

/* ────────────── COLUMNS ── */
[data-testid="stHorizontalBlock"] { gap: 18px !important; align-items: stretch !important; }

/* ────────────── FOOTER ── */
.foot {
    display: flex; align-items: center; justify-content: center;
    gap: 0.5rem;
    font-size: 0.62rem; color: var(--text-muted);
    font-weight: 400; padding: 0.5rem 0 0;
    letter-spacing: 0.07em; text-transform: uppercase;
}
.foot-dot { color: var(--maroon-muted); }
</style>
""", unsafe_allow_html=True)


# ── BACKEND ─────────────────────────────────────────────
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=DURATION)
    y = librosa.util.normalize(y)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.nan_to_num(np.mean(pitch))
    energy = np.mean(librosa.feature.rms(y=y))
    features = np.hstack([mfcc_mean, pitch_mean, energy])
    norm = np.linalg.norm(features)
    return features / norm if norm != 0 else features

def load_dataset():
    features, labels = [], []
    label_names = os.listdir(DATASET_PATH)
    label_map = {label: i for i, label in enumerate(label_names)}
    for label in label_names:
        for file in os.listdir(os.path.join(DATASET_PATH, label)):
            try:
                feat = extract_features(os.path.join(DATASET_PATH, label, file))
                features.append(feat); labels.append(label_map[label])
            except:
                continue
    return np.array(features), np.array(labels), label_map

def train_model(X, y):
    m = KNeighborsClassifier(n_neighbors=1)
    m.fit(X, y)
    return m

def predict(file_path, model, label_map):
    feat = extract_features(file_path).reshape(1, -1)
    pred = model.predict(feat)[0]
    dist, _ = model.kneighbors(feat)
    conf = round(100 / (1 + dist[0][0]), 2)
    return {v: k for k, v in label_map.items()}[pred], conf

SUGGESTIONS = {
    "neh":   ("🍼", "Baby may be hungry — try offering a feed."),
    "owh":   ("😴", "Baby seems sleepy — try gentle rocking or swaddling."),
    "eh":    ("💨", "Baby needs to burp — try an upright hold and gentle patting."),
    "eairh": ("🤸", "Possible gas discomfort — try bicycle leg movements."),
    "heh":   ("🌡️", "Baby may be uncomfortable — check diaper or room temperature."),
}
MEANINGS = dict(neh="Hunger", owh="Sleepy", eh="Needs to Burp", eairh="Gas", heh="Discomfort")


# ── LOAD DATA ────────────────────────────────────────────
X, y_arr, label_map = load_dataset()
dataset_ok = len(X) > 0
if dataset_ok:
    model = train_model(X, y_arr)


# ══════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════

# ── HEADER ───────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <div class="header-left">
    <div class="header-logo">🍼</div>
    <div>
      <div class="header-title">CryNova</div>
      <div class="header-sub">Dunstan Baby Language &nbsp;·&nbsp; Cry Classifier</div>
    </div>
  </div>
  <div class="header-pills">
    <span class="pill">NEH · Hunger</span>
    <span class="pill">OWH · Sleepy</span>
    <span class="pill">EH · Burp</span>
    <span class="pill">EAIRH · Gas</span>
    <span class="pill">HEH · Discomfort</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── STATUS ───────────────────────────────────────────────
if dataset_ok:
    st.markdown(
        f'<div class="badge-wrap"><span class="badge">✓ &nbsp;{len(X)} training samples loaded</span></div>',
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="badge-wrap"><span class="badge badge-err">✗ &nbsp;Dataset not found — check the dataset/ folder</span></div>',
        unsafe_allow_html=True
    )
    st.stop()


# ── TWO COLUMNS ──────────────────────────────────────────
col_left, col_right = st.columns(2, gap="medium")

# ── LEFT: Input ──────────────────────────────────────────
with col_left:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-title">
          <span class="panel-icon">♪</span>
          Input Audio
        </div>
        """, unsafe_allow_html=True)

        option = st.radio("input", ["Upload Audio File", "Record Live Audio"],
                          horizontal=True, label_visibility="hidden")

        audio_file_path = None

        if option == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "wav", type=["wav"], label_visibility="hidden"
            )
            if uploaded_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.write(uploaded_file.read())
                audio_file_path = tmp.name
                st.audio(audio_file_path)
        else:
            audio = audiorecorder("🎙  Start Recording", "⏹  Stop Recording")
            if len(audio) > 0:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio.export(tmp.name, format="wav")
                audio_file_path = tmp.name
                st.audio(audio_file_path)

# ── RIGHT: Results ───────────────────────────────────────
with col_right:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-title">
          <span class="panel-icon">◎</span>
          Analysis Result
        </div>
        """, unsafe_allow_html=True)

        if audio_file_path:
            label, confidence = predict(audio_file_path, model, label_map)
            icon, sug_text = SUGGESTIONS.get(label, ("💬", "No suggestion available."))
            conf_int = min(int(confidence), 100)
            meaning  = MEANINGS.get(label, label.title())

            st.markdown(f"""
            <div class="result-card">
              <div class="result-sub">Detected cry type</div>
              <div class="result-label">{label.upper()}</div>
              <div class="result-meaning">{meaning}</div>
              <div class="conf-label">Confidence &mdash; {confidence}%</div>
              <div class="conf-track">
                <div class="conf-fill" style="width:{conf_int}%"></div>
              </div>
            </div>
            <div class="sug-card">
              <span class="sug-icon">{icon}</span>
              <div>
                <div class="sug-title">Suggested Action</div>
                <div class="sug-text">{sug_text}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">👂</div>
              <div class="empty-title">Awaiting Audio</div>
              <div class="empty-text">Upload or record a baby cry<br>to receive an analysis</div>
            </div>
            """, unsafe_allow_html=True)

# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div class="foot">
  CryNova
  <span class="foot-dot">·</span>
  Prototype
  <span class="foot-dot">·</span>
  Dunstan Baby Language Model
  <span class="foot-dot">·</span>
  Built with care for little ones
</div>
""", unsafe_allow_html=True)