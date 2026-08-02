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

# Imports and setup
st.set_page_config(page_title="CryNova", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800;900&family=Nunito:wght@400;500;600;700;800&display=swap');

:root {
    --pink:          #F48FB1;
    --pink-deep:     #C2185B;
    --pink-mid:      #EC407A;
    --pink-light:    #FCE4EC;
    --pink-pale:     #FFF0F5;
    --blue:          #81D4FA;
    --blue-deep:     #0277BD;
    --blue-mid:      #29B6F6;
    --blue-light:    #E1F5FE;
    --blue-pale:     #F0FAFF;
    --text-primary:  #1A1A2E;
    --text-secondary:#3D2B4A;
    --text-muted:    #6B5B7A;
    --border-pink:   #F48FB1;
    --border-blue:   #81D4FA;
    --white:         #FFFFFF;
    --surface:       #FFFFFF;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif;
    color: var(--text-primary);
    font-size: 16px;
}

/* Force all streamlit text to be dark — prevent white-on-light issues */
.stApp p, .stApp span, .stApp div, .stApp label,
.stApp .stMarkdown, [data-testid="stText"],
[data-testid="stWidgetLabel"] > div,
[data-testid="stWidgetLabel"] p {
    color: var(--text-primary) !important;
}

.stApp {
    background: linear-gradient(135deg, var(--pink-pale) 0%, var(--blue-pale) 100%);
    min-height: 100vh;
}

#MainMenu, footer, header,
[data-testid="stDecoration"],
[data-testid="stToolbar"] { display: none !important; visibility: hidden; }

.block-container {
    padding: 1.8rem 2.6rem 2rem !important;
    max-width: 100% !important;
}

[data-testid="stVerticalBlock"] { gap: 1.2rem !important; }
[data-testid="stVerticalBlockBorderWrapper"] { padding: 0 !important; }

/* ── HEADER ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 2rem;
    background: linear-gradient(135deg, var(--pink-deep) 0%, var(--blue-deep) 100%);
    border-radius: 12px;
    margin-bottom: 0.5rem;
}
.header-left {
    display: flex;
    align-items: center;
    gap: 1.2rem;
}
.header-logo {
    width: 48px; height: 48px;
    background-color: rgba(255,255,255,0.2);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    flex-shrink: 0;
    border: 2px solid rgba(255,255,255,0.4);
}
.header-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 0.02em;
    line-height: 1;
}
.header-sub {
    font-size: 0.78rem;
    color: rgba(255,255,255,0.8);
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 5px;
}
.header-pills {
    display: flex; gap: 0.5rem;
    flex-wrap: wrap;
    justify-content: flex-end;
    align-items: center;
}
.pill {
    font-size: 0.72rem; font-weight: 700;
    padding: 0.3rem 0.9rem;
    border-radius: 20px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
    border: 2px solid rgba(255,255,255,0.5);
    color: #FFFFFF;
    background-color: rgba(255,255,255,0.15);
}

/* ── STATUS BADGE ── */
.badge-wrap { margin-bottom: 0.2rem; }
.badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background-color: var(--blue-light);
    border: 2px solid var(--blue-mid);
    color: var(--blue-deep);
    font-weight: 700; font-size: 0.85rem;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    letter-spacing: 0.02em;
}
.badge-err {
    background-color: var(--pink-light);
    border-color: var(--pink-mid);
    color: var(--pink-deep);
}

/* ── PANEL CARDS ── */
[data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] {
    background-color: var(--white) !important;
    border-radius: 12px !important;
    padding: 1.6rem 1.8rem !important;
    gap: 1rem !important;
    border: 2px solid var(--border-pink) !important;
    box-shadow: 0 4px 20px rgba(244,143,177,0.12) !important;
}

/* ── PANEL TITLE ── */
.panel-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text-primary);
    padding-bottom: 0.6rem;
    border-bottom: 2.5px solid var(--pink-mid);
    margin-bottom: 0.1rem;
    letter-spacing: 0.01em;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.panel-icon {
    width: 24px; height: 24px;
    background: linear-gradient(135deg, var(--pink-mid), var(--blue-mid));
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.65rem; color: #FFFFFF;
    flex-shrink: 0;
    font-style: normal;
    font-weight: 800;
}

/* ── RADIO ── */
.stRadio { margin: 0 !important; }
div[role="radiogroup"] { gap: 1.2rem !important; margin-top: 0 !important; }
div[role="radiogroup"] label {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.95rem !important;
    font-weight: 700 !important;
    color: var(--text-primary) !important;
}
.stRadio [data-testid="stWidgetLabel"] { display: none !important; }

/* ── FILE UPLOADER ── */
[data-testid="stFileUploader"] > section > div,
[data-testid="stFileUploaderDropzone"] {
    background-color: var(--blue-pale) !important;
    border: 2px dashed var(--blue-mid) !important;
    border-radius: 8px !important;
    padding: 0.8rem 1rem !important;
}
[data-testid="stFileUploader"] * {
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: linear-gradient(135deg, var(--pink-mid), var(--blue-mid)) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 20px !important;
    font-family: 'Nunito', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    padding: 0.35rem 1rem !important;
}

/* ── AUDIO PLAYER ── */
audio { width: 100%; border-radius: 6px; height: 36px; }

/* ── RESULT CARD ── */
.result-card {
    background: linear-gradient(135deg, var(--pink-light) 0%, var(--blue-light) 100%);
    border: 2px solid var(--pink);
    border-radius: 12px;
    padding: 1.6rem 1.8rem;
}
.result-sub {
    font-size: 0.72rem;
    color: var(--pink-deep);
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.18em;
    margin-bottom: 0.3rem;
}
.result-label {
    font-family: 'Playfair Display', serif;
    font-size: 3.8rem;
    font-weight: 900;
    color: var(--pink-deep);
    line-height: 1;
    margin-bottom: 0.35rem;
    letter-spacing: 0.06em;
}
.result-meaning {
    font-size: 0.95rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--blue-deep);
    margin-bottom: 1.2rem;
}
.conf-label {
    font-size: 0.78rem;
    font-weight: 700;
    color: #1A1A2E;
    letter-spacing: 0.06em;
    margin-bottom: 0.4rem;
    text-transform: uppercase;
}
.conf-track {
    background-color: rgba(0,0,0,0.1);
    border-radius: 4px;
    height: 7px;
    overflow: hidden;
}
.conf-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--pink-mid), var(--blue-mid));
}

/* ── SUGGESTION ── */
.sug-card {
    background-color: var(--pink-light);
    border: 2px solid var(--border-pink);
    border-left: 5px solid var(--pink-mid);
    border-radius: 8px;
    padding: 1.1rem 1.2rem;
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-top: 0.8rem;
}
.sug-title {
    font-size: 0.7rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--pink-deep);
    margin-bottom: 0.25rem;
}
.sug-text {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--text-primary);
    line-height: 1.6;
}

/* ── EMPTY STATE ── */
.empty {
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    text-align: center; padding: 3rem 1rem;
    gap: 0.5rem;
}
.empty-icon {
    width: 56px; height: 56px;
    background: linear-gradient(135deg, var(--pink-light), var(--blue-light));
    border: 2px solid var(--border-pink);
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    font-weight: 900;
    color: var(--pink-deep);
    margin-bottom: 0.5rem;
}
.empty-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--text-secondary);
}
.empty-text {
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-muted);
    line-height: 1.7;
}

/* ── COLUMNS ── */
[data-testid="stHorizontalBlock"] { gap: 20px !important; align-items: stretch !important; }

/* ── FOOTER ── */
.foot {
    display: flex; align-items: center; justify-content: center;
    gap: 0.6rem;
    font-size: 0.75rem; color: var(--text-muted);
    font-weight: 700; padding: 0.6rem 0 0;
    letter-spacing: 0.08em; text-transform: uppercase;
}
.foot-dot { color: var(--pink-mid); }
</style>
""", unsafe_allow_html=True)


# Backend
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
    label_names = sorted(os.listdir(DATASET_PATH))
    label_map = {label: i for i, label in enumerate(label_names)}

    for label in label_names:
        folder_path = os.path.join(DATASET_PATH, label)
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            try:
                feat = extract_features(file_path)
                features.append(feat)
                labels.append(label_map[label])
            except:
                continue

    return np.array(features), np.array(labels), label_map


def train_model(X, y):
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X, y)
    return model


def predict(file_path, model, label_map):
    feat = extract_features(file_path).reshape(1, -1)
    pred = model.predict(feat)[0]

    distances, _ = model.kneighbors(feat, n_neighbors=5)
    avg_distance = np.mean(distances)

    confidence = round(max(40, min(95, 100 - avg_distance * 50)), 2)

    inv_map = {v: k for k, v in label_map.items()}
    return inv_map[pred], confidence


# Cry type mappings
SUGGESTIONS = {
    "neh":   ("Baby may be hungry — try offering a feed."),
    "owh":   ("Baby seems sleepy — try gentle rocking or swaddling."),
    "eairh": ("Possible gas discomfort — try bicycle leg movements."),
    "heh":   ("Baby may be uncomfortable — check diaper or room temperature."),
}
MEANINGS = dict(neh="Hunger", owh="Sleepy", eairh="Pain", heh="Discomfort")


# LOAD DATA
X, y_arr, label_map = load_dataset()
dataset_ok = len(X) > 0
if dataset_ok:
    model = train_model(X, y_arr)


# User interface

# Header
st.markdown("""
<div class="app-header">
  <div class="header-left">
    <div class="header-logo">N</div>
    <div>
      <div class="header-title">CryNova</div>
      <div class="header-sub">Dunstan Baby Language &nbsp;&middot;&nbsp; Cry Classifier</div>
    </div>
  </div>
  <div class="header-pills">
    <span class="pill">NEH &middot; Hunger</span>
    <span class="pill">OWH &middot; Sleepy</span>
    <span class="pill">EAIRH &middot; Gas</span>
    <span class="pill">HEH &middot; Discomfort</span>
  </div>
</div>
""", unsafe_allow_html=True)

# Layout
col_left, col_right = st.columns(2, gap="medium")

# Input section
with col_left:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-title">
          <span class="panel-icon">A</span>
          Input Audio
        </div>
        """, unsafe_allow_html=True)

        option = st.radio("input", ["Upload Audio File", "Record Live Audio"],
                          horizontal=True, label_visibility="hidden")

        audio_file_path = None

        if option == "Upload Audio File":
            uploaded_file = st.file_uploader("wav", type=["wav"], label_visibility="hidden")
            if uploaded_file:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                tmp.write(uploaded_file.read())
                audio_file_path = tmp.name
                st.audio(audio_file_path)
        else:
            audio = audiorecorder("Start Recording", "Stop Recording")
            if len(audio) > 0:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                audio.export(tmp.name, format="wav")
                audio_file_path = tmp.name
                st.audio(audio_file_path)

# Results section
with col_right:
    with st.container(border=True):
        st.markdown("""
        <div class="panel-title">
          <span class="panel-icon">R</span>
          Analysis Result
        </div>
        """, unsafe_allow_html=True)

        if audio_file_path:
            label, confidence = predict(audio_file_path, model, label_map)
            sug_text = SUGGESTIONS.get(label, "No suggestion available.")
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
              <div>
                <div class="sug-title">Suggested Action</div>
                <div class="sug-text">{sug_text}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="empty">
              <div class="empty-icon">~</div>
              <div class="empty-title">Awaiting Audio</div>
              <div class="empty-text">Upload or record a baby cry<br>to receive an analysis</div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="foot">
  CryNova
  <span class="foot-dot">&middot;</span>
  Prototype
  <span class="foot-dot">&middot;</span>
  Dunstan Baby Language Model
  <span class="foot-dot">&middot;</span>
  Built with care for little ones
</div>
""", unsafe_allow_html=True)