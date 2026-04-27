import os
import numpy as np
import librosa
import streamlit as st
from sklearn.neighbors import KNeighborsClassifier
import tempfile
from audiorecorder import audiorecorder

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = "dataset"
DURATION = 3

# -------------------------------
# FEATURE EXTRACTION
# -------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=DURATION)

    # Normalize
    y = librosa.util.normalize(y)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc.T, axis=0)

    # Pitch
    pitch = librosa.yin(y, fmin=50, fmax=300)
    pitch_mean = np.nan_to_num(np.mean(pitch))

    # Energy
    energy = np.mean(librosa.feature.rms(y=y))

    # Combine
    features = np.hstack([mfcc_mean, pitch_mean, energy])

    # Normalize feature vector
    norm = np.linalg.norm(features)
    if norm != 0:
        features = features / norm

    return features

# -------------------------------
# LOAD DATASET
# -------------------------------
def load_dataset():
    features = []
    labels = []

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

# -------------------------------
# TRAIN MODEL
# -------------------------------
def train_model(X, y):
    model = KNeighborsClassifier(n_neighbors=5)  # better for larger dataset
    model.fit(X, y)
    return model

# -------------------------------
# RECORD AUDIO
# -------------------------------
def record_audio():
    audio = audiorecorder("Click to record", "Recording...")

    if len(audio) > 0:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio.export(temp_file.name, format="wav")
        return temp_file.name

    return None

# -------------------------------
# PREDICTION
# -------------------------------
def predict(file_path, model, label_map):
    features = extract_features(file_path)
    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    # Use multiple neighbors for better confidence
    distances, indices = model.kneighbors(features, n_neighbors=5)

    avg_distance = np.mean(distances)

    # smoother confidence
    confidence = max(40, min(95, 100 - avg_distance * 50))

    inv_map = {v: k for k, v in label_map.items()}
    label = inv_map[prediction]

    return label, round(confidence, 2)

# -------------------------------
# SUGGESTIONS
# -------------------------------
def get_suggestion(label):
    suggestions = {
        "neh": "Baby may be hungry. Try feeding.",
        "owh": "Baby may be sleepy. Try soothing.",
        "eh": "Baby may need to burp.",
        "eairh": "Possible gas discomfort.",
        "heh": "Check diaper or temperature."
    }
    return suggestions.get(label, "No suggestion available.")

# -------------------------------
# UI
# -------------------------------
st.title("CryNova - Baby Cry Analyzer")

st.markdown("### Dunstan-inspired Cry Classification System")

# Load dataset
X, y, label_map = load_dataset()

if len(X) == 0:
    st.error("Dataset not loaded. Check folder.")
else:
    st.success(f"Loaded {len(X)} samples")

    model = train_model(X, y)

    st.markdown("---")

    # INPUT SECTION
    st.subheader("Input")

    option = st.radio("Choose input method:", ["Upload Audio", "Record Audio"])

    audio_file_path = None

    col1, col2 = st.columns(2)

    with col1:
        if option == "Upload Audio":
            uploaded_file = st.file_uploader("Upload .wav", type=["wav"])
            if uploaded_file:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(uploaded_file.read())
                audio_file_path = temp_file.name

    with col2:
        if option == "Record Audio":
            audio_file_path = record_audio()

    # PROCESS + OUTPUT
    if audio_file_path:
        st.markdown("### Audio Preview")
        st.audio(audio_file_path)

        label, confidence = predict(audio_file_path, model, label_map)

        st.markdown("---")
        st.subheader("Result")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Detected Cry Type")
            st.success(label.upper())

        with col2:
            st.markdown("#### Confidence")
            st.progress(int(confidence))
            st.write(f"{confidence}%")

        st.markdown("---")
        st.subheader("Suggested Action")
        st.info(get_suggestion(label))
