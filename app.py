import os
import numpy as np
import librosa
import streamlit as st
import sounddevice as sd
from scipy.io.wavfile import write
from sklearn.neighbors import KNeighborsClassifier
import tempfile

# -------------------------------
# CONFIG
# -------------------------------
DATASET_PATH = "dataset"
SAMPLE_RATE = 22050
DURATION = 3

# -------------------------------
# FEATURE EXTRACTION (BLOCK 3 FINAL)
# -------------------------------
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=DURATION)

    # Normalize audio
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

    label_names = os.listdir(DATASET_PATH)
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
    model = KNeighborsClassifier(n_neighbors=1)  # better for small data
    model.fit(X, y)
    return model

# -------------------------------
# RECORD AUDIO
# -------------------------------
def record_audio():
    st.write("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    write(temp_file.name, SAMPLE_RATE, recording)

    return temp_file.name

# -------------------------------
# PREDICTION
# -------------------------------
def predict(file_path, model, label_map):
    features = extract_features(file_path)
    features = features.reshape(1, -1)

    prediction = model.predict(features)[0]

    # confidence (distance-based)
    distances, _ = model.kneighbors(features)
    confidence = round(100 / (1 + distances[0][0]), 2)

    inv_map = {v: k for k, v in label_map.items()}
    label = inv_map[prediction]

    return label, confidence

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
st.title("CryNova - Dunstan Cry Prototype")

st.write("Prototype system for classifying baby cries into Dunstan-inspired categories.")

# Load dataset
X, y, label_map = load_dataset()

if len(X) == 0:
    st.error("Dataset not loaded. Check dataset folder.")
else:
    st.success(f"Loaded {len(X)} samples")

    model = train_model(X, y)

    st.subheader("Choose Input Method")

    option = st.radio("Input type:", ["Upload Audio", "Record Audio"])

    audio_file_path = None

    if option == "Upload Audio":
        uploaded_file = st.file_uploader("Upload .wav file", type=["wav"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.write(uploaded_file.read())
            audio_file_path = temp_file.name
            st.audio(audio_file_path)

    elif option == "Record Audio":
        if st.button("Record"):
            audio_file_path = record_audio()
            st.audio(audio_file_path)

    if audio_file_path:
        label, confidence = predict(audio_file_path, model, label_map)

        st.subheader("Result")
        st.write(f"Detected: {label.upper()}")
        st.write(f"Confidence: {confidence}%")

        st.subheader("Suggestion")
        st.write(get_suggestion(label))
