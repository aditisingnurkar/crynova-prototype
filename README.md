# CryNova – Baby Cry Classifier Prototype

## Problem Statement

Interpreting why a baby is crying is often challenging, especially for first-time parents and caregivers. Since infants cannot communicate verbally, identifying their needs usually involves trying multiple solutions before finding the right one.

CryNova aims to make this process easier by analyzing infant cries and predicting their likely cause using the Dunstan Baby Language framework. By combining audio signal processing with machine learning, the application provides quick predictions along with simple care suggestions.

---

## Overview

CryNova is a web application that classifies baby cries based on their acoustic characteristics. It supports both uploaded audio files and live recordings, extracts relevant audio features, and predicts the most likely reason for the cry using a K-Nearest Neighbors (KNN) classifier.

The project was developed as a prototype for the **Prototype Competition 2026** and focuses on providing a simple, accessible interface that can assist parents and caregivers in understanding an infant's immediate needs.

---

## UI Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/09c8b2a1-447f-466c-a311-a97ed3a7fdda" width="48%">
  <img src="https://github.com/user-attachments/assets/05c5aec1-3a30-42e9-a014-2980ab372495" width="48%">
</p>

---

## Features

* Upload a `.wav` audio file or record a baby's cry directly through the browser.
* Extracts acoustic features including:

  * Mel-Frequency Cepstral Coefficients (MFCCs)
  * Fundamental frequency (Pitch) using the YIN algorithm
  * Root Mean Square (RMS) Energy
* Classifies infant cries using a K-Nearest Neighbors (KNN) machine learning model.
* Displays the predicted cry type along with a confidence score.
* Provides practical care suggestions based on the detected cry.
* Clean and responsive interface built with Streamlit and custom CSS.

---

## Cry Type Mapping

| Cry Sound | Meaning    | Suggested Action                      |
| --------- | ---------- | ------------------------------------- |
| **NEH**   | Hunger     | Offer a feed.                         |
| **OWH**   | Sleepy     | Rock or swaddle the baby.             |
| **EAIRH** | Gas / Pain | Try bicycle leg movements.            |
| **HEH**   | Discomfort | Check the diaper or room temperature. |

---

## How It Works

```text
Audio Input
      │
      ▼
Audio Preprocessing
      │
      ▼
Feature Extraction
(MFCC + Pitch + RMS)
      │
      ▼
KNN Classification
      │
      ▼
Cry Type Prediction
      │
      ▼
Confidence Score
      │
      ▼
Suggested Care Action
```

---

## Technologies Used

| Category             | Technologies            |
| -------------------- | ----------------------- |
| Programming Language | Python                  |
| Web Framework        | Streamlit               |
| UI Design            | Custom CSS, HTML        |
| Audio Processing     | Librosa, Pydub, FFmpeg  |
| Machine Learning     | Scikit-Learn (KNN)      |
| Audio Recording      | Streamlit Audiorecorder |

---

## Project Structure

```text
CryNova/
│
├── app.py
├── README.md
│
└── dataset/
    ├── neh/
    ├── owh/
    ├── eairh/
    └── heh/
```

The `dataset` folder contains categorized infant cry samples used to train the KNN classifier during application startup.

---

## Installation

Clone the repository and install the required Python packages:

```bash
pip install streamlit numpy librosa scikit-learn pydub st-audiorecorder
```

Install **FFmpeg** separately and ensure it is correctly configured on your system, as it is required for audio processing and conversion.

---

## Running the Application

Start the Streamlit server by running:

```bash
streamlit run app.py
```

The application will launch in your default web browser.

---

## Technical Details

The application extracts three primary acoustic features from each audio sample:

* **MFCCs (Mel-Frequency Cepstral Coefficients):** Capture the spectral characteristics of the baby's cry.
* **Pitch (YIN Algorithm):** Estimates the fundamental frequency of the audio signal.
* **RMS Energy:** Measures the intensity of the cry.

These features are combined into a single feature vector and passed to a **K-Nearest Neighbors (KNN)** classifier, which predicts the most likely cry category. Based on the prediction, the application displays a confidence score and an appropriate care suggestion.

---

## Key Takeaway

CryNova demonstrates how audio signal processing and machine learning can be applied to a real-world healthcare problem. By analyzing infant cry patterns and mapping them to common physical needs, the application provides caregivers with quick insights through a simple and user-friendly interface.
