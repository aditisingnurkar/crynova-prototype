# CryNova – Baby Cry Classifier Prototype

## Problem Statement

Parents and caregivers often struggle to understand why a baby is crying, especially when infants cannot communicate their needs verbally. CryNova helps bridge this gap by analyzing baby cries and predicting their most likely cause using the Dunstan Baby Language framework.

---

## Overview

CryNova is a web application that classifies infant cries using audio signal processing and a K-Nearest Neighbors (KNN) machine learning model. Users can either upload a `.wav` file or record a baby's cry directly through the browser. The application analyzes the audio, predicts the cry type, and provides a practical care suggestion along with a confidence score.

---

## UI Preview

<p align="center">
  <img src="https://github.com/user-attachments/assets/09c8b2a1-447f-466c-a311-a97ed3a7fdda" width="48%">
  <img src="https://github.com/user-attachments/assets/05c5aec1-3a30-42e9-a014-2980ab372495" width="48%">
</p>

---

## Features

* Upload a `.wav` audio file or record audio directly from the browser.
* Extracts MFCC, Pitch (YIN), and RMS Energy features.
* Classifies cries using a K-Nearest Neighbors (KNN) model.
* Displays the predicted cry type with a confidence score.
* Provides care suggestions based on the detected cry.
* Simple and responsive interface built with Streamlit.

---

## Cry Type Mapping

| Cry Sound | Meaning    | Suggested Action                      |
| --------- | ---------- | ------------------------------------- |
| **NEH**   | Hunger     | Offer a feed.                         |
| **OWH**   | Sleepy     | Rock or swaddle the baby.             |
| **EAIRH** | Gas / Pain | Try bicycle leg movements.            |
| **HEH**   | Discomfort | Check the diaper or room temperature. |

---

## Workflow

```text
Audio Input
      │
      ▼
Feature Extraction
(MFCC + Pitch + RMS)
      │
      ▼
KNN Classification
      │
      ▼
Cry Prediction
      │
      ▼
Suggested Care Action
```

---

## Technologies Used

* Python
* Streamlit
* Scikit-Learn (KNN)
* Librosa
* Pydub
* FFmpeg
* Streamlit Audiorecorder
* HTML & CSS

---

## Installation

Install the required dependencies:

```bash
pip install streamlit numpy librosa scikit-learn pydub st-audiorecorder
```

Install **FFmpeg** separately and ensure it is configured correctly for audio processing.

---

## Running the Application

```bash
streamlit run app.py
```

---

## Key Takeaway

CryNova demonstrates how audio signal processing and machine learning can be applied to classify infant cries and provide caregivers with quick, meaningful insights through a simple and easy-to-use interface.
