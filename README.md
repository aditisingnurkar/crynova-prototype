CryNova: Baby Cry Classifier Prototype
CryNova is an intelligent, empathetic web application built to analyze infant cries and translate them into actionable insights for parents and caregivers.
Based on the Dunstan Baby Language framework, the application utilizes audio signal processing and machine learning to identify specific acoustic patterns
(reflexes) that signal a baby's core physical needs.

Developed as a winning concept for the Prototype Competition 2026, CryNova features a highly scannable, minimalist dashboard utilizing a soft, pastel aesthetic
to provide clarity at a glance for stressed parents.

| Feature                         | Description                                                                                                                                                                             
| ------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
| Dual Input Modes                | Upload a pre-recorded `.wav` audio file or record a baby's cry live directly through the web browser.                                                                                   
| Acoustic Feature Extraction     | Extracts Mel-Frequency Cepstral Coefficients (MFCCs), fundamental frequency (pitch using the YIN algorithm), and Root Mean 
|                                 |  Square (RMS) energy to create a comprehensive audio profile. 
| Machine Learning Classifier     | Employs a localized K-Nearest Neighbors (KNN) model trained dynamically on pre-categorized infant vocalizations to classify
|                                 | different types of baby cries.                              |
| Intuitive Visual Feedback       | Dynamic confidence trackers and clean typography make results easy to interpret, even in high-stress situations.                                                                        
| Actionable Guidance             | Provides instant, practical pediatric suggestions based on the detected cry type, such as swaddling, feeding, burping, or 
|                                 | bicycle leg exercises.                                        


## Dunstan Baby Language Mapping

| Cry Sound | Meaning | Suggested Action |
|-----------|---------|------------------|
| NEH       | Hunger  | Offer a feed.    |
| OWH       | Sleepy  | Rock or swaddle  |
|                     |  the baby.       |
| EAIRH     | Gas/Pain| Try bicycle leg  |
|                     |  movements.      |
| HEH       | Discomf | Check diaper     |

## Technical Architecture & Stack

| Layer                 | Technologies                              | Purpose 
| **Frontend & UI**     | Streamlit, Custom CSS/HTML, Audiorecorder | Provides an interactive web interface, custom styling, and browser-based audio recording. 
| **Audio Processing**  | Librosa, Pydub, FFmpeg                    | Handles audio preprocessing, normalization, feature extraction, and format conversion. 
| **Machine Learning**  | Scikit-Learn (KNN)                        | Classifies infant cries based on extracted acoustic features. |

### Technology Details

| Technology          | Role 
| **Streamlit**       | Builds a responsive, single-page web application with minimal overhead. 
| **Custom CSS/HTML** | Creates a clean, modern interface using custom layouts, typography, and pastel color themes. 
| **Audiorecorder**   | Enables real-time baby cry recording directly within the browser. 
| **Librosa**         | Extracts MFCCs, pitch (YIN), and RMS energy features from audio samples. 
| **Scikit-Learn**    | Implements the K-Nearest Neighbors (KNN) classifier for cry-type prediction. 
| **Pydub**           | Simplifies audio manipulation and processing workflows. 
| **FFmpeg**          | Supports reliable audio conversion and export across different formats and platforms. 

Getting Started
1. Prerequisites
Ensure you have Python 3.8+ installed on your system. You will also need to download and install FFmpeg on your local machine to handle automated audio conversions.

2. Directory Structure
For the code to initialize correctly and train its model, prepare your directory layout exactly as follows:

📁 CryNova/
│
├── 📄 app.py               # Main application source code
├── 📄 README.md            # Documentation
│
└── 📁 dataset/             # Training data folder
    ├── 📁 neh/             # .wav samples of hunger cries
    ├── 📁 owh/             # .wav samples of sleepy cries
    ├── 📁 eairh/           # .wav samples of gas/pain cries
    └── 📁 heh/             # .wav samples of discomfort cries

Installation
Clone or copy the source code file into your workspace as app.py.

Install the necessary Python packages using pip:

pip install streamlit numpy librosa scikit-learn pydub st-audiorecorder

Running the Application
Launch the local web development server by executing:
streamlit run app.py


