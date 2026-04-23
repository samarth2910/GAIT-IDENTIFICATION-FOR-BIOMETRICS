# 🚶‍♂️ Gait Recognition System (Open-Set Biometric Identification)

A deep learning–based gait recognition system that identifies individuals from walking patterns using pose estimation and temporal modeling.
The system is designed for **open-set recognition**, meaning it can also detect **unknown / intruder subjects** and perform **clinical biomechanical analysis**.

---

## 📸 Screenshots

### ❌ Unknown / Intruder Detection
![Unknown Detection](assets/Screenshot%20(205).png)
> System rejects an unrecognised subject — entropy too high (0.626 > 0.600), flagged as **Unknown / Intruder**

---

### 🏥 Poor Gait Health Score
![Poor Health Score](assets/Screenshot%20(253).png)
> System detects significant gait abnormalities and assigns a low Gait Health Score

---

### 📈 Temporal Gait Analysis
![Temporal Analysis](assets/Screenshot%20(254).png)
> Joint angle trajectories over time for all 7 biomechanical features across the walking video

---

### 🦿 Biomechanical Radar + Symmetry
![Radar and Symmetry](assets/Screenshot%20(255).png)
> Deviation-from-normal radar chart alongside the left–right gait symmetry index

---

## 📌 Overview

This project uses **pose-based gait features** extracted from video and feeds them into an **LSTM neural network** for classification.
To improve real-world robustness, a **multi-stage decision system** is applied using:

- Entropy-based uncertainty rejection
- Confidence thresholding
- Margin-based separation
- Weighted voting across sequences
- Sequence quality filtering

---

## 🧠 Key Features

- ✅ **Deep Learning Model (LSTM)** for temporal gait analysis
- ✅ **Open-set recognition** (detects unknown / intruder people)
- ✅ **Entropy-based uncertainty filtering**
- ✅ **Weighted voting across sequences**
- ✅ **Sequence quality filtering (noise removal)**
- ✅ **Per-identity adaptive thresholds**
- ✅ **Biomechanical gait analysis** (symmetry, ROM, clinical range flags)
- ✅ **Gait abnormality detection** (crouch gait, foot drop, Trendelenburg, etc.)
- ✅ **Gait Health Score (0–100%)**
- ✅ **Skeleton overlay video export**
- ✅ **Downloadable JSON analysis report**
- ✅ **Persistent audit log with CSV export**
- ✅ **Real-time Streamlit interface**

---

## 🏗️ System Architecture

```
Video Input
   ↓
MediaPipe Pose Extraction (33 landmarks)
   ↓
Feature Engineering (7 joint-angle gait features)
   ↓
Sliding-Window Sequence Generation (len=60, stride=6)
   ↓
Sequence Quality Filtering
   ↓
LSTM Model (Deep Learning)
   ↓
Decision Engine:
   - Margin Gate
   - Entropy Gate
   - Confidence Gate
   - Agreement Gate
   ↓
Final Output (Identity / Unknown)
   +
Biomechanical Analysis & Abnormality Detection
```

---

## 📊 Features Used

The model uses **7 biomechanical gait features**:

| # | Feature                    | Description             |
|---|----------------------------|-------------------------|
| 1 | Right knee flexion         | Hip–Knee–Ankle angle    |
| 2 | Left knee flexion          | Hip–Knee–Ankle angle    |
| 3 | Right hip flexion          | Shoulder–Hip–Knee angle |
| 4 | Left hip flexion           | Shoulder–Hip–Knee angle |
| 5 | Right ankle dorsiflexion   | Knee–Ankle–Foot angle   |
| 6 | Left ankle dorsiflexion    | Knee–Ankle–Foot angle   |
| 7 | Torso tilt                 | Body lean from vertical |

---

## 🔍 Decision Logic (Open-Set Detection)

The system applies **4 cascaded rejection gates**:

1. **Margin Gate** — Ensures separation between top-2 predicted classes
2. **Entropy Gate** — Rejects flat/uncertain softmax distributions
3. **Confidence Gate** — Ensures minimum prediction strength
4. **Agreement Gate** — Ensures consistency across all sequences

Per-identity threshold overrides are supported for fine-grained tuning.

---

## 🦿 Biomechanical Analysis

- Left–right symmetry index per joint pair
- Range of motion (ROM) per feature
- Clinical range flagging (normal walking norms)
- Radar chart deviation profile
- Temporal joint angle trajectories

**Detected Abnormality Patterns:**

| Pattern | Type |
|---|---|
| Crouch gait | Bilateral |
| Stiff-knee gait | Bilateral |
| Bilateral foot drop | Bilateral |
| Vaulting / equinus gait | Bilateral |
| Hip flexion deficit | Bilateral |
| Trendelenburg / trunk sway | Bilateral |
| Antalgic (pain-avoidance) gait | Asymmetry |
| Right/Left-sided foot drop | Unilateral |
| Right/Left-sided knee stiffness | Unilateral |

---

## 🧪 Technologies Used

- Python 3.11
- TensorFlow / Keras
- OpenCV
- MediaPipe
- NumPy / SciPy / Pandas
- Plotly
- Streamlit

---

## 🚀 Installation

```bash
git clone https://github.com/samarth2910/GAIT-IDENTIFICATION-FOR-BIOMETRICS.git
cd GAIT-IDENTIFICATION-FOR-BIOMETRICS
pip install -r requirements.txt
```

---

## ⚙️ Step 1 — Train the Model

Prepare your video dataset in this structure:

```
videos1/
  train/
    PersonA_1.mp4
    PersonA_2.mp4
    PersonB_1.mp4
  test/
    PersonA_3.mp4
    PersonB_3.mp4
```

> Filename format: `IdentityName_videoNumber.mp4`

Then run:

```bash
python ga.py
```

This will:
- Extract 7 joint-angle features from each video using MediaPipe
- Train the LSTM model
- Save model files to `models/`
- Evaluate on test videos and print accuracy + confusion matrix

---

## ▶️ Step 2 — Run the App

```bash
streamlit run st_enhanced.py
```

Then:
- Upload a walking video (MP4, AVI, MOV)
- System identifies the person or flags as **Unknown / Intruder**
- View biomechanical analysis, abnormality detection, and health score
- Download JSON report or skeleton overlay video

---

## 📁 Project Structure

```
├── models/                        # Trained model files (not in Git — generate via ga.py)
│   ├── gait_lstm_videos1_baseline.keras
│   ├── label_encoder_videos1_baseline.pkl
│   └── norm_videos1_baseline.pkl
├── videos1/                       # Training/test videos (not in Git)
│   ├── train/
│   └── test/
├── assets/                        # Screenshots for README
├── ga.py                          # Model training & evaluation script
├── st_enhanced.py                 # Streamlit web app
├── requirements.txt               # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚠️ Notes

- Model files (`.keras`, `.pkl`) are **not included** — run `ga.py` to generate them
- Dataset videos are **excluded** from the repository
- Ensure the person walks clearly and fully in frame for best results
- Minimum ~5 seconds of clear walking video recommended

---

## 📈 Future Improvements

- BiLSTM / Attention-based models
- Multi-view gait recognition
- Real-time CCTV integration
- Larger dataset for better generalization
- Gait-based health monitoring over time

---

## 👨‍💻 Author

**Samarth Shetty**

---

⭐ If you found this useful, consider starring the repo!
