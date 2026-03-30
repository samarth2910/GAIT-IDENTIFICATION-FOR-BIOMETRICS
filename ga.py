import os
import cv2
import pickle
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

# =====================================================
# PATHS & CONFIG (UPDATED)
# =====================================================

TRAIN_DIR = "videos1/train"
TEST_DIR  = "videos1/test"
MODEL_DIR = "models"

MODEL_PATH   = os.path.join(MODEL_DIR, "gait_lstm_videos1_baseline.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_videos1_baseline.pkl")
NORM_PATH    = os.path.join(MODEL_DIR, "norm_videos1_baseline.pkl")

VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv")

SEQ_LEN = 60
STRIDE  = 6
CONFIDENCE_THRESHOLD = 0.31

os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# GEOMETRIC FEATURES
# =====================================================

def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    deg = abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg

def torso_tilt(ls, rs, lh, rh):
    shoulder = (np.array(ls) + np.array(rs)) / 2
    hip = (np.array(lh) + np.array(rh)) / 2
    v = shoulder - hip
    vertical = np.array([0, -1])
    cos = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6)
    cos = np.clip(cos, -1, 1)
    return np.degrees(np.arccos(cos))

# =====================================================
# SEQUENCE EXTRACTION
# =====================================================

def extract_sequences(video_path, seq_len=SEQ_LEN, stride=STRIDE):
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return []

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not res.pose_landmarks:
            continue

        lm = res.pose_landmarks.landmark
        P = mp_pose.PoseLandmark
        def p(i): return [lm[i].x, lm[i].y]

        try:
            rs,rh,rk,ra,rf = p(P.RIGHT_SHOULDER),p(P.RIGHT_HIP),p(P.RIGHT_KNEE),p(P.RIGHT_ANKLE),p(P.RIGHT_FOOT_INDEX)
            ls,lh,lk,la,lf = p(P.LEFT_SHOULDER), p(P.LEFT_HIP), p(P.LEFT_KNEE), p(P.LEFT_ANKLE), p(P.LEFT_FOOT_INDEX)

            frames.append([
                angle(rh,rk,ra),
                angle(lh,lk,la),
                angle(rs,rh,rk),
                angle(ls,lh,lk),
                angle(rk,ra,rf),
                angle(lk,la,lf),
                torso_tilt(ls,rs,lh,rh)
            ])
        except:
            continue

    cap.release()

    sequences = []
    for i in range(0, len(frames) - seq_len, stride):
        sequences.append(frames[i:i+seq_len])

    return sequences

# =====================================================
# TRAIN MODEL (NEW MODEL ONLY)
# =====================================================

if not (os.path.exists(MODEL_PATH) and os.path.exists(ENCODER_PATH) and os.path.exists(NORM_PATH)):

    print("\n🚀 Training NEW Vanilla LSTM on videos1 (Baseline)...")

    X, y = [], []

    for file in os.listdir(TRAIN_DIR):
        if not file.lower().endswith(VIDEO_EXTS):
            continue

        print("Loading train video:", file)
        label = "_".join(file.split("_")[:-1])  # ✅ FIXED
        video_path = os.path.join(TRAIN_DIR, file)

        for seq in extract_sequences(video_path):
            X.append(seq)
            y.append(label)

    if len(X) == 0:
        raise RuntimeError("❌ No training data found!")

    X = np.array(X, dtype=np.float32)

    mean = X.mean(axis=(0,1))
    std  = X.std(axis=(0,1)) + 1e-6
    X = (X - mean) / std

    le = LabelEncoder()
    y_cat = to_categorical(le.fit_transform(y))

    model = Sequential([
        Input(shape=(X.shape[1], X.shape[2])),
        LSTM(96, return_sequences=True),
        Dropout(0.4),
        LSTM(48),
        Dropout(0.3),
        Dense(len(le.classes_), activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(
        X, y_cat,
        epochs=30,
        batch_size=8,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=1
    )

    model.save(MODEL_PATH)
    pickle.dump(le, open(ENCODER_PATH, "wb"))
    pickle.dump((mean, std), open(NORM_PATH, "wb"))

    print("✅ NEW MODEL TRAINED & SAVED")

else:
    print("⚠️ New model already exists. Skipping training.")

# =====================================================
# LOAD MODEL
# =====================================================

model = load_model(MODEL_PATH)
le = pickle.load(open(ENCODER_PATH, "rb"))
mean, std = pickle.load(open(NORM_PATH, "rb"))

# =====================================================
# VIDEO-LEVEL TESTING
# =====================================================

y_true, y_pred = [], []

print("\n--- 🕵️‍♂️ TESTING ON UNSEEN VIDEOS ---")

for file in os.listdir(TEST_DIR):
    if not file.lower().endswith(VIDEO_EXTS):
        continue

    print("Loading test video:", file)
    true_label = "_".join(file.split("_")[:-1])  # ✅ FIXED
    video_path = os.path.join(TEST_DIR, file)

    seqs = extract_sequences(video_path)
    if len(seqs) == 0:
        continue

    X_test = (np.array(seqs, dtype=np.float32) - mean) / std
    preds = model.predict(X_test, verbose=0)

    avg_probs = np.mean(preds, axis=0)
    best_idx = np.argmax(avg_probs)
    best_conf = avg_probs[best_idx]
    pred_name = le.inverse_transform([best_idx])[0]

    print(f"{file:20s} → Pred: {pred_name} ({best_conf*100:.1f}%)")

    y_true.append(true_label)
    y_pred.append(pred_name)

# =====================================================
# METRICS
# =====================================================

print("\n--- FINAL METRICS (videos1 baseline) ---")
print("Accuracy:", accuracy_score(y_true, y_pred))

print(classification_report(
    y_true, y_pred,
    labels=le.classes_,
    target_names=le.classes_,
    zero_division=0
))

cm = confusion_matrix(y_true, y_pred, labels=le.classes_)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Gait Recognition – videos1 Baseline")
plt.show()
