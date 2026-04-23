import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import pickle
import tempfile
import os
import time
import json
import logging
from datetime import datetime
from collections import Counter
from scipy.stats import entropy as scipy_entropy
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

_log = logging.getLogger(__name__)

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="Gait Recognition System",
    page_icon="🚶",
    layout="wide"
)

st.title("Gait Recognition System")
st.markdown(
    "**Open-Set Detection · Entropy Rejection · Weighted Voting · "
    "Sequence Quality Filtering · Biomechanical Analysis**"
)
st.caption(
    "7 joint-angle features · LSTM model · MediaPipe pose estimation"
)

# ==========================================
# PATHS
# ==========================================
MODEL_DIR    = "models"
MODEL_PATH   = os.path.join(MODEL_DIR, "gait_lstm_videos1_baseline.keras")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_videos1_baseline.pkl")
NORM_PATH    = os.path.join(MODEL_DIR, "norm_videos1_baseline.pkl")
LOG_PATH     = "gait_audit_log.jsonl"

# ==========================================
# GLOBAL CONFIG
# ==========================================
SEQ_LEN              = 60
STRIDE               = 6
MAX_PROCESS_TIME_SEC = 40

ENTROPY_REJECT_THRESHOLD = 0.75
MIN_MARGIN               = 0.18
MIN_AGREEMENT            = 0.50
MIN_BEST_CONF            = 0.45
SEQ_STD_MIN              = 0.5
SEQ_STD_MAX              = 60.0

# Clinical reference ranges for the RAW 3-point interior angles from angle().
# A straight knee = ~170-180°; peak swing-phase flexion = ~110-120°.
# These are NOT the conventional "flexion" amounts (which are 180° - interior angle).
#
#   Knee  (hip-knee-ankle)         : 130-165°  mean ~148°
#   Hip   (shoulder-hip-knee)      : 150-175°  mean ~162°
#   Ankle (knee-ankle-foot)        : 85-120°   mean ~102°
#   Torso tilt (from vertical, °)  : 0-10°     mean ~4°
CLINICAL_RANGES = {
    "Right knee flexion":       (130, 165),
    "Left knee flexion":        (130, 165),
    "Right hip flexion":        (150, 175),
    "Left hip flexion":         (150, 175),
    "Right ankle dorsiflexion": (85,  120),
    "Left ankle dorsiflexion":  (85,  120),
    "Torso tilt":               (0,   10),
}

FEATURE_NAMES = [
    "Right knee flexion",
    "Left knee flexion",
    "Right hip flexion",
    "Left hip flexion",
    "Right ankle dorsiflexion",
    "Left ankle dorsiflexion",
    "Torso tilt",
]

# Symmetry pairs: (left_idx, right_idx, label)
SYMMETRY_PAIRS = [
    (1, 0, "Knee flexion"),
    (3, 2, "Hip flexion"),
    (5, 4, "Ankle dorsiflexion"),
]

# ==========================================
# PER-IDENTITY OVERRIDES
# ==========================================
PER_IDENTITY_THRESHOLDS = {
    "Om": {"min_agreement": 0.45},
}

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(ENCODER_PATH, "rb") as f:
        le = pickle.load(f)
    with open(NORM_PATH, "rb") as f:
        mean, std = pickle.load(f)
    return model, le, mean, std


# ==========================================
# FEATURE HELPERS
# ==========================================
def angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    rad = (np.arctan2(c[1] - b[1], c[0] - b[0])
           - np.arctan2(a[1] - b[1], a[0] - b[0]))
    deg = abs(rad * 180.0 / np.pi)
    return 360 - deg if deg > 180 else deg


def torso_tilt(ls, rs, lh, rh):
    shoulder = (np.array(ls) + np.array(rs)) / 2
    hip      = (np.array(lh) + np.array(rh)) / 2
    v        = shoulder - hip
    vertical = np.array([0, -1])
    cos_val  = np.dot(v, vertical) / (np.linalg.norm(v) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_val, -1, 1)))


# ==========================================
# SKELETON DRAWING
# ==========================================
def draw_skeleton_on_frame(frame, landmarks, mp_pose, feature_values=None):
    """
    Draw pose skeleton on frame with joint angle annotations.
    Returns annotated frame (BGR).
    """
    h, w = frame.shape[:2]
    annotated = frame.copy()

    connections = mp_pose.POSE_CONNECTIONS
    lm = landmarks.landmark
    P  = mp_pose.PoseLandmark

    # Draw connections
    for conn in connections:
        start_idx, end_idx = conn
        x1 = int(lm[start_idx].x * w)
        y1 = int(lm[start_idx].y * h)
        x2 = int(lm[end_idx].x * w)
        y2 = int(lm[end_idx].y * h)
        cv2.line(annotated, (x1, y1), (x2, y2), (0, 255, 128), 2)

    # Draw landmark dots
    for lm_pt in lm:
        cx = int(lm_pt.x * w)
        cy = int(lm_pt.y * h)
        cv2.circle(annotated, (cx, cy), 4, (255, 80, 80), -1)
        cv2.circle(annotated, (cx, cy), 4, (255, 255, 255), 1)

    # Annotate key joint angles
    if feature_values is not None:
        key_joints = [
            (P.RIGHT_KNEE, 0, "RK"),
            (P.LEFT_KNEE,  1, "LK"),
            (P.RIGHT_HIP,  2, "RH"),
            (P.LEFT_HIP,   3, "LH"),
        ]
        for joint_id, feat_idx, label in key_joints:
            if feat_idx < len(feature_values):
                cx = int(lm[joint_id].x * w)
                cy = int(lm[joint_id].y * h)
                text = f"{label}:{feature_values[feat_idx]:.0f}"
                cv2.putText(
                    annotated, text,
                    (cx + 6, cy - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1, cv2.LINE_AA
                )

    return annotated


# ==========================================
# SEQUENCE QUALITY FILTER
# ==========================================
def is_good_sequence(seq, std_min, std_max):
    arr      = np.array(seq)
    mean_std = arr.std(axis=0).mean()
    return std_min < mean_std < std_max


# ==========================================
# WEIGHTED VOTING
# ==========================================
def weighted_vote(preds):
    vote_weights = {}
    for pred in preds:
        cls = int(np.argmax(pred))
        vote_weights[cls] = vote_weights.get(cls, 0.0) + float(pred[cls])
    winner    = max(vote_weights, key=vote_weights.get)
    total_w   = sum(vote_weights.values())
    agreement = vote_weights[winner] / total_w if total_w > 0 else 0.0
    return winner, agreement


# ==========================================
# OPEN-SET DECISION ENGINE
# ==========================================
def make_decision(avg_probs, preds, le,
                  slider_entropy, slider_conf, slider_margin, slider_agreement):
    debug = {}
    sorted_idx  = np.argsort(avg_probs)
    best_idx    = int(sorted_idx[-1])
    second_idx  = int(sorted_idx[-2])
    best_conf   = float(avg_probs[best_idx])
    second_conf = float(avg_probs[second_idx])
    margin      = best_conf - second_conf
    best_name   = le.inverse_transform([best_idx])[0]

    raw_entropy        = float(scipy_entropy(avg_probs))
    max_entropy_val    = float(np.log(len(avg_probs)))
    normalized_entropy = raw_entropy / max_entropy_val if max_entropy_val > 0 else 1.0

    best_name_key = next(
        (k for k in PER_IDENTITY_THRESHOLDS
         if k.strip().lower() == best_name.strip().lower()),
        None
    )
    ov         = PER_IDENTITY_THRESHOLDS.get(best_name_key, {}) if best_name_key else {}
    max_ent    = ov.get("max_entropy",   slider_entropy)
    min_conf   = ov.get("min_conf",      slider_conf)
    min_margin = ov.get("min_margin",    slider_margin)
    min_agr    = ov.get("min_agreement", slider_agreement)

    voted_class, agreement = weighted_vote(preds)
    voted_name = le.inverse_transform([voted_class])[0]

    debug.update({
        "best_name": best_name, "voted_name": voted_name,
        "best_conf": best_conf, "second_conf": second_conf,
        "margin": margin, "normalized_entropy": normalized_entropy,
        "agreement": agreement,
        "max_ent_used": max_ent, "min_conf_used": min_conf,
        "min_margin_used": min_margin, "min_agr_used": min_agr,
    })

    if margin < min_margin:
        debug["reject_reason"] = f"Margin too small ({margin:.3f} < {min_margin:.3f})"
        return "Unknown", debug["reject_reason"], debug
    if normalized_entropy > max_ent:
        debug["reject_reason"] = f"Entropy too high ({normalized_entropy:.3f} > {max_ent:.3f})"
        return "Unknown", debug["reject_reason"], debug
    if best_conf < min_conf:
        debug["reject_reason"] = f"Confidence too low ({best_conf:.3f} < {min_conf:.3f})"
        return "Unknown", debug["reject_reason"], debug
    if agreement < min_agr:
        debug["reject_reason"] = f"Vote agreement too low ({agreement:.3f} < {min_agr:.3f})"
        return "Unknown", debug["reject_reason"], debug

    debug["reject_reason"] = None
    return voted_name, "Accepted", debug


# ==========================================
# BIOMECHANICAL ANALYSIS
# ==========================================
def compute_biomechanics(frames_array):
    """
    Returns a dict with per-feature stats, symmetry scores, and clinical flags.
    frames_array: np.array shape (N_frames, 7)
    """
    results = {}

    # Per-feature mean / std / ROM
    for i, name in enumerate(FEATURE_NAMES):
        col  = frames_array[:, i]
        results[name] = {
            "mean": float(np.mean(col)),
            "std":  float(np.std(col)),
            "min":  float(np.min(col)),
            "max":  float(np.max(col)),
            "rom":  float(np.max(col) - np.min(col)),
        }

    # Symmetry index: |L - R| / ((L + R) / 2) * 100
    symmetry = {}
    for l_idx, r_idx, label in SYMMETRY_PAIRS:
        l_mean = results[FEATURE_NAMES[l_idx]]["mean"]
        r_mean = results[FEATURE_NAMES[r_idx]]["mean"]
        avg = (l_mean + r_mean) / 2.0 if (l_mean + r_mean) > 0 else 1.0
        si  = abs(l_mean - r_mean) / avg * 100.0
        symmetry[label] = {
            "left_mean":  l_mean,
            "right_mean": r_mean,
            "symmetry_index": si,
            "symmetric": si < 10.0,   # <10% = clinically symmetric
        }
    results["symmetry"] = symmetry

    # Clinical range flags
    flags = []
    for name, (lo, hi) in CLINICAL_RANGES.items():
        mean_val = results[name]["mean"]
        if mean_val < lo:
            flags.append({
                "feature": name,
                "value":   round(mean_val, 1),
                "issue":   "below normal range",
                "range":   f"{lo}–{hi}°",
            })
        elif mean_val > hi:
            flags.append({
                "feature": name,
                "value":   round(mean_val, 1),
                "issue":   "above normal range",
                "range":   f"{lo}–{hi}°",
            })

    results["clinical_flags"] = flags
    return results


# ==========================================
# ABNORMALITY DETECTION (v2)
# ==========================================

def is_above_range(val: float, lo: float, hi: float, tol: float = 10.0) -> bool:
    """True when val exceeds the upper bound by at least tol degrees."""
    return val > (hi + tol)


def is_below_range(val: float, lo: float, hi: float, tol: float = 10.0) -> bool:
    """True when val is below the lower bound by at least tol degrees."""
    return val < (lo - tol)


def _deviation_score(val: float, lo: float, hi: float) -> float:
    """
    Normalised absolute deviation from the clinical midpoint.
    0.0  = perfectly centred in the normal band.
    0.5  = exactly at the band edge.
    1.0  = one full band-width outside normal range (clipped here).
    """
    span = hi - lo
    if span <= 0:
        return 0.0
    return float(np.clip(abs(val - (lo + hi) / 2.0) / span, 0.0, 1.5))


def _mean_deviation(bio_stats: dict, features: list) -> float:
    """
    Average deviation score computed only over the clinically relevant
    features listed for a given rule — NOT over all 7 features.
    """
    scores = []
    for feat in features:
        lo, hi = CLINICAL_RANGES[feat]
        scores.append(_deviation_score(bio_stats[feat]["mean"], lo, hi))
    return float(np.mean(scores)) if scores else 0.0


# Severity order used for escalation
_SEVERITY_ORDER  = {"Mild": 0, "Moderate": 1, "Severe": 2}
_SEVERITY_LABELS = {0: "Mild", 1: "Moderate", 2: "Severe"}

# ── Rule table ─────────────────────────────────────────────────────────────────
#
# Each rule has:
#   severity          – base severity label
#   check(bio)        – callable returning bool; bio is the bio_stats dict
#   relevant_features – features used ONLY for per-pattern confidence scoring
#   description       – clinical description shown in the UI
#   recommendation    – action recommendation
#
# Bilateral rules require both sides to fire.
# Unilateral variants fire when only one side is abnormal.

ABNORMALITY_RULES: dict = {

    # ── Bilateral patterns ────────────────────────────────────────────────────

    "Crouch gait": {
        "severity": "Moderate",
        "check": lambda bio: (
            is_below_range(bio["Right knee flexion"]["mean"],
                           *CLINICAL_RANGES["Right knee flexion"]) and
            is_below_range(bio["Left knee flexion"]["mean"],
                           *CLINICAL_RANGES["Left knee flexion"])
        ),
        "relevant_features": ["Right knee flexion", "Left knee flexion"],
        "description": (
            "Excessive knee and hip flexion maintained throughout the gait cycle. "
            "Common in cerebral palsy, hamstring spasticity, or hip flexion contracture."
        ),
        "recommendation": "Physiotherapy assessment for hamstring tightness or spasticity.",
    },

    "Stiff-knee gait": {
        "severity": "Mild",
        "check": lambda bio: (
            bio["Right knee flexion"]["rom"] < 25 and
            bio["Left knee flexion"]["rom"] < 25
        ),
        "relevant_features": ["Right knee flexion", "Left knee flexion"],
        "description": (
            "Reduced knee range of motion during swing phase — knee does not flex enough "
            "to clear the foot. Associated with rectus femoris overactivity."
        ),
        "recommendation": "Check rectus femoris muscle tone; consider orthotics.",
    },

    "Bilateral foot drop": {
        "severity": "Moderate",
        "check": lambda bio: (
            is_below_range(bio["Right ankle dorsiflexion"]["mean"],
                           *CLINICAL_RANGES["Right ankle dorsiflexion"]) and
            is_below_range(bio["Left ankle dorsiflexion"]["mean"],
                           *CLINICAL_RANGES["Left ankle dorsiflexion"])
        ),
        "relevant_features": ["Right ankle dorsiflexion", "Left ankle dorsiflexion"],
        "description": (
            "Bilateral reduced ankle dorsiflexion — foot cannot clear the ground during "
            "swing. Bilateral presentation may indicate peripheral neuropathy or myopathy."
        ),
        "recommendation": "Neurological referral; bilateral ankle-foot orthosis (AFO) may be indicated.",
    },

    "Vaulting / equinus gait": {
        "severity": "Mild",
        "check": lambda bio: (
            # Strong plantar-flexion on both sides…
            is_above_range(bio["Right ankle dorsiflexion"]["mean"],
                           *CLINICAL_RANGES["Right ankle dorsiflexion"], tol=15) and
            is_above_range(bio["Left ankle dorsiflexion"]["mean"],
                           *CLINICAL_RANGES["Left ankle dorsiflexion"], tol=15) and
            # …accompanied by compensatory knee flexion reduction
            bio["Right knee flexion"]["mean"] < 155 and
            bio["Left knee flexion"]["mean"] < 155
        ),
        "relevant_features": [
            "Right ankle dorsiflexion", "Left ankle dorsiflexion",
            "Right knee flexion", "Left knee flexion",
        ],
        "description": (
            "Excessive plantar-flexion (high ankle angle) during stance with compensatory "
            "knee restriction. May indicate limb-length discrepancy or plantar-flexion contracture."
        ),
        "recommendation": "Assess for limb-length discrepancy or plantar-flexion contracture.",
    },

    "Hip flexion deficit": {
        "severity": "Mild",
        "check": lambda bio: (
            is_below_range(bio["Right hip flexion"]["mean"],
                           *CLINICAL_RANGES["Right hip flexion"]) and
            is_below_range(bio["Left hip flexion"]["mean"],
                           *CLINICAL_RANGES["Left hip flexion"])
        ),
        "relevant_features": ["Right hip flexion", "Left hip flexion"],
        "description": (
            "Reduced hip angle throughout the cycle — limited forward stride length. "
            "Associated with hip arthritis, iliopsoas tightness, or pain."
        ),
        "recommendation": "Hip mobility assessment; evaluate for iliopsoas tightness.",
    },

    "Trendelenburg / trunk sway": {
        "severity": "Moderate",
        "check": lambda bio: bio["Torso tilt"]["mean"] > 12,
        "relevant_features": ["Torso tilt"],
        "description": (
            "Excessive lateral trunk lean — usually due to hip abductor weakness "
            "causing the pelvis to drop on the swing side."
        ),
        "recommendation": "Hip abductor strengthening programme; rule out hip pathology.",
    },

    # ── Asymmetry / antalgic ──────────────────────────────────────────────────

    "Antalgic (pain-avoidance) gait": {
        "severity": "Mild",
        "check": lambda bio: (
            bio["symmetry"]["Knee flexion"]["symmetry_index"] > 15 or
            bio["symmetry"]["Hip flexion"]["symmetry_index"] > 15
        ),
        "relevant_features": [
            "Right knee flexion", "Left knee flexion",
            "Right hip flexion",  "Left hip flexion",
        ],
        "description": (
            "Marked left-right asymmetry in knee or hip angles, indicating the subject "
            "is favouring one side to avoid pain on the other."
        ),
        "recommendation": "Clinical pain assessment; rule out joint pathology or injury.",
    },

    # ── Unilateral variants ───────────────────────────────────────────────────

    "Right-sided foot drop": {
        "severity": "Mild",
        "check": lambda bio: (
            is_below_range(bio["Right ankle dorsiflexion"]["mean"],
                           *CLINICAL_RANGES["Right ankle dorsiflexion"]) and
            not is_below_range(bio["Left ankle dorsiflexion"]["mean"],
                               *CLINICAL_RANGES["Left ankle dorsiflexion"])
        ),
        "relevant_features": ["Right ankle dorsiflexion"],
        "description": (
            "Unilateral right ankle dorsiflexion deficit — common in peroneal nerve palsy "
            "or L4/L5 radiculopathy on the right side."
        ),
        "recommendation": "Targeted right-side neurological evaluation; single AFO may suffice.",
    },

    "Left-sided foot drop": {
        "severity": "Mild",
        "check": lambda bio: (
            is_below_range(bio["Left ankle dorsiflexion"]["mean"],
                           *CLINICAL_RANGES["Left ankle dorsiflexion"]) and
            not is_below_range(bio["Right ankle dorsiflexion"]["mean"],
                               *CLINICAL_RANGES["Right ankle dorsiflexion"])
        ),
        "relevant_features": ["Left ankle dorsiflexion"],
        "description": (
            "Unilateral left ankle dorsiflexion deficit — common in peroneal nerve palsy "
            "or L4/L5 radiculopathy on the left side."
        ),
        "recommendation": "Targeted left-side neurological evaluation; single AFO may suffice.",
    },

    "Right-sided knee stiffness": {
        "severity": "Mild",
        "check": lambda bio: (
            bio["Right knee flexion"]["rom"] < 25 and
            bio["Left knee flexion"]["rom"] >= 25
        ),
        "relevant_features": ["Right knee flexion"],
        "description": "Unilateral right knee range-of-motion deficit during swing phase.",
        "recommendation": "Right knee assessment; check for joint effusion or pain.",
    },

    "Left-sided knee stiffness": {
        "severity": "Mild",
        "check": lambda bio: (
            bio["Left knee flexion"]["rom"] < 25 and
            bio["Right knee flexion"]["rom"] >= 25
        ),
        "relevant_features": ["Left knee flexion"],
        "description": "Unilateral left knee range-of-motion deficit during swing phase.",
        "recommendation": "Left knee assessment; check for joint effusion or pain.",
    },
}


def detect_abnormalities(bio_stats: dict) -> list:
    """
    Run all ABNORMALITY_RULES against bio_stats.

    Returns a list of dicts, each containing:
        pattern        – rule name
        severity       – "Mild" | "Moderate" | "Severe"  (may be escalated)
        description    – clinical description
        recommendation – suggested action
        confidence     – 0–1 float (mean deviation on relevant features only)
        laterality     – "Bilateral" | "Right" | "Left" | "N/A"
    """
    detected = []

    for name, rule in ABNORMALITY_RULES.items():
        try:
            triggered: bool = rule["check"](bio_stats)
        except Exception as exc:
            _log.warning("Abnormality rule '%s' raised an error: %s", name, exc)
            continue

        if not triggered:
            continue

        # Per-pattern confidence using only the clinically relevant features
        relevant   = rule.get("relevant_features", FEATURE_NAMES)
        confidence = float(np.clip(_mean_deviation(bio_stats, relevant), 0.0, 1.0))

        # Infer laterality from rule name
        if name.startswith("Right"):
            laterality = "Right"
        elif name.startswith("Left"):
            laterality = "Left"
        elif name == "Antalgic (pain-avoidance) gait":
            laterality = "N/A"
        else:
            laterality = "Bilateral"

        detected.append({
            "pattern":        name,
            "severity":       rule["severity"],
            "description":    rule["description"],
            "recommendation": rule["recommendation"],
            "confidence":     round(confidence, 3),
            "laterality":     laterality,
        })

    # ── Severity escalation ───────────────────────────────────────────────────
    # When 3+ distinct patterns co-occur, escalate every Moderate → Severe
    # to reflect compounding clinical significance.
    if len(detected) >= 3:
        for d in detected:
            current     = _SEVERITY_ORDER.get(d["severity"], 0)
            d["severity"] = _SEVERITY_LABELS[min(current + 1, 2)]

    # Sort: highest severity first, then by confidence descending
    detected.sort(
        key=lambda d: (
            -_SEVERITY_ORDER.get(d["severity"], 0),
            -d["confidence"],
        )
    )

    return detected


# ==========================================
# GAIT HEALTH SCORE
# ==========================================
def compute_gait_health_score(bio_stats, detected_abnormalities):
    """
    Composite Gait Health Score (0–100%).

    Four equally-weighted components (25% each):

    1. Symmetry   – mean symmetry index across bilateral joint pairs.
                    SI=0% → 100 pts, SI≥20% → 0 pts (linear).
    2. ROM        – each feature's ROM vs expected walking ROM; mean coverage.
    3. Clinical   – fraction of features whose mean falls within the normal band.
    4. Smoothness – coefficient of variation (std/mean) per feature; lower = smoother.

    Abnormality penalty: −5 pts per Mild, −10 per Moderate, −15 per Severe pattern.
    Final score is clamped to [0, 100].
    """
    EXPECTED_ROM = {
        "Right knee flexion":       40,
        "Left knee flexion":        40,
        "Right hip flexion":        25,
        "Left hip flexion":         25,
        "Right ankle dorsiflexion": 30,
        "Left ankle dorsiflexion":  30,
        "Torso tilt":               8,
    }

    # ── Component 1: Symmetry ──────────────────────────────────────
    sym_scores = []
    for vals in bio_stats["symmetry"].values():
        si = vals["symmetry_index"]
        sym_scores.append(max(0.0, 100.0 - si * 5.0))
    sym_component = float(np.mean(sym_scores)) if sym_scores else 100.0

    # ── Component 2: ROM ──────────────────────────────────────────
    rom_scores = []
    for feat, expected in EXPECTED_ROM.items():
        actual = bio_stats[feat]["rom"]
        rom_scores.append(min(actual / max(expected, 1), 1.0) * 100.0)
    rom_component = float(np.mean(rom_scores))

    # ── Component 3: Clinical range compliance ────────────────────
    in_range = 0
    total    = len(FEATURE_NAMES)
    for feat in FEATURE_NAMES:
        lo, hi = CLINICAL_RANGES[feat]
        if lo <= bio_stats[feat]["mean"] <= hi:
            in_range += 1
    clinical_component = (in_range / total) * 100.0

    # ── Component 4: Smoothness ───────────────────────────────────
    smooth_scores = []
    for feat in FEATURE_NAMES:
        mean_val = bio_stats[feat]["mean"]
        std_val  = bio_stats[feat]["std"]
        if mean_val == 0:
            smooth_scores.append(100.0)
            continue
        cv = std_val / abs(mean_val)
        if cv < 0.02:
            score = 60.0
        elif cv <= 0.15:
            score = 60.0 + (cv - 0.02) / 0.13 * 40.0
        else:
            score = max(0.0, 100.0 - (cv - 0.15) / 0.35 * 100.0)
        smooth_scores.append(score)
    smooth_component = float(np.mean(smooth_scores))

    # ── Weighted composite ────────────────────────────────────────
    raw_score = (
        0.30 * sym_component
        + 0.25 * rom_component
        + 0.25 * clinical_component
        + 0.20 * smooth_component
    )

    # ── Abnormality penalty ───────────────────────────────────────
    penalty_map = {"Mild": 5, "Moderate": 10, "Severe": 15}
    penalty = sum(penalty_map.get(a["severity"], 5) for a in detected_abnormalities)

    final_score = float(np.clip(raw_score - penalty, 0, 100))

    # ── Qualitative band ──────────────────────────────────────────
    if final_score >= 85:
        band, band_color = "Excellent", "green"
    elif final_score >= 70:
        band, band_color = "Good", "green"
    elif final_score >= 55:
        band, band_color = "Fair", "orange"
    elif final_score >= 40:
        band, band_color = "Poor", "red"
    else:
        band, band_color = "Critical", "red"

    return {
        "score":              round(final_score, 1),
        "band":               band,
        "band_color":         band_color,
        "sym_component":      round(sym_component,      1),
        "rom_component":      round(rom_component,      1),
        "clinical_component": round(clinical_component, 1),
        "smooth_component":   round(smooth_component,   1),
        "penalty":            penalty,
    }


# ==========================================
# AUDIT LOG
# ==========================================
def append_audit_log(entry: dict):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def load_audit_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


# ==========================================
# PLOTLY CHARTS
# ==========================================
def plot_temporal_features(frames_array):
    """Multi-panel temporal plot for all 7 joint angles."""
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=FEATURE_NAMES + [""],
        shared_xaxes=True,
        vertical_spacing=0.07,
    )
    colors    = px.colors.qualitative.Safe
    positions = [(1,1),(1,2),(2,1),(2,2),(3,1),(3,2),(4,1)]
    for i, (name, pos) in enumerate(zip(FEATURE_NAMES, positions)):
        fig.add_trace(
            go.Scatter(
                y=frames_array[:, i],
                mode="lines",
                line=dict(color=colors[i % len(colors)], width=1.5),
                name=name,
                showlegend=False,
            ),
            row=pos[0], col=pos[1],
        )
    fig.update_layout(
        height=700,
        title_text="Joint angle trajectories over time (degrees)",
        template="plotly_white",
        margin=dict(t=60, b=40),
    )
    fig.update_xaxes(title_text="Frame", row=4, col=1)
    fig.update_xaxes(title_text="Frame", row=4, col=2)
    return fig


def plot_sequence_confidence_timeline(preds, le):
    """Bar chart of per-sequence top-class confidence."""
    top_confs  = [float(np.max(p)) for p in preds]
    top_labels = [le.inverse_transform([int(np.argmax(p))])[0] for p in preds]
    unique_ids = sorted(set(le.classes_))
    color_map  = {name: px.colors.qualitative.Plotly[i % 10]
                  for i, name in enumerate(unique_ids)}
    bar_colors = [color_map.get(l, "#aaa") for l in top_labels]

    fig = go.Figure(go.Bar(
        x=list(range(len(top_confs))),
        y=top_confs,
        marker_color=bar_colors,
        text=top_labels,
        textposition="outside",
        textfont=dict(size=9),
    ))
    fig.add_hline(
        y=MIN_BEST_CONF,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Min conf ({MIN_BEST_CONF})",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Per-sequence confidence (colour = predicted identity)",
        xaxis_title="Sequence index",
        yaxis_title="Confidence",
        yaxis_range=[0, 1.05],
        template="plotly_white",
        height=320,
    )
    return fig


def plot_symmetry(symmetry_dict):
    """Horizontal bar chart of left-right symmetry index (%)."""
    labels = list(symmetry_dict.keys())
    values = [symmetry_dict[l]["symmetry_index"] for l in labels]
    colors = ["#2ecc71" if v < 10 else "#e74c3c" for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
    ))
    fig.add_vline(
        x=10, line_dash="dash", line_color="orange",
        annotation_text="10% threshold",
        annotation_position="top right",
    )
    fig.update_layout(
        title="Gait symmetry index (lower is more symmetric)",
        xaxis_title="Symmetry index (%)",
        template="plotly_white",
        height=280,
        xaxis_range=[0, max(values + [15]) * 1.2],
    )
    return fig


def plot_radar_feature_profile(bio_stats):
    """
    Deviation-from-normal radar chart.

    Each spoke = normalised absolute deviation from the clinical midpoint:
        deviation = |subject_mean - midpoint| / (hi - lo)
    0.0 = perfect normal, 0.5 = at band edge, 1.0 = one band-width outside.
    """
    feature_order = FEATURE_NAMES[:6]
    labels = [
        n.replace(" flexion", "").replace(" dorsiflexion", " DF")
        for n in feature_order
    ]

    deviations  = []
    hover_texts = []
    bar_colors  = []

    for n in feature_order:
        lo, hi  = CLINICAL_RANGES[n]
        span    = hi - lo
        mid     = (lo + hi) / 2.0
        val     = bio_stats[n]["mean"]

        dev = abs(val - mid) / span if span > 0 else 0.0
        dev = float(np.clip(dev, 0.0, 1.0))
        deviations.append(dev)

        if val > hi:
            direction = f"↑ {val - hi:.1f}° above normal range"
        elif val < lo:
            direction = f"↓ {lo - val:.1f}° below normal range"
        else:
            direction = "✓ within normal range"

        hover_texts.append(
            f"<b>{n}</b><br>"
            f"Subject: {val:.1f}°<br>"
            f"Normal range: {lo}–{hi}°  (mid {mid:.1f}°)<br>"
            f"{direction}<br>"
            f"Deviation score: {dev:.3f}"
        )

        if dev < 0.30:
            bar_colors.append("rgba(46, 204, 113, 0.85)")
        elif dev < 0.60:
            bar_colors.append("rgba(230, 126, 34, 0.85)")
        else:
            bar_colors.append("rgba(231, 76, 60, 0.85)")

    dev_closed    = deviations  + [deviations[0]]
    labels_closed = labels      + [labels[0]]
    hover_closed  = hover_texts + [hover_texts[0]]

    RING_DEFS = [
        (0.30, "rgba(46,204,113,0.08)",  "rgba(46,204,113,0.5)",  "Normal band edge (0.3)"),
        (0.60, "rgba(230,126, 34,0.08)", "rgba(230,126,34,0.5)",  "Mild deviation (0.6)"),
        (1.00, "rgba(231, 76, 60,0.08)", "rgba(231,76, 60,0.5)",  "Significant deviation (1.0)"),
    ]

    fig = go.Figure()

    for r_val, fill, stroke, ring_label in reversed(RING_DEFS):
        fig.add_trace(go.Scatterpolar(
            r=[r_val] * 6 + [r_val],
            theta=labels_closed,
            fill="toself",
            fillcolor=fill,
            line=dict(color=stroke, width=1, dash="dot"),
            name=ring_label,
            showlegend=True,
            hoverinfo="skip",
        ))

    fig.add_trace(go.Scatterpolar(
        r=[0.0] * 6 + [0.0],
        theta=labels_closed,
        mode="markers",
        marker=dict(color="#2980b9", size=10, symbol="circle"),
        name="Clinical normal (0 deviation)",
        hoverinfo="skip",
    ))

    mean_dev   = float(np.mean(deviations))
    poly_color = (
        "rgba(46,204,113,0.30)"  if mean_dev < 0.30 else
        "rgba(230,126,34,0.30)"  if mean_dev < 0.60 else
        "rgba(231,76,60,0.30)"
    )
    line_color = (
        "#27ae60" if mean_dev < 0.30 else
        "#e67e22" if mean_dev < 0.60 else
        "#c0392b"
    )

    fig.add_trace(go.Scatterpolar(
        r=dev_closed,
        theta=labels_closed,
        fill="toself",
        fillcolor=poly_color,
        line=dict(color=line_color, width=2.5),
        mode="lines+markers",
        marker=dict(color=bar_colors, size=9, line=dict(color="white", width=1)),
        name=f"Subject (mean dev: {mean_dev:.2f})",
        text=hover_closed,
        hoverinfo="text",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1.0],
                tickvals=[0.0, 0.3, 0.6, 1.0],
                ticktext=["Normal", "Mild", "Moderate", "Severe"],
                tickfont=dict(size=9, color="#555"),
                gridcolor="rgba(0,0,0,0.08)",
                linecolor="rgba(0,0,0,0.15)",
            ),
            angularaxis=dict(
                tickfont=dict(size=11),
                linecolor="rgba(0,0,0,0.2)",
            ),
            bgcolor="rgba(248,249,250,0.5)",
        ),
        showlegend=True,
        title=dict(
            text=(
                "Gait deviation from clinical normal<br>"
                "<sup>Centre = normal  |  Outer = abnormal  |  "
                "Closer to centre is better</sup>"
            ),
            font=dict(size=14),
        ),
        template="plotly_white",
        height=460,
        legend=dict(
            orientation="h",
            y=-0.18,
            font=dict(size=10),
            itemsizing="constant",
        ),
        margin=dict(t=80, b=80),
    )
    return fig


def plot_prediction_heatmap(preds, le):
    """Heatmap: sequences × classes softmax probability."""
    classes     = le.classes_
    mat         = np.array(preds)
    max_display = min(80, len(mat))
    mat_disp    = mat[:max_display]

    fig = px.imshow(
        mat_disp,
        labels=dict(x="Identity", y="Sequence index", color="Probability"),
        x=classes,
        color_continuous_scale="RdYlGn",
        zmin=0, zmax=1,
        aspect="auto",
    )
    fig.update_layout(
        title=f"Softmax heatmap (first {max_display} sequences × all classes)",
        height=420,
        template="plotly_white",
    )
    return fig


def plot_feature_boxplots(frames_array):
    """Box plots for all 7 features with clinical range overlays."""
    fig    = go.Figure()
    colors = px.colors.qualitative.Safe
    for i, name in enumerate(FEATURE_NAMES):
        fig.add_trace(go.Box(
            y=frames_array[:, i],
            name=name.replace(" flexion", "").replace(" dorsiflexion", " DF"),
            boxmean="sd",
            marker_color=colors[i % len(colors)],
        ))
        lo, hi = CLINICAL_RANGES[name]
        fig.add_hrect(
            y0=lo, y1=hi,
            fillcolor="rgba(0,200,100,0.07)",
            line_width=0,
            annotation_text="" if i > 0 else "Normal range",
            annotation_position="top left",
        )
    fig.update_layout(
        title="Feature distribution across all frames (green band = normal range)",
        yaxis_title="Degrees",
        template="plotly_white",
        height=420,
        showlegend=False,
    )
    return fig


# ==========================================
# REPORT GENERATION
# ==========================================
def build_report(filename, final_name, reject_reason,
                 debug, bio_stats, n_good, n_total, elapsed,
                 health_score=None, detected_abnormalities=None):
    report = {
        "timestamp":      datetime.utcnow().isoformat() + "Z",
        "source_file":    filename,
        "result": {
            "identity":       final_name,
            "accepted":       final_name != "Unknown",
            "reject_reason":  reject_reason,
        },
        "decision_metrics": {
            "confidence":         round(debug["best_conf"],           4),
            "second_best":        round(debug["second_conf"],         4),
            "margin":             round(debug["margin"],              4),
            "normalized_entropy": round(debug["normalized_entropy"],  4),
            "vote_agreement":     round(debug["agreement"],           4),
        },
        "sequences": {
            "quality_sequences": n_good,
            "total_sequences":   n_total,
            "processing_time_s": round(elapsed, 2),
        },
        "biomechanics": {
            feat: {
                "mean_deg": round(bio_stats[feat]["mean"], 2),
                "std_deg":  round(bio_stats[feat]["std"],  2),
                "rom_deg":  round(bio_stats[feat]["rom"],  2),
            }
            for feat in FEATURE_NAMES
        },
        "symmetry": {
            label: {
                "symmetry_index_%": round(vals["symmetry_index"], 2),
                "symmetric":        vals["symmetric"],
            }
            for label, vals in bio_stats["symmetry"].items()
        },
        "clinical_flags": bio_stats["clinical_flags"],
        "gait_health_score": health_score or {},
        "detected_abnormalities": [
            {
                "pattern":        a["pattern"],
                "severity":       a["severity"],
                "laterality":     a.get("laterality", "N/A"),
                "confidence":     a.get("confidence", 0.0),
                "recommendation": a["recommendation"],
            }
            for a in (detected_abnormalities or [])
        ],
    }
    return report


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader(
        "Upload walking video",
        type=["mp4", "avi", "mov", "MOV"]
    )

    st.divider()
    st.subheader("Threshold Tuning")
    st.caption("Per-identity overrides take priority over these sliders.")

    entropy_thresh = st.slider(
        "Max entropy (global fallback)",
        0.30, 0.90, ENTROPY_REJECT_THRESHOLD, 0.01,
        help="Normalized Shannon entropy. Higher = flatter distribution = likely unknown."
    )
    min_conf_slider = st.slider(
        "Min confidence (global fallback)",
        0.30, 0.95, MIN_BEST_CONF, 0.01
    )
    min_margin_slider = st.slider(
        "Min margin – best minus 2nd best (global fallback)",
        0.05, 0.50, MIN_MARGIN, 0.01
    )
    min_agreement_slider = st.slider(
        "Min vote agreement (global fallback)",
        0.30, 0.90, MIN_AGREEMENT, 0.01
    )

    st.divider()
    st.caption("Sequence quality filter")
    std_min = st.number_input("Min feature std (degrees)", 0.0,  5.0,  SEQ_STD_MIN, 0.05)
    std_max = st.number_input("Max feature std (degrees)", 5.0, 120.0, SEQ_STD_MAX, 1.0)

    st.divider()
    st.subheader("Output options")
    save_skeleton_video = st.checkbox(
        "Export skeleton overlay video", value=False,
        help="Saves an annotated MP4 with pose skeleton and joint angles drawn on each frame."
    )
    show_audit_log = st.checkbox("Show audit log", value=False)


# ==========================================
# MAIN LOGIC
# ==========================================
if uploaded_file is not None:
    model, le, mean, std = load_resources()

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded_file.read())
    tmp.flush()

    mp_pose = mp.solutions.pose
    pose    = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap    = cv2.VideoCapture(tmp.name)
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = []

    skeleton_frames = []
    frame_h         = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w         = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    st.divider()
    status_box   = st.info("Extracting gait features…")
    progress_bar = st.progress(0.0)
    start_time   = time.time()

    # ===============================
    # FEATURE EXTRACTION
    # ===============================
    while cap.isOpened():
        elapsed = time.time() - start_time
        if elapsed >= MAX_PROCESS_TIME_SEC:
            break
        progress_bar.progress(min(elapsed / MAX_PROCESS_TIME_SEC, 1.0))

        ret, frame = cap.read()
        if not ret:
            break

        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            if save_skeleton_video:
                skeleton_frames.append(frame)
            continue

        lm = results.pose_landmarks.landmark
        P  = mp_pose.PoseLandmark

        def p(i):
            return [lm[i].x, lm[i].y]

        try:
            rs = p(P.RIGHT_SHOULDER); ls = p(P.LEFT_SHOULDER)
            rh = p(P.RIGHT_HIP);      lh = p(P.LEFT_HIP)
            rk = p(P.RIGHT_KNEE);     lk = p(P.LEFT_KNEE)
            ra = p(P.RIGHT_ANKLE);    la = p(P.LEFT_ANKLE)
            rf = p(P.RIGHT_FOOT_INDEX); lf = p(P.LEFT_FOOT_INDEX)

            feature_values = [
                angle(rh, rk, ra),
                angle(lh, lk, la),
                angle(rs, rh, rk),
                angle(ls, lh, lk),
                angle(rk, ra, rf),
                angle(lk, la, lf),
                torso_tilt(ls, rs, lh, rh),
            ]
            frames.append(feature_values)

            if save_skeleton_video:
                annotated = draw_skeleton_on_frame(
                    frame, results.pose_landmarks, mp_pose, feature_values
                )
                skeleton_frames.append(annotated)

        except Exception:
            if save_skeleton_video:
                skeleton_frames.append(frame)
            continue

    cap.release()
    status_box.success(f"Extraction complete — {len(frames)} frames captured.")
    progress_bar.empty()

    # ===============================
    # BUILD & FILTER SEQUENCES
    # ===============================
    raw_sequences  = []
    good_sequences = []
    bad_count      = 0

    for i in range(0, len(frames) - SEQ_LEN, STRIDE):
        seq = frames[i: i + SEQ_LEN]
        raw_sequences.append(seq)
        if is_good_sequence(seq, std_min, std_max):
            good_sequences.append(seq)
        else:
            bad_count += 1

    st.subheader("Final Result")

    if len(good_sequences) == 0:
        st.error("Not enough quality gait sequences. Try a longer or clearer walking clip.")
    else:
        X         = (np.array(good_sequences, dtype=np.float32) - mean) / std
        preds     = model.predict(X, verbose=0)
        avg_probs = np.mean(preds, axis=0)

        final_name, reject_reason, debug = make_decision(
            avg_probs, preds, le,
            slider_entropy   = entropy_thresh,
            slider_conf      = min_conf_slider,
            slider_margin    = min_margin_slider,
            slider_agreement = min_agreement_slider,
        )

        best_conf         = debug["best_conf"]
        second_conf       = debug["second_conf"]
        margin            = debug["margin"]
        norm_ent          = debug["normalized_entropy"]
        agreement         = debug["agreement"]
        best_name         = debug["best_name"]
        voted_name        = debug["voted_name"]
        max_ent_used      = debug["max_ent_used"]
        min_conf_used     = debug["min_conf_used"]
        min_margin_used   = debug["min_margin_used"]
        min_agr_used      = debug["min_agr_used"]
        identity_override = best_name.strip().lower() in [
            k.strip().lower() for k in PER_IDENTITY_THRESHOLDS
        ]

        total_time = time.time() - start_time

        # ── Biomechanical analysis ─────────────────────────────────────────
        frames_array = np.array(frames, dtype=np.float32)
        bio_stats    = compute_biomechanics(frames_array)

        # ── Abnormality detection + health score ──────────────────────────
        detected_abnormalities = detect_abnormalities(bio_stats)
        health_score           = compute_gait_health_score(bio_stats, detected_abnormalities)

        # ── Audit log ─────────────────────────────────────────────────────
        audit_entry = {
            "timestamp":     datetime.utcnow().isoformat() + "Z",
            "filename":      uploaded_file.name,
            "result":        final_name,
            "confidence":    round(best_conf, 4),
            "margin":        round(margin, 4),
            "entropy":       round(norm_ent, 4),
            "agreement":     round(agreement, 4),
            "reject":        reject_reason,
            "seqs_used":     len(good_sequences),
            "health_score":  health_score["score"],
            "abnormalities": [a["pattern"] for a in detected_abnormalities],
        }
        append_audit_log(audit_entry)

        # ── Result cards ──────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if final_name != "Unknown":
                st.success(f"### Identity: {final_name}")
            else:
                st.error("### Unknown / Intruder")
                if reject_reason:
                    st.caption(f"Rejected: {reject_reason}")
                st.caption(f"Closest match: {best_name}")

        with col2:
            st.metric("Confidence",      f"{best_conf * 100:.2f}%")
            st.metric("Entropy (norm.)", f"{norm_ent:.3f}")

        with col3:
            n_asym_flags = sum(
                1 for v in bio_stats["symmetry"].values()
                if not v["symmetric"]
            )
            st.metric("Symmetry flags",       f"{n_asym_flags} / 3")
            st.metric("Clinical range flags", len(bio_stats["clinical_flags"]))

        with col4:
            st.metric("Processing time",       f"{total_time:.1f}s")
            st.metric("Quality seqs / Total",  f"{len(good_sequences)} / {len(raw_sequences)}")

        # ── Gait Health Score banner ───────────────────────────────────────
        score_val   = health_score["score"]
        score_band  = health_score["band"]
        score_color = health_score["band_color"]
        n_patterns  = len(detected_abnormalities)

        if score_color == "green":
            st.success(
                f"### 🏃 Gait Health Score: **{score_val}%** — {score_band}"
                + (f"  |  {n_patterns} pattern(s) detected" if n_patterns else "  |  No abnormalities detected")
            )
        elif score_color == "orange":
            st.warning(
                f"### 🏃 Gait Health Score: **{score_val}%** — {score_band}"
                + (f"  |  {n_patterns} pattern(s) detected" if n_patterns else "")
            )
        else:
            st.error(
                f"### 🏃 Gait Health Score: **{score_val}%** — {score_band}"
                + (f"  |  {n_patterns} pattern(s) detected" if n_patterns else "")
            )

        # Score component breakdown
        with st.expander("Health score breakdown", expanded=False):
            hc1, hc2, hc3, hc4, hc5 = st.columns(5)
            hc1.metric("Overall",    f"{score_val}%",
                       help="Composite 0–100% score")
            hc2.metric("Symmetry",   f"{health_score['sym_component']}%",
                       help="Left-right balance (30%)")
            hc3.metric("ROM",        f"{health_score['rom_component']}%",
                       help="Range of motion coverage (25%)")
            hc4.metric("Clinical",   f"{health_score['clinical_component']}%",
                       help="Features in normal range (25%)")
            hc5.metric("Smoothness", f"{health_score['smooth_component']}%",
                       help="Temporal consistency (20%)")
            if health_score["penalty"] > 0:
                st.caption(
                    f"Abnormality penalty applied: −{health_score['penalty']} pts "
                    f"({n_patterns} pattern(s) × severity weight)"
                )

        # ── Decision gates ─────────────────────────────────────────────────
        with st.expander("Decision gates & debug", expanded=True):
            colA, colB, colC, colD = st.columns(4)
            with colA:
                st.metric("Best confidence", f"{best_conf * 100:.2f}%")
                st.metric("Second best",     f"{second_conf * 100:.2f}%")
            with colB:
                st.metric("Margin",         f"{margin:.4f}")
                st.metric("Vote agreement", f"{agreement:.4f}")
            with colC:
                st.metric("Norm. entropy",  f"{norm_ent:.4f}")
                st.metric("Sequences used", len(good_sequences))
            with colD:
                st.metric("Seqs dropped", bad_count)
                st.metric("Voted name",   voted_name)

            if identity_override:
                st.info(
                    f"Per-identity override active for: **{best_name}** (agreement gate only)"
                )

            override_tag = "  [override]" if identity_override else ""
            gates = {
                "Margin gate": (
                    "Pass" if margin    >= min_margin_used else "Fail",
                    f"{margin:.3f} >= {min_margin_used:.3f}{override_tag}",
                ),
                "Entropy gate": (
                    "Pass" if norm_ent  <= max_ent_used   else "Fail",
                    f"{norm_ent:.3f} <= {max_ent_used:.3f}",
                ),
                "Confidence gate": (
                    "Pass" if best_conf >= min_conf_used  else "Fail",
                    f"{best_conf:.3f} >= {min_conf_used:.3f}",
                ),
                "Agreement gate": (
                    "Pass" if agreement >= min_agr_used   else "Fail",
                    f"{agreement:.3f} >= {min_agr_used:.3f}{override_tag}",
                ),
            }
            st.table({k: list(v) for k, v in gates.items()})

        # ── Class probabilities ────────────────────────────────────────────
        with st.expander("Class probabilities"):
            prob_dict = {
                le.classes_[i]: f"{float(avg_probs[i]) * 100:.2f}%"
                for i in np.argsort(avg_probs)[::-1]
            }
            st.table(prob_dict)

        # ── Weighted voting breakdown ──────────────────────────────────────
        with st.expander("Weighted voting breakdown"):
            vote_weights_display = {}
            for pred in preds:
                cls  = int(np.argmax(pred))
                name = le.inverse_transform([cls])[0]
                vote_weights_display[name] = (
                    vote_weights_display.get(name, 0.0) + float(pred[cls])
                )
            total_w = sum(vote_weights_display.values())
            vote_table = {
                k: f"{v:.3f} ({v / total_w * 100:.1f}%)"
                for k, v in sorted(
                    vote_weights_display.items(), key=lambda x: x[1], reverse=True
                )
            }
            st.table(vote_table)

        # ── Temporal feature plots ─────────────────────────────────────────
        st.subheader("📈 Temporal gait analysis")
        st.plotly_chart(
            plot_temporal_features(frames_array),
            use_container_width=True,
        )

        # ── Feature box plots ──────────────────────────────────────────────
        st.plotly_chart(
            plot_feature_boxplots(frames_array),
            use_container_width=True,
        )

        # ── Sequence confidence timeline ───────────────────────────────────
        st.subheader("🔍 Per-sequence decision analysis")
        st.plotly_chart(
            plot_sequence_confidence_timeline(preds, le),
            use_container_width=True,
        )

        # ── Prediction heatmap ─────────────────────────────────────────────
        with st.expander("Softmax probability heatmap"):
            st.plotly_chart(
                plot_prediction_heatmap(preds, le),
                use_container_width=True,
            )

        # ── Biomechanical analysis ─────────────────────────────────────────
        st.subheader("🦿 Biomechanical analysis")
        col_r, col_s = st.columns([1, 1])
        with col_r:
            st.plotly_chart(
                plot_radar_feature_profile(bio_stats),
                use_container_width=True,
            )
        with col_s:
            st.plotly_chart(
                plot_symmetry(bio_stats["symmetry"]),
                use_container_width=True,
            )

        # ── Abnormality detection panel ────────────────────────────────────
        st.markdown("**🔬 Gait abnormality detection**")

        if detected_abnormalities:
            for ab in detected_abnormalities:
                sev  = ab["severity"]
                lat  = ab.get("laterality", "Bilateral")
                conf = ab.get("confidence", 0.0)
                icon = "🟡" if sev == "Mild" else ("🟠" if sev == "Moderate" else "🔴")

                lat_badge = (
                    f" · 🦵 {lat}" if lat not in ("Bilateral", "N/A") else
                    (" · ↔ Bilateral" if lat == "Bilateral" else "")
                )
                conf_badge = f" · Confidence: {conf:.0%}"

                with st.expander(
                    f"{icon} {ab['pattern']}  —  {sev}{lat_badge}{conf_badge}",
                    expanded=True
                ):
                    st.markdown(f"**What it means:** {ab['description']}")
                    st.info(f"💡 **Recommendation:** {ab['recommendation']}")
        else:
            st.success("✅ No significant gait abnormality patterns detected.")

        st.divider()

        # ── Clinical flags ─────────────────────────────────────────────────
        st.markdown("**Clinical range assessment**")
        if bio_stats["clinical_flags"]:
            flag_df = pd.DataFrame(bio_stats["clinical_flags"])
            st.dataframe(flag_df, use_container_width=True)
        else:
            st.success("All features within normal clinical walking ranges.")

        # ── Symmetry detail table ──────────────────────────────────────────
        with st.expander("Left–right symmetry detail"):
            sym_rows = []
            for label, vals in bio_stats["symmetry"].items():
                sym_rows.append({
                    "Joint":          label,
                    "Left mean (°)":  round(vals["left_mean"],       2),
                    "Right mean (°)": round(vals["right_mean"],      2),
                    "SI (%)":         round(vals["symmetry_index"],  2),
                    "Status":         "✅ Symmetric" if vals["symmetric"] else "⚠️ Asymmetric",
                })
            st.dataframe(pd.DataFrame(sym_rows), use_container_width=True)

        # ── Feature stats table ────────────────────────────────────────────
        with st.expander("Full feature statistics"):
            stat_rows = []
            for feat in FEATURE_NAMES:
                lo, hi = CLINICAL_RANGES[feat]
                m = bio_stats[feat]["mean"]
                stat_rows.append({
                    "Feature":      feat,
                    "Mean (°)":     round(m, 2),
                    "Std (°)":      round(bio_stats[feat]["std"], 2),
                    "ROM (°)":      round(bio_stats[feat]["rom"], 2),
                    "Normal range": f"{lo}–{hi}°",
                    "In range":     "✅" if lo <= m <= hi else "⚠️",
                })
            st.dataframe(pd.DataFrame(stat_rows), use_container_width=True)

        # ── Skeleton overlay video export ──────────────────────────────────
        if save_skeleton_video and skeleton_frames:
            with st.spinner("Writing skeleton overlay video…"):
                out_path = tempfile.mktemp(suffix="_skeleton.mp4")
                fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
                writer   = cv2.VideoWriter(
                    out_path, fourcc, fps,
                    (frame_w, frame_h)
                )
                for sf in skeleton_frames:
                    if sf.shape[0] != frame_h or sf.shape[1] != frame_w:
                        sf = cv2.resize(sf, (frame_w, frame_h))
                    writer.write(sf)
                writer.release()

                with open(out_path, "rb") as vf:
                    video_bytes = vf.read()

            st.subheader("🎥 Skeleton overlay video")
            st.video(video_bytes)
            st.download_button(
                label="Download skeleton video",
                data=video_bytes,
                file_name=f"skeleton_{uploaded_file.name}",
                mime="video/mp4",
            )

        # ── Downloadable JSON report ───────────────────────────────────────
        st.subheader("📄 Analysis report")
        report = build_report(
            uploaded_file.name, final_name, reject_reason,
            debug, bio_stats, len(good_sequences), len(raw_sequences), total_time,
            health_score=health_score,
            detected_abnormalities=detected_abnormalities,
        )
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="Download JSON report",
            data=report_json,
            file_name=f"gait_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
        )

        with st.expander("Preview report"):
            st.json(report)

        # ── Features used ──────────────────────────────────────────────────
        with st.expander("Features used"):
            st.markdown(
                "| # | Feature | Description |\n"
                "|---|---------|-------------|\n"
                "| 1 | Right knee flexion       | Hip–Knee–Ankle angle |\n"
                "| 2 | Left knee flexion        | Hip–Knee–Ankle angle |\n"
                "| 3 | Right hip flexion        | Shoulder–Hip–Knee angle |\n"
                "| 4 | Left hip flexion         | Shoulder–Hip–Knee angle |\n"
                "| 5 | Right ankle dorsiflexion | Knee–Ankle–Foot angle |\n"
                "| 6 | Left ankle dorsiflexion  | Knee–Ankle–Foot angle |\n"
                "| 7 | Torso tilt               | Trunk lean from vertical |\n"
            )

# ==========================================
# AUDIT LOG VIEWER
# ==========================================
if show_audit_log:
    st.divider()
    st.subheader("📋 Prediction audit log")
    log_entries = load_audit_log()
    if log_entries:
        log_df = pd.DataFrame(log_entries)
        st.dataframe(log_df, use_container_width=True)

        accepted = log_df[log_df["result"] != "Unknown"]
        st.markdown(
            f"**Total predictions:** {len(log_df)}  |  "
            f"**Accepted:** {len(accepted)}  |  "
            f"**Rejected:** {len(log_df) - len(accepted)}"
        )

        if len(accepted) > 0:
            identity_counts = accepted["result"].value_counts()
            fig_hist = px.bar(
                identity_counts,
                title="Accepted predictions by identity",
                labels={"value": "Count", "index": "Identity"},
                template="plotly_white",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        st.download_button(
            label="Download full audit log (CSV)",
            data=log_df.to_csv(index=False),
            file_name="gait_audit_log.csv",
            mime="text/csv",
        )
    else:
        st.info("No audit log entries yet. Run a prediction to populate it.")

# ==========================================
# LANDING PAGE
# ==========================================
else:
    st.info("Upload a walking video to begin.")

    with st.expander("System overview"):
        st.markdown("""
        **Recognition pipeline**
        1. MediaPipe extracts 33 body landmarks per frame.
        2. 7 joint angles are computed (bilateral knees, hips, ankles, torso tilt).
        3. Sliding-window sequences (length 60, stride 6) are normalised and fed to the LSTM.
        4. Four cascaded rejection gates enforce open-set detection.

        **Rejection gates (in order)**
        - Gate 1 – Margin: top-2 confidence gap. Primary intruder signal.
        - Gate 2 – Entropy: flat softmax distribution → unknown person.
        - Gate 3 – Confidence: top class probability floor.
        - Gate 4 – Agreement: weighted vote convergence.

        **Abnormality detection (v2)**
        - Bilateral patterns: crouch, stiff-knee, bilateral foot drop, equinus, hip deficit, Trendelenburg.
        - Unilateral patterns: right/left foot drop, right/left knee stiffness.
        - Asymmetry pattern: antalgic (pain-avoidance) gait.
        - Per-pattern confidence scored on clinically relevant features only.
        - Severity escalation when ≥3 patterns co-occur.
        - Laterality badge shown per finding.

        **Biomechanical analysis**
        - Temporal plots of all 7 joint angles over time.
        - Left–right symmetry index with clinical thresholds.
        - Feature profile radar chart vs normal walking ranges.
        - Clinical range flagging for each feature.
        - Feature distribution box plots.

        **Decision analysis**
        - Per-sequence confidence timeline bar chart.
        - Softmax heatmap (sequences × classes).

        **Export**
        - Skeleton overlay video with joint angle annotations.
        - Downloadable JSON analysis report.
        - Persistent audit log with CSV export.
        """)