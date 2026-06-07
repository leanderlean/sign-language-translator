import numpy as np
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'asl_landmarks_final.csv')
JSON1_PATH = os.path.join(BASE_DIR, '..', 'asl_alphabet.json')
JSON2_PATH = os.path.join(BASE_DIR, '..', 'asl_alphabet_original.json')
MSASL_PATH = os.path.join(DATA_DIR, 'msasl_greetings.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, 'encoder.pkl')
SCALER_PATH = os.path.join(BASE_DIR, 'scaler.pkl')

RANDOM_STATE = 42

KEY_PAIRS = [
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (4, 8), (4, 12), (4, 16), (4, 20),
    (8, 12), (12, 16), (16, 20),
    (3, 7), (7, 11), (11, 15), (15, 19),
    (2, 6), (6, 10), (10, 14), (14, 18),
]

def compute_distances(pts):
    dists = []
    for i, j in KEY_PAIRS:
        d = np.linalg.norm(pts[i] - pts[j])
        dists.append(d)
    return np.array(dists)

def compute_angles(pts):
    angles = []
    for i, j, k in [(0, 4, 8), (4, 8, 12), (8, 12, 16), (12, 16, 20),
                    (0, 5, 9), (5, 9, 13), (9, 13, 17),
                    (1, 5, 9), (2, 6, 10), (3, 7, 11)]:
        v1 = pts[i] - pts[j]
        v2 = pts[k] - pts[j]
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norm == 0:
            angles.append(0)
        else:
            cos_a = np.clip(np.dot(v1, v2) / norm, -1, 1)
            angles.append(np.arccos(cos_a))
    return np.array(angles)

def extract_features(row_vals):
    pts = np.array(row_vals).reshape(21, 3)
    raw = np.array(row_vals)
    dists = compute_distances(pts)
    angles = compute_angles(pts)
    wrist = pts[0]
    rel = (pts - wrist).flatten()
    return np.concatenate([raw, rel, dists, angles])

def augment(pts, label):
    samples = [pts]
    noise = np.random.normal(0, 0.005, pts.shape)
    samples.append(pts + noise)
    return samples, [label] * len(samples)

def load_csv(path):
    df = pd.read_csv(path)
    feature_cols = [c for c in df.columns if c != 'label']
    data = df[feature_cols].values
    labels = df['label'].values
    return data, labels

def load_json(path):
    with open(path) as f:
        records = json.load(f)
    data = np.array([r['features'] for r in records])
    labels = np.array([r['label'] for r in records])
    return data, labels

def build_feature_matrix(raw_data, labels):
    X_list, y_list = [], []
    for i in range(len(raw_data)):
        pts = raw_data[i].reshape(21, 3)
        aug_pts, aug_labels = augment(pts, labels[i])
        for ap, al in zip(aug_pts, aug_labels):
            feats = extract_features(ap.flatten())
            X_list.append(feats)
            y_list.append(al)
    return np.array(X_list), np.array(y_list)

# ─── Load all data sources ─────────────────────────────────────
print("=" * 60)
print("Loading datasets...")
print("=" * 60)

csv_data, csv_labels = load_csv(CSV_PATH)
print(f"CSV  (Kaggle) : {len(csv_data)} samples, {len(set(csv_labels))} classes")

json1_data, json1_labels = load_json(JSON1_PATH)
print(f"JSON1 (alphabet): {len(json1_data)} samples, {len(set(json1_labels))} classes")

json2_data, json2_labels = load_json(JSON2_PATH)
print(f"JSON2 (original): {len(json2_data)} samples, {len(set(json2_labels))} classes")

msasl_data, msasl_labels = None, None
if os.path.exists(MSASL_PATH):
    msasl_data, msasl_labels = load_json(MSASL_PATH)
    print(f"MSASL (greetings): {len(msasl_data)} samples, {len(set(msasl_labels))} classes")
else:
    print(f"MSASL (greetings): not found (run preprocess_msasl.py first)")

# ─── Phase 1: Train on CSV, validate on JSON ──────────────────
print("\n" + "=" * 60)
print("Phase 1: Real-World Validation (Train on CSV, Validate on JSON)")
print("=" * 60)

X_csv, y_csv = build_feature_matrix(csv_data, csv_labels)
X_json1, y_json1 = build_feature_matrix(json1_data, json1_labels)
X_json2, y_json2 = build_feature_matrix(json2_data, json2_labels)

# Combine both JSON sources
X_json = np.concatenate([X_json1, X_json2])
y_json = np.concatenate([y_json1, y_json2])

# Find overlapping classes between CSV and JSON
csv_classes = set(csv_labels)
json_classes = set(np.concatenate([json1_labels, json2_labels]))
overlap = sorted(csv_classes & json_classes)
extra_json = sorted(json_classes - csv_classes)
extra_csv = sorted(csv_classes - json_classes)
print(f"Overlap classes: {overlap} ({len(overlap)})")
print(f"JSON-only classes (not in CSV): {extra_json or 'none'}")
print(f"CSV-only classes (not in JSON): {extra_csv or 'none'}")

# Train on all CSV data
encoder_phase1 = LabelEncoder()
y_csv_enc = encoder_phase1.fit_transform(y_csv)

scaler_phase1 = StandardScaler()
X_csv_s = scaler_phase1.fit_transform(X_csv)

rf_phase1 = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)
rf_phase1.fit(X_csv_s, y_csv_enc)

# Validate on JSON overlap classes only
mask_overlap = np.isin(y_json, overlap)
X_json_overlap = X_json[mask_overlap]
y_json_overlap = y_json[mask_overlap]

# Encode JSON labels using the CSV-trained encoder (only overlap classes present)
# Filter to only labels the encoder knows about
known_mask = np.isin(y_json_overlap, encoder_phase1.classes_)
X_json_overlap = X_json_overlap[known_mask]
y_json_overlap = y_json_overlap[known_mask]

y_json_enc = encoder_phase1.transform(y_json_overlap)
X_json_s = scaler_phase1.transform(X_json_overlap)

y_json_pred = rf_phase1.predict(X_json_s)
json_acc = np.mean(y_json_pred == y_json_enc)
print(f"\nReal-world accuracy on JSON data: {json_acc:.3f} ({len(X_json_overlap)} samples)")
print()
present_labels = np.unique(y_json_enc)
present_names = encoder_phase1.classes_[present_labels]
print(classification_report(y_json_enc, y_json_pred, labels=present_labels, target_names=present_names))

# ─── Phase 2: Combined training ────────────────────────────────
print("\n" + "=" * 60)
print("Phase 2: Combined Training (CSV + JSON1 + JSON2)")
print("=" * 60)

# Combine all raw data
sources_data = [csv_data, json1_data, json2_data]
sources_labels = [csv_labels, json1_labels, json2_labels]
if msasl_data is not None:
    sources_data.append(msasl_data)
    sources_labels.append(msasl_labels)

all_data = np.concatenate(sources_data)
all_labels = np.concatenate(sources_labels)

print(f"Combined dataset: {len(all_data)} samples, {len(set(all_labels))} classes")
print(f"Classes: {sorted(set(all_labels))}")

X_all, y_all = build_feature_matrix(all_data, all_labels)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_encoded, test_size=0.15, random_state=RANDOM_STATE, stratify=y_encoded
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

rf = RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=-1, random_state=RANDOM_STATE)
rf.fit(X_train_s, y_train)

y_pred = rf.predict(X_test_s)
acc = np.mean(y_pred == y_test)

print(f'\nCombined test accuracy: {acc:.3f}')
print()
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Per-class accuracy for key letters
print("\nPer-class accuracy (key letters):")
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
for cls in ['A', 'E'] + (['HELLO', 'I LOVE YOU', 'NO', 'YES'] if 'HELLO' in encoder.classes_ else []):
    if cls in report:
        prec = report[cls]['precision']
        recl = report[cls]['recall']
        f1 = report[cls]['f1-score']
        print(f"  {cls:15s}: precision={prec:.3f}, recall={recl:.3f}, f1={f1:.3f}")

# ─── Save model ────────────────────────────────────────────────
joblib.dump({
    'model_rf': rf, 'encoder': encoder, 'scaler': scaler,
    'classes': encoder.classes_, 'n_classes': len(encoder.classes_)
}, MODEL_PATH)
print(f'\nModel saved to: {MODEL_PATH}')
