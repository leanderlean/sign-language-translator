import numpy as np
import pandas as pd
import json
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CSV_PATH = os.path.join(DATA_DIR, 'asl_landmarks_final.csv')
JSON1_PATH = os.path.join(BASE_DIR, '..', 'asl_alphabet.json')
JSON2_PATH = os.path.join(BASE_DIR, '..', 'asl_alphabet_original.json')
MSASL_PATH = os.path.join(DATA_DIR, 'msasl_greetings.json')
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

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

def compute_finger_curls(pts):
    """Measure how curled each finger is: distance from fingertip to MCP joint."""
    fingers = [(4, 0), (8, 0), (12, 0), (16, 0), (20, 0)]
    return np.array([np.linalg.norm(pts[tip] - pts[base]) for tip, base in fingers])

def extract_features(row_vals):
    pts = np.array(row_vals).reshape(21, 3)
    raw = np.array(row_vals)
    dists = compute_distances(pts)
    angles = compute_angles(pts)
    wrist = pts[0]
    rel = (pts - wrist).flatten()
    curls = compute_finger_curls(pts)
    return np.concatenate([raw, rel, dists, angles, curls])

def rotate_landmarks(pts, angle_deg):
    """Rotate landmarks around Z axis (camera plane)."""
    angle = np.radians(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    wrist = pts[0]
    centered = pts - wrist
    rotated = centered @ rot.T
    return rotated + wrist

def scale_landmarks(pts, scale_factor):
    """Scale landmarks relative to wrist."""
    wrist = pts[0]
    centered = pts - wrist
    scaled = centered * scale_factor
    return scaled + wrist

def augment(pts, label, n_aug=5):
    samples = [pts]
    labels = [label]
    rng = np.random.RandomState(abs(hash(str(pts[:3].tolist()) + str(label))) % 2**31)
    for _ in range(n_aug):
        aug = pts.copy()
        aug = rotate_landmarks(aug, rng.uniform(-15, 15))
        aug = scale_landmarks(aug, rng.uniform(0.85, 1.15))
        noise = rng.normal(0, rng.uniform(0.002, 0.008), aug.shape)
        aug = aug + noise
        samples.append(aug)
        labels.append(label)
    return samples, labels

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

def build_feature_matrix(raw_data, labels, n_aug=5):
    X_list, y_list = [], []
    for i in range(len(raw_data)):
        pts = raw_data[i].reshape(21, 3)
        aug_pts, aug_labels = augment(pts, labels[i], n_aug=n_aug)
        for ap, al in zip(aug_pts, aug_labels):
            feats = extract_features(ap.flatten())
            X_list.append(feats)
            y_list.append(al)
    return np.array(X_list), np.array(y_list)

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
    print("MSASL (greetings): not found")

# Combine all raw data
sources_data = [csv_data, json1_data, json2_data]
sources_labels = [csv_labels, json1_labels, json2_labels]
if msasl_data is not None:
    sources_data.append(msasl_data)
    sources_labels.append(msasl_labels)

all_data = np.concatenate(sources_data)
all_labels = np.concatenate(sources_labels)

print(f"\nCombined raw dataset: {len(all_data)} samples, {len(set(all_labels))} classes")
print(f"Classes: {sorted(set(all_labels))}")

# Use more augmentation for classes with fewer samples
label_counts = pd.Series(all_labels).value_counts()
median_count = label_counts.median()
n_aug_map = {label: max(10, int(median_count / count * 3)) for label, count in label_counts.items() if count < median_count}
print(f"\nClasses needing extra augmentation: {len(n_aug_map)}")
for label, aug in sorted(n_aug_map.items()):
    print(f"  {label}: {label_counts[label]} samples -> {aug}x augmentation")

# Build feature matrix with adaptive augmentation
X_list, y_list = [], []
counts_used = {}
for i in range(len(all_data)):
    label = all_labels[i]
    pts = all_data[i].reshape(21, 3)
    n_aug = n_aug_map.get(label, 5)
    aug_pts, aug_labels = augment(pts, label, n_aug=n_aug)
    for ap, al in zip(aug_pts, aug_labels):
        feats = extract_features(ap.flatten())
        X_list.append(feats)
        y_list.append(al)
    counts_used[label] = counts_used.get(label, 0) + 1

X_all = np.array(X_list)
y_all = np.array(y_list)

print(f"\nAfter augmentation: {len(X_all)} samples")
print(f"Min samples per class: {min(pd.Series(y_all).value_counts())}")
print(f"Max samples per class: {max(pd.Series(y_all).value_counts())}")

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_all)

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_encoded, test_size=0.15, random_state=RANDOM_STATE, stratify=y_encoded
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Hyperparameter tuning with limited grid
print("\n" + "=" * 60)
print("Hyperparameter tuning...")
print("=" * 60)

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [15, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
}

base_rf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=RANDOM_STATE)

# Use a smaller grid search on a stratified subset for speed
grid = GridSearchCV(
    base_rf, param_grid,
    cv=3, scoring='balanced_accuracy', n_jobs=-1, verbose=1
)
grid.fit(X_train_s, y_train)

rf = grid.best_estimator_
print(f"Best params: {grid.best_params_}")
print(f"Best CV score: {grid.best_score_:.4f}")

y_pred = rf.predict(X_test_s)
acc = np.mean(y_pred == y_test)
print(f'\nTest accuracy: {acc:.4f}')
print()
print(classification_report(y_test, y_pred, target_names=encoder.classes_))

# Per-class accuracy
print("\nPer-class accuracy:")
report = classification_report(y_test, y_pred, target_names=encoder.classes_, output_dict=True)
for cls in sorted(report.keys()):
    if cls not in ['accuracy', 'macro avg', 'weighted avg']:
        prec = report[cls]['precision']
        recl = report[cls]['recall']
        f1 = report[cls]['f1-score']
        support = report[cls]['support']
        print(f"  {cls:15s}: prec={prec:.3f} recall={recl:.3f} f1={f1:.3f} support={int(support)}")

# Save model
joblib.dump({
    'model_rf': rf, 'encoder': encoder, 'scaler': scaler,
    'classes': encoder.classes_, 'n_classes': len(encoder.classes_)
}, MODEL_PATH)
print(f'\nModel saved to: {MODEL_PATH}')
