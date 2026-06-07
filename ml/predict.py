import sys
import json
import numpy as np
import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

pkg = joblib.load(MODEL_PATH)
rf = pkg['model_rf']
encoder = pkg['encoder']
scaler = pkg['scaler']
classes = pkg['classes']

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
    fingers = [(4, 0), (8, 0), (12, 0), (16, 0), (20, 0)]
    return np.array([np.linalg.norm(pts[tip] - pts[base]) for tip, base in fingers])

def extract_features(row_vals):
    pts = np.array(row_vals).reshape(21, 3)
    raw = row_vals
    dists = compute_distances(pts)
    angles = compute_angles(pts)
    wrist = pts[0]
    rel = (pts - wrist).flatten()
    curls = compute_finger_curls(pts)
    return np.concatenate([raw, rel, dists, angles, curls])

def predict(landmarks):
    feats = extract_features(landmarks)
    feats_s = scaler.transform(feats.reshape(1, -1))
    probs = rf.predict_proba(feats_s)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    results = [
        {'label': classes[i], 'confidence': float(probs[i])}
        for i in top3_idx
    ]
    return results

if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        import csv
        csv_path = os.path.join(BASE_DIR, 'data', 'asl_landmarks_final.csv')
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            row = next(reader)
        landmarks = [float(row[f'{c}{i}']) for i in range(21) for c in ['x','y','z']]
        print(f'Actual: {row["label"]}')
        print(json.dumps(predict(landmarks), indent=2))
    else:
        data = json.loads(sys.stdin.read())
        result = predict(data['landmarks'])
        print(json.dumps(result))
