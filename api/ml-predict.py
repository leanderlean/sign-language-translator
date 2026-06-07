import json
import os
from http.server import BaseHTTPRequestHandler
import numpy as np
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'ml', 'model.pkl')

pkg = joblib.load(MODEL_PATH)
rf = pkg['model_rf']
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

def extract_features(row_vals):
    pts = np.array(row_vals).reshape(21, 3)
    raw = row_vals
    dists = compute_distances(pts)
    angles = compute_angles(pts)
    wrist = pts[0]
    rel = (pts - wrist).flatten()
    return np.concatenate([raw, rel, dists, angles])

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length)
        try:
            data = json.loads(body)
            landmarks = data.get('landmarks')
            if not isinstance(landmarks, list) or len(landmarks) == 0:
                raise ValueError('landmarks array is required')
            feats = extract_features(landmarks)
            feats_s = scaler.transform(feats.reshape(1, -1))
            probs = rf.predict_proba(feats_s)[0]
            top3_idx = np.argsort(probs)[-3:][::-1]
            results = [{'label': classes[i], 'confidence': float(probs[i])} for i in top3_idx]
            response = json.dumps({'ok': True, 'predictions': results})
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
        except Exception as e:
            response = json.dumps({'error': 'ML prediction failed', 'message': str(e)})
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(response.encode())
