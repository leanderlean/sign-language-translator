#!/usr/bin/env python3
"""Export the trained RandomForest model to a compact JSON file for JS inference."""
import json
import joblib
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model.pkl')

pkg = joblib.load(MODEL_PATH)
rf = pkg['model_rf']
scaler = pkg['scaler']
classes = pkg['classes'].tolist() if hasattr(pkg['classes'], 'tolist') else list(pkg['classes'])

def export_tree(tree):
    """Export a single DecisionTree to a compact dict."""
    t = tree.tree_
    # value.shape = (nodes, n_classes, n_outputs=1) → flatten last dim
    values = t.value.reshape(t.value.shape[0], -1).tolist()
    return {
        'children_left': t.children_left.tolist(),
        'children_right': t.children_right.tolist(),
        'feature': t.feature.tolist(),
        'threshold': t.threshold.tolist(),
        'value': values,
    }

forest = [export_tree(est) for est in rf.estimators_]

model_json = {
    'n_classes': int(rf.n_classes_),
    'n_features': int(rf.n_features_in_),
    'classes': classes,
    'n_estimators': len(forest),
    'trees': forest,
    'scaler_mean': scaler.mean_.tolist(),
    'scaler_scale': scaler.scale_.tolist(),
}

OUT_PATH = os.path.join(BASE_DIR, 'model.json')
with open(OUT_PATH, 'w') as f:
    json.dump(model_json, f, separators=(',', ':'))

print(f'Exported {len(forest)} trees to {OUT_PATH}')
print(f'  Nodes total: {sum(len(t["children_left"]) for t in forest)}')
print(f'  Classes: {len(classes)}')
print(f'  Features: {rf.n_features_in_}')
print(f'  File size: {os.path.getsize(OUT_PATH) / 1024 / 1024:.1f} MB')
