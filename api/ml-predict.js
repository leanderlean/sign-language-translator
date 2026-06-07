import fs from 'fs'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))
const MODEL_PATH = path.join(__dirname, '..', 'ml', 'model.json')

const model = JSON.parse(fs.readFileSync(MODEL_PATH, 'utf-8'))

const KEY_PAIRS = [
  [0, 4], [0, 8], [0, 12], [0, 16], [0, 20],
  [4, 8], [4, 12], [4, 16], [4, 20],
  [8, 12], [12, 16], [16, 20],
  [3, 7], [7, 11], [11, 15], [15, 19],
  [2, 6], [6, 10], [10, 14], [14, 18],
]

const ANGLE_TRIPLES = [
  [0, 4, 8], [4, 8, 12], [8, 12, 16], [12, 16, 20],
  [0, 5, 9], [5, 9, 13], [9, 13, 17],
  [1, 5, 9], [2, 6, 10], [3, 7, 11],
]

function dist(a, b) {
  return Math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
}

function computeDistances(pts) {
  return KEY_PAIRS.map(([i, j]) => dist(pts[i], pts[j]))
}

function computeAngles(pts) {
  return ANGLE_TRIPLES.map(([i, j, k]) => {
    const v1 = [pts[i][0] - pts[j][0], pts[i][1] - pts[j][1], pts[i][2] - pts[j][2]]
    const v2 = [pts[k][0] - pts[j][0], pts[k][1] - pts[j][1], pts[k][2] - pts[j][2]]
    const dot = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
    const n1 = Math.sqrt(v1[0] ** 2 + v1[1] ** 2 + v1[2] ** 2)
    const n2 = Math.sqrt(v2[0] ** 2 + v2[1] ** 2 + v2[2] ** 2)
    if (n1 === 0 || n2 === 0) return 0
    const cos = Math.max(-1, Math.min(1, dot / (n1 * n2)))
    return Math.acos(cos)
  })
}

function extractFeatures(rowVals) {
  const pts = []
  for (let i = 0; i < 21; i++) {
    pts.push([rowVals[i * 3], rowVals[i * 3 + 1], rowVals[i * 3 + 2]])
  }
  const raw = rowVals
  const wrist = pts[0]
  const rel = pts.flatMap(p => [p[0] - wrist[0], p[1] - wrist[1], p[2] - wrist[2]])
  const dists = computeDistances(pts)
  const angles = computeAngles(pts)
  return [...raw, ...rel, ...dists, ...angles]
}

function predict(landmarks) {
  const feats = extractFeatures(landmarks)
  const scaled = feats.map((v, i) => (v - model.scaler_mean[i]) / model.scaler_scale[i])
  const probs = new Array(model.n_classes).fill(0)

  for (const tree of model.trees) {
    let node = 0
    while (tree.children_left[node] !== -1) {
      node = scaled[tree.feature[node]] <= tree.threshold[node]
        ? tree.children_left[node]
        : tree.children_right[node]
    }
    const counts = tree.value[node]
    for (let c = 0; c < counts.length; c++) {
      probs[c] += counts[c]
    }
  }

  const total = probs.reduce((s, v) => s + v, 0)
  const normalized = probs.map(v => v / total)
  const indices = normalized.map((v, i) => i).sort((a, b) => normalized[b] - normalized[a])

  return indices.slice(0, 3).map(i => ({
    label: model.classes[i],
    confidence: normalized[i],
  }))
}

export default function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }
  try {
    const { landmarks } = req.body || {}
    if (!Array.isArray(landmarks) || landmarks.length === 0) {
      return res.status(400).json({ error: 'landmarks array is required' })
    }
    if (landmarks.length !== 63) {
      return res.status(400).json({ error: 'landmarks must contain 63 values (21 × 3)' })
    }
    const predictions = predict(landmarks)
    res.json({ ok: true, predictions })
  } catch (error) {
    console.error('ML prediction failed:', error)
    res.status(500).json({ error: 'ML prediction failed', message: error.message })
  }
}
