import { requireSupabaseConfig, supabaseRequest, ensureGesture } from './_supabase.js'

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' })
  }
  try {
    requireSupabaseConfig()
    const {
      label, landmarks, features, numHands = 1, handedness = 'Right',
      imagePath = null, source = 'webcam', quality = 0, approved = true,
      uploaderTag = 'anonymous', sessionId = null
    } = req.body || {}
    const normalizedLabel = String(label || '').trim().toUpperCase()
    if (!normalizedLabel) return res.status(400).json({ error: 'label is required' })
    if (!Array.isArray(landmarks) || landmarks.length === 0) return res.status(400).json({ error: 'landmarks array is required' })
    if (!Array.isArray(features) || features.length === 0) return res.status(400).json({ error: 'features array is required' })
    const gesture = await ensureGesture(normalizedLabel, Number(numHands) || 1)
    const insertedSamples = await supabaseRequest('/samples', {
      method: 'POST',
      body: JSON.stringify({
        gesture_id: gesture.id, uploader_tag: String(uploaderTag || 'anonymous'),
        session_id: sessionId, num_hands: Number(numHands) || 1, handedness,
        landmarks, features, image_path: imagePath, source,
        quality: Number.isFinite(Number(quality)) ? Number(quality) : 0,
        approved: Boolean(approved)
      })
    })
    res.status(201).json({ ok: true, gesture, sample: Array.isArray(insertedSamples) ? insertedSamples[0] : insertedSamples })
  } catch (error) {
    console.error('Failed to save training sample:', error)
    res.status(error.status || 500).json({ error: 'Failed to save training sample', message: error.message })
  }
}
