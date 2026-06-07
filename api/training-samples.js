import { requireSupabaseConfig, listAllSamples, deleteSamplesByGestureName } from './_supabase.js'

export default async function handler(req, res) {
  if (req.method === 'GET') {
    try {
      requireSupabaseConfig()
      const samples = await listAllSamples()
      const normalized = Array.isArray(samples) ? samples.map(s => ({
        id: s.id, gesture_id: s.gesture_id, gesture_name: s.gestures?.name || null,
        num_hands: s.num_hands, handedness: s.handedness, landmarks: s.landmarks,
        features: s.features, image_path: s.image_path, source: s.source,
        quality: s.quality, created_at: s.created_at
      })) : []
      res.json({ ok: true, count: normalized.length, samples: normalized })
    } catch (error) {
      console.error('Failed to fetch samples:', error)
      res.status(error.status || 500).json({ error: 'Failed to fetch samples', message: error.message })
    }
    return
  }

  if (req.method === 'DELETE') {
    try {
      requireSupabaseConfig()
      const { label } = req.body || {}
      if (!label) return res.status(400).json({ error: 'label is required' })
      const deleted = await deleteSamplesByGestureName(label)
      res.json({ ok: true, deleted })
    } catch (error) {
      console.error('Failed to delete samples:', error)
      res.status(error.status || 500).json({ error: 'Failed to delete samples', message: error.message })
    }
    return
  }

  res.status(405).json({ error: 'Method not allowed' })
}
