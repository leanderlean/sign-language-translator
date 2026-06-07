import { requireSupabaseConfig, listAllSamples } from './_supabase.js'

export default async function handler(req, res) {
  if (req.method !== 'GET') return res.status(405).json({ error: 'Method not allowed' })
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
}
