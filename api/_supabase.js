const SUPABASE_URL = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL || ''
const SUPABASE_KEY = process.env.VITE_SUPABASE_PUBLISHABLE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY || ''

function requireSupabaseConfig() {
  if (!SUPABASE_URL || !SUPABASE_KEY) {
    throw new Error('Supabase environment variables are missing. Set SUPABASE_URL and SUPABASE_KEY (or VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY) in your environment.')
  }
}

async function supabaseRequest(path, options = {}) {
  const response = await fetch(`${SUPABASE_URL}/rest/v1${path}`, {
    ...options,
    headers: {
      apikey: SUPABASE_KEY,
      Authorization: `Bearer ${SUPABASE_KEY}`,
      'Content-Type': 'application/json',
      Prefer: 'return=representation',
      ...(options.headers || {})
    }
  })
  const text = await response.text()
  let data = null
  if (text) {
    try { data = JSON.parse(text) } catch { data = text }
  }
  if (!response.ok) {
    const message = data?.message || data?.error || text || `Supabase request failed with ${response.status}`
    const error = new Error(message)
    error.status = response.status
    error.details = data
    throw error
  }
  return data
}

async function ensureGesture(label, numHands = 1) {
  const normalized = String(label || '').trim().toUpperCase()
  const lookup = await supabaseRequest(`/gestures?name=eq.${encodeURIComponent(normalized)}&select=id,name,num_hands&limit=1`, { method: 'GET' })
  if (Array.isArray(lookup) && lookup.length > 0) return lookup[0]
  const created = await supabaseRequest('/gestures', {
    method: 'POST',
    body: JSON.stringify({ name: normalized, owner_tag: 'backend', is_public: true, num_hands: numHands })
  })
  return Array.isArray(created) ? created[0] : created
}

async function listAllSamples() {
  return supabaseRequest('/samples?select=id,gesture_id,num_hands,handedness,landmarks,features,image_path,source,quality,created_at,gestures(name)&order=created_at.asc', { method: 'GET' })
}

async function deleteSamplesByGestureName(label) {
  const normalized = String(label || '').trim().toUpperCase()
  const lookup = await supabaseRequest(`/gestures?name=eq.${encodeURIComponent(normalized)}&select=id&limit=1`, { method: 'GET' })
  if (!Array.isArray(lookup) || lookup.length === 0) return 0
  const gestureId = lookup[0].id
  const result = await supabaseRequest(`/samples?gesture_id=eq.${gestureId}`, { method: 'DELETE' })
  await supabaseRequest(`/gestures?id=eq.${gestureId}`, { method: 'DELETE' })
  return Array.isArray(result) ? result.length : 1
}

export { requireSupabaseConfig, supabaseRequest, ensureGesture, listAllSamples, deleteSamplesByGestureName }
