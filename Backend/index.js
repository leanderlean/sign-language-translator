import 'dotenv/config'
import express from 'express'
import cors from 'cors'
import helmet from 'helmet'
import morgan from 'morgan'
import path from 'path'
import { fileURLToPath } from 'url'
import { spawn } from 'child_process'
import { fetch as undiciFetch } from 'undici'

// ensure `fetch` is available (some Node deployments use older runtimes)
if (!globalThis.fetch) {
    globalThis.fetch = undiciFetch
}

const app = express()
app.use(express.json({ limit: '10mb' }))
// Security headers
app.use(helmet())

// Logging
app.use(morgan(process.env.LOG_FORMAT || 'combined'))

// CORS origin control (comma-separated list in ALLOWED_ORIGINS)
const ALLOWED_ORIGINS = process.env.ALLOWED_ORIGINS ? process.env.ALLOWED_ORIGINS.split(',').map(s => s.trim()) : null;
app.use(cors({
    origin: function (origin, callback) {
        if (!ALLOWED_ORIGINS || !origin) return callback(null, true);
        if (ALLOWED_ORIGINS.includes(origin)) return callback(null, true);
        return callback(new Error('CORS origin not allowed'));
    }
}))
const PORT = process.env.PORT || process.env.BACKEND_PORT || 3000

const SUPABASE_URL = process.env.VITE_SUPABASE_URL || process.env.SUPABASE_URL || ''
const SUPABASE_KEY = process.env.VITE_SUPABASE_PUBLISHABLE_KEY || process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY || ''

function requireSupabaseConfig(res) {
    if (!SUPABASE_URL || !SUPABASE_KEY) {
        res.status(500).json({
            error:
                'Supabase environment variables are missing. Set SUPABASE_URL and SUPABASE_KEY (or VITE_SUPABASE_URL / VITE_SUPABASE_PUBLISHABLE_KEY) in your environment.'
        })
        return false
    }
    return true
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
        try {
            data = JSON.parse(text)
        } catch (err) {
            data = text
        }
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
    const lookup = await supabaseRequest(`/gestures?name=eq.${encodeURIComponent(normalized)}&select=id,name,num_hands&limit=1`, {
        method: 'GET'
    })

    if (Array.isArray(lookup) && lookup.length > 0) {
        return lookup[0]
    }

    const created = await supabaseRequest('/gestures', {
        method: 'POST',
        body: JSON.stringify({
            name: normalized,
            owner_tag: 'backend',
            is_public: true,
            num_hands: numHands
        })
    })

    return Array.isArray(created) ? created[0] : created
}

async function listAllSamples() {
    return supabaseRequest(
        '/samples?select=id,gesture_id,num_hands,handedness,landmarks,features,image_path,source,quality,created_at,gestures(name)&order=created_at.asc',
        { method: 'GET' }
    )
}

async function listApprovedSamples(gestureName = null) {
    if (!gestureName) {
        return supabaseRequest(
            '/samples?approved=eq.true&select=id,gesture_id,num_hands,handedness,landmarks,features,image_path,source,quality,created_at,gestures(name)&order=created_at.asc',
            { method: 'GET' }
        )
    }

    const gestures = await supabaseRequest(
        `/gestures?name=eq.${encodeURIComponent(String(gestureName).trim().toUpperCase())}&select=id,name&limit=1`,
        { method: 'GET' }
    )

    if (!Array.isArray(gestures) || gestures.length === 0) {
        return []
    }

    return supabaseRequest(
        `/samples?approved=eq.true&gesture_id=eq.${gestures[0].id}&select=id,gesture_id,num_hands,handedness,landmarks,features,image_path,source,quality,created_at,gestures(name)&order=created_at.asc`,
        { method: 'GET' }
    )
}

app.get('/health', (req, res) => {
    res.json({ ok: true })
})

app.post('/api/training/sample', async (req, res) => {
    try {
        if (!requireSupabaseConfig(res)) return

        const {
            label,
            landmarks,
            features,
            numHands = 1,
            handedness = 'Right',
            imagePath = null,
            source = 'webcam',
            quality = 0,
            approved = false,
            uploaderTag = 'anonymous',
            sessionId = null
        } = req.body || {}

        const normalizedLabel = String(label || '').trim().toUpperCase()
        if (!normalizedLabel) {
            return res.status(400).json({ error: 'label is required' })
        }

        if (!Array.isArray(landmarks) || landmarks.length === 0) {
            return res.status(400).json({ error: 'landmarks array is required' })
        }

        if (!Array.isArray(features) || features.length === 0) {
            return res.status(400).json({ error: 'features array is required' })
        }

        const gesture = await ensureGesture(normalizedLabel, Number(numHands) || 1)

        const insertedSamples = await supabaseRequest('/samples', {
            method: 'POST',
            body: JSON.stringify({
                gesture_id: gesture.id,
                uploader_tag: String(uploaderTag || 'anonymous'),
                session_id: sessionId,
                num_hands: Number(numHands) || 1,
                handedness,
                landmarks,
                features,
                image_path: imagePath,
                source,
                quality: Number.isFinite(Number(quality)) ? Number(quality) : 0,
                approved: Boolean(approved)
            })
        })

        res.status(201).json({
            ok: true,
            gesture,
            sample: Array.isArray(insertedSamples) ? insertedSamples[0] : insertedSamples
        })
    } catch (error) {
        console.error('Failed to save training sample:', error)
        res.status(error.status || 500).json({
            error: 'Failed to save training sample',
            message: error.message
        })
    }
})

async function handleExportApprovedSamples(req, res) {
    try {
        if (!requireSupabaseConfig(res)) return

        const gesture = req.query.gesture ? String(req.query.gesture) : null
        const samples = await listApprovedSamples(gesture)

        const normalized = Array.isArray(samples)
            ? samples.map((sample) => ({
                id: sample.id,
                gesture_id: sample.gesture_id,
                gesture_name: sample.gestures?.name || null,
                num_hands: sample.num_hands,
                handedness: sample.handedness,
                landmarks: sample.landmarks,
                features: sample.features,
                image_path: sample.image_path,
                source: sample.source,
                quality: sample.quality,
                created_at: sample.created_at
            }))
            : []

        res.json({
            ok: true,
            count: normalized.length,
            gesture: gesture,
            samples: normalized
        })
    } catch (error) {
        console.error('Failed to export approved samples:', error)
        res.status(error.status || 500).json({
            error: 'Failed to export approved samples',
            message: error.message
        })
    }
}

app.get('/api/training/export', handleExportApprovedSamples)

app.get('/api/training/export/approved', handleExportApprovedSamples)

app.get('/api/training/samples', async (req, res) => {
    try {
        if (!requireSupabaseConfig(res)) return
        const samples = await listAllSamples()
        const normalized = Array.isArray(samples)
            ? samples.map(s => ({
                id: s.id,
                gesture_id: s.gesture_id,
                gesture_name: s.gestures?.name || null,
                num_hands: s.num_hands,
                handedness: s.handedness,
                landmarks: s.landmarks,
                features: s.features,
                image_path: s.image_path,
                source: s.source,
                quality: s.quality,
                created_at: s.created_at
            }))
            : []
        res.json({ ok: true, count: normalized.length, samples: normalized })
    } catch (error) {
        console.error('Failed to fetch samples:', error)
        res.status(error.status || 500).json({ error: 'Failed to fetch samples', message: error.message })
    }
})

app.get("/", (req, res) => {
    res.send("Hello from the backend")
})

// Serve frontend when built (Vite outputs to ../dist)
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
if (process.env.NODE_ENV === 'production') {
    const distPath = path.join(__dirname, '..', 'dist');
    app.use(express.static(distPath));
    app.get(/.*/, (req, res) => res.sendFile(path.join(distPath, 'index.html')));
}

// ML inference endpoint — calls Python predict script
app.post('/api/ml/predict', (req, res) => {
    const { landmarks } = req.body || {}
    if (!Array.isArray(landmarks) || landmarks.length === 0) {
        return res.status(400).json({ error: 'landmarks array is required' })
    }

    const py = spawn('python', [
        'ml/predict.py'
    ], { cwd: path.join(__dirname, '..') })

    let stdout = '', stderr = ''
    py.stdout.on('data', d => stdout += d)
    py.stderr.on('data', d => stderr += d)
    py.on('close', code => {
        if (code !== 0) {
            return res.status(500).json({ error: 'ML prediction failed', details: stderr })
        }
        try {
            const result = JSON.parse(stdout)
            res.json({ ok: true, predictions: result })
        } catch {
            res.status(500).json({ error: 'Invalid response from model', raw: stdout })
        }
    })
    py.stdin.end(JSON.stringify({ landmarks }))
})

// Error handler
app.use((err, req, res, next) => {
    console.error('Unhandled error:', err && err.stack ? err.stack : err);
    if (res.headersSent) return next(err);
    res.status(err.status || 500).json({ error: err.message || 'Internal Server Error' });
});

const server = app.listen(PORT, () => {
    console.log(`Server is running on port: http://localhost:${PORT}`)
})

// Graceful shutdown
function shutdown() {
    console.log('Shutting down server...');
    server.close(() => {
        console.log('HTTP server closed.');
        process.exit(0);
    });
    setTimeout(() => {
        console.error('Forcing shutdown.');
        process.exit(1);
    }, 10000).unref();
}

process.on('SIGTERM', shutdown);
process.on('SIGINT', shutdown);

