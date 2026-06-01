Deployment

Quick steps to deploy backend and frontend:

- Backend (Node.js):

  1. Copy environment variables into `Backend/.env` from `Backend/.env.example`.
  2. Install and start:

```bash
cd Backend
npm install
npm start
```

  - Use `PORT` to override the listening port.
  - Set `ALLOWED_ORIGINS` to a comma-separated list of allowed frontend origins.

- Frontend (Vite):

```bash
npm install
npm run build
npm run preview # or serve the `dist` directory with your static host
```

Production tip: build the frontend (`npm run build`) and set `NODE_ENV=production` for the backend; the backend will serve `dist/` automatically when `NODE_ENV=production`.

Optional: use containerization (Docker) or deploy to platform providers (Vercel/Netlify for frontend, Fly/Render/Heroku for backend).
