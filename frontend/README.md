# PaperVLM-Agent Frontend

This is the Netlify-ready Vite + React frontend for PaperVLM-Agent.
It is currently a static SPA, so it does not require a Netlify framework adapter.

## Local Development

```powershell
cd frontend
npm install
npm run dev
```

## Build

```powershell
cd frontend
npm run build
```

## Backend API

The frontend can run in a static demo mode without a backend. To connect it to a Python backend, set:

```text
VITE_PAPERVLM_API_BASE_URL=https://your-backend.example.com
```

Expected backend endpoints:

- `POST /api/process-pdf`
- `POST /api/ask`
- `POST /api/ask-visual`

Do not put `DASHSCOPE_API_KEY` in the frontend. It must stay on the backend.
