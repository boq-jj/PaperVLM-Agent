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

The frontend can run in a static demo mode without a backend. To use real PDF
processing and question answering, start the Python backend from the project
root:

```powershell
python scripts\run_api.py --host 127.0.0.1 --port 8000
```

Then set the frontend API base URL to:

```text
VITE_PAPERVLM_API_BASE_URL=http://127.0.0.1:8000
```

For deployed frontend environments such as Netlify, set
`VITE_PAPERVLM_API_BASE_URL` to the public URL of the deployed Python backend.

The repository backend provides:

- `GET /health`
- `POST /api/process-pdf`
- `POST /api/ask`
- `POST /api/ask-visual`

Do not put `DASHSCOPE_API_KEY` in the frontend. It must stay on the backend.
