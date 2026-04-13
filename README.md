# Hoof Hearted 🏇

AI-powered horse racing handicapper. Fetches live race program data and returns an AI analysis.

## Stack

### API
- **FastAPI** + **Jinja2** + **HTMX** (python web app)
- **OpenAI SDK** → GitHub Models endpoint (`gpt-4o` with `gpt-5` fallback)
- `uv` for dependency management

### UI
- **React 19** + **TypeScript** + **Vite**
- **TanStack Query** server side state mgmt
- **terminal.css** 

## Setup

**Prerequisites:** Python 3.13+, Node 18+, [`uv`](https://docs.astral.sh/uv/), a GitHub PAT with `models:read` scope.

```bash
git clone <repo-url>
cd hoof-hearted-app
uv sync
cp .env.example .env   # then fill in your GITHUB_TOKEN
```

## Run

**API**
```bash
uv run uvicorn main:app --reload
```
API runs at [http://localhost:8000](http://localhost:8000)
API docs at [http://localhost:8000/docs](http://localhost:8000/docs)

**UI**
```bash
cd ui
npm install
npm run dev
```
UI runs at [http://localhost:5173](http://localhost:5173)

## Workflow

1. **`GET /tracks`** — fetch today's available tracks
2. **`GET /program/{track_id}/{race_n}`** — fetch and cache a race program
3. **`POST /analyze/{track_id}/{race_n}`** — run AI analysis (uses cache if available)

`track_id` is the BRIS code (e.g. `kee`, `op`, `cd`) — case insensitive.

Supported tracks: Keeneland (`kee`), Santa Anita (`sa`), Aqueduct (`aqu`), Oaklawn Park (`op`), Churchill Downs (`cd`)

## Environment

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub PAT with `models:read` scope |
| `MODEL` | Primary model (default: `gpt-4o`) |
| `MODEL_UPGRADE` | Fallback on rate limit (default: `gpt-5`) |
| `MODEL_PROTOTYPE` | Optional dev model (default: `gpt-4o-mini`) |
| `DEV` | Set to `true` to use `MODEL_PROTOTYPE` |
| `VITE_API_URL` | UI API base URL (default: `http://localhost:8000`) |