# Hoof Hearted 🏇

AI-powered horse racing handicapper. Fetches live race program data and returns an AI analysis.

## Stack

### API
- **FastAPI** + **Jinja2** + **HTMX** (python web app)
- **OpenAI SDK** → [OpenRouter](https://openrouter.ai) (free-tier `:free` models, no card required)
- `uv` for dependency management

### UI
- **React 19** + **TypeScript** + **Vite**
- **TanStack Query** server side state mgmt
- **terminal.css** 

## Setup

**Prerequisites:** Python 3.13+, Node 18+, [`uv`](https://docs.astral.sh/uv/), an [OpenRouter](https://openrouter.ai) account.

```bash
git clone <repo-url>
cd hoof-hearted-app
uv sync
cp .env.example .env   # then fill in your OPENROUTER_API_KEY (openrouter.ai/settings/keys)
```

## Run

**API**
```bash
uv run uvicorn app.main:app --reload
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

Supported tracks: Santa Anita (`sa`), Churchill Downs (`cd`), Belmont Park (`bel`), Keeneland (`kee`),
Aqueduct (`aqu`), Oaklawn Park (`op`), Del Mar (`dmr`)

## Environment

| Variable | Description |
|---|---|
| `OPENROUTER_API_KEY` | OpenRouter API key — create at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) |
| `MODEL` | Primary model (default: `z-ai/glm-5.2:free`, verified live for prompt-compliant output). OpenRouter's own `openrouter/free` auto-router was tried and rejected — it can route to a non-chat model. |
| `MODEL_UPGRADE` | Tried once if the primary is rate-limited and a same-model retry also fails (default: `google/gemma-4-31b-it:free`, a different lab) |
| `MODEL_PROTOTYPE` | Optional dev model (default: `z-ai/glm-5.2:free`) |
| `DEV` | Set to `true` to use `MODEL_PROTOTYPE` |
| `OPENROUTER_SITE_URL` / `OPENROUTER_SITE_NAME` | Optional attribution headers for OpenRouter's leaderboards — cosmetic only |
| `VITE_API_URL` | UI API base URL (default: `http://localhost:8000`) |
| `ANALYZE_API_KEY` | Optional shared secret required as `X-API-Key` on `/analyze*` routes. Unset by default; set before deploying anywhere public. |
| `ALLOWED_ORIGINS` | Comma-separated extra CORS origins, beyond `localhost:5173`/`4173` |
| `CACHE_TTL_SECONDS` | How long a fetched race program stays cached in memory (default: `300`) |

**Free tier limits:** OpenRouter's `:free` models allow 50 requests/day per account by default,
or 1000/day once you've purchased $10+ in lifetime credit (the free models themselves stay $0
either way — the credit only raises the daily cap). Free-tier 429s are often a transient
shared-provider-pool limit rather than the daily cap — `analyze()`/`analyze_program_json()`
honor the provider's `Retry-After` hint and retry the same model once before falling back to
`MODEL_UPGRADE`. The free model roster itself rotates roughly weekly; if `MODEL` or
`MODEL_UPGRADE` stop working, check [openrouter.ai/models?max_price=0](https://openrouter.ai/models?max_price=0)
for a current replacement.