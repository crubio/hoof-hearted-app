# Hoof Hearted 🏇

AI-powered horse racing handicapper. Paste raw race program data, get an analysis.

## Stack

- **FastAPI** + **Jinja2** + **HTMX** + **Pico CSS**
- **OpenAI SDK** → GitHub Models endpoint (`gpt-4o` with `gpt-5` fallback)
- `uv` for dependency management

## Setup

**Prerequisites:** Python 3.13+, [`uv`](https://docs.astral.sh/uv/), a GitHub PAT with `models:read` scope.

```bash
git clone <repo-url>
cd hoof-hearted-app
uv sync
cp .env.example .env   # then fill in your GITHUB_TOKEN
```

## Run

```bash
uv run uvicorn main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), paste race data, click **Analyze**.

## Environment

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub PAT with `models:read` scope |
| `MODEL` | Primary model (default: `gpt-4o`, 50 req/day) |
| `MODEL_UPGRADE` | Fallback on rate limit (default: `gpt-5`, 8 req/day) |
| `MODEL_PROTOTYPE` | Optional dev model (e.g. `gpt-4o-mini`, 150 req/day) |
