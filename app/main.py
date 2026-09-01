import time
import markdown as md
import httpx
from pathlib import Path
from fastapi import Depends, FastAPI, Form, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
import os
from typing import Optional
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.scraper import get_todays_tracks, check_api_health, get_track_and_race_program, FILTERED_TRACKS, todays_race_date

from app.analyzer import analyze, analyze_program_json, AnalyzerError, TOKEN as ANALYZER_TOKEN

# Anchor static/template dirs to this file's location, not the process's working directory --
# `uvicorn app.main:app` from outside the repo root would otherwise fail at import time.
BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(
    title="Hoof Hearted API",
    description="""
Horse racing analysis API powered by publicly available program data and AI.

## Workflow
1. **`/tracks`** — get today's available tracks
2. **`/program/{track_id}/{race_n}`** — fetch and cache a race program
3. **`/analyze/{track_id}/{race_n}`** — run AI analysis on a race (uses cache if available)

## Notes
- `track_id` is the BRIS code (e.g. `kee`, `op`, `cd`)
- Race programs are cached in memory for a few minutes, per process
- The legacy `/analyze` POST endpoint accepts raw text and returns HTML (used by the HTMX app)
- Set `ANALYZE_API_KEY` to require an `X-API-Key` header on the LLM-spending `/analyze*` routes
    """,
    version="0.2.0",
    contact={
        "name": "Hoof Hearted API",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    *[o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",")],  # comma-separated list
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o for o in origins if o],  # filter out empty strings
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# In-memory cache with a short TTL -- race data (scratches, odds) changes during the day,
# so this only saves the redundant refetch between /program and /analyze on the same race,
# not a lasting snapshot. Not shared across uvicorn workers; fine for the single-worker beta.
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "300"))
program_cache = {}  # cache_key -> (fetched_at_monotonic, program_data)

# Optional shared-secret gate for the LLM-spending endpoints. Unset by default so local dev
# keeps working with no extra setup; set ANALYZE_API_KEY to stop any visitor from being able
# to drain the shared daily model quota.
ANALYZE_API_KEY = os.getenv("ANALYZE_API_KEY")


async def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if ANALYZE_API_KEY and x_api_key != ANALYZE_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


def _normalize_track_id(track_id: str) -> str:
    """Lowercase and validate against the supported track list before it's used
    anywhere -- as a cache key or interpolated into the upstream TwinSpires URL."""
    normalized = track_id.strip().lower()
    if normalized not in FILTERED_TRACKS:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown track_id '{track_id}'. Supported: {', '.join(sorted(FILTERED_TRACKS))}",
        )
    return normalized


def _validate_race_n(race_n: int) -> None:
    if not (1 <= race_n <= 20):
        raise HTTPException(status_code=400, detail="race_n must be between 1 and 20")


def _cache_get(cache_key: str):
    entry = program_cache.get(cache_key)
    if entry is None:
        return None
    fetched_at, data = entry
    if time.monotonic() - fetched_at > CACHE_TTL_SECONDS:
        del program_cache[cache_key]
        return None
    return data


def _cache_set(cache_key: str, data) -> None:
    program_cache[cache_key] = (time.monotonic(), data)


async def _fetch_program_upstream(track_id: str, race_n: int):
    """Fetch from TwinSpires, mapping a 404 there to a 404 here instead of a bare 500."""
    race_date = todays_race_date()
    try:
        return race_date, await get_track_and_race_program(track_id, race_n, race_date)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            raise HTTPException(status_code=404, detail="No program found for that track/race/date") from e
        print(f"Upstream error fetching program: {e}")
        raise HTTPException(status_code=502, detail="Upstream data provider error") from e
    except httpx.HTTPError as e:
        print(f"Upstream request failed fetching program: {e}")
        raise HTTPException(status_code=502, detail="Upstream data provider is unreachable") from e


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health/api")
async def api_health_check():
    """
    Check if public data is available, and whether an LLM token is configured.
    Returns 200 if healthy, 503 if unhealthy, and includes details about response time and any errors.

    Note: `llm_configured` only checks that OPENROUTER_API_KEY is set, not that it is still valid --
    verifying the token against the model API would spend part of the shared daily quota.
    """
    health_status = await check_api_health()
    health_status["llm_configured"] = bool(ANALYZER_TOKEN)
    status_code = 200 if health_status["status"] == "healthy" else 503
    return JSONResponse(content=health_status, status_code=status_code)

@app.get("/tracks")
async def tracks():
    '''
    Tracks fetched from public APIs, ordered by next up and coming race.

    Each track includes its BRIS code, track name, and available race numbers.
    Use the `brisCode` as the `track_id` in subsequent calls.
    '''
    print("Received request for today's tracks")
    try:
        tracks_data = await get_todays_tracks()
        return JSONResponse(content=tracks_data)
    except httpx.HTTPError as e:
        print(f"Upstream error fetching tracks: {e}")
        return JSONResponse(content={"error": "Upstream data provider is unreachable"}, status_code=502)
    except Exception as e:
        print(f"Unexpected error fetching tracks: {e}")
        return JSONResponse(content={"error": "Unexpected server error"}, status_code=500)

@app.get("/program/{track_id}/{race_n}")
async def program(track_id: str, race_n: int):
    '''
    Fetches a full program from publicly available APIs for a specific track and race number.

    - **track_id**: BRIS track code (e.g. `kee`, `op`, `cd`) — case insensitive
    - **race_n**: Race number (e.g. `6`)

    The result is cached in memory for a few minutes. Calling `/analyze/{track_id}/{race_n}`
    afterward will use the cached data and skip this fetch.
    '''
    track_id = _normalize_track_id(track_id)
    _validate_race_n(race_n)
    print(f"Received request for program data: track_id={track_id}, race_n={race_n}")

    race_date, program_data = await _fetch_program_upstream(track_id, race_n)
    cache_key = f"{race_date}_{track_id}_{race_n}"
    _cache_set(cache_key, program_data)

    return JSONResponse(content=program_data)

@app.post("/analyze/{track_id}/{race_n}", dependencies=[Depends(require_api_key)])
async def analyze_race_program(track_id: str, race_n: int):
    '''
    Runs AI handicapping analysis on a race program.

    - **track_id**: BRIS track code (e.g. `kee`, `op`, `cd`) — case insensitive
    - **race_n**: Race number (e.g. `6`)

    If the program is already cached (via `/program/{track_id}/{race_n}`), it will
    be used directly. Otherwise the program is fetched automatically first.

    ### Response
    ```json
    {
      "success": true,
      "meta": {
        "track": "KEE",
        "race": 6,
        "cache_hit": true,
        "model": "z-ai/glm-5.2:free",
        "tokens": { "prompt": 7160, "completion": 6513, "total": 13673 },
        "elapsed_ms": 31500
      },
      "analysis": "## Contenders\\n..."
    }
    ```
    Free-tier models are noticeably slower than a paid API — 20-40s per analysis is typical.
    Analysis is returned as a markdown string. If `ANALYZE_API_KEY` is set, requires a matching
    `X-API-Key` header — this endpoint spends the shared daily model quota.
    '''
    track_id = _normalize_track_id(track_id)
    _validate_race_n(race_n)

    race_date = todays_race_date()
    cache_key = f"{race_date}_{track_id}_{race_n}"

    cached = _cache_get(cache_key)
    cache_hit = cached is not None
    if not cache_hit:
        print(f"Cache miss for {cache_key}, fetching program data")
        _, program_data = await _fetch_program_upstream(track_id, race_n)
        _cache_set(cache_key, program_data)
    else:
        program_data = cached

    try:
        # analyze_program_json uses the synchronous OpenAI client -- run it off the event
        # loop so one slow analysis (3-10s) doesn't stall every other in-flight request.
        result = await run_in_threadpool(analyze_program_json, program_data)

        return JSONResponse(
            content={
                "success": True,
                "analysis": result["text"],
                "meta": {
                    "track": track_id.upper(),
                    "race": race_n,
                    "cache_hit": cache_hit,
                    "model": result["model"],
                    "tokens": {
                        "prompt": result["prompt_tokens"],
                        "completion": result["completion_tokens"],
                        "total": result["total_tokens"],
                    },
                    "elapsed_ms": result["elapsed_ms"],
                }
            },
            headers={
                "X-Model-Used": result["model"],
                "X-Tokens-Prompt": str(result["prompt_tokens"]),
                "X-Tokens-Completion": str(result["completion_tokens"]),
                "X-Tokens-Total": str(result["total_tokens"]),
                "X-Response-Time-Ms": str(result["elapsed_ms"]),
            }
        )
    except AnalyzerError as e:
        return JSONResponse(
            content={"success": False, "error": str(e)},
            status_code=e.status_code,
        )
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        return JSONResponse(
            content={"success": False, "error": "Unexpected server error"},
            status_code=500,
        )

@app.post("/analyze", response_class=HTMLResponse, dependencies=[Depends(require_api_key)])
async def analyze_race(request: Request, race_data: str = Form(...), data_only: bool = False):
    '''
    **Legacy endpoint** — used by the HTMX web app.

    Accepts raw race program text via form POST and returns an HTML partial
    for HTMX to swap into the page. Pass `data_only=true` to get a JSON
    response instead (markdown string).

    Deprecated: Not intended for direct API use — use `/analyze/{track_id}/{race_n}` instead.
    If `ANALYZE_API_KEY` is set, requires a matching `X-API-Key` header.
    '''
    try:
        # Same rationale as the JSON endpoint: keep the blocking OpenAI call off the event loop.
        result = await run_in_threadpool(analyze, race_data)
        result_html = md.markdown(result["text"], extensions=["tables", "nl2br"])

        # use data_only True to return a JSON response.
        if data_only:
            return JSONResponse(content={"markdown": result["text"]}, headers={
                "X-Model-Used": result["model"],
                "X-Tokens-Prompt": str(result["prompt_tokens"]),
                "X-Tokens-Completion": str(result["completion_tokens"]),
                "X-Tokens-Total": str(result["total_tokens"]),
                "X-Response-Time-Ms": str(result["elapsed_ms"]),
            })

        return templates.TemplateResponse(
            "partials/analysis.html",
            {"request": request, "result": result_html},
            headers={
                "X-Model-Used": result["model"],
                "X-Tokens-Prompt": str(result["prompt_tokens"]),
                "X-Tokens-Completion": str(result["completion_tokens"]),
                "X-Tokens-Total": str(result["total_tokens"]),
                "X-Response-Time-Ms": str(result["elapsed_ms"]),
            },
        )
    except AnalyzerError as e:
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": str(e)},
            status_code=200,  # 200 so HTMX swaps the error partial in normally
        )
    except Exception as e:
        print(f"Unexpected error in legacy /analyze: {e}")
        return templates.TemplateResponse(
            "partials/error.html",
            {"request": request, "message": "Unexpected server error. Please try again."},
            status_code=200,  # 200 so HTMX swaps the error partial in normally
        )
