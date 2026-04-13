import markdown as md
from fastapi import FastAPI, Form, Request
import json
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from app.scraper import get_todays_tracks, check_api_health, get_track_and_race_program

from app.analyzer import analyze, analyze_program_json, AnalyzerError

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
- Race programs are cached in memory for the lifetime of the server process
- The legacy `/analyze` POST endpoint accepts raw text and returns HTML (used by the HTMX app)
    """,
    version="0.2.0",
    contact={
        "name": "Hoof Hearted API",
    },
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:4173",  # Vite preview
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# simple caching for beta version.
program_cache = {}

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/health/api")
async def api_health_check():
    """
    Check if public data is available.
    Returns 200 if healthy, 503 if unhealthy, and includes details about response time and any errors.
    """
    health_status = await check_api_health()
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
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "error_type": type(e).__name__},
            status_code=500
          )
    
@app.get("/program/{track_id}/{race_n}")
async def program(track_id: str, race_n: int):
    '''
    Fetches a full program from publicly available APIs for a specific track and race number.
     
    - **track_id**: BRIS track code (e.g. `kee`, `op`, `cd`) — case insensitive
    - **race_n**: Race number (e.g. `6`)

    The result is cached in memory. Calling `/analyze/{track_id}/{race_n}` afterward
    will use the cached data and skip this fetch.
    '''
    print(f"Received request for program data: track_id={track_id}, race_n={race_n}")
    try:
        program_data = await get_track_and_race_program(track_id, race_n)

        # Cache for later analysis
        cache_key = f"{track_id.lower()}_{race_n}"
        program_cache[cache_key] = program_data

        return JSONResponse(content=program_data)
    except Exception as e:
        return JSONResponse(
            content={"error": str(e), "error_type": type(e).__name__},
            status_code=500
          )
    
@app.post("/analyze/{track_id}/{race_n}")
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
        "model": "gpt-4o",
        "tokens": { "prompt": 1200, "completion": 800, "total": 2000 },
        "elapsed_ms": 3100
      },
      "analysis": "## Selections\\n..."
    }
    ```
    Analysis is returned as a markdown string.
    '''

    cache_key = f"{track_id.lower()}_{race_n}"

    try:
      if cache_key not in program_cache:
            print(f"Cache miss for {cache_key}, fetching program data")
            program_cache[cache_key] = await get_track_and_race_program(track_id, race_n)

      result = analyze_program_json(program_cache[cache_key])

      return JSONResponse(
        content={
            "success": True,
            "analysis": result["text"],
            "meta": {
                "track": track_id.upper(),
                    "race": race_n,
                    "cached_as": cache_key,
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
        content={"error": str(e)},
        status_code=200  # Return 200 so HTMX can handle it gracefully
      )
    except Exception as e:
      return JSONResponse(
        content={"error": str(e), "error_type": type(e).__name__},
        status_code=500
      )

@app.post("/analyze", response_class=HTMLResponse)
async def analyze_race(request: Request, race_data: str = Form(...), data_only: bool = False):
    '''
    **Legacy endpoint** — used by the HTMX web app.

    Accepts raw race program text via form POST and returns an HTML partial
    for HTMX to swap into the page. Pass `data_only=true` to get a JSON
    response instead (markdown string).

    Deprecated: Not intended for direct API use — use `/analyze/{track_id}/{race_n}` instead.
    '''
    try:
        result = analyze(race_data)
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
            status_code=200,  # Return 200 so HTMX swaps it in normally
        )
