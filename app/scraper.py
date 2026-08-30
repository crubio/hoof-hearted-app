import httpx
from typing import List, Dict, Any, Optional
from datetime import date


TWINSPIRES_BASE_URL = "https://www.twinspires.com"
TODAYS_TRACKS_URL = f"{TWINSPIRES_BASE_URL}/adw/todays-tracks?affid=2800&sortOrder=nextUp"
TRACK_PROGRAM_URL_TEMPLATE = f"{TWINSPIRES_BASE_URL}/adw/track/{{track_id}}/program?affid=2800"
TIMEOUT = 10.0  # seconds
# track_id -> track name. These are hardcoded tracks for demo.
FILTERED_TRACKS = {
    "sa": "Santa Anita",
    "cd": "Churchill Downs",
    "bel": "Belmont Park",
    "kee": "Keeneland",
    "aqu": "Aqueduct",
    "op": "Oaklawn Park",
    "dmr": "Del Mar"
}


HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.twinspires.com/",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

async def check_api_health() -> Dict[str, Any]:
    """
    Check if the TwinSpires API is accessible and returning valid data.
    
    Returns:
        Dict with status, response_time, and track_count
    """
    import time
    
    try:
        start_time = time.time()
        async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
            response = await client.get(TODAYS_TRACKS_URL)
            elapsed_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 200:
                data = response.json() 
                return {
                    "status": "healthy",
                    "status_code": 200,
                    "response_time_ms": elapsed_ms,
                    "track_count": len(data) if isinstance(data, list) else 0,
                    "data_type": type(data).__name__
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "response_time_ms": elapsed_ms,
                    "error": f"Unexpected status code: {response.status_code}"
                }
    except httpx.TimeoutException:
        return {
            "status": "timeout",
            "error": "Request timed out after 10 seconds"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "error_type": type(e).__name__
        }


async def get_todays_tracks() -> List[Dict[str, Any]]:
    """
    Fetch today's tracks ordered by next up from TwinSpires.
    
    Note: This endpoint doesn't require authentication for basic track data.
    If you get 403/401 errors, you may need to add session cookies.
    
    Returns:
        List of track dictionaries containing name, brisCode, currentRaceNumber, status, etc.
    """
    async with httpx.AsyncClient(
        timeout=30.0,
        headers=HEADERS,
        follow_redirects=True
    ) as client:
        response = await client.get(TODAYS_TRACKS_URL)
        
        if response.status_code != 200:
            print(f"Response headers: {response.headers}")
            print(f"Response text: {response.text[:1000]}")
        
        response.raise_for_status()
        
        data = response.json()
        filtered_data = [
            track for track in data
            if track.get("brisCode", "").lower() in FILTERED_TRACKS
        ]

        if not isinstance(data, list):
            raise ValueError(f"Unexpected response format, got {type(data).__name__}")
        
        return filtered_data

async def get_track_and_race_program(track_id: str, race_n: int = 1, race_date: Optional[str] = None) -> Dict[str, Any]:
    """
    Fetch the program json for a track. This is valid the day fetched.

    Example:
    - URL to build: https://www.twinspires.com/apigw/cdux-program-api/programs/racedate/2026-04-11/track/aqu/type/TB/race/8
    - Build url with {YYY-MM-DDDD}, {brisCode}, {currentRaceNumber}
    - If no currentRaceNumber, there might be a finalRaceNumber, indicating the race alraedy happened.
    - track, race & runners all in one dataset. This is essentially what we're pasting in the current python app.
    
    Returns:
        Dict containing detailed track program information.
    """
    # Resolve per-call, not at import — a long-running server must not pin to its boot date.
    race_date = race_date or date.today().isoformat()
    print(f"Fetching program for track_id={track_id}, race_n={race_n}, date={race_date}")
    url = f"{TWINSPIRES_BASE_URL}/apigw/cdux-program-api/programs/racedate/{race_date}/track/{track_id}/type/TB/race/{race_n}"
    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()