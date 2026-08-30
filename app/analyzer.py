import os
import re
import time
import json
from datetime import date
from typing import Dict, Any, Optional
from openai import OpenAI, RateLimitError, AuthenticationError, APIError
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("GITHUB_TOKEN")
DEV_MODE = os.getenv("DEV", "false").lower() == "true"
LLM = os.getenv("MODEL_PROTOTYPE", "gpt-4o-mini") if DEV_MODE else os.getenv("MODEL", "gpt-4o")
LLM_UPGRADE = os.getenv("MODEL_UPGRADE", "gpt-5")


def _current_year() -> int:
    """Racing age is calendar year minus foaling year, so this must not be hardcoded."""
    return date.today().year


class AnalyzerError(Exception):
    """Clean user-facing error from the analyzer."""
    pass


def _looks_like_hallucination(text: str) -> bool:
    """Basic sanity check — flag output that is too short or contains
    non-Latin garbage characters (Cyrillic, CJK, control chars, etc.)"""
    if len(text.strip()) < 100:
        return True
    # Flag if more than 5% of chars are outside printable ASCII + common punctuation
    non_latin = re.findall(r'[^\x00-\x7F\u2013\u2014\u2019\u201c\u201d]', text)
    if len(non_latin) / max(len(text), 1) > 0.05:
        return True
    return False


def _validate_input(text: str) -> None:
    """Reject input that doesn't look like race program data before hitting the API."""
    stripped = text.strip()

    if len(stripped) < 30:
        raise AnalyzerError(
            "Input is too short. Please paste the full race program data including "
            "the race header (race type, purse, distance, surface) and horse entries."
        )

    # Require at least one odds-style token — universal to any race format
    has_odds = bool(re.search(r'M:\s*\d|\d/\d', stripped))

    if not has_odds:
        raise AnalyzerError(
            "Input appears incomplete. Make sure to include the full race header "
            "(race type, purse, distance, surface) and horse entries with at least "
            "post positions and morning line odds (e.g. M: 5/2)."
        )

# This prompt is for the all in one python app that renders templating based on markdown returned
# from analysis. This is deprecated but can still be used as a standalone app if spun up that way.
SYSTEM_PROMPT = """You are an expert horse racing handicapper. You will be given raw race program
data copied from a race-day program page.

## Input Structure

The input always begins with a race header block (track name, race number, post time, race type,
purse, age/sex conditions, distance, surface). This may come as a separate paste or precede the
data block.

The data section may optionally begin with a `#` token followed by column names — one per line.
Whether or not the `#` is present, identify the data view by recognizing column name keywords
in the input (e.g. "AVG SPD", "prime power", "early pace", "Jockey", "angles", "MED/WT/EQP").
Use whichever column names are present to map values to fields. Be flexible — column names may
appear as part of the text or as a header block; either way, use them to understand the data shape.

After any header/column names, horse entries appear as numbered blocks with one value per line.
Map each value to its field using the detected column order.

The user may paste one or more of these data views:

- **Summary** — columns include: ODDS, PL, Runner, DAYS OFF, RUN STYLE, AVG SPD, BACK SPD, SPD LR, AVG CLS, PRM PWR, W% JKY, W% TRN, $ — the richest single view
- **Speed** — columns include: ODDS, Runner, run style, Average Speed, Average Distance, best speed — speed figures with field ranks
- **Class** — columns include: ODDS, Runner, days off, prime power, last class, Average Class — class ratings with ranks
- **Pace** — columns include: ODDS, Runner, run style, early pace 1, early pace 2, late pace — pace figures with ranks
- **Adv** — columns include: ODDS, PL, Runner, Jockey, Trainer, Sire / Dam — full jockey/trainer win records and breeding
- **Basic** — columns include: ODDS, ML, PL, Runner, MED/WT/EQP, Jockey, Trainer — odds, equipment, jockey, trainer names only
- **Tips** — columns include: ODDS, PL, Runner, angles — expert angle tags (Hot Trainer, Top Pick, Clocker Special, etc.)

Run Style codes (letter + early speed points 0–8):
- **E** (Early) — vies for the early lead; typically cannot rate behind a pace setter
- **E/P** (Early/Presser) — runs 2nd-3rd within a few lengths early; unlike E, can rate behind a pace setter
- **P** (Presser) — middle-of-pack early, tries to run down the leader; rarely challenges for the lead early
- **S** (Sustain/Closer) — runs at the back of the pack early before closing
- **NA** (Not Available) — first-time starter or insufficient data to assess preferred run style

The number following the style letter (0–8) is the Early Speed Points rating: measures early speed
ability based on running position and beaten lengths at the first call of recent races.
Higher = more early speed shown. E.g. E6 = Early runner with high early speed; S0 = Closer with
no early speed points recorded.
Missing figures (— or blank) indicate a first-time starter or incomplete record.
Scratches may be indicated in the data.

## Output Format

For each race, return markdown with:

1. **Selections** — top 3 horses with brief justification. If data is thin (basic view only),
   weight jockey/trainer records and odds movement heavily.
2. **Single** — one horse to anchor multi-race bets, only if clearly justified.
3. **Race header** — include all available: track, race #, race type, purse, distance, surface,
   conditions. Note any fields not provided.
4. **Data available** — one line listing which views were detected (by their column headers) and
   any key fields absent. Sets expectations for analysis depth.
5. **Horse-by-horse breakdown** — analyze only the fields present. Note standout positives and
   negatives across whatever is available: speed/pace figure trends, class ratings, prime power,
   run style matchup, jockey/trainer records, days off, equipment changes, expert angles, and
   overlays (site odds significantly higher than ML = value). Distill into one or two concise sentences per horse.

At the end include:
- **Win bet** — the single strongest horse and why, if one clearly stands out.
- **Value/Overlay** — any horse where site odds are notably higher than ML that represents value.
- **Exotic use** — which horses to include in exactas, trifectas, or superfectas (wide vs. singled).

Format everything in clean markdown. Be concise but analytical — think like a sharp bettor,
not a tout. Prioritize numbers over opinions. Flag high-variance or missing-data horses clearly.

IMPORTANT: Only use the data provided in the input. Do not invent statistics, horse names,
jockey names, or any details not present in the raw data. If a field is missing, say so.

If the input is missing critical race context — specifically race type, purse, distance, or
surface — do not attempt an analysis. Instead, respond with a single short markdown paragraph
explaining what is missing and asking the user to re-paste the complete race header and horse
entries before you can proceed.
"""

# This the prompt for json driven analysis. This data comes directly from the API and should not
# require any user copy/pasta. We assume the model can understand JSON.
PROGRAM_ANALYSIS_PROMPT = """You are an expert horse racing handicapper analyzing structured race data from TwinSpires.

Analyze the race program JSON and provide:

1. **Selections** — top 3 horses to bet, with data-driven justification based on:
   - Speed figures, pace analysis, and recent form
   - Class ratings and competition level
   - Jockey/trainer statistics and win percentages
   - Value overlays (current odds vs. expected probability)
   - Running style and pace matchup

2. **Win bet** — single strongest horse if one clearly stands out, with reasoning
3. **Value plays** — horses where odds represent value based on their chances
4. **Exotic recommendations** — exacta, trifecta, superfecta suggestions (who to key, spread, single)

Format in clean markdown. Be concise and analytical. Focus on the data provided.
If critical fields are missing (speed figures, odds, recent races), note it explicitly.

IMPORTANT: Only analyze data present in the JSON. Do not invent statistics or horse details."""


def analyze_program_json(program_json: Dict[str, Any]) -> dict:
    """Analyze structured race program data in JSON format. Returns dict with keys: text, model, prompt_tokens, completion_tokens, total_tokens, elapsed_ms."""
    token = TOKEN
    model = LLM
    model_upgrade = LLM_UPGRADE

    client = OpenAI(
      base_url="https://models.inference.ai.azure.com",
      api_key=token,
    )

    filtered_data = _filter_program_data(program_json)

    user_message = f"Analyze this race and provide betting recommendations:\n\n```json\n{json.dumps(filtered_data, indent=2, separators=(",", ":"))}\n```"

    messages = [
        {"role": "system", "content": PROGRAM_ANALYSIS_PROMPT},
        {"role": "user", "content": user_message}
    ]
    
    params = {
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 3000,
    }

    try:
        start = time.monotonic()
        try:
            response = client.chat.completions.create(model=model, **params)
        except RateLimitError:
            # Primary model rate-limited — retry with upgrade model
            model = model_upgrade
            response = client.chat.completions.create(model=model, **params)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        content = response.choices[0].message.content or ""

        if _looks_like_hallucination(content):
            raise AnalyzerError(
                "The model returned an unexpected response. "
                "Please try again or check the input data for completeness."
            )

        usage = response.usage
        return {
            "text": content,
            "model": model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "elapsed_ms": elapsed_ms,
        }
    except AnalyzerError:
        raise
    except AuthenticationError:
        raise AnalyzerError(
            "Auth failed"
        )
    except RateLimitError:
        raise AnalyzerError(
            "Models are rate-limited. Try again later."
        )
    except APIError as e:
        raise AnalyzerError(f"API error: {e.message}")
    except Exception as e:
        raise AnalyzerError(f"Unexpected error: {str(e)}")  

def analyze(raw_text: str) -> dict:
    """Returns a dict with keys: text, model, prompt_tokens, completion_tokens, total_tokens, elapsed_ms."""
    _validate_input(raw_text)

    token = TOKEN
    dev_mode = DEV_MODE
    model = LLM
    model_upgrade = LLM_UPGRADE

    client = OpenAI(
        base_url="https://models.inference.ai.azure.com",
        api_key=token,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]

    params = dict(
        messages=messages,
        temperature=0.3,
        max_tokens=1500,
    )

    try:
        start = time.monotonic()
        try:
            response = client.chat.completions.create(model=model, **params)
        except RateLimitError:
            # Primary model rate-limited — retry with upgrade model
            model = model_upgrade
            response = client.chat.completions.create(model=model, **params)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        content = response.choices[0].message.content or ""

        if _looks_like_hallucination(content):
            raise AnalyzerError(
                "The model returned an unexpected response. "
                "Please try again or check your input format."
            )

        usage = response.usage
        return {
            "text": content,
            "model": model,
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
            "elapsed_ms": elapsed_ms,
        }

    except AnalyzerError:
        raise
    except AuthenticationError:
        raise AnalyzerError(
            "GitHub token authentication failed. "
            "Check that GITHUB_TOKEN is set and has the 'models:read' permission."
        )
    except RateLimitError:
        raise AnalyzerError(
            "Both models are rate-limited. You've hit the daily request cap — try again later."
        )
    except APIError as e:
        raise AnalyzerError(f"API error: {e.message}")
    except Exception as e:
        raise AnalyzerError(f"Unexpected error: {str(e)}")

def _filter_program_data(program_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip noise from raw TwinSpires program JSON before sending to LLM.
    Keeps only handicapping-relevant fields to reduce token usage.
    """
    race = program_data.get("race", {})
    track = program_data.get("track", {})

    filtered = {
        "track": {
            "name": track.get("trackName"),
            "country": track.get("country"),
            "raceDate": track.get("raceDate"),
        },
        "race": {
            "raceNumber": race.get("raceNumber"),
            "raceType": race.get("raceType"),
            "raceCondition": race.get("raceCondition"),
            "purse": race.get("purse"),
            "distance": race.get("distanceFormatted"),
            "surface": race.get("surface"),
            "ageRestriction": race.get("ageRestriction"),
            "claimingPrice": race.get("claimingPrice"),
            "speedPar": race.get("speedPar"),
            "pacePars": {
                "early1": race.get("paceParEarly1"),
                "early2": race.get("paceParEarly2"),
                "late": race.get("paceParLatePace"),
            },
            "programSelections": race.get("programSelections"),
            # Track bias — very useful context for the model
            "trackBias": {
                "speedBias": race.get("meetSpeedBias"),        # >70 = strong speed bias
                "wireToWire": race.get("meetWireToWire"),      # % front runners winning
                "winByOddsZone": {
                    "short": race.get("winPerShort"),          # <5/2
                    "mid": race.get("winPerMid"),              # 5/2-9/1
                    "long": race.get("winPerLong"),            # 10/1+
                },
                "runStyleImpact": {
                    "E":  {"impact": race.get("meetRs1Impact"),  "pct": race.get("meetRs1Percent"),  "flag": race.get("meetRs1Plusser")},
                    "EP": {"impact": race.get("meetRs2Impact"),  "pct": race.get("meetRs2Percent"),  "flag": race.get("meetRs2Plusser")},
                    "P":  {"impact": race.get("meetRs3Impact"),  "pct": race.get("meetRs3Percent"),  "flag": race.get("meetRs3Plusser")},
                    "S":  {"impact": race.get("meetRs4Impact"),  "pct": race.get("meetRs4Percent"),  "flag": race.get("meetRs4Plusser")},
                },
                "postPositionImpact": {
                    "PP1": {"impact": race.get("meetPost1Impact"), "flag": race.get("meetPost1Plusser")},
                    "PP2": {"impact": race.get("meetPost2Impact"), "flag": race.get("meetPost2Plusser")},
                    "PP3": {"impact": race.get("meetPost3Impact"), "flag": race.get("meetPost3Plusser")},
                    "PP4+": {"impact": race.get("meetPost4Impact"), "flag": race.get("meetPost4Plusser")},
                },
            },
        },
        "runners": [],
    }

    for interest in race.get("interest", []):
        ml_odds = interest.get("morningLineOdds")
        for r in interest.get("runner", []):
            if r.get("scratchIndicator") == "Y":
                continue  # skip scratches

            # Detect surface switchers (prior races on turf when today is dirt)
            surface_switches = [
                r.get(f"surface{i}Back") for i in range(1, 5)
                if r.get(f"surface{i}Back") is not None
            ]

            runner = {
                # Identity
                "programNumber": r.get("programNumber"),
                "horseName": r.get("horseName"),
                "postPosition": r.get("postPosition"),
                "morningLineOdds": ml_odds,
                "sex": r.get("sex"),
                "age": _current_year() - r["yearOfBirth"] if r.get("yearOfBirth") else None,
                "medication": r.get("medication"),
                "weight": r.get("weight"),
                "apprenticeAllowance": r.get("apprenticeWeightAllowance"),

                # Speed figures
                "speed": {
                    "avgSpeed": r.get("avgSpeed"),
                    "avgSpeedRank": r.get("avgSpeedRank"),
                    "avgSpeedLast3": r.get("avgSpeedLast3"),
                    "backSpeed": r.get("backSpeed"),           # best speed ever (Brisnet)
                    "backSpeedRank": r.get("backSpeedRank"),
                    "speedLastRace": r.get("speedLastRace"),
                    "speedLastRaceRank": r.get("speedLastRaceRank"),
                    "recentSpeeds": [                          # chronological: 1Back = most recent
                        r.get("finalSpeed1Back"),
                        r.get("finalSpeed2Back"),
                        r.get("finalSpeed3Back"),
                        r.get("finalSpeed4Back"),
                    ],
                },

                # Pace figures
                "pace": {
                    "avgE1": r.get("averagePaceE1"),
                    "avgE2": r.get("averagePaceE2"),
                    "avgLate": r.get("averagePaceLp"),
                    "bestE1atDistSurf": r.get("bestSpeedE1AtDistanceSurface"),
                    "bestE2atDistSurf": r.get("bestSpeedE2AtDistanceSurface"),
                },

                # Class
                "class": {
                    "primePower": r.get("primePower"),
                    "primePowerRank": r.get("primePowerRank"),
                    "avgClass": r.get("averageClass"),
                    "avgClassRank": r.get("averageClassRank"),
                    "currentClass": r.get("currentClass"),
                    "lastClass": r.get("lastClass"),
                    "raceRatings": [
                        r.get("raceRating1Back"),
                        r.get("raceRating2Back"),
                        r.get("raceRating3Back"),
                    ],
                    "horseClaimPrice": r.get("horseClaimPrice"),  # vs race claimingPrice = class drop/rise
                },

                # Form / fitness
                "form": {
                    "daysOff": r.get("daysOff"),
                    "speedPoints": r.get("speedPoints"),       # early speed measure (0-8)
                    "priorRunStyle": r.get("priorRunStyle"),   # E, E/P, P, S, NA
                    "distSurfaceFit": [                        # True = same dist/surface as today
                        r.get("sameDistanceSurface1Back"),
                        r.get("sameDistanceSurface2Back"),
                        r.get("sameDistanceSurface3Back"),
                    ],
                    "priorSurfaces": surface_switches or None,  # flag turf-to-dirt switches
                    "mudSpeed": r.get("mudSpeed") or None,
                    "totalEarnings": r.get("totalEarnings"),
                    "earningsRank": r.get("totalEarningsRank"),
                },

                # Connections
                "jockey": {
                    "name": f"{r.get('jockeyFirstName', '')} {r.get('jockeyLastName', '')}".strip(),
                    "winPct": r.get("jockeyWinPercent"),
                    "winPctRank": r.get("jockeyWinPercentRank"),
                    "wins": r.get("jockeyWins"),
                    "starts": r.get("jockeyStarts"),
                },
                "trainer": {
                    "name": f"{r.get('trainerFirstName', '')} {r.get('trainerLastName', '')}".strip(),
                    "winPct": r.get("trainerWinPercent"),
                    "winPctRank": r.get("trainerWinPercentRank"),
                    "wins": r.get("trainerWins"),
                    "starts": r.get("trainerStarts"),
                },

                # Model probability (Brisnet Power Rating)
                "model": {
                    "plProbability": r.get("plProbability"),   # win probability 0-1
                    "plFairValueOdds": r.get("plFairValueOdds"),
                    "plPredScore": r.get("plPredScore"),
                    "plMaxOdds": r.get("plMaxOdds"),           # fade above this
                    "plSuperValueOdds": r.get("plSuperValueOdds"),  # strong value threshold
                },

                # Breeding (only meaningful context)
                "sire": r.get("sire"),
                "dam": r.get("dam"),
            }
            filtered["runners"].append(runner)

    return filtered