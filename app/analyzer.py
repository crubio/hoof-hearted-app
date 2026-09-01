import os
import re
import time
import json
from datetime import datetime
from typing import Dict, Any
from openai import OpenAI, RateLimitError, AuthenticationError, APIError
from dotenv import load_dotenv

from app.scraper import RACETRACK_TZ

load_dotenv()

# OpenRouter (https://openrouter.ai) -- OpenAI-SDK compatible, gives access to genuinely free
# ":free"-tagged models. The free roster rotates roughly weekly, and OpenRouter's own
# "openrouter/free" auto-router alias was tested and rejected: it can route a request to a
# non-chat model (observed landing on a content-safety classifier that refused the whole
# task). MODEL/MODEL_UPGRADE below are two specific, verified-working chat models from
# different labs instead -- update them if OpenRouter's roster moves on; check current free
# models at https://openrouter.ai/models?max_price=0 or via GET /api/v1/models.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
TOKEN = os.getenv("OPENROUTER_API_KEY")
DEV_MODE = os.getenv("DEV", "false").lower() == "true"
LLM = os.getenv("MODEL_PROTOTYPE", "z-ai/glm-5.2:free") if DEV_MODE else os.getenv("MODEL", "z-ai/glm-5.2:free")
LLM_UPGRADE = os.getenv("MODEL_UPGRADE", "google/gemma-4-31b-it:free")

OPENROUTER_SITE_URL = os.getenv("OPENROUTER_SITE_URL", "")
OPENROUTER_SITE_NAME = os.getenv("OPENROUTER_SITE_NAME", "Hoof Hearted")


def _current_year() -> int:
    """Racing age is calendar year minus foaling year, so this must not be hardcoded.
    Uses the racetrack's own timezone (Pacific) rather than the server's local time --
    matters only in the few hours around New Year's, but costs nothing to get right."""
    return datetime.now(RACETRACK_TZ).year


def _completion_params(model: str, messages: list, max_tokens: int, temperature: float = 0.3) -> dict:
    """OpenAI's gpt-5 and o-series reasoning models (o1, o3, o4-mini, ...) reject `temperature`
    and `max_tokens` -- they only accept `max_completion_tokens`. Match only that specific
    naming pattern, not any model starting with "o" -- "openrouter/free" and "openai/gpt-4o"
    both start with "o" too but are NOT reasoning models and need the standard params."""
    if model.startswith("gpt-5") or re.match(r"^o[0-9]", model):
        return {"messages": messages, "max_completion_tokens": max_tokens}
    return {"messages": messages, "temperature": temperature, "max_tokens": max_tokens}


def _extra_headers() -> dict:
    """OpenRouter attribution headers -- optional, only sent if configured."""
    headers = {"X-Title": OPENROUTER_SITE_NAME}
    if OPENROUTER_SITE_URL:
        headers["HTTP-Referer"] = OPENROUTER_SITE_URL
    return headers


def _retry_delay_seconds(e: RateLimitError, default: float = 3.0, cap: float = 8.0) -> float:
    """Free-tier 429s on OpenRouter are frequently a transient shared-provider-pool limit
    (observed: `limit_source: upstream_provider_shared_pool`, `retry_after_seconds: 5`) that
    clears in seconds -- not the account's daily cap. Honor the provider's own hint instead
    of guessing, capped so a single request can't stall the whole handler indefinitely."""
    try:
        header_val = e.response.headers.get("retry-after")
        if header_val:
            return min(float(header_val), cap)
    except Exception:
        pass
    return default


def _create_completion_with_fallback(client: OpenAI, model: str, model_upgrade: str, messages: list, max_tokens: int):
    """Call the primary model; on a rate limit, briefly honor the provider's Retry-After hint
    and retry the SAME model once (most free-tier 429s are a transient shared-pool limit that
    clears quickly, and retrying the same model avoids landing on a worse one). If that also
    rate-limits, fall back to `model_upgrade`. Returns (response, model_actually_used)."""
    def _call(model_name: str):
        return client.chat.completions.create(
            model=model_name, extra_headers=_extra_headers(), **_completion_params(model_name, messages, max_tokens)
        )

    try:
        return _call(model), model
    except RateLimitError as e:
        time.sleep(_retry_delay_seconds(e))
        try:
            return _call(model), model
        except RateLimitError:
            if model_upgrade == model:
                raise
            return _call(model_upgrade), model_upgrade


class AnalyzerError(Exception):
    """Clean user-facing error from the analyzer, carrying an HTTP status code
    the API layer can return directly instead of always answering 200."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.status_code = status_code


def _looks_like_hallucination(text: str, min_headings: int = 0) -> bool:
    """Basic sanity check on model output before it reaches the user: too short, non-Latin
    garbage, or (when min_headings is set) missing most of the required section headings --
    catches a malformed response even when it happens to be long enough to pass the length check."""
    stripped = text.strip()
    if len(stripped) < 100:
        return True
    # Flag if more than 5% of chars are outside printable ASCII + common punctuation
    non_latin = re.findall(r'[^\x00-\x7F\u2013\u2014\u2019\u201c\u201d]', text)
    if len(non_latin) / max(len(text), 1) > 0.05:
        return True
    if min_headings and stripped.count("## ") < min_headings:
        return True
    return False


def _validate_input(text: str) -> None:
    """Reject input that doesn't look like race program data before hitting the API."""
    stripped = text.strip()

    if len(stripped) < 30:
        raise AnalyzerError(
            "Input is too short. Please paste the full race program data including "
            "the race header (race type, purse, distance, surface) and horse entries.",
            status_code=400,
        )

    # Require at least one odds-style token -- universal to any race format
    has_odds = bool(re.search(r'M:\s*\d|\d/\d', stripped))

    if not has_odds:
        raise AnalyzerError(
            "Input appears incomplete. Make sure to include the full race header "
            "(race type, purse, distance, surface) and horse entries with at least "
            "post positions and morning line odds (e.g. M: 5/2).",
            status_code=400,
        )

# This prompt is for the all in one python app that renders templating based on markdown returned
# from analysis. This is deprecated but can still be used as a standalone app if spun up that way.
SYSTEM_PROMPT = """You are an expert horse racing handicapper. You will be given raw race program
data copied from a race-day program page.

## Input Structure

The input always begins with a race header block (track name, race number, post time, race type,
purse, age/sex conditions, distance, surface). This may come as a separate paste or precede the
data block.

The data section may optionally begin with a `#` token followed by column names -- one per line.
Whether or not the `#` is present, identify the data view by recognizing column name keywords
in the input (e.g. "AVG SPD", "prime power", "early pace", "Jockey", "angles", "MED/WT/EQP").
Use whichever column names are present to map values to fields. Be flexible -- column names may
appear as part of the text or as a header block; either way, use them to understand the data shape.

After any header/column names, horse entries appear as numbered blocks with one value per line.
Map each value to its field using the detected column order.

The user may paste one or more of these data views:

- **Summary** -- columns include: ODDS, PL, Runner, DAYS OFF, RUN STYLE, AVG SPD, BACK SPD, SPD LR, AVG CLS, PRM PWR, W% JKY, W% TRN, $ -- the richest single view
- **Speed** -- columns include: ODDS, Runner, run style, Average Speed, Average Distance, best speed -- speed figures with field ranks
- **Class** -- columns include: ODDS, Runner, days off, prime power, last class, Average Class -- class ratings with ranks
- **Pace** -- columns include: ODDS, Runner, run style, early pace 1, early pace 2, late pace -- pace figures with ranks
- **Adv** -- columns include: ODDS, PL, Runner, Jockey, Trainer, Sire / Dam -- full jockey/trainer win records and breeding
- **Basic** -- columns include: ODDS, ML, PL, Runner, MED/WT/EQP, Jockey, Trainer -- odds, equipment, jockey, trainer names only
- **Tips** -- columns include: ODDS, PL, Runner, angles -- expert angle tags (Hot Trainer, Top Pick, Clocker Special, etc.)

Run Style codes (letter + early speed points 0-8):
- **E** (Early) -- vies for the early lead; typically cannot rate behind a pace setter
- **E/P** (Early/Presser) -- runs 2nd-3rd within a few lengths early; unlike E, can rate behind a pace setter
- **P** (Presser) -- middle-of-pack early, tries to run down the leader; rarely challenges for the lead early
- **S** (Sustain/Closer) -- runs at the back of the pack early before closing
- **NA** (Not Available) -- first-time starter or insufficient data to assess preferred run style

The number following the style letter (0-8) is the Early Speed Points rating: measures early speed
ability based on running position and beaten lengths at the first call of recent races.
Higher = more early speed shown. E.g. E6 = Early runner with high early speed; S0 = Closer with
no early speed points recorded.
Missing figures (-- or blank) indicate a first-time starter or incomplete record.
Scratches may be indicated in the data.

## Output Format

For each race, return markdown with:

1. **Selections** -- top 3 horses with brief justification. If data is thin (basic view only),
   weight jockey/trainer records and odds movement heavily.
2. **Single** -- one horse to anchor multi-race bets, only if clearly justified.
3. **Race header** -- include all available: track, race #, race type, purse, distance, surface,
   conditions. Note any fields not provided.
4. **Data available** -- one line listing which views were detected (by their column headers) and
   any key fields absent. Sets expectations for analysis depth.
5. **Horse-by-horse breakdown** -- analyze only the fields present. Note standout positives and
   negatives across whatever is available: speed/pace figure trends, class ratings, prime power,
   run style matchup, jockey/trainer records, days off, equipment changes, expert angles, and
   overlays (site odds significantly higher than ML = value). Distill into one or two concise sentences per horse.

At the end include:
- **Win bet** -- the single strongest horse and why, if one clearly stands out.
- **Value/Overlay** -- any horse where site odds are notably higher than ML that represents value.
- **Exotic use** -- which horses to include in exactas, trifectas, or superfectas (wide vs. singled).

Format everything in clean markdown. Be concise but analytical -- think like a sharp bettor,
not a tout. Prioritize numbers over opinions. Flag high-variance or missing-data horses clearly.

IMPORTANT: Only use the data provided in the input. Do not invent statistics, horse names,
jockey names, or any details not present in the raw data. If a field is missing, say so.

If the input is missing critical race context -- specifically race type, purse, distance, or
surface -- do not attempt an analysis. Instead, respond with a single short markdown paragraph
explaining what is missing and asking the user to re-paste the complete race header and horse
entries before you can proceed.
"""

# JSON-driven prompt: this data comes directly from the API and requires no user copy/paste.
# Field semantics (rank direction, sentinel zeros, array ordering, odds notation, coupled
# entries) are documented inline below since the raw JSON keys alone don't convey them.
PROGRAM_ANALYSIS_PROMPT = """You are an expert horse racing handicapper analyzing structured race data from
TwinSpires. Your job is to be RIGHT, not to always have a pick. A handicapper who bets every race loses to
takeout. Passing is a normal, frequently correct output — do not force picks to fill the format.

## Field reference (read this before analyzing)

Ranks: every "*Rank" field is 1 = best in the field, null = unranked/no qualifying data.
Percentages: jockey.winPct, trainer.winPct, and model.plProbability are decimal fractions
(0.1139 = 11.4%), not whole numbers.

race.fieldSize is the number of live (non-scratched) runners actually in this race; race.scratches
lists any horses removed from it. The runners array already excludes scratches — never analyze or
recommend a scratched horse even if a gap in post positions or programSelections suggests one was
entered.

SENTINEL ZERO — important: in avgSpeed, backSpeed, avgE1, avgE2, avgLate, bestE1atDistSurf, and
bestE2atDistSurf, a value of exactly 0 means "no qualifying starts at today's exact distance+surface,"
NOT a real figure of zero. Never call a horse slow or pace-weak because of a 0 in these seven fields
specifically — say instead that the figure is unavailable at today's distance/surface and lean on
speed.recentSpeeds, speed.speedLastRace, speed.avgSpeedLast3, class.primePower, and model.plProbability
instead. class.primePower is NOT part of this sentinel — it is Brisnet's overall power rating and stays
a real, usable number even when avgSpeed/backSpeed/currentClass read 0 for the same horse; only treat
primePower as missing when it is also 0. class.currentClass IS part of the sentinel (0 = no qualifying
start at today's distance/surface, same as avgSpeed/backSpeed). (speedPoints legitimately runs 0–8
including 0; that field is not affected by this rule.)

Array order: speed.recentSpeeds, class.raceRatings, and form.distSurfaceFit are all ordered
most-recent-first (index 0 = last race, 1 = two back, 2 = three back). recentSpeeds has a 4th slot
that is always null — ignore it; raceRatings and distSurfaceFit have no 4th slot. distSurfaceFit true =
the horse ran at today's exact distance+surface that time. form.priorSurfaces is null in this feed far
more often than not — that means prior-surface data simply isn't available, not that the horse never
switched surfaces; don't conclude anything from an absent priorSurfaces.

race.speedPar / race.pacePars.{early1,early2,late} are the benchmark figures a horse typically needs to
win THIS class and distance. Compare each contender's own figures against these pars, not only against
each other — a horse that ranks 1st in a slow field can still be well below par.

race.trackBias describes the current MEET at this track (not this specific race, and not "today"):
speedBias > 70 = surface is playing fast/speed-favoring; wireToWire = % of races won gate-to-wire;
winByOddsZone.{short,mid,long} = meet win% for morning-line favorites/mid-price/longshots — a season
trend, not this race's odds. runStyleImpact.{E,EP,P,S}.impact and postPositionImpact.*.impact are
win-rate multipliers (1.00 = neutral; ~1.15+ is a real edge); flag "++" = strong, "+" = mild, "" = none.
Use trackBias only as a small directional nudge, and say out loud when you're applying it — it should
never override plProbability or a horse's own figures by more than a modest amount.

Odds format warning: morningLineOdds and model.plFairValueOdds are strings in TWO different notations
in the same dataset — fractional ("5/2" = 2.5-to-1) and bare-number ("35" = 35-to-1). Convert every odds
string to implied probability (1 / (decimal_odds + 1)) before comparing anything numerically, and don't
misread "35" as 3.5.

Coupled entries: runners sharing the same bettingNumber run as ONE wagering interest (an entry or
field horse) even though each has its own programNumber and its own row here — listed separately in
model.coupledWith. Discuss them individually in Contenders, but never present "back one half, fade the
other" as a real bet in Selections/Exotics — for win/place/show purposes a coupled entry wins or loses
together.

model.plProbability is the vendor's own calibrated win-probability model — a wider, more data-rich model
than the individual fields shown to you. It has already been renormalized to sum to ~1.0 across the
runners you see here (after removing scratches), so use it directly without further adjustment for field
size. Treat plProbability as your PRIMARY prior for each horse. You may adjust it using today's pace
shape, trackBias, or a clear figure trend, but keep adjustments modest (nudge a 3rd choice into a
co-favorite, don't turn a longshot into the horse to beat) and name the one reason for every adjustment.
Do not re-derive an unrelated opinion straight from the speed/class figures and present it as independent
evidence — those figures already feed plProbability, so re-reading them and moving your number by a wide
margin is double-counting, not new analysis. model.plFairValueOdds is the vendor's separate fair-odds
figure and may not exactly match a recomputed odds-from-plProbability — treat plProbability as the
number to reason from, and plFairValueOdds only as a rough cross-check, not a contradiction to resolve.
model.plMaxOdds and model.plSuperValueOdds are typically null in this feed — ignore them.

class.horseClaimPrice compares to race.claimingPrice to read a class move — but check race.raceCondition
first: a claim price roughly double the base claimingPrice commonly signals a state-bred allowance
mentioned in that condition text, not a genuine class rise.

race.programSelections lists the TRACK's own public handicapper picks — not yours. You may note
agreement or disagreement with them in passing, but never cite them as evidence for your own selections
or present them as your own analysis.

sire and dam are name strings only, nothing else. Treat them purely as identifiers — never attribute a
racing characteristic, surface preference, or reputation to a sire or dam; you have no data field for
that and would be inventing it.

## Required steps, in order

1. **Contenders** — one line per runner, EVERY runner in the array, no skipping: program #, name,
   morning-line odds, plProbability (as a %), primePowerRank, priorRunStyle, and the single biggest
   positive or negative. Flag any horse with no recentSpeeds, no speedLastRace, and no primePower as a
   first-time starter / HIGH VARIANCE — judge it on daysOff, jockey/trainer win%, and morning line only.
2. **Pace & bias read** — count E / E-P runners with meaningful speedPoints (priorRunStyle "NA" means
   unknown, not "no speed" — treat those horses as unproven, not slow). Compare early contenders'
   avgE1/avgE2 to pacePars. Call the shape (lone speed / honest pace / contested / speed duel) and say
   who it favors. Apply trackBias as a small adjustment and state that you did.
3. **Selections** — give each a confidence label: STRONG (clear support from at least two independent
   data points, not just plProbability restated), LEAN (some support, real uncertainty), or SPECULATIVE
   (thin data or a price-only case). A race with no STRONG or LEAN horse should end in a PASS.
4. **Verdict** — first line under this heading, bolded: either "BET" naming the play, or
   "PASS — no play." Bet only when a contender's adjusted probability clearly beats the probability
   implied by its own morning-line price, or one STRONG horse is priced generously enough to be worth
   backing outright — say so in words rather than picking an arbitrary odds cutoff. If neither holds
   anywhere in the field, PASS and give the one main reason. Do not fill this section with a bet just
   to have one.
5. **Value note** — compare plProbability/plFairValueOdds against morningLineOdds only. Always call it
   "the morning line," never "today's odds," "the board," or "the market" — you have no live tote odds
   and must not imply otherwise or claim to know how prices are moving.
6. **Exotics** — only if the verdict is BET. Assume standard base bets ($1 exacta / $0.50 trifecta /
   $0.20 superfecta) unless told otherwise, and show the math: combinations × base = cost. Cap total
   suggested spend at $24 unless you explicitly justify more. Skip trifecta/superfecta structures below
   7 live runners, or when 3+ horses carry similar confidence labels with no real separation. If the
   verdict is PASS, write "Skipped — no bet on this race."
7. **Data gaps** — short list: which contenders are missing recentSpeeds/speedLastRace/primePower
   entirely, which fields hit the sentinel-zero case above, and anything else you couldn't evaluate
   because the field was null.

## Output format

Use `##` headings for each of the seven sections above, in order. Use bullet lists, not markdown
tables — they will not render as tables for the reader. Refer to every horse as
`#<programNumber> <name>` on first mention in each section, consistently.

## Hard rules

- Use only horse, jockey, trainer, sire, and dam names exactly as given in the JSON. Never attribute a
  fact to a sire/dam beyond its name.
- Never mention live odds, the tote board, money/odds movement, or workouts/equipment/trip notes that
  are not fields in the JSON.
- If a number isn't in the data, say the claim can't be made — do not estimate or round from memory.
- Be concise. This is a wagering tool, not an essay — no filler sentences, no restating the JSON back
  as prose."""


def analyze_program_json(program_json: Dict[str, Any]) -> dict:
    """Analyze structured race program data in JSON format. Returns dict with keys: text, model, prompt_tokens, completion_tokens, total_tokens, elapsed_ms."""
    token = TOKEN
    model = LLM
    model_upgrade = LLM_UPGRADE

    client = OpenAI(
      base_url=OPENROUTER_BASE_URL,
      api_key=token,
    )

    filtered_data = _filter_program_data(program_json)

    if not filtered_data["runners"]:
        raise AnalyzerError(
            "No usable runner data in this race program (all entries scratched or missing). "
            "Try a different race.",
            status_code=422,
        )

    # No indent -- the model doesn't need pretty-printing and it costs ~2k tokens/request.
    user_message = f"Analyze this race and provide betting recommendations:\n\n```json\n{json.dumps(filtered_data, separators=(',', ':'))}\n```"

    messages = [
        {"role": "system", "content": PROGRAM_ANALYSIS_PROMPT},
        {"role": "user", "content": user_message}
    ]

    # 8000: an 11-runner field with a full per-horse Contenders line plus six reasoning
    # sections routinely needs 4000-7000+ completion tokens on the free models tested --
    # 4000 was observed truncating mid-Contenders on a real run.
    max_tokens = 8000

    try:
        start = time.monotonic()
        response, model = _create_completion_with_fallback(client, model, model_upgrade, messages, max_tokens)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        content = response.choices[0].message.content or ""

        if response.choices[0].finish_reason == "length":
            content += "\n\n*[Analysis was cut off before completion -- try again for a full response.]*"

        if _looks_like_hallucination(content, min_headings=5):
            raise AnalyzerError(
                "The model returned an unexpected response. "
                "Please try again or check the input data for completeness.",
                status_code=502,
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
            "OpenRouter authentication failed. Check that OPENROUTER_API_KEY is set correctly.",
            status_code=401,
        )
    except RateLimitError:
        raise AnalyzerError(
            "OpenRouter free-tier requests are rate-limited (50/day by default, 1000/day with $10+ lifetime credit). Try again later.",
            status_code=429,
        )
    except APIError as e:
        raise AnalyzerError(f"API error: {e.message}", status_code=502)
    except Exception as e:
        raise AnalyzerError(f"Unexpected error: {str(e)}", status_code=500)

def analyze(raw_text: str) -> dict:
    """Returns a dict with keys: text, model, prompt_tokens, completion_tokens, total_tokens, elapsed_ms."""
    _validate_input(raw_text)

    token = TOKEN
    model = LLM
    model_upgrade = LLM_UPGRADE

    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=token,
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": raw_text},
    ]

    max_tokens = 1500

    try:
        start = time.monotonic()
        response, model = _create_completion_with_fallback(client, model, model_upgrade, messages, max_tokens)
        elapsed_ms = round((time.monotonic() - start) * 1000)

        content = response.choices[0].message.content or ""

        if response.choices[0].finish_reason == "length":
            content += "\n\n*[Analysis was cut off before completion -- try again for a full response.]*"

        if _looks_like_hallucination(content):
            raise AnalyzerError(
                "The model returned an unexpected response. "
                "Please try again or check your input format.",
                status_code=502,
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
            "OpenRouter authentication failed. Check that OPENROUTER_API_KEY is set correctly.",
            status_code=401,
        )
    except RateLimitError:
        raise AnalyzerError(
            "OpenRouter free-tier requests are rate-limited (50/day by default, 1000/day with $10+ lifetime credit). You've hit the daily request cap -- try again later.",
            status_code=429,
        )
    except APIError as e:
        raise AnalyzerError(f"API error: {e.message}", status_code=502)
    except Exception as e:
        raise AnalyzerError(f"Unexpected error: {str(e)}", status_code=500)

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
            # Track bias -- very useful context for the model
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

    # Flatten (interest, runner) pairs so coupled entries (shared bettingNumber, e.g. an
    # "entry" or field horse) can be cross-referenced, then sort by post position -- the
    # raw payload orders interests lexically by program number ("1","10","11","2",...),
    # not by post, which is what a handicapper actually reads races in.
    pairs = []
    scratches = []
    for interest in race.get("interest", []):
        ml_odds = interest.get("morningLineOdds")
        betting_number = interest.get("bettingNumber")
        runners_in_interest = interest.get("runner", [])
        coupled_numbers = [r.get("programNumber") for r in runners_in_interest] if len(runners_in_interest) > 1 else []
        for r in runners_in_interest:
            if r.get("scratchIndicator") == "Y":
                scratches.append({"programNumber": r.get("programNumber"), "horseName": r.get("horseName")})
                continue  # skip scratches
            pairs.append((ml_odds, betting_number, coupled_numbers, r))

    pairs.sort(key=lambda p: (p[3].get("postPosition") is None, p[3].get("postPosition")))

    for ml_odds, betting_number, coupled_numbers, r in pairs:
        # Detect surface switchers (prior races on turf when today is dirt)
        surface_switches = [
            r.get(f"surface{i}Back") for i in range(1, 5)
            if r.get(f"surface{i}Back") is not None
        ]
        program_number = r.get("programNumber")
        coupled_with = [n for n in coupled_numbers if n != program_number] or None

        runner = {
            # Identity
            "programNumber": program_number,
            "horseName": r.get("horseName"),
            "postPosition": r.get("postPosition"),
            "bettingNumber": betting_number,
            "coupledWith": coupled_with,       # other programNumbers sharing this bettingNumber
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
                "backSpeed": r.get("backSpeed"),           # best speed at today's dist/surface (Brisnet)
                "backSpeedRank": r.get("backSpeedRank"),
                "speedLastRace": r.get("speedLastRace"),
                "speedLastRaceRank": r.get("speedLastRaceRank"),
                "recentSpeeds": [                          # chronological: index 0 = most recent
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
                "bestLateAtDistSurf": r.get("bestSpeedLateAtDistanceSurface"),
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
                "plProbability": r.get("plProbability"),   # win probability 0-1; renormalized below
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

    # plProbability is a calibrated distribution that sums to 1.0 across the FULL field.
    # Scratches remove probability mass from that sum, so raw values here would understate
    # every survivor's true chance -- renormalize across the runners actually being analyzed.
    prob_sum = sum(
        rr["model"]["plProbability"] for rr in filtered["runners"]
        if rr["model"]["plProbability"] is not None
    )
    if prob_sum:
        for rr in filtered["runners"]:
            p = rr["model"]["plProbability"]
            if p is not None:
                rr["model"]["plProbability"] = round(p / prob_sum, 4)

    filtered["race"]["fieldSize"] = len(filtered["runners"])
    filtered["race"]["scratches"] = scratches or None

    return filtered
