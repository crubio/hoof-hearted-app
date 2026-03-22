import os
import re
import time
from openai import OpenAI, RateLimitError, AuthenticationError, APIError
from dotenv import load_dotenv


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

load_dotenv()

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

1. **Race header** — include all available: track, race #, race type, purse, distance, surface,
   conditions. Note any fields not provided.
2. **Data available** — one line listing which views were detected (by their column headers) and
   any key fields absent. Sets expectations for analysis depth.
3. **Horse-by-horse breakdown** — analyze only the fields present. Note standout positives and
   negatives across whatever is available: speed/pace figure trends, class ratings, prime power,
   run style matchup, jockey/trainer records, days off, equipment changes, expert angles, and
   overlays (site odds significantly higher than ML = value).
4. **Key angles** — based on available data:
   - Speed or pace figure trends (improving, declining, flat) if present
   - Pace scenario (lone speed, contested, closers' race) if run style or pace data present
   - Overlays: ML vs site odds divergence
   - Equipment, layoff, or angle flags if present
5. **Selections** — top 3 horses with brief justification. If data is thin (basic view only),
   weight jockey/trainer records and odds movement heavily.
6. **Single** — one horse to anchor multi-race bets, only if clearly justified.

At the end include:
- **Win bet** — the single strongest horse and why, if one clearly stands out.
- **Value/Overlay** — any horse where site odds are notably higher than ML that represents value.
- **Exotic use** — which horses to include in exactas, trifectas, or superfectas (wide vs. singled).

If 3 or more races are provided, also suggest:
- **Pick 3 ticket** — best 3-race sequence, legs, combos, cost at $1 base.
- **Pick 5 ticket** — best 5-race sequence, legs, combos, cost at $0.50 base.
- Note that all costs are approximations — actual payouts are pari-mutuel and set at post time.

Format everything in clean markdown. Be concise but analytical — think like a sharp bettor,
not a tout. Prioritize numbers over opinions. Flag high-variance or missing-data horses clearly.

IMPORTANT: Only use the data provided in the input. Do not invent statistics, horse names,
jockey names, or any details not present in the raw data. If a field is missing, say so.

If the input is missing critical race context — specifically race type, purse, distance, or
surface — do not attempt an analysis. Instead, respond with a single short markdown paragraph
explaining what is missing and asking the user to re-paste the complete race header and horse
entries before you can proceed.
"""


def analyze(raw_text: str) -> dict:
    """Returns a dict with keys: text, model, prompt_tokens, completion_tokens, total_tokens, elapsed_ms."""
    _validate_input(raw_text)

    token = os.getenv("GITHUB_TOKEN")
    dev_mode = os.getenv("DEV", "false").lower() == "true"
    model = os.getenv("MODEL_PROTOTYPE", "gpt-4o-mini") if dev_mode else os.getenv("MODEL", "gpt-4o")
    model_upgrade = os.getenv("MODEL_UPGRADE", "gpt-5")

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
