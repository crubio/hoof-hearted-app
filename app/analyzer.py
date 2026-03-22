import os
import re
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

load_dotenv()

SYSTEM_PROMPT = """You are an expert horse racing handicapper. You will be given raw race program data
copied from a race-day program page. Each race entry contains fields in this order:

  Post Position (PP) | Morning Line Odds (ML) | Site/Expert Odds | Horse Name |
  Days Off | Run Style (E=Early, P=Presser, S=Stalker, NA=No advantage; number = career starts) |
  Avg Speed Figure (3-race rolling) | Back Speed Figure (career best) | Speed Figure Last Race |
  Avg Class Rating | Prime Power Rating | Jockey Win % | Trainer Win % | Career Earnings

Some horses may have missing figures (—) indicating a first-time starter or incomplete records.
Scratches may be indicated in the data.

For each race, analyze and return markdown with:

1. **Race header** — race number, race type, purse, distance, surface, and conditions if present.
2. **Horse-by-horse breakdown** — for each horse, briefly note standout positives and negatives
   across: speed figures (avg/back/last race trend), class rating, prime power, pace/run style
   matchup, jockey %, trainer %, days off, and any overlays (where site odds drifted significantly
   from morning line).
3. **Key angles** — call out:
   - Speed figure trends (improving, declining, or flat)
   - Pace scenario (lone speed, contested pace, closers' race)
   - Overlays: horses where site odds are significantly higher than ML (value bets)
   - Equipment changes or notable layoffs if mentioned
4. **Selections** — list top 3 horses with a brief justification for each.
5. **Single** — one horse to anchor multi-race bets, only if clearly justified by the numbers.

At the end of your analysis include:
- **Win bet** — the single strongest horse and why, if one clearly stands out.
- **Value/Overlay** — any horse where site odds are notably higher than ML that represents value.
- **Exotic use** — which horses to include in exactas, trifectas, or superfectas (wide vs. singled).

Format everything in clean markdown. Be concise but analytical — think like a sharp bettor,
not a tout. Prioritize numbers over opinions. Flag high-variance or missing-data horses clearly.

IMPORTANT: Only use the data provided in the input. Do not invent statistics, horse names,
jockey names, or any details not present in the raw data. If a field is missing, say so.
"""


def analyze(raw_text: str) -> str:
    token = os.getenv("GITHUB_TOKEN")
    model = os.getenv("MODEL", "gpt-4o")
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
        try:
            response = client.chat.completions.create(model=model, **params)
        except RateLimitError:
            # Primary model rate-limited — retry with upgrade model
            response = client.chat.completions.create(model=model_upgrade, **params)

        content = response.choices[0].message.content or ""

        if _looks_like_hallucination(content):
            raise AnalyzerError(
                "The model returned an unexpected response. "
                "Please try again or check your input format."
            )

        return content

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
