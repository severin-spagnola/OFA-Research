"""
LLM Variant Generator
=====================
Takes top-performing parameter sets from grid search and asks an LLM to
generate structural variants — not just parameter tweaks, but novel
combinations and ideas the grid wouldn't explore.

Uses Anthropic Claude (claude-sonnet-4-20250514) as the sole LLM backend.

Usage:
    # Called by overfit_search.py --llm
    # Or standalone:
    python llm_variant.py --input results/search_XXXXXX.json
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent


def _get_client():
    """Get the Anthropic Claude client."""
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not anthropic_key:
        return None, None, None

    import anthropic
    # Disable SDK's built-in retries — we handle 429 ourselves in
    # call_llm() with longer backoff (20-40s) to avoid hammering.
    return "anthropic", anthropic.Anthropic(
        api_key=anthropic_key, max_retries=0,
    ), "claude-sonnet-4-20250514"


# ─── Prompt ───────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a quantitative trading strategy designer specializing in MES (Micro E-mini S&P 500) \
futures intraday gap-fill strategies.

You will be given the top-performing parameter configurations from a grid search, along with \
their performance metrics. Your job is to generate NEW parameter combinations that the grid \
search would not have explored — specifically:

1. Non-obvious parameter interactions (e.g., tight stops + wide runners)
2. Extreme but potentially profitable configurations
3. Regime-specific tuning (e.g., low-vol vs high-vol adaptations)
4. Asymmetric setups (e.g., aggressive longs, conservative shorts)

IMPORTANT CONSTRAINTS:
- Output ONLY valid JSON
- Each variant must use the exact parameter names from the input
- All numeric values must be within reasonable ranges (documented below)
- Generate diverse variants — don't cluster around the winners

PARAMETER RANGES:
- m1_extrema_distance: 5.0 to 30.0 (distance from overnight H/L in MES points)
- gap_max_age_days: 1.0 to 21.0 (how long a gap stays tradeable)
- m1_max_sl_pts: 8.0 to 30.0 (max stop loss in points)
- partial_profit: 6.0 to 25.0 (first profit target in points)
- runner_r: 0.5 to 3.0 (runner take-profit as multiple of risk)
- penetration: 1.0 to 6.0 (entry trigger depth into gap body)
- entry_style: "wick" or "body"
- enable_m2: true or false (breakout model)
- enable_m3: true or false (midpoint fade model)
- sl_buffer: 1.0 to 6.0 (SL placement buffer past wick)
- min_sl_pts: 4.0 to 12.0 (minimum stop distance)
- m2_breakout_min: 8.0 to 20.0 (min breakout candle size)
- immediate_entry: true or false
- m1_gap_cutoff_pt_hour: 3 to 7 (gap valid after this PT hour)
- m3_start_pt_hour: 6 to 9 (M3 starts after this PT hour)
"""


def build_user_prompt(top_params: list[dict], top_fitness: list[dict],
                      n_variants: int = 10) -> str:
    """Build the user prompt from top candidates."""
    sections = []

    sections.append("## TOP CANDIDATES FROM GRID SEARCH\n")
    for i, (params, fitness) in enumerate(zip(top_params, top_fitness)):
        sections.append(f"### Candidate #{i+1}")
        sections.append(f"Fitness: {fitness.get('fitness', 0):.3f} | "
                        f"Sharpe: {fitness.get('sharpe', 0):.2f} | "
                        f"PF: {fitness.get('profit_factor', 0):.2f} | "
                        f"WR: {fitness.get('win_rate', 0):.1f}% | "
                        f"Trades: {fitness.get('n_trades', 0)} | "
                        f"P&L: ${fitness.get('total_pnl', 0):,.0f} | "
                        f"MaxDD: ${fitness.get('max_dd', 0):,.0f}")
        sections.append(f"Parameters: {json.dumps(params, indent=2)}")
        sections.append("")

    sections.append(f"\n## YOUR TASK")
    sections.append(f"Generate exactly {n_variants} new parameter variants.")
    sections.append("Focus on:")
    sections.append("1. Combinations the grid didn't explore")
    sections.append("2. At least 2 'aggressive' variants (tight stops, high runner_r)")
    sections.append("3. At least 2 'conservative' variants (wide stops, low runner_r)")
    sections.append("4. At least 1 'M1-only' variant (enable_m2=false, enable_m3=false)")
    sections.append("5. At least 1 variant with unusual gap_max_age (very short or very long)")
    sections.append("")
    sections.append("Respond with a JSON object: {\"variants\": [{...}, ...]}")
    sections.append("Each variant is a dict with the exact parameter names shown above.")

    return "\n".join(sections)


# ─── LLM Call ─────────────────────────────────────────────────────────────────

def call_llm(system: str, user: str, temperature: float = 0.8,
             max_retries: int = 5) -> str:
    """Call Anthropic Claude and return the response text.

    Adds random jitter (0-5s) before each call to desynchronize parallel
    workers hitting the API simultaneously.

    Retries up to max_retries times on 429/529 rate-limit or overload errors.
    """
    import random, time as _time

    jitter = random.uniform(0, 5.0)
    _time.sleep(jitter)

    provider, client, model = _get_client()

    if provider is None:
        raise RuntimeError(
            "No LLM backend found. Set ANTHROPIC_API_KEY."
        )

    retries = 0
    while True:
        try:
            _call_t0 = _time.time()
            response = client.messages.create(
                model=model,
                max_tokens=8000,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            in_tok = response.usage.input_tokens
            out_tok = response.usage.output_tokens
            print(f"    [LLM] Anthropic Claude: {_time.time() - _call_t0:.1f}s "
                  f"({in_tok}in {out_tok}out)")
            return response.content[0].text

        except Exception as e:
            err_str = str(e).lower()
            status = getattr(e, "status_code", None) or getattr(e, "status", None)
            is_rate_limit = (
                status == 429
                or "rate" in err_str
                or "429" in err_str
                or "too many requests" in err_str
                or "overloaded" in err_str
                or status == 529
            )
            if is_rate_limit:
                retries += 1
                if retries > max_retries:
                    raise RuntimeError(
                        f"LLM rate-limited {max_retries} times, giving up. "
                        f"Last error: {e}"
                    )
                wait = 30.0 + random.uniform(-10.0, 10.0)  # 20-40s
                print(f"  [anthropic rate limit] Retry {retries}/{max_retries} "
                      f"in {wait:.0f}s ...")
                _time.sleep(wait)
                continue
            # Non-rate-limit error — re-raise
            raise


def parse_variants(response_text: str) -> list[dict]:
    """Parse LLM response into variant dicts."""
    # Try to extract JSON from the response
    text = response_text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            print(f"Failed to parse LLM response as JSON")
            return []

    if isinstance(data, dict) and "variants" in data:
        return data["variants"]
    elif isinstance(data, list):
        return data
    else:
        return [data]


# ─── Public API ───────────────────────────────────────────────────────────────

def generate_llm_variants(
    top_params,  # list of ParamSet or dicts
    top_fitness,  # list of FitnessResult or dicts
    n_variants: int = 10,
):
    """Generate LLM structural variants from top candidates.

    Returns list of ParamSet objects ready for backtesting.
    """
    from overfit_search import ParamSet

    # Convert to dicts if needed
    param_dicts = [
        p.to_dict() if hasattr(p, "to_dict") else (asdict(p) if hasattr(p, "__dataclass_fields__") else p)
        for p in top_params
    ]
    fitness_dicts = [
        f.to_dict() if hasattr(f, "to_dict") else (asdict(f) if hasattr(f, "__dataclass_fields__") else f)
        for f in top_fitness
    ]

    prompt = build_user_prompt(param_dicts, fitness_dicts, n_variants)

    print(f"Calling LLM for {n_variants} structural variants...")
    try:
        response = call_llm(SYSTEM_PROMPT, prompt)
    except Exception as e:
        print(f"LLM call failed: {e}")
        return []

    variants = parse_variants(response)
    print(f"LLM returned {len(variants)} variants")

    # Convert to ParamSet objects
    param_sets = []
    for v in variants:
        try:
            # Only keep keys that ParamSet accepts
            valid_keys = set(ParamSet.__dataclass_fields__.keys())
            filtered = {k: v[k] for k in v if k in valid_keys}
            ps = ParamSet(**filtered)
            param_sets.append(ps)
        except (TypeError, KeyError) as e:
            print(f"  Skipping invalid variant: {e}")
            continue

    print(f"Parsed {len(param_sets)} valid ParamSets")
    return param_sets


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Variant Generator")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to search results JSON")
    parser.add_argument("--n", type=int, default=10,
                        help="Number of variants to generate")
    args = parser.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    candidates = data["candidates"][:3]
    top_params = [c["params"] for c in candidates]
    top_fitness = [c["fitness"] for c in candidates]

    variants = generate_llm_variants(top_params, top_fitness, n_variants=args.n)

    if variants:
        print("\nGenerated variants:")
        for i, v in enumerate(variants):
            print(f"  {i+1}. {v.label}")
            print(f"     {json.dumps(v.to_dict(), indent=None)[:120]}...")
    else:
        print("No variants generated.")


if __name__ == "__main__":
    main()
