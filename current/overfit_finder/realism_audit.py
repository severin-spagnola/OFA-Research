"""
LLM Realism Auditor
====================
Sends strategy source code + backtest results to GPT-4o to check for
implementation flaws: look-ahead bias, unrealistic fill assumptions,
temporal artifacts, and cost modeling issues.

Does NOT flag overfitting — that's intentional in this pipeline.

Usage:
    # Called by overfit_search.py --audit
    # Or standalone:
    python realism_audit.py --input results/search_XXXXXX.json
"""
from __future__ import annotations

import json
import time as tm
from dataclasses import dataclass, asdict, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_STRATEGY_PATH = _REPO_ROOT / "archived" / "fvg_gap" / "mes_friend_strategy_base.py"

# ─── Code section extraction ────────────────────────────────────────────────

# Line ranges for the 4 critical sections the auditor needs to review
STRATEGY_CODE_SECTIONS = {
    "strategy_constants_and_realism_knobs": (53, 94),
    "m1_gap_entry_logic": (552, 641),
    "m2_breakout_entry_logic": (378, 550),
    "trade_simulation_and_costs": (644, 823),
}


def extract_strategy_sections(strategy_path: str = None) -> dict[str, str]:
    """Extract the relevant strategy source code sections for audit review.

    Returns dict mapping section_name -> source code string.
    """
    path = Path(strategy_path) if strategy_path else _STRATEGY_PATH
    lines = path.read_text().splitlines()

    sections = {}
    for name, (start, end) in STRATEGY_CODE_SECTIONS.items():
        # Convert to 0-indexed
        section_lines = lines[start - 1:end]
        sections[name] = "\n".join(f"{start + i}: {line}"
                                   for i, line in enumerate(section_lines))

    return sections


# ─── Prompt ──────────────────────────────────────────────────────────────────

AUDIT_SYSTEM_PROMPT = """\
You are a quantitative strategy auditor reviewing a MES (Micro E-mini S&P 500) \
futures gap-fill backtest implementation.

CRITICAL CONTEXT: This strategy INTENTIONALLY overfits to the last 60-90 days. \
Overfitting is the DESIGN. Do NOT flag overfitting, curve-fitting, small sample \
sizes, or lack of out-of-sample testing as issues.

Your job is to review the backtest CODE and RESULTS for IMPLEMENTATION FLAWS \
that would make the backtest unrealistic compared to live trading:

1. LOOK-AHEAD BIAS / DATA LEAKAGE
   - Does the entry logic use future price data that wouldn't be available at \
decision time?
   - Are gaps detected using candles that haven't occurred yet at the time of \
the trade signal?
   - Does the fill simulation reference bars after the entry bar for entry \
price determination?

2. FILL ASSUMPTIONS
   - Is the entry price achievable within the entry candle's OHLC range?
   - Are limit orders assumed to fill at exact prices without considering \
book depth or market impact?
   - Does the sim verify that the entry bar actually traded at/through the \
entry price?
   - For wick entries: is the wick penetration check realistic?

3. SLIPPAGE & COST MODELING
   - Is slippage per side realistic for MES? (MES tick = 0.25 pts = $1.25; \
typical slippage is 0-0.25 pts per side for market orders)
   - Are round-trip fees realistic? ($1.24/contract is standard retail for MES)
   - Is position sizing reasonable? (MES margin ~$1,500/contract)

4. TEMPORAL ARTIFACTS
   - Are overnight session boundaries (18:00-09:30 ET) handled correctly?
   - Could DST transitions cause incorrect time window calculations?
   - Are weekend/holiday gaps handled properly (not confused with FVG gaps)?
   - Is the gap "age" filter applied correctly across session boundaries?

5. STOP LOSS / EXIT REALISM
   - Within a 1-minute bar, if both SL and TP are breached, which is assumed \
to fill? Is this realistic?
   - Are EOD flat exits priced at close or at a specific time?
   - Are partial exits (scaling out) priced realistically?
   - Does the runner portion get a realistic exit?

DO NOT flag:
- Overfitting or curve-fitting (intentional)
- Small sample sizes (expected for 60-90 day windows)
- Strategy logic choices (entry model selection, parameter values)
- Performance metrics being "too good" or "too bad"
- Lack of out-of-sample testing

Respond with ONLY a JSON object (no markdown, no explanation outside JSON):
{
    "verdict": "PASS" | "FAIL" | "REVIEW",
    "summary": "One paragraph summary of overall assessment",
    "findings": [
        {
            "category": "look_ahead" | "fill_assumption" | "slippage" | \
"temporal_artifact" | "sl_exit_realism",
            "severity": "critical" | "warning" | "info",
            "description": "What the issue is",
            "code_reference": "Which function/line range",
            "recommendation": "What to fix"
        }
    ]
}

Use "PASS" if no critical or warning issues found.
Use "REVIEW" if warnings but no critical issues.
Use "FAIL" if any critical issues found.
"""


# ─── Dataclasses ─────────────────────────────────────────────────────────────

@dataclass
class AuditFinding:
    """One specific issue found by the auditor."""
    category: str
    severity: str
    description: str
    code_reference: str
    recommendation: str


@dataclass
class AuditReport:
    """Structured audit report for one candidate."""
    candidate_rank: int
    params_label: str
    verdict: str          # PASS, FAIL, REVIEW, ERROR
    findings: list[AuditFinding] = field(default_factory=list)
    summary: str = ""
    audit_time_seconds: float = 0.0
    model_used: str = "gpt-4o"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def has_critical(self) -> bool:
        return any(f.severity == "critical" for f in self.findings)


# ─── User prompt builder ────────────────────────────────────────────────────

def build_audit_prompt(
    strategy_sections: dict[str, str],
    params: dict,
    fitness: dict,
    summary: dict,
) -> str:
    """Build the user prompt containing all context the auditor needs."""
    parts = []

    parts.append("## STRATEGY SOURCE CODE\n")
    parts.append("Below are the 4 critical sections of the backtest implementation.\n")

    section_descriptions = {
        "strategy_constants_and_realism_knobs":
            "Global constants: slippage, fees, SL buffer, position sizing, gap parameters",
        "m1_gap_entry_logic":
            "M1 gap-fill entry: detects gap penetration, sets entry price, SL, TP",
        "m2_breakout_entry_logic":
            "M2 breakout entry: detects breakout candles, retracement entries",
        "trade_simulation_and_costs":
            "Trade simulator: fills entries on 1m bars, manages partials, applies costs",
    }

    for name, code in strategy_sections.items():
        desc = section_descriptions.get(name, name)
        parts.append(f"### {desc}")
        parts.append(f"```python\n{code}\n```\n")

    parts.append("\n## CANDIDATE PARAMETERS\n")
    parts.append("These are the specific parameter values used for this backtest run:")
    parts.append(f"```json\n{json.dumps(params, indent=2)}\n```\n")

    parts.append("\n## BACKTEST RESULTS\n")
    parts.append("Performance metrics for this candidate:")
    fitness_display = {k: v for k, v in fitness.items()
                      if k not in ("trade_count_penalty", "dd_penalty",
                                   "concentration_penalty")}
    parts.append(f"```json\n{json.dumps(fitness_display, indent=2)}\n```\n")

    if summary:
        key_summary = {k: v for k, v in summary.items()
                       if k in ("total_trades", "wins", "losses", "win_rate",
                                "total_pnl", "max_drawdown",
                                "model_breakdown")}
        if key_summary:
            parts.append("Trade summary:")
            parts.append(f"```json\n{json.dumps(key_summary, indent=2, default=str)}\n```\n")

    parts.append("\n## BACKTEST STRUCTURE\n")
    parts.append("The backtest iterates trading days. For each day:")
    parts.append("1. Load overnight (18:00 ET prev day to 09:30 ET) H/L from 1m candles")
    parts.append("2. Scan 15m candles from gap cutoff time, detect gaps via precomputed index")
    parts.append("3. For each valid gap: check penetration on the 15m candle, create Trade")
    parts.append("4. Simulate trade on 1m bars: check SL/TP bar-by-bar, apply partials")
    parts.append("5. Apply slippage + commission at entry and exit")
    parts.append("6. Flat all positions at 16:00 ET if not already closed")
    parts.append("\nReview the code sections above for implementation correctness.")

    return "\n".join(parts)


# ─── LLM call + parsing ─────────────────────────────────────────────────────

def _parse_audit_response(text: str, rank: int, label: str) -> AuditReport:
    """Parse LLM JSON response into AuditReport."""
    text = text.strip()

    # Handle markdown code blocks
    if "```json" in text:
        text = text.split("```json")[1].split("```")[0].strip()
    elif "```" in text:
        text = text.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(text[start:end])
        else:
            return AuditReport(
                candidate_rank=rank,
                params_label=label,
                verdict="ERROR",
                summary=f"Failed to parse LLM response as JSON: {text[:200]}",
            )

    findings = []
    for f in data.get("findings", []):
        findings.append(AuditFinding(
            category=f.get("category", "unknown"),
            severity=f.get("severity", "info"),
            description=f.get("description", ""),
            code_reference=f.get("code_reference", ""),
            recommendation=f.get("recommendation", ""),
        ))

    return AuditReport(
        candidate_rank=rank,
        params_label=label,
        verdict=data.get("verdict", "ERROR"),
        findings=findings,
        summary=data.get("summary", ""),
    )


# ─── Public API ──────────────────────────────────────────────────────────────

def audit_candidates(
    results: list,        # list[SearchResult] from overfit_search
    top_n: int = 3,
) -> list[AuditReport]:
    """Run LLM realism audit on the top N search candidates.

    Returns list of AuditReport objects.
    """
    from llm_variant import call_llm

    # Extract strategy source once
    sections = extract_strategy_sections()

    reports = []
    for i, r in enumerate(results[:top_n]):
        rank = r.rank
        label = r.params.label
        params = r.params.to_dict()
        fitness = r.fitness.to_dict()
        summary = r.summary

        print(f"  Auditing #{rank} ({label})...")
        t0 = tm.time()

        prompt = build_audit_prompt(sections, params, fitness, summary)

        try:
            response = call_llm(AUDIT_SYSTEM_PROMPT, prompt, temperature=0.2)
            report = _parse_audit_response(response, rank, label)
        except Exception as e:
            report = AuditReport(
                candidate_rank=rank,
                params_label=label,
                verdict="ERROR",
                summary=f"LLM call failed: {e}",
            )

        report.audit_time_seconds = round(tm.time() - t0, 1)
        reports.append(report)

        # Brief delay between calls to avoid rate limits
        if i < top_n - 1:
            tm.sleep(1)

    return reports


# ─── Output ──────────────────────────────────────────────────────────────────

def print_audit_results(reports: list[AuditReport]):
    """Pretty-print audit reports to console."""
    print("\n" + "=" * 70)
    print("LLM REALISM AUDIT RESULTS")
    print("=" * 70)

    for report in reports:
        verdict_marker = {
            "PASS": "PASS",
            "FAIL": "** FAIL **",
            "REVIEW": "REVIEW",
            "ERROR": "ERROR",
        }.get(report.verdict, report.verdict)

        print(f"\n#{report.candidate_rank} | {report.params_label} | {verdict_marker}")
        print(f"  {report.summary}")

        if report.findings:
            for f in report.findings:
                severity_prefix = {
                    "critical": "!!",
                    "warning": " !",
                    "info": "  ",
                }.get(f.severity, "  ")
                print(f"  {severity_prefix} [{f.category}] {f.description}")
                if f.code_reference:
                    print(f"       Ref: {f.code_reference}")
                if f.recommendation:
                    print(f"       Fix: {f.recommendation}")

        print(f"  (audit took {report.audit_time_seconds}s)")

    print()


def save_audit_results(reports: list[AuditReport], output_dir: Path) -> Path:
    """Save audit reports as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    out = {
        "audit_time": ts,
        "n_candidates_audited": len(reports),
        "reports": [r.to_dict() for r in reports],
    }

    path = output_dir / f"audit_{ts}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Audit results saved: {path}")
    return path


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LLM Realism Auditor")
    parser.add_argument("--input", type=str, required=True,
                        help="Path to search results JSON")
    parser.add_argument("--top", type=int, default=3,
                        help="Number of top candidates to audit")
    args = parser.parse_args()

    # Load search results and reconstruct minimal SearchResult-like objects
    with open(args.input) as f:
        data = json.load(f)

    from dataclasses import dataclass as dc

    @dc
    class _Params:
        _d: dict
        def to_dict(self): return self._d
        @property
        def label(self):
            return self._d.get("label", "candidate")

    @dc
    class _Fitness:
        _d: dict
        def to_dict(self): return self._d

    @dc
    class _Result:
        rank: int
        params: _Params
        fitness: _Fitness
        summary: dict

    results = []
    for c in data["candidates"][:args.top]:
        results.append(_Result(
            rank=c["rank"],
            params=_Params(c["params"]),
            fitness=_Fitness(c["fitness"]),
            summary=c.get("summary", {}),
        ))

    reports = audit_candidates(results, top_n=args.top)
    print_audit_results(reports)
    save_audit_results(reports, Path(args.input).parent)


if __name__ == "__main__":
    main()
