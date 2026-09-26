"""Calibrate the volatility-error and transaction-cost axes from public market data.

``findings.misspecification_sweep`` finds that a +10% relative volatility error costs
2.7x-7.1x (median 4.2x) what 5bp of proportional cost does, and that the ordering flips at
50bp with daily rebalancing. Both magnitudes -- "+10%" and "index ~1-5bp vs single stock
~25-50bp" -- were asserted. This module measures them from public data, so the paper can
state how often the adverse error actually occurs and which cost level an instrument is at.

Volatility misspecification
---------------------------
A desk that sells an S&P 500 option at implied volatility and delta-hedges at that same
volatility experiences, over the following month, the ratio

    rho_t = RV_t / IV_t,         IV_t = VIX_t / 100,
    RV_t  = sqrt( (252 / 21) * sum_{k=1}^{21} r_{t+k}^2 ),       r_j = ln(S_j / S_{j-1}),

the sum running over the 21 S&P 500 trading days after the close at which VIX_t is fixed.
rho is exactly the sweep's ``vol_ratio`` = sigma_realised / sigma_hedge when the hedge is
run at implied vol. rho > 1 is adverse for a short-gamma hedger; rho < 1 is a gain. VIX
embeds a variance risk premium, so E[rho] < 1 is expected -- what matters for the paper is
how often, how far, and *when* rho > 1.

Cost regime
-----------
The simulator charges ``c * S * |trade|`` per rebalance, so c is a proportional one-way
cost: a half-spread when crossing the quote is the only friction. For an instrument
quoted one tick wide,

    c = (tick / 2) / price.

ES: tick 0.25 index points = $12.50 on the $50 multiplier (CME Rule 35802.B-C). SPY: tick
$0.01 since decimalization was completed on 2001-04-09. Single-stock spreads cannot be
computed from free data; they are transcribed from the literature in ``LITERATURE`` with
the table and page they came from, and only simple arithmetic (full -> half) is applied.

Data and licensing
------------------
Raw vendor files are cached OUTSIDE git-tracked paths (checked with ``git check-ignore``
before anything is written) and are never written to the repository. FRED's S&P 500
series is "Copyright (c) 2016, S&P Dow Jones Indices LLC ... Reproduction of S&P 500 in
any form is prohibited"; only derived summary statistics go into the JSON. The S&P 500
close series is spliced: FRED (S&P DJI's licensed feed) wherever FRED has it -- its last
ten years -- and Yahoo Finance's ``^GSPC`` before that. Yahoo is queried directly at the
chart endpoint that the ``yfinance`` package wraps, so no dependency is added to the
pinned environment. Cboe's own SPX and VIX histories are downloaded as cross-checks, and
the headline statistics are recomputed on the Cboe splice to show the vendor choice does
not matter.

Run::

    python -m experiments.market_calibration --cache-dir <dir outside the repo> \\
        --json paper/market_calibration.json

Numbers are pinned to ``SAMPLE_END``: later observations are ignored, so a rerun
reproduces the published statistics unless a vendor revises history -- which the
per-series SHA-256 fingerprints in the JSON will reveal.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import io
import json
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = [
    "LITERATURE",
    "RhoFrame",
    "build_rho",
    "cost_regime",
    "forward_realised_vol",
    "forward_realised_vol_calendar",
    "half_spread_bp",
    "run",
    "summarise",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

# ``experiments/runs/`` is gitignored. Any other location inside the repository is refused.
DEFAULT_CACHE_DIR = REPO_ROOT / "experiments" / "runs" / "market_data"

# ---- Sample --------------------------------------------------------------------------
SAMPLE_START = dt.date(1990, 1, 2)   # first VIX observation
SAMPLE_END = dt.date(2026, 9, 25)    # pinned vintage: last S&P 500 close at retrieval
YAHOO_START = dt.date(1989, 12, 1)   # a margin before SAMPLE_START; only 1990+ is used

# ---- Realised-volatility convention ---------------------------------------------------
# VIX is a 30-calendar-day quantity. 21 trading days is the usual match (30 * 252/365 =
# 20.7); the calendar-matched variant is reported as a sensitivity check.
HORIZON = 21
HORIZON_3M = 63                      # for the 3-month index (FRED VXVCLS)
PERIODS_PER_YEAR = 252
CALENDAR_DAYS = 30

# ---- Statistics ----------------------------------------------------------------------
QUANTILES = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)
ABOVE = (1.0, 1.1, 1.25, 1.5, 2.0)   # 1.1 and 1.25 are the sweep's adverse levels
BELOW = (0.9, 0.8)                   # 0.9 and 0.8 are the sweep's favourable levels

# Windows select on the date t at which the hedge is struck (VIX observed); the realised
# window then runs up to 21 trading days past the window's end.
STRESS_WINDOWS: dict[str, tuple[dt.date, dt.date]] = {
    "gfc_2008_calendar_year": (dt.date(2008, 1, 1), dt.date(2008, 12, 31)),
    "gfc_acute_2008-09_to_2009-03": (dt.date(2008, 9, 1), dt.date(2009, 3, 31)),
    "covid_2020-02_to_2020-03": (dt.date(2020, 2, 1), dt.date(2020, 3, 31)),
    "bear_market_2022": (dt.date(2022, 1, 1), dt.date(2022, 12, 31)),
}
SUBPERIODS: dict[str, tuple[dt.date, dt.date]] = {
    "1990s": (dt.date(1990, 1, 1), dt.date(1999, 12, 31)),
    "2000s": (dt.date(2000, 1, 1), dt.date(2009, 12, 31)),
    "2010s": (dt.date(2010, 1, 1), dt.date(2019, 12, 31)),
    "2020s": (dt.date(2020, 1, 1), SAMPLE_END),
    # VIX before its 2003 methodology change is Cboe's retrospective SPX-based
    # calculation, not a value disseminated in real time. Split to show it does not matter.
    "pre_2004_retrospective_vix": (dt.date(1990, 1, 1), dt.date(2003, 12, 31)),
    "from_2004": (dt.date(2004, 1, 1), SAMPLE_END),
}
VIX_BUCKET_EDGES = (0.0, 15.0, 20.0, 25.0, 30.0, 40.0, np.inf)

# Overlapping monthly windows are autocorrelated by construction, and volatility regimes
# persist for months. A one-year circular block keeps both inside each block.
BOOTSTRAP_BLOCK = 252
BOOTSTRAP_BLOCK_SENSITIVITY = (63, 504)   # a reviewer will ask "why 252?"
BOOTSTRAP_REPS = 2_000
BOOTSTRAP_SEED = 2026

ADVERSE_EPISODE_THRESHOLD = 1.25     # the sweep's largest adverse level
EPISODE_GAP = HORIZON                # adverse days more than a month apart = new episode

# ---- Cost regime ---------------------------------------------------------------------
ES_TICK = 0.25                       # index points (CME Rule 35802.C)
ES_MULTIPLIER = 50.0                 # USD per index point (CME Rule 35802.B)
ES_START = dt.date(1998, 1, 1)       # first full calendar year after the 1997 launch
SPY_TICK = 0.01                      # USD minimum quoting increment since decimal pricing
DECIMALIZATION_COMPLETE = dt.date(2001, 4, 9)
SWEEP_COSTS = (0.0001, 0.0005, 0.0025, 0.005)   # findings.misspecification_sweep

RETRIEVED = "2026-09-26"             # when the literature below was opened and read

# ---- Downloads -----------------------------------------------------------------------
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={}"
CBOE_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{}_History.csv"
YAHOO_CHART = (
    "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    "?period1={p1}&period2={p2}&interval=1d&events=history"
)
# FRED and Cboe accept an honest client string. Yahoo's endpoint rejects non-browser
# clients (HTTP 429); ``yfinance`` works around the same thing by browser impersonation.
HONEST_AGENT = "dhbench-market-calibration/1.0 (+https://github.com/Adi7710/deep-hedging-benchmark)"
BROWSER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
)

SERIES_NOTES = {
    "vix_fred": {
        "page": "https://fred.stlouisfed.org/series/VIXCLS",
        "note": "CBOE Volatility Index: VIX. 'Copyright, 2016, Chicago Board Options "
                "Exchange, Inc. Reprinted with permission.'",
    },
    "vix3m_fred": {
        "page": "https://fred.stlouisfed.org/series/VXVCLS",
        "note": "CBOE S&P 500 3-Month Volatility Index, from 2007-12-04. Same Cboe "
                "copyright notice as VIXCLS.",
    },
    "spx_fred": {
        "page": "https://fred.stlouisfed.org/series/SP500",
        "note": "S&P Dow Jones Indices LLC; 10 years of daily history only. 'Reproduction "
                "of S&P 500 in any form is prohibited except with the prior written "
                "permission of S&P Dow Jones Indices LLC.' Raw data is not redistributed.",
    },
    "spx_yahoo": {
        "page": "https://finance.yahoo.com/quote/%5EGSPC/history/",
        "note": "Undocumented Yahoo Finance chart endpoint (the one yfinance wraps). Used "
                "only before FRED's coverage starts. Raw data is not redistributed.",
    },
    "spy_yahoo": {
        "page": "https://finance.yahoo.com/quote/SPY/history/",
        "note": "SPY closes, for the SPY half-spread only. Not redistributed.",
    },
    "vix_cboe": {
        "page": "https://www.cboe.com/tradable_products/vix/vix_historical_data/",
        "note": "Cross-check of FRED VIXCLS. Cboe: data 'furnished without responsibility "
                "for accuracy'. VIX before 2003 is the current SPX-based methodology "
                "applied retrospectively; the original OEX-based index is published as VXO.",
    },
    "spx_cboe": {
        "page": "https://www.cboe.com/tradable_products/vix/vix_historical_data/",
        "note": "Cboe's SPX close history (same CDN as VIX_History.csv). Cross-check and "
                "alternative splice only.",
    },
}


# ======================================================================================
# Literature -- transcribed values, each with where it came from
# ======================================================================================

LITERATURE: dict[str, dict] = {
    "cme_rule_35802": {
        "citation": "CME Rulebook Chapter 358 (E-mini S&P 500 futures), Rules 35802.B-C, "
                    "as filed by CME with the CFTC (Exhibit A, 2016 submission).",
        "url": "https://www.cftc.gov/filings/orgrules/rule030416cmedcm003.pdf",
        "retrieved": RETRIEVED,
        "locator": "Exhibit A, pdf p. 47",
        "quote": "The minimum price increment shall be 0.25 Index points, equal to $12.50 "
                 "per contract [...] The unit of trading shall be $50.00 times the Index.",
        "note": "The struck-through prior rule text on the same page also specifies 0.25 "
                "index points. CME's contract-specification web page could not be opened: "
                "cmegroup.com returned HTTP 403 ('IP address is blocked due to suspected "
                "web scraping') on 2026-09-26. Worth a 30-second manual check in a browser.",
    },
    "kirilenko_kyle_samadi_tuzun": {
        "citation": "Kirilenko, Kyle, Samadi and Tuzun, 'The Flash Crash: High-Frequency "
                    "Trading in an Electronic Market', J. Finance 72(3), 2017. CFTC version "
                    "of 5 May 2014.",
        "url": "https://www.cftc.gov/sites/default/files/idc/groups/public/"
               "@economicanalysis/documents/file/oce_flashcrash0314.pdf",
        "retrieved": RETRIEVED,
        "locator": "pdf p. 9",
        "quote": "The minimum price increment, or 'tick' size, of the E-mini is 0.25 index "
                 "points, or $12.50; a price move of one tick represents a fluctuation of "
                 "about 2.5 basis points.",
        "note": "At S&P ~1,000 in May 2010: 0.25/1000 = 2.5bp full tick, 1.25bp half.",
    },
    "budish_cramton_shim": {
        "citation": "Budish, Cramton and Shim, 'The High-Frequency Trading Arms Race', "
                    "Q. J. Econ. 130(4), 2015. NBER working version of 17 Sep 2013.",
        "url": "https://conference.nber.org/confer/2013/MDf13/Budish_Cramton_Shim.pdf",
        "retrieved": RETRIEVED,
        "locator": "pdf pp. 13 (fn. 10), 18, 26",
        "quotes": [
            "ES tick sizes are 0.25 index points, whereas SPY tick sizes are 0.10 index "
            "points.",
            "buying one security while selling the other entails paying half the bid-ask "
            "spread in each market, constituting 0.175 index points in total.",
            "the minimum ES bid-ask spread is substantially larger than the minimum SPY "
            "bid-ask spread (0.25 index points versus 0.10 index points)",
        ],
        "note": "0.175 = 0.125 (half of one ES tick) + 0.05 (half of one SPY cent, in "
                "index points): their 2005-2011 ES/SPY data are treated as quoted at the "
                "minimum tick.",
    },
    "sec_2024_tick_rule": {
        "citation": "SEC Release No. 34-101070, 'Regulation NMS: Minimum Pricing Increments, "
                    "Access Fees, and Transparency of Better Priced Orders', 18 Sep 2024.",
        "url": "https://www.sec.gov/files/rules/final/2024/34-101070.pdf",
        "retrieved": RETRIEVED,
        "spy_nov_2023": {
            "locator": "footnote 801, pdf p. 183",
            "quote": "As of Nov. 30, 2023, the last sale price for SPY was $454.30 and its "
                     "average bid-ask spread over the previous 30 trading days was $0.0105.",
            "price_usd": 454.30,
            "avg_quoted_spread_usd": 0.0105,
        },
        "dollar_volume_by_quoted_spread_2023": {
            "locator": "Table 3, pdf p. 252 (WRDS Intraday Indicators, all 2023 days)",
            "twaqs_le_0.011_usd": {"share_volume_pct": 65.2, "dollar_volume_pct": 31.5,
                                   "avg_n_stocks": 1782},
            "twaqs_0.011_to_0.015_usd": {"share_volume_pct": 9.1, "dollar_volume_pct": 15.3,
                                         "avg_n_stocks": 638},
            "twaqs_gt_0.15_usd": {"share_volume_pct": 2.8, "dollar_volume_pct": 13.0,
                                  "avg_n_stocks": 2177},
        },
        "percentage_spread_2023": {
            "locator": "pdf p. 418",
            "quote": "For each stock-day, divide its TWAQS by its price to measure its "
                     "percentage spread. Only 1% of this sample has a percentage spread "
                     "below 0.015%",
            "p01_full_spread_bp": 1.5,
        },
        "note": "The same release amends Rule 612 to a $0.005 increment for stocks with "
                "TWAQS <= $0.015. SPY qualified on these numbers; if the smaller tick is in "
                "force its minimum half-spread halves. Nothing here relies on it.",
    },
    "hagstromer_sp500_2015": {
        "citation": "Hagstromer, 'Bias in the effective bid-ask spread', J. Financial "
                    "Economics 142(1), 2021. Working version v12 of 25 Sep 2019.",
        "url": "https://www.hec.edu/sites/default/files/documents/overestEspr-v12.pdf",
        "retrieved": RETRIEVED,
        "locator": "Table 1, row 'Quoted spread (bps)', pdf p. 18",
        "sample": "All 506 S&P 500 constituents, 7-11 Dec 2015 (TRTH). NBBO spread just "
                  "before each trade over the midpoint, dollar-weighted per stock; "
                  "percentiles are across stocks.",
        "full_quoted_spread_bp": {
            "mean_dollar_weighted": 3.55, "p05": 1.71, "p25": 2.46, "p50": 3.23,
            "p75": 4.92, "p95": 9.13,
        },
    },
    "collver_sec_2014": {
        "citation": "Collver, 'A characterization of market quality for small "
                    "capitalization US equities', SEC staff memorandum, September 2014.",
        "url": "https://www.sec.gov/marketstructure/research/small_cap_liquidity.pdf",
        "retrieved": RETRIEVED,
        "locator": "Table 2, panel 'Median Relative Quoted Spread (%)', pdf p. 7",
        "sample": "US-listed, US-domiciled common stocks, price >= $2 and market cap < $5bn, "
                  "every 2013 trading day; duration-weighted NBBO spread (MIDAS SIP), first "
                  "five minutes excluded. Large caps (> $5bn) are outside this sample.",
        "cap_bins_usd_m": ["<100", "100-250", "250-500", "500-1000", "1000-2000", "2000-5000"],
        "price_bins_usd": ["2-5.99", "6-9.99", "10-19.99", "20-39.99", ">=40"],
        # rows = price bins, columns = cap bins, full relative quoted spread in percent
        "median_relative_quoted_spread_pct": [
            [1.749, 0.460, 0.292, 0.234, 0.208, 0.237],
            [1.867, 0.558, 0.254, 0.155, 0.135, 0.128],
            [2.079, 0.727, 0.332, 0.166, 0.099, 0.070],
            [2.641, 0.984, 0.492, 0.231, 0.132, 0.066],
            [1.823, 1.226, 0.706, 0.374, 0.178, 0.097],
        ],
    },
    "frazzini_israel_moskowitz_2018": {
        "citation": "Frazzini, Israel and Moskowitz, 'Trading Costs', working paper, "
                    "August 2018 (SSRN 3229719).",
        "url": "https://spinup-000d1a-wp-offload-media.s3.amazonaws.com/faculty/wp-content/"
               "uploads/sites/3/2021/08/Trading-Cost.pdf",
        "retrieved": RETRIEVED,
        "locator": "Table II Panel A and text, pdf pp. 15, 18-19",
        "sample": "USD 1.7tn of one institutional manager's live executions, Aug 1998 - "
                  "Jun 2016, 21 developed markets. Large cap = Russell 1000 benchmark, small "
                  "cap = below it (typically Russell 2000). Costs are one-way versus the "
                  "arrival price, i.e. spread plus impact at institutional size.",
        "market_impact_bp": {"mean_all": 9.97, "median_all": 6.18,
                             "value_weighted_all": 15.14, "mean_large_cap": 8.90,
                             "mean_small_cap": 18.95},
        "avg_quoted_spread_at_arrival_bp": 21.33,
    },
    "fed_feds_note_2020": {
        "citation": "Federal Reserve Board, FEDS Notes, 25 Sep 2020, 'What Do Quoted "
                    "Spreads Tell Us About Machine Trading at Times of Market Stress?'",
        "url": "https://federalreserve.gov/econres/notes/feds-notes/what-do-quoted-spreads-"
               "tell-us-about-machine-trading-market-stress-march-2020-accessible-20200925.htm",
        "retrieved": RETRIEVED,
        "quote": "There was also some increase in quoted spreads in the equity futures "
                 "market, albeit to a lesser extent.",
        "note": "Qualitative only: E-mini spreads widened in March 2020, less than in "
                "Treasury and FX futures. No usable number is given.",
    },
    "sec_decimalization": {
        "citation": "Unger (Acting Chairman, SEC), testimony on the effects of "
                    "decimalization, 24 May 2001.",
        "url": "https://www.sec.gov/newsroom/speeches-statements/"
               "052401tslu-testimony-effects-decimalization-securities-markets",
        "retrieved": RETRIEVED,
        "quote": "the full implementation of decimal pricing on April 9, 2001",
    },
}


# ======================================================================================
# Data -- download once into a cache that git will never see, parse, splice
# ======================================================================================

def _assert_cache_is_untracked(cache_dir: Path) -> None:
    """Refuse to cache raw vendor data anywhere git would commit it.

    The S&P 500 series may not be reproduced; one careless ``git add .`` would publish it.
    Outside the repository is always fine. Inside it, the path must be gitignored.
    """
    try:
        cache_dir.resolve().relative_to(REPO_ROOT)
    except ValueError:
        return
    probe = cache_dir.resolve() / "probe.raw"
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "check-ignore", "-q", str(probe)],
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise SystemExit(
            f"git not found; cannot verify {cache_dir} is ignored. Use --cache-dir "
            "outside the repository."
        ) from exc
    if result.returncode != 0:
        raise SystemExit(
            f"refusing to cache raw market data in a git-tracked path: {cache_dir}\n"
            "use a directory outside the repository, or under experiments/runs/"
        )


def _fetch(name: str, url: str, cache_dir: Path, refresh: bool, agent: str) -> tuple[str, dict]:
    """Return (text, provenance) for ``url``, downloading only if not already cached.

    Provenance -- URL, UTC retrieval time, SHA-256 and size of the raw bytes -- is kept
    in a sidecar so a cached rerun reports when the data was actually retrieved.
    """
    raw_path = cache_dir / f"{name}.raw"
    meta_path = cache_dir / f"{name}.meta.json"
    if refresh or not raw_path.exists() or not meta_path.exists():
        request = urllib.request.Request(url, headers={"User-Agent": agent, "Accept": "*/*"})
        with urllib.request.urlopen(request, timeout=120) as response:
            body = response.read()
        raw_path.write_bytes(body)
        meta = {
            "url": url,
            "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "raw_sha256": hashlib.sha256(body).hexdigest(),
            "raw_bytes": len(body),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    body = raw_path.read_bytes()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if hashlib.sha256(body).hexdigest() != meta["raw_sha256"]:
        raise SystemExit(f"cached {raw_path} does not match its recorded hash; use --refresh")
    return body.decode("utf-8"), meta


def _parse_fred(text: str) -> dict[dt.date, float]:
    """FRED ``fredgraph.csv``: ``observation_date,<ID>``; missing values are '' or '.'."""
    rows = csv.reader(io.StringIO(text))
    next(rows)
    out = {}
    for row in rows:
        if len(row) >= 2 and row[1].strip() not in ("", "."):
            out[dt.date.fromisoformat(row[0])] = float(row[1])
    return out


def _parse_cboe(text: str, column: str) -> dict[dt.date, float]:
    """Cboe ``*_History.csv``: ``DATE`` as MM/DD/YYYY plus value columns."""
    out = {}
    for row in csv.DictReader(io.StringIO(text)):
        value = row[column].strip()
        if value:
            out[dt.datetime.strptime(row["DATE"], "%m/%d/%Y").date()] = float(value)
    return out


def _parse_yahoo(text: str) -> dict[dt.date, float]:
    """Yahoo chart JSON. Bars are stamped at the session open (13:30-14:30 UTC), so the
    UTC calendar date is the trading date."""
    result = json.loads(text)["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    return {
        dt.datetime.fromtimestamp(t, dt.timezone.utc).date(): float(c)
        for t, c in zip(result["timestamp"], closes)
        if c is not None
    }


def _yahoo_url(symbol: str, start: dt.date, end: dt.date) -> str:
    def epoch(d: dt.date) -> int:
        return int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp())
    return YAHOO_CHART.format(
        symbol=urllib.parse.quote(symbol), p1=epoch(start), p2=epoch(end) + 2 * 86_400
    )


def _window(series: dict[dt.date, float], start: dt.date, end: dt.date) -> dict[dt.date, float]:
    return {d: v for d, v in series.items() if start <= d <= end}


def load_data(cache_dir: Path, refresh: bool, end: dt.date) -> tuple[dict, dict]:
    """Download (or reuse) every series; return (series by name, provenance by name)."""
    _assert_cache_is_untracked(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    tag = end.isoformat()
    plan = {
        "vix_fred": (FRED_CSV.format("VIXCLS"), HONEST_AGENT, _parse_fred),
        "vix3m_fred": (FRED_CSV.format("VXVCLS"), HONEST_AGENT, _parse_fred),
        "spx_fred": (FRED_CSV.format("SP500"), HONEST_AGENT, _parse_fred),
        "vix_cboe": (CBOE_CSV.format("VIX"), HONEST_AGENT,
                     lambda t: _parse_cboe(t, "CLOSE")),
        "spx_cboe": (CBOE_CSV.format("SPX"), HONEST_AGENT,
                     lambda t: _parse_cboe(t, "SPX")),
        f"spx_yahoo_{tag}": (_yahoo_url("^GSPC", YAHOO_START, end), BROWSER_AGENT,
                             _parse_yahoo),
        f"spy_yahoo_{tag}": (_yahoo_url("SPY", YAHOO_START, end), BROWSER_AGENT,
                             _parse_yahoo),
    }
    series, provenance = {}, {}
    for name, (url, agent, parse) in plan.items():
        text, meta = _fetch(name, url, cache_dir, refresh, agent)
        key = name.removesuffix(f"_{tag}")
        series[key] = _window(parse(text), YAHOO_START, end)
        provenance[key] = {**meta, **SERIES_NOTES[key]}
    return series, provenance


def splice(authoritative: dict[dt.date, float], fallback: dict[dt.date, float]
           ) -> tuple[dict[dt.date, float], dict]:
    """Authoritative values wherever they exist; fallback strictly before they start.

    Inside the authoritative range, a date missing there but present in the fallback is
    filled from the fallback -- otherwise the next return would silently span two days.
    The count is reported and expected to be zero.
    """
    first = min(authoritative)
    out = {d: v for d, v in fallback.items() if d < first}
    filled = [d for d in fallback if d >= first and d not in authoritative]
    out.update({d: fallback[d] for d in filled})
    out.update(authoritative)
    return out, {"authoritative_from": first.isoformat(), "gap_days_filled": len(filled)}


def _as_arrays(series: dict[dt.date, float], start: dt.date, end: dt.date
               ) -> tuple[list[dt.date], np.ndarray, np.ndarray]:
    days = sorted(d for d in series if start <= d <= end)
    return days, np.array(days, dtype="datetime64[D]"), np.array([series[d] for d in days])


def _fingerprint(series: dict[dt.date, float], start: dt.date, end: dt.date) -> dict:
    """SHA-256 of the in-sample (date, value) pairs: detects vendor revisions on rerun
    without publishing the data itself."""
    items = sorted((d, v) for d, v in series.items() if start <= d <= end)
    text = "".join(f"{d.isoformat()},{v:.6f}\n" for d, v in items)
    return {
        "n": len(items),
        "first": items[0][0].isoformat() if items else None,
        "last": items[-1][0].isoformat() if items else None,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
    }


def _compare(a: dict[dt.date, float], b: dict[dt.date, float], start: dt.date,
             end: dt.date) -> dict:
    """Agreement between two vendors' closes over the range both cover. Levels are not
    reported, only the size of each disagreement in basis points."""
    start = max(start, min(a), min(b))
    end = min(end, max(a), max(b))
    common = sorted(d for d in set(a) & set(b) if start <= d <= end)
    if not common:
        return {"n_common": 0}
    rel = np.array([a[d] / b[d] - 1.0 for d in common])
    worst = int(np.argmax(np.abs(rel)))
    return {
        "range": [start.isoformat(), end.isoformat()],
        "n_common": len(common),
        "n_only_first": sum(1 for d in a if start <= d <= end and d not in b),
        "n_only_second": sum(1 for d in b if start <= d <= end and d not in a),
        "n_disagree_above_1bp": int((np.abs(rel) > 1e-4).sum()),
        "n_disagree_above_10bp": int((np.abs(rel) > 1e-3).sum()),
        "max_abs_rel_diff_bp": float(abs(rel[worst]) * 1e4),
        "date_of_max": common[worst].isoformat(),
        "disagreements_above_10bp": [
            {"date": d.isoformat(), "rel_diff_bp": round(float(r * 1e4), 2)}
            for d, r in zip(common, rel) if abs(r) > 1e-3
        ],
    }


# ======================================================================================
# Volatility misspecification
# ======================================================================================

def forward_realised_vol(closes: np.ndarray, horizon: int = HORIZON,
                         periods_per_year: int = PERIODS_PER_YEAR) -> np.ndarray:
    """RV_i = sqrt( periods_per_year / horizon * sum_{k=1}^{horizon} r_{i+k}^2 ).

    Not demeaned (the variance-swap and VIX convention). ``NaN`` where the forward window
    runs past the last close. Computed from a cumulative sum of squared returns,
    C_k = sum_{j<=k} r_j^2, as C_{i+h} - C_i.
    """
    r = np.diff(np.log(closes))
    c = np.concatenate(([0.0], np.cumsum(r * r)))
    n = closes.size
    out = np.full(n, np.nan)
    if n > horizon:
        out[: n - horizon] = np.sqrt(periods_per_year / horizon * (c[horizon:] - c[: n - horizon]))
    return out


def forward_realised_vol_calendar(dates: np.ndarray, closes: np.ndarray,
                                  days: int = CALENDAR_DAYS) -> np.ndarray:
    """Calendar-matched variant: RV_i^2 = (365 / days) * sum of r_j^2 over trading days
    j with t_i < t_j <= t_i + days calendar days.

    Matches VIX's own convention exactly (VIX^2 is annualised variance over 30 calendar
    days on a 365-day year). ``NaN`` unless t_i + days is inside the sample.
    """
    ordinal = dates.astype("datetime64[D]").astype(np.int64)
    r = np.diff(np.log(closes))
    c = np.concatenate(([0.0], np.cumsum(r * r)))
    last = np.searchsorted(ordinal, ordinal + days, side="right") - 1
    ok = ordinal + days <= ordinal[-1]
    out = np.full(closes.size, np.nan)
    idx = np.nonzero(ok)[0]
    out[idx] = np.sqrt(365.0 / days * (c[last[idx]] - c[idx]))
    return out


@dataclass(frozen=True)
class RhoFrame:
    """rho_t = RV_t / IV_t on every S&P trading day t with a quote and a full window."""

    dates: np.ndarray       # datetime64[D], the day the hedge is struck
    position: np.ndarray    # index of t in the S&P 500 trading calendar
    iv: np.ndarray          # implied vol, decimal
    rv: np.ndarray          # forward realised vol, decimal
    rho: np.ndarray
    horizon: int

    def select(self, mask: np.ndarray) -> "RhoFrame":
        return RhoFrame(self.dates[mask], self.position[mask], self.iv[mask],
                        self.rv[mask], self.rho[mask], self.horizon)


def build_rho(days: list[dt.date], dates: np.ndarray, closes: np.ndarray,
              implied_pct: dict[dt.date, float], horizon: int = HORIZON,
              rv: np.ndarray | None = None) -> RhoFrame:
    """Align implied vol to the S&P calendar and form rho.

    Days with an implied-vol print but no S&P close (US holidays on which Cboe computed
    VIX during global trading hours) are dropped: there is no close to hedge from.
    """
    if rv is None:
        rv = forward_realised_vol(closes, horizon)
    iv = np.array([implied_pct.get(d, np.nan) for d in days]) / 100.0
    ok = np.isfinite(rv) & np.isfinite(iv)
    return RhoFrame(dates[ok], np.nonzero(ok)[0], iv[ok], rv[ok], rv[ok] / iv[ok], horizon)


KEY_STATS: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda x: float(np.mean(x)),
    "median": lambda x: float(np.median(x)),
    "p_rho_gt_1.0": lambda x: float(np.mean(x > 1.0)),
    "p_rho_gt_1.1": lambda x: float(np.mean(x > 1.1)),
    "p_rho_gt_1.25": lambda x: float(np.mean(x > 1.25)),
    "p_rho_lt_0.9": lambda x: float(np.mean(x < 0.9)),
}


def summarise(frame: RhoFrame) -> dict:
    """Distribution of rho: moments, quantiles, and the frequency of each sweep level.

    ``n_independent_windows_approx`` counts the distinct horizon-length blocks of the
    trading calendar the strike days fall in: ~n/21 for consecutive days, n for a
    non-overlapping subsample. It is the honest sample size for a window's frequencies.
    ``vol_points_error`` is 100 * (RV - IV): the error in the units the sweep's original
    '22% realised vs 20% hedged' example was quoted in.
    """
    rho = frame.rho
    if rho.size == 0:
        return {"n": 0}
    err = 100.0 * (frame.rv - frame.iv)
    return {
        "n": int(rho.size),
        "n_independent_windows_approx": int(np.unique(frame.position // frame.horizon).size),
        "first": str(frame.dates[0]),
        "last": str(frame.dates[-1]),
        "mean": float(rho.mean()),
        "sd": float(rho.std(ddof=1)) if rho.size > 1 else None,
        "min": float(rho.min()),
        "max": float(rho.max()),
        "quantiles": {f"p{round(q * 100):02d}": float(np.quantile(rho, q)) for q in QUANTILES},
        "p_rho_gt": {f"{k}": float(np.mean(rho > k)) for k in ABOVE},
        "p_rho_lt": {f"{k}": float(np.mean(rho < k)) for k in BELOW},
        "mean_implied_vol_pct": float(100.0 * frame.iv.mean()),
        "mean_realised_vol_pct": float(100.0 * frame.rv.mean()),
        "vol_points_error": {
            "mean": float(err.mean()),
            **{f"p{round(q * 100):02d}": float(np.quantile(err, q)) for q in (0.05, 0.5, 0.95)},
        },
    }


def block_bootstrap_ci(x: np.ndarray, block: int = BOOTSTRAP_BLOCK,
                       reps: int = BOOTSTRAP_REPS, seed: int = BOOTSTRAP_SEED) -> dict:
    """95% percentile intervals for KEY_STATS from a circular block bootstrap.

    Each replicate concatenates ceil(n / block) blocks of ``block`` consecutive days with
    uniformly random (circular) starts, truncated to n. i.i.d. resampling would treat
    ~9,000 heavily overlapping windows as independent and overstate precision ~20-fold.
    """
    rng = np.random.default_rng(seed)
    n = x.size
    k = -(-n // block)
    offsets = np.arange(block)
    draws = {name: np.empty(reps) for name in KEY_STATS}
    for b in range(reps):
        starts = rng.integers(0, n, size=k)
        sample = x[((starts[:, None] + offsets[None, :]) % n).ravel()[:n]]
        for name, stat in KEY_STATS.items():
            draws[name][b] = stat(sample)
    return {
        "method": f"circular block bootstrap, block {block} trading days, {reps} replicates, "
                  f"numpy default_rng seed {seed}",
        **{name: [float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))]
           for name, v in draws.items()},
    }


def non_overlapping(frame: RhoFrame, step: int) -> dict:
    """Every ``step``-th trading day, so no two windows share a return.

    Offset 0 (from the first day of the S&P calendar) is reported in full; the range of
    each key statistic across all ``step`` offsets shows the choice of offset is harmless.
    """
    per_offset = [frame.select(frame.position % step == o) for o in range(step)]
    ranges = {
        name: [min(stat(f.rho) for f in per_offset), max(stat(f.rho) for f in per_offset)]
        for name, stat in KEY_STATS.items()
    }
    return {
        "step_trading_days": step,
        "offset_0": summarise(per_offset[0]),
        "key_stat_range_across_offsets": ranges,
    }


def _mask(frame: RhoFrame, start: dt.date, end: dt.date) -> np.ndarray:
    return (frame.dates >= np.datetime64(start)) & (frame.dates <= np.datetime64(end))


def by_vix_level(frame: RhoFrame) -> dict:
    """Is the adverse tail a high-VIX phenomenon? Bucket on VIX at the strike date."""
    out = {}
    vix = 100.0 * frame.iv
    for lo, hi in zip(VIX_BUCKET_EDGES[:-1], VIX_BUCKET_EDGES[1:]):
        f = frame.select((vix >= lo) & (vix < hi))
        label = f"vix_{lo:g}_to_{hi:g}" if np.isfinite(hi) else f"vix_ge_{lo:g}"
        out[label] = {
            "n": int(f.rho.size),
            "share_of_sample": float(f.rho.size / frame.rho.size),
            **({name: stat(f.rho) for name, stat in KEY_STATS.items()} if f.rho.size else {}),
        }
    return out


def adverse_episodes(frame: RhoFrame, threshold: float = ADVERSE_EPISODE_THRESHOLD,
                     gap: int = EPISODE_GAP) -> dict:
    """Cluster days with rho > threshold into episodes and rank them by peak rho.

    A new episode starts when the next adverse day is more than ``gap`` trading days
    after the previous one. VIX at the peak day answers *when* the adverse tail happens:
    at calm-looking levels just before a shock, or deep inside a crisis.
    """
    idx = np.nonzero(frame.rho > threshold)[0]
    if idx.size == 0:
        return {"threshold": threshold, "n_episodes": 0}
    groups = np.split(idx, np.nonzero(np.diff(frame.position[idx]) > gap)[0] + 1)
    episodes = []
    for g in groups:
        peak = g[np.argmax(frame.rho[g])]
        episodes.append({
            "first_adverse_day": str(frame.dates[g[0]]),
            "last_adverse_day": str(frame.dates[g[-1]]),
            "n_adverse_days": int(g.size),
            "peak_day": str(frame.dates[peak]),
            "peak_rho": round(float(frame.rho[peak]), 3),
            "vix_at_peak": round(float(100.0 * frame.iv[peak]), 1),
            "realised_vol_at_peak_pct": round(float(100.0 * frame.rv[peak]), 1),
        })
    episodes.sort(key=lambda e: -e["peak_rho"])
    peak_vix = np.array([e["vix_at_peak"] for e in episodes])
    median_vix = float(np.median(100.0 * frame.iv))
    extreme = [e for e in episodes if e["peak_rho"] > 2.0]
    return {
        "threshold": threshold,
        "gap_trading_days": gap,
        "n_adverse_days": int(idx.size),
        "n_episodes": len(episodes),
        "n_episodes_peak_rho_gt_1.5": sum(e["peak_rho"] > 1.5 for e in episodes),
        "n_episodes_peak_rho_gt_2": len(extreme),
        "median_vix_at_episode_peak": float(np.median(peak_vix)),
        "median_vix_full_sample": median_vix,
        "share_of_episodes_peaking_below_median_vix": float(np.mean(peak_vix < median_vix)),
        "n_episodes_peak_rho_gt_2_with_vix_below_median":
            sum(e["vix_at_peak"] < median_vix for e in extreme),
        "top_10": episodes[:10],
    }


def concentration(frame: RhoFrame) -> dict:
    """How much of the adverse tail sits inside the named stress windows?

    Compares each window's share of adverse days with its share of all days; a ratio
    well above one means the tail is concentrated there.
    """
    inside = np.zeros(frame.rho.size, dtype=bool)
    for start, end in STRESS_WINDOWS.values():
        inside |= _mask(frame, start, end)
    out = {
        "windows": sorted(STRESS_WINDOWS),
        "share_of_all_days_inside": float(inside.mean()),
    }
    for thr in (1.1, 1.25, 1.5):
        adverse = frame.rho > thr
        share = float(adverse[inside].sum() / adverse.sum()) if adverse.any() else None
        out[f"rho_gt_{thr}"] = {
            "share_of_adverse_days_inside": share,
            "concentration_ratio": share / float(inside.mean()) if share is not None else None,
            "p_inside": float(adverse[inside].mean()),
            "p_outside": float(adverse[~inside].mean()),
        }
    return out


def vendor_sensitivity(primary: RhoFrame, alternative: RhoFrame) -> dict:
    """Headline statistics recomputed on the other S&P 500 splice, on common days."""
    common, ia, ib = np.intersect1d(primary.dates, alternative.dates, return_indices=True)
    a, b = primary.rho[ia], alternative.rho[ib]
    diffs = {name: abs(stat(a) - stat(b)) for name, stat in KEY_STATS.items()}
    delta = np.abs(a - b)
    worst = int(np.argmax(delta))
    return {
        "n_common_days": int(common.size),
        "abs_diff_in_key_stats": diffs,
        "max_abs_diff_in_any_key_stat": max(diffs.values()),
        "n_days_rho_differs_by_more_than_0.01": int((delta > 0.01).sum()),
        "max_abs_rho_diff": float(delta[worst]),
        "date_of_max_rho_diff": str(common[worst]),
    }


# ======================================================================================
# Cost regime
# ======================================================================================

def half_spread_bp(tick: float, price: np.ndarray | float) -> np.ndarray | float:
    """c = (tick / 2) / price, in basis points: the one-way cost of crossing a one-tick
    quote, which is what ``c`` in the simulator's ``c * S * |trade|`` represents."""
    return 0.5 * tick / np.asarray(price) * 1e4


def _sig4(x: float) -> float:
    """Four significant figures. A tick-over-price statistic at full precision inverts to
    the exact index close, which is the vendor's data rather than a derived statistic."""
    return float(f"{x:.4g}")


def _level_summary(days: list[dt.date], values: np.ndarray) -> dict:
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    years = np.array([d.year for d in days])
    return {
        "first": days[0].isoformat(),
        "last": days[-1].isoformat(),
        "at_last": _sig4(values[-1]),
        "median": _sig4(np.median(values)),
        "min": _sig4(values[lo]), "min_date": days[lo].isoformat(),
        "max": _sig4(values[hi]), "max_date": days[hi].isoformat(),
        "median_by_year": {str(int(y)): _sig4(np.median(values[years == y]))
                           for y in np.unique(years)},
    }


def cost_regime(spx_days: list[dt.date], spx: np.ndarray, spy_series: dict[dt.date, float],
                end: dt.date) -> dict:
    """One-tick half-spreads for the index hedging instruments, and what the sweep's cost
    levels would mean in ES ticks; single-stock levels derived from ``LITERATURE``."""
    es_idx = [i for i, d in enumerate(spx_days) if d >= ES_START]
    es_days = [spx_days[i] for i in es_idx]
    es_bp = half_spread_bp(ES_TICK, spx[es_idx])
    s_end = float(spx[-1])

    spy_days, _, spy = _as_arrays(spy_series, DECIMALIZATION_COMPLETE, end)
    spy_bp = half_spread_bp(SPY_TICK, spy)
    spx_by_day = dict(zip(spx_days, spx))
    ratio = np.array([p / spx_by_day[d] for d, p in zip(spy_days, spy) if d in spx_by_day])

    sec = LITERATURE["sec_2024_tick_rule"]["spy_nov_2023"]
    hag = LITERATURE["hagstromer_sp500_2015"]["full_quoted_spread_bp"]
    col = LITERATURE["collver_sec_2014"]
    table = np.array(col["median_relative_quoted_spread_pct"]) * 100.0 / 2.0   # -> half, bp
    fim = LITERATURE["frazzini_israel_moskowitz_2018"]["market_impact_bp"]

    def rng_(block: np.ndarray) -> list[float]:
        return [round(float(block.min()), 2), round(float(block.max()), 2)]

    return {
        "definition": "one-way proportional cost c = (tick/2)/price in bp: crossing a quote "
                      "one tick wide. Commissions, fees and market impact are excluded.",
        "es": {
            "tick_index_points": ES_TICK,
            "tick_value_usd": ES_TICK * ES_MULTIPLIER,
            "source": "LITERATURE.cme_rule_35802",
            "one_tick_half_spread_bp": _level_summary(es_days, es_bp),
            "sweep_levels_at_last_close": {
                f"{c * 1e4:g}bp": {
                    "half_spread_index_points": _sig4(c * s_end),
                    "equivalent_quoted_width_es_ticks": _sig4(2.0 * c * s_end / ES_TICK),
                    "multiple_of_one_tick_half_spread": _sig4(c * 1e4 / float(es_bp[-1])),
                }
                for c in SWEEP_COSTS
            },
        },
        "spy": {
            "tick_usd": SPY_TICK,
            "from": DECIMALIZATION_COMPLETE.isoformat(),
            "one_cent_half_spread_bp": _level_summary(spy_days, spy_bp),
            "spy_over_spx_ratio_range": [_sig4(ratio.min()), _sig4(ratio.max())],
            "sec_measured_nov_2023": {
                "avg_quoted_spread_usd": sec["avg_quoted_spread_usd"],
                "price_usd": sec["price_usd"],
                "full_spread_bp": sec["avg_quoted_spread_usd"] / sec["price_usd"] * 1e4,
                "half_spread_bp": 0.5 * sec["avg_quoted_spread_usd"] / sec["price_usd"] * 1e4,
            },
        },
        "single_stocks_half_spread_bp": {
            "sp500_constituents_dec_2015": {k: v / 2.0 for k, v in hag.items()},
            "mid_cap_1bn_to_5bn_2013_median_range": rng_(table[:, 4:6]),
            "small_cap_100m_to_1bn_2013_median_range": rng_(table[:, 1:4]),
            "micro_cap_below_100m_2013_median_range": rng_(table[:, 0:1]),
            "sources": ["LITERATURE.hagstromer_sp500_2015", "LITERATURE.collver_sec_2014"],
        },
        "institutional_all_in_one_way_bp": {
            "large_cap_mean_market_impact": fim["mean_large_cap"],
            "small_cap_mean_market_impact": fim["mean_small_cap"],
            "source": "LITERATURE.frazzini_israel_moskowitz_2018",
        },
    }


# ======================================================================================
# Driver
# ======================================================================================

def run(cache_dir: Path, end: dt.date, spx_vendor: str, refresh: bool) -> dict:
    """Load, cross-check, compute. Returns the JSON-ready record; prints nothing."""
    s, provenance = load_data(cache_dir, refresh, end)
    other = "cboe" if spx_vendor == "yahoo" else "yahoo"
    spx, splice_meta = splice(s["spx_fred"], s[f"spx_{spx_vendor}"])
    spx_alt, _ = splice(s["spx_fred"], s[f"spx_{other}"])
    days, dates, closes = _as_arrays(spx, SAMPLE_START, end)
    days_alt, dates_alt, closes_alt = _as_arrays(spx_alt, SAMPLE_START, end)

    frame = build_rho(days, dates, closes, s["vix_fred"])
    alt_frame = build_rho(days_alt, dates_alt, closes_alt, s["vix_fred"])
    cal_frame = build_rho(days, dates, closes, s["vix_fred"],
                          rv=forward_realised_vol_calendar(dates, closes))
    frame_3m = build_rho(days, dates, closes, s["vix3m_fred"], horizon=HORIZON_3M,
                         rv=forward_realised_vol(closes, HORIZON_3M))
    both, i1m, _ = np.intersect1d(frame.dates, frame_3m.dates, return_indices=True)
    mask_1m_common = np.zeros(frame.rho.size, dtype=bool)
    mask_1m_common[i1m] = True

    vix_in_sample = _window(s["vix_fred"], SAMPLE_START, end)
    spx_day_set = set(days)
    no_close = sorted(d for d in vix_in_sample if d not in spx_day_set)
    no_vix = sorted(d for d in days if d not in vix_in_sample)
    vix_vs_cboe = [abs(s["vix_fred"][d] - s["vix_cboe"][d])
                   for d in s["vix_fred"].keys() & s["vix_cboe"].keys()
                   if SAMPLE_START <= d <= end]
    fred_from = dt.date.fromisoformat(splice_meta["authoritative_from"])

    vol = {
        "definition": "rho_t = RV_t / (VIX_t/100); RV_t = sqrt(252/21 * sum_{k=1..21} "
                      "r_{t+k}^2), r = daily log return of the S&P 500 close; rho > 1 is "
                      "adverse for a short-gamma hedger hedging at implied vol",
        "full_sample": summarise(frame),
        "full_sample_ci95": {
            **block_bootstrap_ci(frame.rho),
            "block_length_sensitivity": {
                f"block_{b}": {k: v for k, v in block_bootstrap_ci(frame.rho, block=b).items()
                               if k in ("median", "p_rho_gt_1.1", "p_rho_gt_1.25")}
                for b in BOOTSTRAP_BLOCK_SENSITIVITY
            },
        },
        "non_overlapping": non_overlapping(frame, HORIZON),
        "stress_windows": {
            name: {"from": a.isoformat(), "to": b.isoformat(),
                   **summarise(frame.select(_mask(frame, a, b)))}
            for name, (a, b) in STRESS_WINDOWS.items()
        },
        "concentration_in_stress_windows": concentration(frame),
        "by_vix_level": by_vix_level(frame),
        "adverse_episodes": adverse_episodes(frame),
        "subperiods": {
            name: summarise(frame.select(_mask(frame, a, b)))
            for name, (a, b) in SUBPERIODS.items()
        },
        "sensitivity": {
            "calendar_matched_30d_365": {
                "definition": "RV^2 = 365/30 * sum of r^2 over trading days in (t, t+30 "
                              "calendar days]",
                **summarise(cal_frame),
            },
            f"spx_vendor_{other}_before_fred": vendor_sensitivity(frame, alt_frame),
        },
        "horizon_3m": {
            "definition": "rho63 = RV over next 63 trading days (252/63 annualised) / "
                          "(VIX3M/100); FRED VXVCLS from 2007-12-04. Matches the sweep's "
                          "3-month contracts; the 1-month rho on the same days is shown "
                          "alongside so horizon and period effects are not confused",
            "rho_3m": summarise(frame_3m.select(np.isin(frame_3m.dates, both))),
            "rho_1m_same_days": summarise(frame.select(mask_1m_common)),
            "rho_3m_non_overlapping": non_overlapping(
                frame_3m.select(np.isin(frame_3m.dates, both)), HORIZON_3M),
        },
    }

    integrity = {
        "spx_splice": {
            "primary_vendor_before_fred": spx_vendor,
            **splice_meta,
            "n_closes_in_sample": len(days),
        },
        "vix_fred_vs_cboe": {
            "n_common": len(vix_vs_cboe),
            "max_abs_diff_index_points": float(max(vix_vs_cboe)),
            "n_diff_above_0.005": int(sum(v > 0.005 for v in vix_vs_cboe)),
        },
        "spx_yahoo_vs_fred": _compare(s["spx_yahoo"], s["spx_fred"], SAMPLE_START, end),
        "spx_cboe_vs_fred": _compare(s["spx_cboe"], s["spx_fred"], SAMPLE_START, end),
        "spx_yahoo_vs_cboe_before_fred": _compare(
            s["spx_yahoo"], s["spx_cboe"], SAMPLE_START, fred_from - dt.timedelta(days=1)),
        "calendar": {
            "vix_days_without_spx_close_dropped": len(no_close),
            "examples_dropped": [d.isoformat() for d in no_close[:5]],
            "spx_days_without_vix": len(no_vix),
        },
        "fingerprints_in_sample": {
            name: _fingerprint(series, SAMPLE_START, end) for name, series in s.items()
        },
        "last_observation": {name: max(series).isoformat() for name, series in s.items()},
    }

    costs = cost_regime(days, closes, s["spy_yahoo"], end)
    full, nov = vol["full_sample"], vol["non_overlapping"]
    headline = {
        "sample": f"{full['first']} to {full['last']} (strike dates)",
        "median_rho": full["quantiles"]["p50"],
        "mean_rho": full["mean"],
        "p_rho_gt_1": full["p_rho_gt"]["1.0"],
        "p_rho_gt_1.1": full["p_rho_gt"]["1.1"],
        "p_rho_gt_1.25": full["p_rho_gt"]["1.25"],
        "p_rho_lt_0.9": full["p_rho_lt"]["0.9"],
        "p_rho_gt_1.1_ci95": vol["full_sample_ci95"]["p_rho_gt_1.1"],
        "p_rho_gt_1.1_non_overlapping_offset_range":
            nov["key_stat_range_across_offsets"]["p_rho_gt_1.1"],
        "es_one_tick_half_spread_bp_at_last": costs["es"]["one_tick_half_spread_bp"]["at_last"],
        "es_one_tick_half_spread_bp_range": [
            costs["es"]["one_tick_half_spread_bp"]["min"],
            costs["es"]["one_tick_half_spread_bp"]["max"],
        ],
        "spy_one_cent_half_spread_bp_at_last":
            costs["spy"]["one_cent_half_spread_bp"]["at_last"],
        "sp500_stock_median_half_spread_bp_2015":
            costs["single_stocks_half_spread_bp"]["sp500_constituents_dec_2015"]["p50"],
        "small_cap_median_half_spread_bp_range_2013":
            costs["single_stocks_half_spread_bp"]["small_cap_100m_to_1bn_2013_median_range"],
    }

    return {
        "generated_by": "python -m experiments.market_calibration",
        "numpy_version": np.__version__,
        "sample_end": end.isoformat(),
        "headline": headline,
        "vol_misspecification": vol,
        "cost_regime": costs,
        "literature": LITERATURE,
        "data_sources": provenance,
        "data_integrity": integrity,
    }


def _print_summary(out: dict) -> None:
    vol, costs, integ = out["vol_misspecification"], out["cost_regime"], out["data_integrity"]
    sp = integ["spx_splice"]
    print(f"\n  S&P 500 closes: FRED from {sp['authoritative_from']}, "
          f"{sp['primary_vendor_before_fred']} before  (n = {sp['n_closes_in_sample']:,})")
    v = integ["vix_fred_vs_cboe"]
    print(f"  VIX: FRED vs Cboe on {v['n_common']:,} days, max |diff| "
          f"{v['max_abs_diff_index_points']:.4f}")
    for key in ("spx_yahoo_vs_fred", "spx_cboe_vs_fred", "spx_yahoo_vs_cboe_before_fred"):
        c = integ[key]
        print(f"  {key:<32} common {c['n_common']:>5,}  >1bp {c['n_disagree_above_1bp']:>3}"
              f"  >10bp {c['n_disagree_above_10bp']:>2}  max {c['max_abs_rel_diff_bp']:7.1f}bp"
              f" ({c['date_of_max']})")

    print("\n  rho = RV(next 21 trading days) / (VIX/100); >1 adverse for a short-gamma hedger\n")
    print(f"  {'':<34}{'n':>6}{'mean':>7}{'p05':>6}{'p50':>6}{'p95':>6}"
          f"{'P>1':>7}{'P>1.1':>7}{'P>1.25':>7}{'P<0.9':>7}")

    def row(label: str, sm: dict) -> None:
        if sm.get("n", 0) == 0:
            print(f"  {label:<34}{0:>6}")
            return
        q, g, l_ = sm["quantiles"], sm["p_rho_gt"], sm["p_rho_lt"]
        print(f"  {label:<34}{sm['n']:>6}{sm['mean']:>7.3f}{q['p05']:>6.2f}{q['p50']:>6.2f}"
              f"{q['p95']:>6.2f}{g['1.0']:>7.1%}{g['1.1']:>7.1%}{g['1.25']:>7.1%}"
              f"{l_['0.9']:>7.1%}")

    row("full sample (overlapping)", vol["full_sample"])
    row("non-overlapping, offset 0", vol["non_overlapping"]["offset_0"])
    for name, sm in vol["stress_windows"].items():
        row(name, sm)
    for name, sm in vol["subperiods"].items():
        row(name, sm)
    row("calendar-matched 30d/365", vol["sensitivity"]["calendar_matched_30d_365"])
    row("3m: rho63 vs VIX3M", vol["horizon_3m"]["rho_3m"])
    row("1m on the same days", vol["horizon_3m"]["rho_1m_same_days"])

    ci = vol["full_sample_ci95"]
    rng = vol["non_overlapping"]["key_stat_range_across_offsets"]
    print(f"\n  95% CI (block bootstrap, {BOOTSTRAP_BLOCK}d):  P>1.1 "
          f"[{ci['p_rho_gt_1.1'][0]:.1%}, {ci['p_rho_gt_1.1'][1]:.1%}]   median "
          f"[{ci['median'][0]:.3f}, {ci['median'][1]:.3f}]")
    for name, b in ci["block_length_sensitivity"].items():
        print(f"      {name:<10} P>1.1 [{b['p_rho_gt_1.1'][0]:.1%}, {b['p_rho_gt_1.1'][1]:.1%}]"
              f"   median [{b['median'][0]:.3f}, {b['median'][1]:.3f}]")
    print(f"  non-overlapping, all 21 offsets:  P>1.1 in "
          f"[{rng['p_rho_gt_1.1'][0]:.1%}, {rng['p_rho_gt_1.1'][1]:.1%}]")
    sens = vol["sensitivity"][[k for k in vol["sensitivity"] if k.startswith("spx_vendor")][0]]
    print(f"  S&P vendor swap: max change in any key statistic "
          f"{sens['max_abs_diff_in_any_key_stat']:.4f}")

    conc = vol["concentration_in_stress_windows"]
    print(f"\n  stress windows hold {conc['share_of_all_days_inside']:.1%} of days;")
    for thr in (1.1, 1.25, 1.5):
        c = conc[f"rho_gt_{thr}"]
        print(f"    rho>{thr:<5} {c['share_of_adverse_days_inside']:.1%} of adverse days inside "
              f"({c['concentration_ratio']:.1f}x);  P inside {c['p_inside']:.1%}, "
              f"outside {c['p_outside']:.1%}")

    print("\n  by VIX at the strike date:")
    for name, b in vol["by_vix_level"].items():
        if b["n"]:
            print(f"    {name:<14} n {b['n']:>5}  median {b['median']:.2f}  "
                  f"P>1.1 {b['p_rho_gt_1.1']:6.1%}  P>1.25 {b['p_rho_gt_1.25']:6.1%}")

    ep = vol["adverse_episodes"]
    print(f"\n  adverse episodes (rho > {ep['threshold']}): {ep['n_episodes']} episodes, "
          f"{ep['n_adverse_days']} days; median VIX at peak {ep['median_vix_at_episode_peak']:.1f}"
          f" vs sample median {ep['median_vix_full_sample']:.1f}")
    print(f"    {ep['n_episodes_peak_rho_gt_2']} peaked above rho = 2; "
          f"{ep['n_episodes_peak_rho_gt_2_with_vix_below_median']} of them from VIX below "
          "its sample median")
    for e in ep["top_10"]:
        print(f"    {e['peak_day']}  rho {e['peak_rho']:5.2f}  VIX {e['vix_at_peak']:5.1f}  "
              f"RV {e['realised_vol_at_peak_pct']:5.1f}  ({e['n_adverse_days']} days)")

    es, spy = costs["es"]["one_tick_half_spread_bp"], costs["spy"]["one_cent_half_spread_bp"]
    print("\n  one-tick quoted half-spread, bp (c in c*S*|trade|):")
    print(f"    ES  0.25pt : {es['at_last']:.3f} at {es['last']};  range "
          f"{es['min']:.3f} ({es['min_date']}) .. {es['max']:.3f} ({es['max_date']})")
    print(f"    SPY $0.01  : {spy['at_last']:.3f} at {spy['last']};  range "
          f"{spy['min']:.3f} .. {spy['max']:.3f};  SEC Nov-2023 measured "
          f"{costs['spy']['sec_measured_nov_2023']['half_spread_bp']:.3f}")
    for lvl, d in costs["es"]["sweep_levels_at_last_close"].items():
        print(f"    sweep {lvl:>5} = ES quote {d['equivalent_quoted_width_es_ticks']:6.1f} ticks "
              f"wide = {d['multiple_of_one_tick_half_spread']:6.1f}x the one-tick half-spread")
    ss = costs["single_stocks_half_spread_bp"]
    print(f"    S&P 500 stocks, Dec 2015: median {ss['sp500_constituents_dec_2015']['p50']:.2f}"
          f", p95 {ss['sp500_constituents_dec_2015']['p95']:.2f}")
    print(f"    2013 medians: mid cap $1-5bn {ss['mid_cap_1bn_to_5bn_2013_median_range']}, "
          f"small $0.1-1bn {ss['small_cap_100m_to_1bn_2013_median_range']}, "
          f"micro <$0.1bn {ss['micro_cap_below_100m_2013_median_range']}")
    inst = costs["institutional_all_in_one_way_bp"]
    print(f"    institutional all-in (impact): large {inst['large_cap_mean_market_impact']}, "
          f"small {inst['small_cap_mean_market_impact']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR,
                        help="where raw downloads are cached (must be untracked by git)")
    parser.add_argument("--end", type=dt.date.fromisoformat, default=SAMPLE_END,
                        help=f"last S&P 500 close used (default {SAMPLE_END})")
    parser.add_argument("--spx-vendor", choices=("yahoo", "cboe"), default="yahoo",
                        help="S&P 500 source before FRED's ten-year window")
    parser.add_argument("--refresh", action="store_true", help="re-download cached files")
    parser.add_argument("--json", metavar="PATH", help="write derived statistics as JSON")
    args = parser.parse_args()

    out = run(args.cache_dir, args.end, args.spx_vendor, args.refresh)
    last = out["data_integrity"]["last_observation"]
    if dt.date.fromisoformat(last["spx_fred"]) < args.end:
        print(f"  WARNING: FRED S&P 500 ends {last['spx_fred']}, before --end {args.end}; "
              "the cache may be stale (use --refresh)")
    _print_summary(out)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, sort_keys=True)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
