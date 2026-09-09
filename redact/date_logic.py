"""Accident-date-anchored date redaction logic.

Rule (per user spec):
  - Every date is compared against the document's "Date of Accident".
  - Within 2 years (730 days) of that anchor -> replaced with a relative
    offset: "[N DAYS POST ACCIDENT]" / "[N DAYS PRE ACCIDENT]".
  - More than 2 years away (e.g. a date of birth) -> fully redacted to
    "[DATE]".
"""
import re
from datetime import date
from typing import Optional

TWO_YEARS_DAYS = 730

MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2, "mar": 3, "march": 3,
    "apr": 4, "april": 4, "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "sept": 9, "september": 9, "oct": 10,
    "october": 10, "nov": 11, "november": 11, "dec": 12, "december": 12,
}

# MM/DD/YY or MM/DD/YYYY
NUMERIC_DATE_RE = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{2,4})\b")

# "Jun 18, 1989" / "June 18 1989"
MONTH_NAME_DATE_RE = re.compile(
    r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b"
)

# OCR sometimes splits "DOB: Sep 29, 1970" across cells (even across a
# blank line), leaving a dangling "DOB: ... 29, 1970" with no month. The
# month is unrecoverable, but it's always a birth date (>2 years from any
# accident) so an approximate month is fine for the distance threshold -
# it still resolves to [DATE]. Uses capturing groups (not a lookbehind,
# which must be fixed-width) so "DOB:" can be separated from the day/year
# by an arbitrary amount of whitespace, including blank lines.
DOB_DAY_YEAR_RE = re.compile(r"DOB:\s*(\d{1,2}),\s*(\d{4})\b")

# Whole-match patterns: the entire regex match is the date span.
DATE_PATTERN_REGEXES = [NUMERIC_DATE_RE, MONTH_NAME_DATE_RE]

# Sub-group patterns: only the captured groups (spanning group 1's start
# to the last group's end) are the date span - the rest of the match is
# context (e.g. "DOB:") that must stay in the output.
DATE_SUBGROUP_REGEXES = [DOB_DAY_YEAR_RE]

ACCIDENT_ANCHOR_RE = re.compile(
    r"date\s+of\s+accident\s*:?\s*(\d{1,2}/\d{1,2}/\d{2,4})", re.IGNORECASE
)


def _pivot_year(yy: int) -> int:
    if yy >= 100:
        return yy
    return 2000 + yy if yy <= 50 else 1900 + yy


def parse_numeric_date(match: re.Match) -> Optional[date]:
    mm, dd, yy = int(match.group(1)), int(match.group(2)), int(match.group(3))
    yyyy = _pivot_year(yy)
    try:
        return date(yyyy, mm, dd)
    except ValueError:
        return None


def parse_month_name_date(match: re.Match) -> Optional[date]:
    month_name, dd, yyyy = match.group(1).lower(), int(match.group(2)), int(match.group(3))
    mm = MONTHS.get(month_name)
    if mm is None:
        return None
    try:
        return date(yyyy, mm, dd)
    except ValueError:
        return None


def parse_day_year_only(match: re.Match) -> Optional[date]:
    dd, yyyy = int(match.group(1)), int(match.group(2))
    try:
        return date(yyyy, 1, min(dd, 28))
    except ValueError:
        return None


def parse_date_text(text: str) -> Optional[date]:
    """Best-effort parse of a date-shaped string using the same patterns
    used for detection."""
    stripped = text.strip()
    m = NUMERIC_DATE_RE.fullmatch(stripped)
    if m:
        return parse_numeric_date(m)
    m = MONTH_NAME_DATE_RE.fullmatch(stripped)
    if m:
        return parse_month_name_date(m)
    m = DOB_DAY_YEAR_RE.fullmatch(stripped) or re.fullmatch(r"(\d{1,2}),\s*(\d{4})", stripped)
    if m:
        return parse_day_year_only(m)
    return None


def find_accident_anchor(text: str) -> Optional[date]:
    """Find the first 'Date of Accident: MM/DD/YY(YY)' occurrence in the
    document and return it as the anchor date all other dates are
    compared against."""
    match = ACCIDENT_ANCHOR_RE.search(text)
    if not match:
        return None
    numeric_match = NUMERIC_DATE_RE.fullmatch(match.group(1))
    if not numeric_match:
        return None
    return parse_numeric_date(numeric_match)


def redact_date(parsed: date, anchor: date) -> str:
    diff_days = (parsed - anchor).days
    if abs(diff_days) > TWO_YEARS_DAYS:
        return "[DATE]"
    if diff_days >= 0:
        return f"[{diff_days} DAYS POST ACCIDENT]"
    return f"[{abs(diff_days)} DAYS PRE ACCIDENT]"


def redact_date_text(original_text: str, anchor: Optional[date]) -> str:
    """Given the original matched date substring, return its bracketed
    replacement. Falls back to a generic [DATE] if it can't be parsed or
    there is no accident anchor to compare against."""
    if anchor is None:
        return "[DATE]"
    parsed = parse_date_text(original_text)
    if parsed is None:
        return "[DATE]"
    return redact_date(parsed, anchor)
