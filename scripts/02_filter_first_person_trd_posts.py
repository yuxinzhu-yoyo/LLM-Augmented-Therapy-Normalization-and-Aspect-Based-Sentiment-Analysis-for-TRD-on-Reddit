#!/usr/bin/env python3
"""TRD Personal-Experience Filter
================================
Given a CSV of Reddit posts that already *mention* treatment-resistant
depression (TRD), keep only those rows where at least **one sentence** is both

* first-person (contains *I, me, my, mine*) **and**
* mentions a TRD-related term from ``LEXICON_PATTERNS``.

The script is **stage 2** of the pipeline - it assumes you already ran the
lexicon search and produced something like ``TRD_all_matches.csv``.  It does *not*
rescan all source CSVs; it focuses on one file.

Outputs
-------
* ``<input>_personal.csv`` - filtered rows (all original columns preserved)
* ``<input>_personal_users.txt`` - unique Reddit authors (if an ``author`` column is present)

Usage
-----
```bash
pip install pandas
python scripts/02_filter_first_person_trd_posts.py --input data/private/TRD_all_matches.csv
```
If ``--input`` is omitted, the default is ``TRD_all_matches.csv`` in the current
working directory.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Lexicon & regexes
# ---------------------------------------------------------------------------
LEXICON_PATTERNS: List[str] = [
    r"treatment[-\s]?resistant depression",
    r"treatment[-\s]?resistant mdd",
    r"\bTRD\b",
    r"refractory depression",
    r"SSRI[-\s]?resistant depression",
    r"incurable depression",
    ##r"failed meds?",
   ##r"failed two meds",
    ##r"medication roulette",
    ##r"nothing[']?s working",
    #r"maximum dose",
   #r"tried everything",
]
LEXICON_REGEX = re.compile(r"(?i)(?:" + "|".join(LEXICON_PATTERNS) + r")")

FIRST_PERSON_REGEX = re.compile(r"\b(?:i(?:'m|'ve|'d|'ll)?|me|my|mine)\b", re.IGNORECASE)
SENTENCE_SPLIT_REGEX = re.compile(r"[.!?\n]+")
TARGET_COLUMNS = ["text", "title", "selftext"]

# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------

def sentence_has_both(sentence: str) -> bool:
    """Return True if *sentence* contains both patterns."""
    return bool(FIRST_PERSON_REGEX.search(sentence) and LEXICON_REGEX.search(sentence))


def row_is_personal(row: pd.Series) -> bool:
    """True if any target column in *row* meets the personal-experience rule."""
    for col in TARGET_COLUMNS:
        if col in row and pd.notna(row[col]):
            text = str(row[col])
            # quick reject - avoids splitting every row unnecessarily
            if not (FIRST_PERSON_REGEX.search(text) and LEXICON_REGEX.search(text)):
                continue
            for sent in SENTENCE_SPLIT_REGEX.split(text):
                if sentence_has_both(sent):
                    return True
    return False

# ---------------------------------------------------------------------------
# 3. Main
# ---------------------------------------------------------------------------

def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Filter TRD matches for personal experience posts.")
    parser.add_argument("--input", default="TRD_all_matches.csv", help="CSV file with TRD matches (default: TRD_all_matches.csv)")
    args = parser.parse_args(argv)

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"ERROR: {input_path} does not exist", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {input_path.name} ...")
    df = pd.read_csv(input_path)

    # Ensure columns exist
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    mask = df.apply(row_is_personal, axis=1)
    personal_df = df[mask].copy()
    print(f"{mask.sum()} personal TRD posts kept out of {len(df)} rows")

    # Outputs
    out_csv = input_path.with_name(input_path.stem + "_personal.csv")
    personal_df.to_csv(out_csv, index=False)
    print(f"{out_csv.name} written")

    if "author" in personal_df.columns:
        users = sorted(personal_df["author"].dropna().unique())
        users_path = input_path.with_name(input_path.stem + "_personal_users.txt")
        users_path.write_text("\n".join(users))
        print(f"{users_path.name} written ({len(users)} unique authors)")


if __name__ == "__main__":
    main()
