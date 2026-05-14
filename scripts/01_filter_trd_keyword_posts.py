#!/usr/bin/env python3
"""Extract Reddit posts mentioning Treatment-Resistant Depression (TRD)

This script scans every CSV file in the specified directory whose filename ends
with "_SR_data.csv". It applies a lexicon-based regular-expression search across
`text`, `title`, and `selftext` columns to identify posts that reference
Treatment-Resistant Depression (TRD) or related concepts.

Outputs
-------
For each source file <basename>_SR_data.csv, a filtered file named
<basename>_TRD_lexicon_matches.csv is written to the same directory containing
only the matching rows (all original columns preserved).

A combined file ``TRD_all_matches.csv`` and a plain-text list of unique authors
``TRD_unique_users.txt`` are also created in the directory.

Usage
-----
```
python scripts/01_filter_trd_keyword_posts.py --data_dir data/private/reddit_exports
```
If no ``--data_dir`` argument is given, the current working directory is used.

Dependencies
------------
- Python 3.8+
- pandas (``pip install pandas``)

"""

import argparse
import glob
import os
import re
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# 1. Define the TRD lexicon and compile a single case-insensitive regex
# ---------------------------------------------------------------------------
LEXICON_PATTERNS = [
    r"treatment resistant depression",
    r"treatment[-\s]?resistant depression",
    r"treatment[-\s]?resistant mdd",
    r"\bTRD\b",
    r"refractory depression",
    r"SSRI[-\s]?resistant depression",
    r"incurable depression",
    r"failed meds",
    r"failed two meds",
    r"medication roulette",
    r"nothing[']?s working",
    r"maximum dose",
    r"tried everything",
    r"difficult[-\s]?to[-\s]?treat mdd",
    r"difficult[-\s]?to[-\s]?treat depression",
    r"difficult to treat depression",
    r"\bDTD\b",
    r"tried every"
]
# Combine into one pattern with non-capturing group; (?i) for case-insensitive
LEXICON_REGEX = re.compile(r"(?i)(?:" + "|".join(LEXICON_PATTERNS) + r")")

TARGET_COLUMNS = ["text", "title", "selftext"]  # in order of preference


# ---------------------------------------------------------------------------
# 2. Helper: determine if any target column contains a lexicon match
# ---------------------------------------------------------------------------

def row_matches(row) -> bool:
    """Return True if any target column in *row* matches the lexicon regex."""
    for col in TARGET_COLUMNS:
        if col in row and pd.notna(row[col]):
            if bool(LEXICON_REGEX.search(str(row[col]))):
                return True
    return False


# ---------------------------------------------------------------------------
# 3. Main routine
# ---------------------------------------------------------------------------

def process_directory(directory: Path) -> None:
    """Process every *_SR_data.csv file inside *directory*."""
    csv_paths = sorted(directory.glob("*_SR_data.csv"))
    if not csv_paths:
        print(f"No *_SR_data.csv files found in {directory}")
        return

    all_matches = []  # collect DataFrames for combined output

    for csv_path in csv_paths:
        print(f"Processing {csv_path.name} ...", flush=True)
        try:
            df = pd.read_csv(csv_path)
        except Exception as exc:
            print(f"  Failed to read: {exc}. Skipping.")
            continue

        # Ensure target columns exist; fill missing with empty strings
        for col in TARGET_COLUMNS:
            if col not in df.columns:
                df[col] = ""

        mask = df.apply(row_matches, axis=1)
        subset = df[mask].copy()
        print(f"  {mask.sum()} matches found.")

        if not subset.empty:
            out_path = csv_path.with_name(csv_path.stem.replace("_SR_data", "_TRD_lexicon_matches") + ".csv")
            subset.to_csv(out_path, index=False)
            print(f"    Written {out_path.name}")
            all_matches.append(subset)
        else:
            print("    No matches; output skipped.")

    # Write combined outputs -----------------------------------------------
    if all_matches:
        combined = pd.concat(all_matches, ignore_index=True)
        combined_path = directory / "TRD_all_matches.csv"
        combined.to_csv(combined_path, index=False)
        print(f"Combined file written: {combined_path.name}")

        # Unique authors -----------------------------------------------------
        author_col = "author" if "author" in combined.columns else None
        if author_col:
            unique_authors = combined[author_col].dropna().unique()
            authors_path = directory / "TRD_unique_users.txt"
            authors_path.write_text("\n".join(sorted(unique_authors)))
            print(f"Unique-author list written: {authors_path.name} ({len(unique_authors)} users)")
        else:
            print("No 'author' column detected in combined dataframe; unique-user list skipped.")


# ---------------------------------------------------------------------------
# 4. Entry point
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(description="Filter Reddit CSVs for TRD mentions.")
    parser.add_argument("--data_dir", default=".", help="Directory containing *_SR_data.csv files (default: current directory)")
    args = parser.parse_args(argv)

    target_dir = Path(args.data_dir).expanduser().resolve()
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"ERROR: {target_dir} is not a valid directory.", file=sys.stderr)
        sys.exit(1)

    process_directory(target_dir)


if __name__ == "__main__":
    main()
