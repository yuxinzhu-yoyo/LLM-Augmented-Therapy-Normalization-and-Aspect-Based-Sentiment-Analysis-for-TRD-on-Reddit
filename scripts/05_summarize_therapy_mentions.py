#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate medication_distribution_with_variants.csv

Rules:
- Strict lexicon counting only.
- Text to search = 'text' if present, else "title + \n + selftext".
- The canonical name (lexicon 'medication') is included as a match token.
- from_misspellings = total mentions from tokens != canonical string.
- exact_mentions = total_mentions - from_misspellings  (i.e., canonical-token hits).
- top_aliases includes every variant (including canonical) as "name:count".
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ----------------------------
# Data loading / preprocessing
# ----------------------------

def load_posts(posts_csv: str,
               id_col: str = "post_id",
               title_col: str = "title",
               body_col: str = "selftext",
               text_col: str = "text") -> pd.DataFrame:
    df = pd.read_csv(posts_csv)
    def build_content(row):
        t = row.get(text_col, None)
        if isinstance(t, str) and t.strip():
            return t
        title = row.get(title_col, "") or ""
        body  = row.get(body_col, "") or ""
        return (str(title) + "\n" + str(body)).strip()
    df["content"] = df.apply(build_content, axis=1)
    return df


def load_lexicon(lexicon_xlsx: str,
                 med_col: str = "medication",
                 class_col: str = "class",
                 variants_col: str = "Misspellings_and_Variants"
                 ) -> Tuple[pd.DataFrame, Dict[str, List[str]], Dict[str, str]]:
    xl  = pd.ExcelFile(lexicon_xlsx)
    lex = xl.parse(xl.sheet_names[0]).copy()
    lex.columns = [str(c).strip() for c in lex.columns]
    if med_col not in lex.columns or variants_col not in lex.columns:
        raise ValueError("Lexicon must have columns 'medication' and 'Misspellings_and_Variants'.")

    # Normalize
    lex[med_col] = lex[med_col].astype(str).str.strip().str.lower()
    if class_col in lex.columns:
        lex[class_col] = lex[class_col].astype(str).fillna("").str.strip()
    else:
        lex[class_col] = ""

    # Canonical -> variants (INCLUDE canonical itself; preserve order; dedupe)
    c2v: Dict[str, List[str]] = {}
    for _, r in lex.iterrows():
        canon = r[med_col]
        raw   = r.get(variants_col, "")
        var_list: List[str] = []
        if isinstance(raw, str):
            var_list = [v.strip().lower() for v in raw.split(",") if str(v).strip()]
        if canon not in var_list:
            var_list.append(canon)
        seen = set(); ordered = []
        for v in var_list:
            if v and v not in seen:
                seen.add(v); ordered.append(v)
        c2v[canon] = ordered

    # Canonical -> class (handy to keep)
    c2class: Dict[str, str] = {str(r[med_col]): str(r[class_col]) for _, r in lex.iterrows()}

    # Return a simple (medication,class) table to preserve lexicon order in output
    return lex[[med_col, class_col]].copy(), c2v, c2class


# ----------------------------
# Counting
# ----------------------------

def build_union_regex(variants: List[str]):
    """
    Case-insensitive union regex with strict word boundaries.
    IMPORTANT: single-character tokens ARE allowed (per requirement).
    """
    if not variants:
        return None
    toks = [re.escape(v) for v in sorted(variants, key=len, reverse=True) if v]
    return re.compile(r'(?i)(?<!\w)(' + "|".join(toks) + r')(?!\w)')


def count_mentions(df_posts: pd.DataFrame,
                   c2v: Dict[str, List[str]]
                   ) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    # Build variant -> canonical and union regex
    v2c = {v: c for c, vs in c2v.items() for v in vs}
    rx  = build_union_regex(list(v2c.keys()))

    totals: Dict[str, int] = {c: 0 for c in c2v}
    variant_counts: Dict[str, Dict[str, int]] = {c: {v: 0 for v in c2v[c]} for c in c2v}

    if rx is None:
        return totals, variant_counts

    for content in df_posts["content"].astype(str):
        for m in rx.finditer(content):
            var   = m.group(0).lower()
            canon = v2c.get(var)
            if canon in totals:
                totals[canon] += 1
                # ensure we count this exact surface token
                if var not in variant_counts[canon]:
                    variant_counts[canon][var] = 0
                variant_counts[canon][var] += 1

    return totals, variant_counts


# ----------------------------
# Output assembly
# ----------------------------

def build_distribution(lex_simple: pd.DataFrame,
                       c2v: Dict[str, List[str]],
                       c2class: Dict[str, str],
                       totals: Dict[str, int],
                       variant_counts: Dict[str, Dict[str, int]],
                       out_csv: str) -> pd.DataFrame:
    """
    exact_mentions = total_mentions - from_misspellings  (canonical hits)
    from_misspellings = total_mentions - canonical_hits
    top_aliases lists every variant (including canonical) as "name:count"
    """
    rows = []
    for _, r in lex_simple.iterrows():
        canon = str(r["medication"])
        cls   = str(r["class"])
        total = int(totals.get(canon, 0))

        # canonical hits are the count for the canonical token
        canonical_hits = int(variant_counts.get(canon, {}).get(canon, 0))
        from_misspellings = max(0, total - canonical_hits)
        exact_mentions = total - from_misspellings  # equals canonical_hits

        # exhaustive name:count for every variant (including canonical), preserving lexicon order
        names_counts = [f"{name}:{variant_counts.get(canon, {}).get(name, 0)}"
                        for name in c2v.get(canon, [])]

        rows.append({
            "class": cls,
            "medication": canon,
            "total_mentions": total,
            "exact_mentions": exact_mentions,
            "from_misspellings": from_misspellings,
            "top_aliases": ", ".join(names_counts),
            "canonical": canon
        })

    out_df = pd.DataFrame(
        rows,
        columns=["class", "medication", "total_mentions", "exact_mentions",
                 "from_misspellings", "top_aliases", "canonical"]
    )
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    return out_df


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--posts_csv", required=True, help="TRD_all_matches_personal.csv")
    ap.add_argument("--lexicon_xlsx", required=True, help="TRDmeds_with_variants_and_misspell.xlsx")
    ap.add_argument("--out_csv", default="medication_distribution_with_variants_v6.csv")
    ap.add_argument("--id_col", default="post_id")
    ap.add_argument("--title_col", default="title")
    ap.add_argument("--body_col", default="selftext")
    ap.add_argument("--text_col", default="text")
    args = ap.parse_args()

    posts = load_posts(args.posts_csv,
                       id_col=args.id_col,
                       title_col=args.title_col,
                       body_col=args.body_col,
                       text_col=args.text_col)

    lex_simple, c2v, c2class = load_lexicon(args.lexicon_xlsx)
    totals, variant_counts    = count_mentions(posts, c2v)
    build_distribution(lex_simple, c2v, c2class, totals, variant_counts, args.out_csv)


if __name__ == "__main__":
    main()

 
