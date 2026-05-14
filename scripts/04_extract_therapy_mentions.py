#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ABSA input builder (STRICT LEXICON VERSION)
- Matches ONLY tokens provided in the XLSX lexicon (variants + canonical for each drug)
- Word-boundary union-regex; case-insensitive
- No brand/dose fallback; no fuzzy heuristics
- Outputs:
  1) annotated_posts.csv (one row per post with set of meds mentioned)
  2) mentions_long.csv (one row per mention with sentence window)
  3) absa_input.csv (text+target for modeling)
  4) absa_counts.csv (per-canonical totals; includes ALL meds with zero if absent)
  5) med_distribution_from_absa.csv (class, medication, total_mentions, exact_mentions, top_aliases "name:count")
"""
import argparse, re
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd

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
        body = row.get(body_col, "") or ""
        return (str(title) + "\n" + str(body)).strip()
    df["content"] = df.apply(build_content, axis=1)
    if id_col not in df.columns:
        df[id_col] = range(1, len(df)+1)
    return df

def load_lexicon(lexicon_xlsx: str,
                 med_col: str = "medication",
                 class_col: str = "class",
                 variants_col: str = "Misspellings_and_Variants"):
    xl = pd.ExcelFile(lexicon_xlsx)
    df = xl.parse(xl.sheet_names[0])
    df.columns = [str(c).strip() for c in df.columns]
    if med_col not in df.columns or variants_col not in df.columns:
        raise ValueError("Lexicon must have 'medication' and 'Misspellings_and_Variants' columns.")
    v2c, c2v, c2class = {}, {}, {}
    for _, row in df.iterrows():
        canonical = str(row.get(med_col, "")).strip().lower()
        if not canonical:
            continue
        c2class[canonical] = str(row.get(class_col, "")) if class_col in df.columns else ""
        variants = row.get(variants_col, "")
        var_list = []
        if isinstance(variants, str):
            var_list = [v.strip().lower() for v in variants.split(",") if str(v).strip()]
        if canonical not in var_list:
            var_list.append(canonical)
        # de-duplicate, preserve order
        seen, final_vars = set(), []
        for v in var_list:
            if v and v not in seen:
                seen.add(v); final_vars.append(v)
        c2v[canonical] = final_vars
        for v in final_vars:
            v2c[v] = canonical
    return v2c, c2v, c2class

def build_union_regex(variants: List[str]):
    toks = [re.escape(v) for v in sorted(variants, key=len, reverse=True) if v]
    return re.compile(r'(?i)(?<!\w)(' + "|".join(toks) + r')(?!\w)') if toks else None

def split_sentences(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    t = re.sub(r'[ \t\r\f\v]+', ' ', text)
    parts = re.split(r'(?<=[\.\!\?])\s+|\n+', t)
    return [p.strip() for p in parts if p and p.strip()]

def window_sentences(sents: List[str], idx: int, before: int = 1, after: int = 1) -> str:
    lo = max(0, idx - before)
    hi = min(len(sents), idx + after + 1)
    return " ".join(sents[lo:hi])

def find_mentions(text: str, rx_union, v2c: Dict[str,str]):
    out = []
    if not isinstance(text, str) or not text.strip():
        return out
    if rx_union is None:
        return out
    for m in rx_union.finditer(text):
        surface = m.group(0)
        key = surface.lower()
        canonical = v2c.get(key, key)
        out.append({"surface": surface, "canonical": canonical, "start": m.start(), "end": m.end()})
    # dedupe by exact span
    seen, uniq = set(), []
    for d in out:
        k = (d["start"], d["end"], d["surface"].lower())
        if k not in seen:
            seen.add(k); uniq.append(d)
    return uniq

def build_windows_for_mentions(text: str, mentions):
    sents = split_sentences(text)
    # establish offsets
    offsets, pos = [], 0
    for s in sents:
        i = text.find(s, pos)
        if i == -1: i = pos
        offsets.append((i, i + len(s))); pos = i + len(s)
    rows = []
    for mi, m in enumerate(mentions):
        sent_idx = 0
        for j, (a, b) in enumerate(offsets):
            if m["start"] >= a and m["start"] < b:
                sent_idx = j; break
        window = window_sentences(sents, sent_idx, before=1, after=1)
        rows.append({"mention_id": mi+1, "surface": m["surface"], "canonical": m["canonical"],
                     "start": m["start"], "end": m["end"], "sent_index": sent_idx,
                     "window_text": window})
    return rows

def run(posts_csv: str, lexicon_xlsx: str, out_dir: str,
        id_col: str = "post_id", title_col: str = "title", body_col: str = "selftext",
        text_col: str = "text"):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    df_posts = load_posts(posts_csv, id_col=id_col, title_col=title_col, body_col=body_col, text_col=text_col)
    v2c, c2v, c2class = load_lexicon(lexicon_xlsx)
    rx_union = build_union_regex(list(v2c.keys()))
    all_canon = list(c2v.keys())

    per_post_annotations, mentions_rows = [], []
    for _, row in df_posts.iterrows():
        pid = row.get(id_col)
        content = row.get("content", "")
        subreddit = row.get("subreddit", "")
        author = row.get("author", "")
        created = row.get("created_at_date", row.get("created_utc", ""))
        mentions = find_mentions(content, rx_union, v2c)
        meds_all = sorted({m["canonical"] for m in mentions}) if mentions else []
        per_post_annotations.append({id_col: pid, "subreddit": subreddit, "author": author,
                                     "created": created, "meds_all_canonical": meds_all,
                                     "n_mentions": len(mentions)})
        for mrow in build_windows_for_mentions(content, mentions):
            mrow.update({id_col: pid, "subreddit": subreddit, "author": author, "created": created})
            mentions_rows.append(mrow)

    # Write base outputs
    df_posts_anno = pd.DataFrame(per_post_annotations)
    df_mentions = pd.DataFrame(mentions_rows)
    df_posts_anno.to_csv(out_dir / "annotated_posts.csv", index=False)
    df_mentions.to_csv(out_dir / "mentions_long.csv", index=False)

    absa_in = df_mentions.rename(columns={"window_text": "text", "surface": "target"})
    cols = ["text", "target", "canonical", id_col, "mention_id", "subreddit", "author", "created", "sent_index"]
    cols = [c for c in cols if c in absa_in.columns]
    absa_in = absa_in[cols]
    absa_in.to_csv(out_dir / "absa_input.csv", index=False)

    # Aggregated counts (include ALL meds with zero if absent)
    totals = {c: 0 for c in all_canon}
    variant_counts = {c: {v: 0 for v in c2v[c]} for c in all_canon}
    for _, r in df_mentions.iterrows():
        c = str(r["canonical"]).strip().lower()
        if c in totals:
            totals[c] += 1
        v = str(r["target"] if "target" in r else r.get("surface", "")).strip().lower()
        if c in variant_counts:
            if v not in variant_counts[c]:
                variant_counts[c][v] = 0
            variant_counts[c][v] += 1

    # Write absa_counts.csv
    counts_df = pd.DataFrame({"canonical": all_canon, "total": [totals[c] for c in all_canon]})
    counts_df.to_csv(out_dir / "absa_counts.csv", index=False)

    # Write a meds distribution with exhaustive top_aliases "name:count"
    rows = []
    for canon in all_canon:
        cls = c2class.get(canon, "")
        alias_pairs = [f"{name}:{variant_counts[canon].get(name, 0)}" for name in c2v[canon]]
        rows.append({
            "class": cls,
            "medication": canon,
            "total_mentions": int(totals[canon]),
            "exact_mentions": int(sum(variant_counts[canon].values())),
            "from_fallback": 0,
            "top_aliases": ", ".join(alias_pairs),
            "canonical": canon
        })
    dist_df = pd.DataFrame(rows, columns=["class","medication","total_mentions","exact_mentions","from_fallback","top_aliases","canonical"])
    dist_df.to_csv(out_dir / "med_distribution_from_absa.csv", index=False)

    print(f"Wrote: {out_dir / 'annotated_posts.csv'} ({len(df_posts_anno)} rows)")
    print(f"Wrote: {out_dir / 'mentions_long.csv'} ({len(df_mentions)} rows)")
    print(f"Wrote: {out_dir / 'absa_input.csv'} ({len(absa_in)} rows)")
    print(f"Wrote: {out_dir / 'absa_counts.csv'} ({len(counts_df)} rows)")
    print(f"Wrote: {out_dir / 'med_distribution_from_absa.csv'} ({len(dist_df)} rows)")

if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--posts_csv', required=True, type=str)
    ap.add_argument('--lexicon_xlsx', required=True, type=str)
    ap.add_argument('--out_dir', required=True, type=str)
    ap.add_argument('--id_col', type=str, default='post_id')
    ap.add_argument('--title_col', type=str, default='title')
    ap.add_argument('--body_col', type=str, default='selftext')
    ap.add_argument('--text_col', type=str, default='text')
    args = ap.parse_args()
    run(**vars(args))
