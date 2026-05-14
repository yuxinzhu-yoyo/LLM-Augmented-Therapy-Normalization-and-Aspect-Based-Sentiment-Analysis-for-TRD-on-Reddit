#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QMisSpell-adapted full pipeline for TRD Reddit posts
- Loads TRD_all_matches_personal.csv
- Implements QMisSpell core (weighted & standard Levenshtein) with Word2Vec neighbors
- Generates misspellings for each seed alias (brand/generic)
- Counts exact alias (single-token & multiword) + misspellings
- Folds everything to canonical medication buckets
- Outputs clean distributions and mapping tables

Author: Yuxin Zhu
"""

import os
import sys
import argparse
import re
from collections import defaultdict, Counter
import pandas as pd
from gensim.models import KeyedVectors


# -----------------------------
# Optional Levenshtein dependency (preferred)
# -----------------------------
try:
    import Levenshtein  # python-Levenshtein
    def lev_ratio(a, b):
        return Levenshtein.ratio(a, b)
    HAVE_LEV = True
except Exception:
    import difflib
    def lev_ratio(a, b):
        return difflib.SequenceMatcher(None, a, b).ratio()
    HAVE_LEV = False



# -----------------------------
# QMisSpell weights (from source)
# -----------------------------
WEIGHTS = {
    0.5: 1.0133116199788939,
    0.7: 0.97859255044477178,
    0.9: 0.95,
    0.3: 1.05,
    0.1: 0.99601150272112082,
}

# -----------------------------
# Medication dictionary (class -> canonical -> aliases)
# Medication dictionary with common class-level extensions.
# -----------------------------
MED_CLASSES = {
    "SSRIs": {
        "fluoxetine": ["fluoxetine", "prozac"],
        "paroxetine": ["paroxetine", "paxil"],
        "sertraline": ["sertraline", "zoloft"],
        "escitalopram": ["escitalopram", "lexapro"],
        "citalopram": ["citalopram", "celexa"],
        "fluvoxamine": ["fluvoxamine", "luvox"],
    },
    "SNRIs": {
        "venlafaxine": ["venlafaxine", "effexor", "effexor xr", "venlafaxine xr"],
        "desvenlafaxine": ["desvenlafaxine", "pristiq"],
        "duloxetine": ["duloxetine", "cymbalta"],
        "levomilnacipran": ["levomilnacipran", "fetzima"],
        "milnacipran": ["milnacipran", "savella"],
    },
    "NDRI": {
        "bupropion": ["bupropion", "wellbutrin", "wellbutrin xl", "wellbutrin sr", "bupropion xl", "bupropion sr"],
    },
    "NaSSA": {
        "mirtazapine": ["mirtazapine", "remeron"],
    },
    "Atypical antidepressants": {
        "trazodone": ["trazodone", "desyrel", "olpetro"],
        "nefazodone": ["nefazodone", "serzone"],
        "vilazodone": ["vilazodone", "viibryd"],
        "vortioxetine": ["vortioxetine", "trintellix", "brintellix"],
    },
    "TCAs": {
        "amitriptyline": ["amitriptyline", "elavil"],
        "nortriptyline": ["nortriptyline", "pamelor", "aventyl"],
        "imipramine": ["imipramine", "tofranil"],
        "desipramine": ["desipramine", "norpramin"],
        "clomipramine": ["clomipramine", "anafranil"],
        "doxepin": ["doxepin", "silenor", "sinequan"],
        "trimipramine": ["trimipramine", "surmontil"],
        "protriptyline": ["protriptyline", "vivactil"],
        "amoxapine": ["amoxapine"],
    },
    "MAOIs": {
        "tranylcypromine": ["tranylcypromine", "parnate"],
        "phenelzine": ["phenelzine", "nardil"],
        "isocarboxazid": ["isocarboxazid", "marplan"],
        "selegiline": ["selegiline", "emsam"],
    },
    "NMDA / Rapid-acting": {
        "ketamine": ["ketamine"],
        "esketamine": ["esketamine", "spravato"],
        "dextromethorphan-bupropion": ["dextromethorphan-bupropion", "dextromethorphan bupropion", "auvelity"],
    },
    "Augmentation (antipsychotics)": {
        "aripiprazole": ["aripiprazole", "abilify"],
        "quetiapine": ["quetiapine", "seroquel"],
        "olanzapine": ["olanzapine", "zyprexa"],
        "risperidone": ["risperidone", "risperdal"],
        "brexpiprazole": ["brexpiprazole", "rexulti"],
        "cariprazine": ["cariprazine", "vraylar"],
        "ziprasidone": ["ziprasidone", "geodon"],
        "lurasidone": ["lurasidone", "latuda"],
        "clozapine": ["clozapine", "clozaril"],
        "asenapine": ["asenapine", "saphris"],
        "iloperidone": ["iloperidone", "fanapt"],
        "paliperidone": ["paliperidone", "invega"],
    },
    "Augmentation (mood stabilizers)": {
        "lithium": ["lithium", "lithobid"],
        "lamotrigine": ["lamotrigine", "lamictal"],
        "valproate": ["valproate", "divalproex", "depakote"],
        "carbamazepine": ["carbamazepine", "tegretol"],
        "oxcarbazepine": ["oxcarbazepine", "trileptal"],
        "topiramate": ["topiramate", "topamax"],
        "gabapentin": ["gabapentin", "neurontin"],
        "pregabalin": ["pregabalin", "lyrica"],
    },
    "Stimulants / ADHD adjuncts": {
        "methylphenidate": ["methylphenidate", "ritalin", "concerta", "metadate", "daytrana", "aptensio"],
        "amphetamine-dextroamphetamine": [
            "adderall", "mixed amphetamine salts",
            "amphetamine dextroamphetamine", "amphetamine-dextroamphetamine",
            "amphetamine"
        ],
        "lisdexamfetamine": ["lisdexamfetamine", "vyvanse"],
        "dextroamphetamine": ["dextroamphetamine", "dexedrine", "zenzedi", "procentra"],
        "atomoxetine": ["atomoxetine", "strattera"],
        "guanfacine": ["guanfacine", "intuniv"],
        "clonidine": ["clonidine", "kapvay", "catapres"],
        "modafinil": ["modafinil", "provigil"],
        "armodafinil": ["armodafinil", "nuvigil"],
    },
    "Anxiolytics / Sedatives": {
        "alprazolam": ["alprazolam", "xanax"],
        "clonazepam": ["clonazepam", "klonopin"],
        "lorazepam": ["lorazepam", "ativan"],
        "diazepam": ["diazepam", "valium"],
        "buspirone": ["buspirone", "buspar"],
        "zolpidem": ["zolpidem", "ambien"],
        "eszopiclone": ["eszopiclone", "lunesta"],
        "zaleplon": ["zaleplon", "sonata"],
        "hydroxyzine": ["hydroxyzine", "atarax", "vistaril"],
    },
    "Other / Misc": {
        "thyroid": ["liothyronine", "cytomel", "levothyroxine", "synthroid", "thyroxine", "t3", "t4"],
        "psilocybin": ["psilocybin", "shrooms", "magic mushrooms"],
        "omega-3": ["omega 3", "omega-3", "fish oil"],
        "sam-e": ["sam-e", "s-adenosylmethionine"],
    }
}

# ------------------------------------------
# Helpers
# ------------------------------------------
def load_keyed_vectors(model_path: str):
    """Load a local gensim KeyedVectors model or word2vec-format vectors."""
    try:
        return KeyedVectors.load(model_path, mmap="r")
    except Exception:
        return KeyedVectors.load_word2vec_format(model_path, binary=True)



def combine_text_columns(df, cols=("title", "selftext", "text")):
    def _combine(row):
        parts = []
        for c in cols:
            if c in df.columns:
                v = row[c]
                if isinstance(v, str):
                    parts.append(v)
        return " ".join(parts)
    return df.apply(_combine, axis=1).fillna("")

def pad(term, required_length):
    out = []
    for i in range(required_length):
        out.append(term[i] if i < len(term) else '?')
    return "".join(out)

def weighted_levenshtein_ratio(variant, seedword):
    # QMisSpell-style sliding window with precomputed WEIGHTS
    v, s = variant, seedword
    if len(v) < len(s):
        v = pad(v, len(s))
    elif len(s) < len(v):
        s = pad(s, len(v))

    window_beg = 0
    window_end = int((len(s) - 1) / 2)
    vals = []
    while window_end <= len(s) and window_end <= len(v):
        relpos = ((window_beg + window_end + 0.0) / 2.0) / len(s)
        if relpos < 0.2:
            relpos = 0.1
        elif relpos < 0.4:
            relpos = 0.3
        elif relpos < 0.6:
            relpos = 0.5
        elif relpos < 0.8:
            relpos = 0.7
        elif relpos < 1.0:
            relpos = 0.9
        r = lev_ratio(s[window_beg:window_end], v[window_beg:window_end])
        vals.append(WEIGHTS[relpos] * r)
        window_beg += 1
        window_end += 1
    if not vals:
        return 0.0
    return sum(vals) / len(vals)

def multi_alias_to_regex(alias):
    # match tokens with spaces/hyphens flexibly
    parts = re.split(r"[\s\-]+", alias.lower())
    parts = [re.escape(p) for p in parts if p]
    return r"\b" + r"[\s\-]+".join(parts) + r"\b"

# ------------------------------------------
# QMisSpell core: BFS over semantic neighbors
# ------------------------------------------
def generate_spelling_variants(seedwordlist, kv: KeyedVectors,
                               semantic_search_length=4000,
                               levenshtein_threshold=0.70,
                               use_weighted=True,
                               restrict_to_vocab=None):
    """
    Faithful to QMisSpell idea:
      - For each seed, expand over most_similar() neighbors (topn=semantic_search_length)
      - Keep those with Levenshtein ratio above threshold (weighted or standard)
      - BFS-style iterative expansion
      - Skip terms containing underscores

    Args:
        seedwordlist: list[str] seed aliases (single-token)
        kv: gensim KeyedVectors
        semantic_search_length: topn for most_similar()
        levenshtein_threshold: threshold
        use_weighted: True -> weighted_levenshtein_ratio; False -> standard ratio
        restrict_to_vocab: optional set[str]; if provided, keep only variants in this set
    Returns:
        dict[str, set[str]]: seed -> set(variants)
    """
    print(f"[INFO] QMisSpell params: topn={semantic_search_length}, threshold={levenshtein_threshold}, "
          f"weighted={use_weighted}, restrict_to_vocab={'yes' if restrict_to_vocab is not None else 'no'}")

    def _score(a, b):
        if use_weighted:
            return max(lev_ratio(a, b), weighted_levenshtein_ratio(a, b))
        else:
            return lev_ratio(a, b)

    vars_map = defaultdict(set)

    for seed in seedwordlist:
        terms_to_expand = [seed]
        all_expanded = set()

        while terms_to_expand:
            t = terms_to_expand.pop(0)
            all_expanded.add(t)
            try:
                similars = kv.most_similar(t, topn=semantic_search_length)
            except Exception as e:
                # For OOV, skip
                continue

            for similar_term, _sim in similars:
                st = str(similar_term)
                if "_" in st:
                    continue
                score = _score(st, seed)
                if score >= levenshtein_threshold:
                    if restrict_to_vocab is not None and st not in restrict_to_vocab:
                        # We only care about variants that actually appear in our corpus
                        continue
                    vars_map[seed].add(st)
                    if st not in all_expanded and st not in terms_to_expand:
                        terms_to_expand.append(st)

    # convert to unique sorted list (optional)
    return {k: set(v) for k, v in vars_map.items()}

# ------------------------------------------
# Main pipeline
# ------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="TRD Reddit medication mining with QMisSpell.")
    ap.add_argument("--csv", required=True, help="Path to TRD_all_matches_personal.csv")
    ap.add_argument("--w2v", required=True, help="Path to word2vec/KeyedVectors model (bin or txt)")
    ap.add_argument("--w2v-binary", action="store_true", help="If provided, loads model as binary")
    ap.add_argument("--lev-threshold", type=float, default=0.70, help="Levenshtein ratio threshold (default 0.70)")
    ap.add_argument("--semantic-topn", type=int, default=4000, help="Topn for most_similar (default 4000)")
    ap.add_argument("--weighted", action="store_true", help="Use weighted Levenshtein (QMisSpell); if not set, use standard")
    ap.add_argument("--outdir", default="./out", help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) Load CSV & build corpus text
    print(f"[INFO] Loading CSV: {args.csv}")
    df = pd.read_csv(args.csv)
    texts = combine_text_columns(df)
    full_text = "\n".join(t.lower() for t in texts)

    # 2) Build mappings
    alias_to_canon = {}
    canon_to_class = {}
    for cls, meds in MED_CLASSES.items():
        for canon, aliases in meds.items():
            canon_to_class[canon] = cls
            for a in aliases:
                alias_to_canon[a.lower()] = canon

    # 3) Tokenize (single-token) & count
    token_re = re.compile(r"[a-z][a-z\-']{2,}")
    token_counts = Counter()
    for t in texts:
        token_counts.update(token_re.findall(t.lower()))
    vocab = set(token_counts.keys())
    print(f"[INFO] Corpus vocab size (single-token): {len(vocab):,}")

    # 4) Seed words for QMisSpell: single-token aliases only
    seedwords = sorted([a for a in alias_to_canon if " " not in a])

    # 5) Load Word2Vec/KeyedVectors
    kv = load_keyed_vectors(args.w2v)

    # 6) QMisSpell: generate misspellings present in corpus
    seed_to_misspellings = generate_spelling_variants(
        seedwordlist=seedwords,
        kv=kv,
        semantic_search_length=args.semantic_topn,
        levenshtein_threshold=args.lev_threshold,
        use_weighted=args.weighted or True,  # default to weighted (faithful to QMisSpell)
        restrict_to_vocab=vocab,
    )

    # 7) Resolve misspelling -> best seed (ties by higher score)
    def best_seed_for_miss(mis):
        # Only consider seeds that produced this miss
        candidates = [s for s, sset in seed_to_misspellings.items() if mis in sset]
        best_s, best_score = None, -1.0
        for s in candidates:
            score = max(lev_ratio(mis, s), weighted_levenshtein_ratio(mis, s))
            if score > best_score:
                best_score = score
                best_s = s
        return best_s, best_score

    misspell_to_best = {}
    for seed, mset in seed_to_misspellings.items():
        for mis in mset:
            if mis in misspell_to_best:
                # compare and keep the better assignment
                curr_seed, curr_score = misspell_to_best[mis]
                new_score = max(lev_ratio(mis, seed), weighted_levenshtein_ratio(mis, seed))
                if new_score > curr_score:
                    misspell_to_best[mis] = (seed, new_score)
            else:
                seed_, score_ = best_seed_for_miss(mis)
                misspell_to_best[mis] = (seed_, score_)

    # 8) Count exact alias mentions
    canon_counts = Counter()
    alias_counts = Counter()
    misspell_counts_by_canon = defaultdict(Counter)

    # single-token aliases
    for alias, canon in alias_to_canon.items():
        if " " in alias:
            continue
        cnt = token_counts.get(alias, 0)
        if cnt:
            canon_counts[canon] += cnt
            alias_counts[alias] += cnt

    # multiword/hyphenated aliases via regex
    for alias, canon in alias_to_canon.items():
        if " " in alias or "-" in alias:
            pat = re.compile(multi_alias_to_regex(alias))
            matches = len(pat.findall(full_text))
            if matches:
                canon_counts[canon] += matches
                alias_counts[alias] += matches

    # 9) Add misspellings (observed in corpus)
    for mis, (seed_alias, score) in misspell_to_best.items():
        cnt = token_counts.get(mis, 0)
        if cnt == 0:
            continue
        canon = alias_to_canon.get(seed_alias, None)
        if canon is None:
            # seed alias should always map to a canonical
            continue
        canon_counts[canon] += cnt
        misspell_counts_by_canon[canon][mis] += cnt

    # 10) Build outputs
    # Distribution table
    rows = []
    for canon, total in canon_counts.most_common():
        cls = canon_to_class.get(canon, "Unknown")
        # alias breakdown
        aliases = sorted([a for a, c in alias_to_canon.items() if c == canon])
        alias_breakdown = [(a, alias_counts.get(a, 0)) for a in aliases if alias_counts.get(a, 0) > 0]
        alias_breakdown = sorted(alias_breakdown, key=lambda x: -x[1])
        miss_breakdown = sorted(misspell_counts_by_canon.get(canon, Counter()).items(), key=lambda x: -x[1])

        rows.append({
            "class": cls,
            "medication": canon,
            "total_mentions": int(total),
            "exact_mentions": int(sum(n for _, n in alias_breakdown)),
            "from_misspellings": int(sum(n for _, n in miss_breakdown)),
            "top_aliases": ", ".join([f"{a}:{n}" for a, n in alias_breakdown[:8]]),
            "top_misspellings": ", ".join([f"{m}:{n}" for m, n in miss_breakdown[:8]]),
        })

    dist_df = pd.DataFrame(rows).sort_values(["total_mentions", "medication"], ascending=[False, True]).reset_index(drop=True)

    # Misspelling mapping table
    map_records = []
    for mis, (seed_alias, score) in sorted(misspell_to_best.items(), key=lambda x: (-token_counts.get(x[0], 0), x[0])):
        canon = alias_to_canon.get(seed_alias, "")
        map_records.append({
            "misspelling": mis,
            "seed_alias": seed_alias,
            "mapped_canonical": canon,
            "similarity_score": round(float(score), 4) if score is not None else None,
            "count": int(token_counts.get(mis, 0)),
        })
    miss_map_df = pd.DataFrame(map_records)

    # Classes & aliases table
    class_rows = []
    for cls, meds in MED_CLASSES.items():
        for canon, aliases in meds.items():
            class_rows.append({"class": cls, "medication": canon, "aliases": ", ".join(sorted(set(a.lower() for a in aliases)))})
    classes_df = pd.DataFrame(class_rows).sort_values(["class", "medication"]).reset_index(drop=True)

    # 11) Save
    dist_path = os.path.join(args.outdir, "medication_distribution_qmisspell.csv")
    miss_map_path = os.path.join(args.outdir, "qmisspell_misspellings_map.csv")
    classes_path = os.path.join(args.outdir, "medication_classes_aliases.csv")

    dist_df.to_csv(dist_path, index=False)
    miss_map_df.to_csv(miss_map_path, index=False)
    classes_df.to_csv(classes_path, index=False)

    print("\n[DONE]")
    print(f"- Distribution: {dist_path}")
    print(f"- Misspelling map: {miss_map_path}")
    print(f"- Classes & aliases: {classes_path}")
    if not HAVE_LEV:
        print("[WARN] python-Levenshtein not found; used difflib as fallback. For best fidelity, run: pip install python-Levenshtein")

if __name__ == "__main__":
    main()
