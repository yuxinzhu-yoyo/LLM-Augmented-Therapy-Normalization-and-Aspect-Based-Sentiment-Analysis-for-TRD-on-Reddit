import os, sys, re, json, requests, traceback
import pandas as pd

# ------------ CONFIG ------------
url = os.getenv("LLM_URL", "https://sesame.bmi.emory.edu/SarkerlabLLM/llama3_70B/v1/completions")
model_name = "meta-llama/Meta-Llama-3-70B-Instruct"
max_tokens = 320
temperature = 0.7
top_p = 0.95

THERAPY_COL = "therapy"
LABEL_COL   = "label"
TEXT_COL    = "text"
ID_COL_OPT  = "tweet_id"     # if present, synthetic rows will have ""
ALLOWED = ["negative", "positive"]
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
# --------------------------------

def request_llm(prompt, seed_idx, attempt=1):
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    r = requests.post(url, headers=headers, json=data, timeout=180)
    try:
        r.raise_for_status()
        j = r.json()
    except Exception:
        open(os.path.join(LOG_DIR, f"seed_{seed_idx:05d}_attempt{attempt}_http.txt"), "w", encoding="utf-8").write(r.text)
        raise
    # support both completions-style and chat-style
    choice = (j.get("choices") or [{}])[0]
    text = choice.get("text") or (choice.get("message") or {}).get("content") or ""
    # log raw JSON for this seed
    open(os.path.join(LOG_DIR, f"seed_{seed_idx:05d}_attempt{attempt}_resp.json"), "w", encoding="utf-8").write(json.dumps(j, indent=2))
    return text

TWEET_RE = re.compile(r'^\s*tweet\s*([1-5])\s*:\s*(?:\[(.*)\]|(.*))\s*$', re.IGNORECASE)

def parse_five(answer_text):
    tweets = {}
    for line in answer_text.splitlines():
        m = TWEET_RE.match(line)
        if m:
            idx = int(m.group(1))
            body = m.group(2) if m.group(2) is not None else m.group(3)
            if body:
                t = body.strip().strip('\'"[]')
                if t:
                    tweets[idx] = t
    return [tweets[k] for k in range(1, 6) if k in tweets]

def backup_variants(seed_text, therapy, label, need):
    seed_text = (seed_text or "").strip() or f"{therapy}: sharing my experience."
    def ensure_therapy(t):
        return t if therapy.lower() in t.lower() else f"{therapy}: {t}"
    pos = ["really helping a lot.", "glad it works for me.", "noticeable relief lately.", "keeping symptoms in check.", "feels like a good fit."]
    neg = ["not working for me.", "annoying side effects.", "no real improvement.", "symptoms still rough.", "wish I had tried something else."]
    tails = pos if str(label).lower()=="positive" else neg
    out=[]
    for i in range(need):
        base = ensure_therapy(seed_text)
        out.append(f"{base.rstrip('.! ')} - {tails[i % len(tails)]}")
    return out

if __name__ == "__main__":
    # usage: python scripts/06_augment_smm4h_training_data.py train.csv train.csv prompt_absa_SMOTE.txt augmented_train.csv 2
    train_file = sys.argv[1]   # few-shot pool
    seed_file  = sys.argv[2]   # rows to augment (can be the same)
    prompt_file = sys.argv[3]
    outfile = sys.argv[4]
    n_shot = int(sys.argv[5])

    train_df = pd.read_csv(train_file)
    seeds_df = pd.read_csv(seed_file)
    prompt_tpl = open(prompt_file, encoding="utf-8").read()

    # only pos/neg
    train_df = train_df[train_df[LABEL_COL].isin(ALLOWED)].copy()
    seeds_df = seeds_df[seeds_df[LABEL_COL].isin(ALLOWED)].copy()

    for col in (THERAPY_COL, LABEL_COL, TEXT_COL):
        if col not in train_df.columns or col not in seeds_df.columns:
            raise ValueError(f"Both files must contain '{col}'")

    out_cols = list(train_df.columns)
    has_id = ID_COL_OPT in out_cols
    synthetic_rows = []

    for i, row in seeds_df.iterrows():
        therapy_val = str(row[THERAPY_COL]); label_val = str(row[LABEL_COL]).lower(); seed_text = str(row[TEXT_COL])
        print(f"\n=== Seed {i+1}/{len(seeds_df)} | therapy={therapy_val} | label={label_val} ===")

        # build prompt
        case_prompt = prompt_tpl
        for col in seeds_df.columns:
            case_prompt = case_prompt.replace('{{'+col+'}}', "" if pd.isna(row[col]) else str(row[col]))

        # few-shot: prefer same therapy+label; fallback label-only
        fewshots = []
        for class_name in ["negative","positive"]:
            pool = train_df[(train_df[LABEL_COL]==class_name) & (train_df[THERAPY_COL]==therapy_val)]
            if len(pool)==0:
                pool = train_df[train_df[LABEL_COL]==class_name]
            k = n_shot if len(pool)>=n_shot else len(pool)
            ex_df = pool.sample(n=k, replace=(len(pool)<n_shot), random_state=len(fewshots)+19)
            for text, label in zip(ex_df[TEXT_COL], ex_df[LABEL_COL]):
                fewshots.append(f'Example {len(fewshots)+1}: "{text}" should be classified as {label}')
        case_prompt = case_prompt.replace('{{example}}', "\n".join(fewshots))

        if "{{" in case_prompt:
            print("Template error: unreplaced placeholders remain; skipping.")
            continue

        # call model and log raw
        try:
            raw = request_llm(case_prompt, seed_idx=i, attempt=1)
        except Exception:
            print(traceback.format_exc())
            continue

        print("MODEL RAW:\n" + raw)
        tweets = parse_five(raw)

        # if fewer than 5, top up deterministically (so you always get rows)
        if len(tweets) < 5:
            tweets += backup_variants(seed_text, therapy_val, label_val, 5 - len(tweets))

        # append rows
        for tw in tweets[:5]:
            new_row = {c: None for c in out_cols}
            if has_id:
                new_row[ID_COL_OPT] = ""
            new_row[THERAPY_COL] = therapy_val
            new_row[LABEL_COL]   = label_val
            new_row[TEXT_COL]    = tw
            synthetic_rows.append(new_row)

        for j, tw in enumerate(tweets[:5], 1):
            print(f"  tweet {j}: {tw}")

    augmented = pd.concat([train_df, pd.DataFrame(synthetic_rows, columns=out_cols)], ignore_index=True)
    augmented.to_csv(outfile, index=False)
    print(f"\nWrote augmented dataset with {len(augmented) - len(train_df)} synthetic rows to: {outfile}")
