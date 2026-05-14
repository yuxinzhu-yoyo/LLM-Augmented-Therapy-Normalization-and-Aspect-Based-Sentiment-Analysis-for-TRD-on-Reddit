# LLM-Augmented Therapy Normalization and Aspect-Based Sentiment Analysis of TRD-Labeled Reddit Discourse

The project characterizes treatment-resistant depression (TRD) therapy discussion in Reddit posts using a curated TRD cohort, lexicon-based therapy normalization, and medication-tagged aspect-based sentiment analysis. 


## Environment


```bash
conda env create -f environment.yml
conda activate trd-absa
python -m pip install -r requirements.txt
```

R and Quarto packages needed for statistical tables and figures are listed in `R_PACKAGES.md`.


## Reproduction Workflow

Run commands from the repository root.

### 1. Build the TRD post cohort

```bash
conda activate trd-absa

python scripts/01_filter_trd_keyword_posts.py \
  --data_dir data/private/reddit_exports

python scripts/02_filter_first_person_trd_posts.py \
  --input data/private/reddit_exports/TRD_all_matches.csv
```

The manuscript analysis uses the cleaned post-level cohort stored locally as `data/private/TRD_new_cohort.csv`.

### 2. Expand and apply the therapy lexicon

```bash
python scripts/03_expand_therapy_lexicon_qmisspell.py \
  --csv data/private/TRD_all_matches_personal.csv \
  --w2v data/private/DSM-language-models-3M/400features_10minwords_5context \
  --outdir data/private/qmisspell_outputs

python scripts/04_extract_therapy_mentions.py \
  --posts_csv data/your_posts.csv \
  --lexicon_xlsx data/reference/your_lexicon.csv \
  --out_dir data/private/out_absa_prep
```

`04_extract_therapy_mentions.py` produces the mention-level ABSA input, including the local 3-sentence context window used for sentiment inference.

### 3. Train and apply the ABSA sentiment model

```bash
python scripts/06_augment_smm4h_training_data.py \
  data/private/train.csv \
  data/private/train.csv \
  data/reference/prompt_absa_SMOTE.txt \
  data/private/train_new.csv \
  2

python scripts/07_train_absa_sentiment_model.py train \
  --train_csv data/private/train_new.csv \
  --val_csv data/private/validation.csv \
  --test_csv data/private/test.csv \
  --out_dir data/private/model_runs \
  --models microsoft/deberta-v3-base \
  --text_col text \
  --target_col therapy \
  --label_col label
```

The training command expects separate training, validation, and held-out test CSVs. If a validation file is not already present, create a local validation split from the SMM4H-derived training data and keep it under `data/private/`.

For inference on Reddit mention windows:

```bash
python scripts/07_train_absa_sentiment_model.py predict \
  --model_dir ... \
  --input_csv data/private/out_absa_prep/absa_input.csv \
  --out_csv data/private/name_how_you_like.csv \
  --text_col text \
  --target_col target
```
