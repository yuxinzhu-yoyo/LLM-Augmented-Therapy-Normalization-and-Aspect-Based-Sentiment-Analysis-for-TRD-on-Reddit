# TRD Reddit Therapy Normalization and Sentiment Analysis

This repository contains the code, non-private reference files, aggregate outputs, and manuscript figure sources for the study:

**LLM-Augmented Therapy Normalization and Aspect-Based Sentiment Analysis of TRD-Labeled Reddit Discourse**

The project characterizes treatment-resistant depression (TRD) therapy discussion in Reddit posts using a curated TRD cohort, lexicon-based therapy normalization, and medication-tagged aspect-based sentiment analysis. The repository is organized for transparent research reuse while excluding raw Reddit text, usernames, row-level private annotations, and local model artifacts that should not be redistributed.

## Repository Scope

Included:

- Analysis scripts used to build the TRD cohort, extract therapy mentions, train and apply sentiment models, compute statistical tests, prepare manual-validation samples, and summarize validation results.
- Non-private reference files, including the therapy lexicon and LLM augmentation prompt.
- Aggregate source tables, model reports, and manuscript-ready figures.
- A manifest of private inputs required for full local reproduction.

Excluded:

- Raw Reddit exports and row-level Reddit-derived files.
- Private manual annotation spreadsheets and private annotation excerpts.
- Row-level Twitter/X training and test data.
- Full transformer checkpoints and large DSM/QMisSpell vector files.

## Environment

The local environment available on this machine is `trd-absa`.

```bash
conda env create -f environment.yml
conda activate trd-absa
python -m pip install -r requirements.txt
```

R and Quarto packages needed for statistical tables and figures are listed in `R_PACKAGES.md`.

## Repository Layout

```text
analysis_notebooks/
  generate_manuscript_figures.qmd
data/
  reference/
  private_input_manifest/
model_reference/
outputs/
  aggregate_absa/
  figures/
  model_reports/
  tables/
scripts/
  01_filter_trd_keyword_posts.py
  ...
  13_summarize_manual_validation.py
```

`data/private/` is intentionally absent from the repository. Create it locally and place the restricted inputs listed in `data/private_input_manifest/PRIVATE_INPUTS.md` before running the full workflow.

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
  --posts_csv data/private/TRD_new_cohort.csv \
  --lexicon_xlsx data/reference/TRDmeds_with_variants_and_misspell.xlsx \
  --out_dir data/private/out_absa_prep_new_ann2
```

`04_extract_therapy_mentions.py` produces the mention-level ABSA input, including the local context window used for sentiment inference.

### 3. Train and apply the ABSA sentiment model

```bash
python scripts/06_augment_smm4h_training_data.py \
  data/private/train_Twitter.csv \
  data/private/train_Twitter.csv \
  data/reference/prompt_absa_SMOTE.txt \
  data/private/train_new_Twitter.csv \
  2

python scripts/07_train_absa_sentiment_model.py train \
  --train_csv data/private/train_new_Twitter.csv \
  --val_csv data/private/validation_Twitter.csv \
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
  --model_dir data/private/aug2_638035_microsoft_deberta_v3-base_best_model \
  --input_csv data/private/out_absa_prep_new_ann2/absa_input.csv \
  --out_csv data/private/predictions_deberta.csv \
  --text_col text \
  --target_col target
```

For the held-out Twitter/X test-set confusion matrix:

```bash
python scripts/07_train_absa_sentiment_model.py predict \
  --model_dir data/private/aug2_638035_microsoft_deberta_v3-base_best_model \
  --input_csv data/private/test.csv \
  --out_csv data/private/twitter_test_predictions_deberta.csv \
  --text_col text \
  --target_col therapy \
  --label_col label
```

The public repository includes only the aggregate confusion matrix in `outputs/model_reports/twitter_test_confusion_matrix_deberta.csv`, not row-level Twitter/X predictions.

### 4. Compute statistical and validation summaries

```bash
Rscript scripts/09_run_sentiment_statistics.R

python scripts/10_audit_domain_shift_lengths.py \
  --twitter data/private/train_new_Twitter.csv \
  --reddit data/private/predictions_deberta.csv \
  --out outputs/tables/domain_shift_length_summary.csv

python scripts/11_generate_sensitivity_tables.py \
  --predictions data/private/predictions_deberta.csv \
  --therapy-map data/reference/medication_distribution_with_variants_FIN.csv \
  --out-dir outputs/tables/revision_sensitivity
```

Manual-validation files are private because they include Reddit-derived excerpts:

```bash
python scripts/12_prepare_manual_validation_samples.py \
  --trd-posts data/private/TRD_new_cohort.csv \
  --predictions data/private/predictions_deberta.csv \
  --out-dir outputs/manual_review_private \
  --construct-n 300 \
  --sentiment-n 100

python scripts/13_summarize_manual_validation.py \
  --construct outputs/manual_review_private/construct_validity_private_sample.csv \
  --sentiment outputs/manual_review_private/reddit_sentiment_validation_private_sample.csv \
  --out-dir outputs/tables/manual_review_summary
```

### 5. Render manuscript figures

```bash
quarto render analysis_notebooks/generate_manuscript_figures.qmd
```

Generated figures are written to `outputs/figures/`. The committed figures are aggregate, publication-ready artifacts and do not contain verbatim Reddit text.

## Public Data and Privacy Notes

This repository is designed to support reproducibility without redistributing sensitive or restricted text. Raw Reddit posts, usernames, post identifiers, and private annotation excerpts are not committed. Aggregated tables and figures are included where they do not expose verbatim user-generated content. Private inputs required for a complete local rerun are listed explicitly in `data/private_input_manifest/PRIVATE_INPUTS.md`.

Before making a public release, review `outputs/`, `data/reference/`, and any newly generated files for row-level text, identifiers, or local paths.

## Citation

Please cite the associated manuscript when using this repository. A formal citation will be added after publication.
