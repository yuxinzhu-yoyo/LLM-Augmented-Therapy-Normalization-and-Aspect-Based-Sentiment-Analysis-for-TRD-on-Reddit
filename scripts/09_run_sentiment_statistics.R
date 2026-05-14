library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(readr)

private_dir <- file.path("data", "private")
table_dir <- file.path("outputs", "tables")
dir.create(table_dir, recursive = TRUE, showWarnings = FALSE)

pred <- read_csv(file.path(private_dir, "predictions_deberta.csv"), show_col_types = FALSE)

# Create a clean sentiment factor from pred_label
pred <- pred %>%
  mutate(
    sentiment = tolower(pred_label),  # start from pred_label, NOT sentiment
    sentiment = recode(
      sentiment,
      "neg"  = "negative",
      "pos"  = "positive",
      "neu"  = "neutral"
    ),
    sentiment = factor(
      sentiment,
      levels = c("negative", "neutral", "positive")
    )
  )

# Quick sanity check
pred %>% count(sentiment)

pred %>%
  count(canonical) %>%
  arrange(desc(n)) %>%
  head()


# Aggregate counts per medication
drug_sent <- pred %>%
  count(canonical, sentiment, name = "n") %>%
  pivot_wider(
    names_from  = sentiment,
    values_from = n,
    values_fill = 0
  ) %>%
  rename(
    n_negative = negative,
    n_neutral  = neutral,
    n_positive = positive
  )

# Binomial tests per drug: H0: P(pos | pos+neg) = 0.5
drug_tests <- drug_sent %>%
  mutate(
    n_nonneutral = n_positive + n_negative
  ) %>%
  filter(n_nonneutral > 0) %>%  # ignore drugs with only neutral mentions
  mutate(
    test = map2(
      n_positive, n_nonneutral,
      ~ binom.test(.x, .y, p = 0.5, alternative = "two.sided")
    ),
    tidied = map(test, broom::tidy)
  ) %>%
  unnest(tidied) %>%
  mutate(
    p_adj    = p.adjust(p.value, method = "BH"),  # FDR across meds
    pos_share = estimate,
    neg_share = 1 - estimate
  ) %>%
  select(
    canonical,
    n_positive, n_negative, n_neutral, n_nonneutral,
    pos_share, conf.low, conf.high,
    p.value, p_adj
  ) %>%
  arrange(p_adj)

# Significant imbalance after FDR
drug_tests_sig <- drug_tests %>%
  filter(p_adj < 0.05)


meds_meta <- read_csv(file.path(private_dir, "out_absa_prep_new_ann2", "med_distribution_from_absa.csv"), show_col_types = FALSE)

pred <- pred %>% left_join(meds_meta, by = "canonical")
class_sent <- pred %>%
  count(class, sentiment, name = "n") %>%
  pivot_wider(
    names_from  = sentiment,
    values_from = n,
    values_fill = 0
  )

tab_class_sent <- class_sent %>%
  select(-class) %>%
  as.matrix()
rownames(tab_class_sent) <- class_sent$class

chisq_res <- chisq.test(tab_class_sent)
chisq_res


classes <- rownames(tab_class_sent)
pairs <- combn(classes, 2, simplify = FALSE)

pair_res <- purrr::map_df(pairs, function(cls) {
  sub_tab <- tab_class_sent[cls, ]
  test    <- chisq.test(sub_tab)
  tibble(
    class1 = cls[1],
    class2 = cls[2],
    p      = test$p.value
  )
}) %>%
  mutate(
    p_adj = p.adjust(p, method = "BH")
  ) %>%
  arrange(p_adj)

pair_res %>% filter(p_adj < 0.05)


#---- 1. Per-drug sentiment counts and binomial tests -----------------------

# If drug_sent not yet created, build it from pred
library(dplyr)
library(tidyr)
library(purrr)
library(broom)
library(readr)
library(tibble)

if (!exists("drug_sent")) {
  drug_sent <- pred %>%
    count(canonical, sentiment, name = "n") %>%
    pivot_wider(
      names_from  = sentiment,
      values_from = n,
      values_fill = 0
    ) %>%
    rename(
      n_negative = negative,
      n_neutral  = neutral,
      n_positive = positive
    )
}

# Binomial tests per drug (if not already run)
if (!exists("drug_tests")) {
  drug_tests <- drug_sent %>%
    mutate(
      n_nonneutral = n_positive + n_negative
    ) %>%
    filter(n_nonneutral > 0) %>%
    mutate(
      test   = map2(
        n_positive, n_nonneutral,
        ~ binom.test(.x, .y, p = 0.5, alternative = "two.sided")
      ),
      tidied = map(test, broom::tidy)
    ) %>%
    unnest(tidied) %>%
    mutate(
      p_adj    = p.adjust(p.value, method = "BH"),
      pos_share = estimate,
      neg_share = 1 - estimate
    ) %>%
    select(
      canonical,
      n_positive, n_negative, n_neutral, n_nonneutral,
      pos_share, conf.low, conf.high,
      p.value, p_adj
    ) %>%
    arrange(p_adj)
}

drug_tests_sig <- drug_tests %>%
  filter(p_adj < 0.05)

# Write per-drug tables
write_csv(drug_sent,
          file.path(table_dir, "table_drug_sentiment_counts.csv"))

write_csv(drug_tests,
          file.path(table_dir, "table_drug_sentiment_binomial_tests.csv"))

write_csv(drug_tests_sig,
          file.path(table_dir, "table_drug_sentiment_binomial_tests_sig.csv"))


#---- 2. Class by sentiment contingency and chi-square stats ----------------

# If class_sent not yet built, build it from pred (requires pred$class)
if (!exists("class_sent")) {
  class_sent <- pred %>%
    count(class, sentiment, name = "n") %>%
    pivot_wider(
      names_from  = sentiment,
      values_from = n,
      values_fill = 0
    )
}


# Add row totals for convenience
class_sent_with_totals <- class_sent %>%
  mutate(total = negative + neutral + positive)

# If chisq_res not yet computed, do it now
tab_class_sent <- class_sent %>%
  select(-class) %>%
  as.matrix()
rownames(tab_class_sent) <- class_sent$class

if (!exists("chisq_res")) {
  chisq_res <- chisq.test(tab_class_sent)
}

# Tidy observed, expected, and standardized residuals into one long table
obs <- chisq_res$observed %>%
  as.data.frame() %>%
  rownames_to_column("class") %>%
  pivot_longer(
    cols      = -class,
    names_to  = "sentiment",
    values_to = "observed"
  )

exp <- chisq_res$expected %>%
  as.data.frame() %>%
  rownames_to_column("class") %>%
  pivot_longer(
    cols      = -class,
    names_to  = "sentiment",
    values_to = "expected"
  )

stdres <- chisq_res$stdres %>%
  as.data.frame() %>%
  rownames_to_column("class") %>%
  pivot_longer(
    cols      = -class,
    names_to  = "sentiment",
    values_to = "std_resid"
  )

class_sent_stats <- obs %>%
  left_join(exp, by = c("class", "sentiment")) %>%
  left_join(stdres, by = c("class", "sentiment")) %>%
  arrange(class, sentiment)

# Write class-level tables
write_csv(class_sent_with_totals,
          file.path(table_dir, "table_class_sentiment_counts.csv"))

write_csv(class_sent_stats,
          file.path(table_dir, "table_class_sentiment_chisq_stats.csv"))


#---- 3. Pairwise class comparisons (if you computed them) ------------------

# If you have not yet created pair_res:
if (!exists("pair_res")) {
  classes <- rownames(tab_class_sent)
  pairs   <- combn(classes, 2, simplify = FALSE)

  pair_res <- purrr::map_df(pairs, function(cls) {
    sub_tab <- tab_class_sent[cls, ]
    test    <- chisq.test(sub_tab)
    tibble(
      class1 = cls[1],
      class2 = cls[2],
      p      = test$p.value
    )
  }) %>%
    mutate(
      p_adj = p.adjust(p, method = "BH")
    ) %>%
    arrange(p_adj)
}

write_csv(pair_res,
          file.path(table_dir, "table_class_pairwise_chisq.csv"))
