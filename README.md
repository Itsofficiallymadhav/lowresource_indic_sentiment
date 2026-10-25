# Cross-Lingual Sentiment and Intent Alignment for Low-Resource Languages

This project explores how sentiment and intent can be modelled across Indian languages, including low-resource ones, by preparing labelled and parallel text in a consistent way so that meaning can later be aligned across languages.

The project uses IndicSentiment for labelled sentiment data,  Samanantar English–Indic parallel text for cross-lingual alignment. Text is cleaned for informal content (emojis, hashtags, URLs) and tokenized with the IndicNLP Library.

The overall work is divided into weekly stages, progressing from dataset acquisition and preprocessing towards modelling and evaluation.

## Week 1 — Data Curation and Preprocessing

Week 1 establishes the data foundation of the project: acquiring the datasets, cleaning and tokenizing their text, validating it, and saving it in a consistent format. No models are loaded or trained in this stage.

### Covered

* Set up the workspace and environment (`datasets`, `transformers`, `indic-nlp-library`)
* Loaded IndicSentiment (Hindi, Bengali, Marathi; validation and test splits) with the Hugging Face `datasets` loader
* Loaded a small Samanantar sample of English–Indic sentence pairs (first 5,000 per language)
* Explored columns, languages, splits, labels and missing values for each dataset
* Cleaned the text: URLs, e-mails, mentions and HTML-like tags removed; hashtag words kept; emojis kept; Indic characters and ZWJ/ZWNJ preserved; digits normalized
* Tokenized the cleaned text with IndicNLP and compared it with plain whitespace splitting
* Checked data quality: missing and empty text, labels, duplicates, train/validation/test overlap, script/language sanity and sentence-pair checks
* Saved the processed data as CSV files and reloaded them to verify

### Datasets

| Dataset | Type | Role |
|---|---|---|
| IndicSentiment | Labelled sentiment (hi, bn, mr) | Sentiment labels; only `validation` and `test` splits exist |
| Samanantar | English–Indic parallel text, no labels | Cross-lingual alignment (small sample only) |
| IndicCorpV2 | Monolingual text, no labels | Not used (very large) |

### Notebook

`Week1/Week1_CrossLingual_Preprocessing.ipynb`

### Output

The Week 1 pipeline turns three raw sources into cleaned, tokenized and validated CSV files in `Week1/data/processed/`:

* `indic_sentiment_processed.csv` — `language, split, text_raw, text, label, tokens`
* `samanantar_parallel_sample.csv` — `language, english_raw, indic_raw, english_text, indic_text`

These files are created locally when the notebook runs and are not committed, because of the dataset licences.

### Limitations

* IndicSentiment has no train split, so sentiment training data has to come from another source.
* The Samanantar sample contains only the first rows of each language and is not representative of the full corpus.


## Setup

```bash
pip install pandas datasets transformers indic-nlp-library
```

## Week 2 — Dataset Preparation and Baseline Classifier

Week 2 links the three languages through a shared review key, builds a fixed train/validation/test split, adds intent labels, and trains a first baseline: a frozen multilingual encoder with simple classifiers on top. The encoder is not fine-tuned.

### Covered

* Built a `review_id` for every review (original split + position within language and split) so that the Hindi, Bengali and Marathi versions of a review are linked
* Checked that the three languages are row-aligned: same ids, same sentiment label per review, and English numbers found in the translation of the same row (98% hi, 97% bn, 96% mr) but not in the neighbouring row (about 1%)
* Left out 4 of the 1,156 reviews, each with a stated reason: 2 with no sentiment label (`test_663`, `test_694`) and 2 repeated reviews (`test_662`, `test_744`)
* Joined the intent labels and assigned class numbers (Sentiment: Negative 0, Positive 1; Intent: Opinion 0, Factual 1)
* Split by review, not by row: 806 train / 173 validation / 173 test reviews per language, identical in all three languages
* Checked the prepared data: no review in more than one part, no text shared between parts, equal counts per language and part, labels equal to Week 1
* Computed review vectors with `xlm-roberta-base` (frozen, mean of the token vectors, maximum 128 tokens)
* Trained a standardiser + logistic regression for each task on the Hindi training reviews only, and chose `C` on the Hindi validation reviews
* Scored Hindi (trained language), Bengali and Marathi (never seen in training) against a majority-class reference

### Files

| File | Content |
|---|---|
| `data/split_manifest.csv` | 1,152 reviews with id, English text, sentiment label and class number, and part (`train` / `val` / `test`) |
| `data/intent_labels.csv` | Intent label and `unsure` flag per review, plus the second-pass columns `reviewed_pass2`, `changed_in_pass2` |
| `data/week2_dataset.csv` | Prepared dataset: 3,456 rows (1,152 reviews × 3 languages) |
| `outputs/vectors_xlm-roberta-base.npz` | Cached review vectors (3,456 × 768) |
| `outputs/metrics.csv`, `outputs/per_class_metrics.csv` | Accuracy, macro-F1 and per-class scores |
| `outputs/run_settings.json` | Settings, chosen `C`, library versions, dataset SHA-256 |

### Notebooks

* `Week_2/Week2_A_DataPrep.ipynb` — builds and validates the dataset (needs the Week 1 file `indic_sentiment_processed.csv`)
* `Week_2/Week2_B_Baseline.ipynb` — computes vectors, trains the classifiers and scores them

### Results

Test reviews (173 per language), macro-F1:

| Task | Model | hi | bn | mr |
|---|---|---|---|---|
| Sentiment | Majority class | 0.335 | 0.335 | 0.335 |
| Sentiment | XLM-R + logistic regression | 0.942 | 0.878 | 0.884 |
| Intent | Majority class | 0.480 | 0.480 | 0.480 |
| Intent | XLM-R + logistic regression | 0.697 | 0.627 | 0.536 |

Sentiment transfers well from Hindi to Bengali and Marathi. Intent does not yet: its accuracy is not above the majority class, and only 13 Factual reviews are in each test language.

### Limitations

* Only two sentiment classes (no Neutral) and two intent classes (no Inquiry) occur in the data
* Intent labels are silver labels, not human-verified, and Factual is rare (8.2% of reviews)
* Only Hindi is used for training, and the encoder is not fine-tuned


### Setup for Week 2

```bash
pip install -r requirements.txt
```

