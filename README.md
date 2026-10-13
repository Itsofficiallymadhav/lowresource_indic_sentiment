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

| Dataset | Type | Role | Licence |
|---|---|---|---|
| IndicSentiment | Labelled sentiment (hi, bn, mr) | Sentiment labels; only `validation` and `test` splits exist | None stated on the dataset card |
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

