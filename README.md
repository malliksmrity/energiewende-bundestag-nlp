# Energiewende in the Bundestag: what German MPs say about energy

**In one sentence:** this project reads 1,161 Bundestag speeches about energy (2009 to 2021) and shows which energy topics came and went.

Project page (charts, plain language): `docs/index.html`, served with GitHub Pages once Pages is enabled for the `/docs` folder.

## The question

How did the language of energy debates in the German parliament change between 2009 and 2021, and can the words of a speech tell us which party the speaker belongs to?

## Who could benefit, and how

Journalists, researchers and government-affairs teams could use this kind of analysis as a first look at how attention shifts before it turns into law (plausible, not tested with any user).

## Data and licence

- GermaParl, German Bundestag plenary protocols as TEI-XML, built by the PolMine project, licence CC BY-SA: https://github.com/PolMine/GermaParlTEI
- Periods 17 to 19 (2009 to 2021). Every 8th session (93 files). Speeches containing one of 16 energy keywords are kept: 1,161 speeches (396, 393 and 372 per period).
- The notebook also fetches 30 whole protocols of period 20 from the Bundestag DIP API. They have no party labels and are not used in the numbers below.
- Raw speeches are downloaded by the script and are not stored in the repository.

## How it works

1. Download the sampled sessions and keep speeches with an energy keyword.
2. Clean the text, tag a few energy terms (Atomkraft, Wasserstoff, Kohleausstieg) with a word list.
3. Find topics with LDA (older, counts words) and BERTopic (newer, uses sentence embeddings).
4. Extra: predict the speaker's party from the words with a TF-IDF and logistic regression model.

## Results

From `python src/energiewende_pipeline.py` (needs internet), saved in `reports/results.json`.

Share of energy speeches mentioning at least one word of the group:

| Group | Period 17 (2009-13) | 18 (2013-17) | 19 (2017-21) |
|---|---|---|---|
| Nuclear | 31.8 % | 26.7 % | 9.9 % |
| Renewables | 44.4 % | 40.7 % | 29.8 % |
| Coal | 4.8 % | 15.0 % | 10.2 % |
| Hydrogen | 2.5 % | 3.1 % | 13.7 % |
| Climate protection | 38.9 % | 47.6 % | 57.8 % |

Party prediction (six parties, 1,158 speeches with a party label and at least 20 per party):

| Test set-up | Always the largest party | Word model accuracy | Macro-F1 |
|---|---|---|---|
| Random split (as in the notebook) | 0.332 | 0.496 | 0.431 |
| Whole sessions held out | 0.355 | 0.522 | 0.490 |
| Train periods 17 and 18, test period 19 | 0.368 | 0.534 | 0.413 |

From the executed notebook (not recomputed by the script): LDA coherence c_v 0.374, BERTopic 0.397. BERTopic assigned 421 of 1,191 documents to its "no topic" bucket.

## Limitations

- A sample of sessions, and keywords decide what counts as an energy speech.
- Word matching cannot tell support from opposition.
- The earlier notebook text said the classifier is "roughly 3x the 0.17 chance baseline". The classes are uneven, so the fair baseline is the largest party at about 0.33, and the model is about 1.5 times better, not 3 times.
- 43 AfD speeches only, and AfD exists only in the last period.
- Topic-model quality was judged by one coherence score and by reading topics. No second reader.
- LDA and BERTopic are not re-run by the script. They need the notebook and its packages.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python src/energiewende_pipeline.py      # downloads about 60 MB, writes reports/ and docs/data.js
# notebook (topic models), see requirements-notebook.txt:
pip install -r requirements-notebook.txt
```

## What is coursework and what I extended

- Coursework (module M508 Big Data Analytics, Gisma): the notebook with LDA, BERTopic, entity tagging, party classifier and evaluation.
- Extended afterwards: the reproducible script, the session-held-out and time-based tests, the baseline correction, the theme trend table, and the static page.
- The API key hard-coded in the original notebook was removed. Check your university's rules before publishing coursework.

## For technical readers

- Parsing is as in the notebook: `sp` elements, paragraph text with `p.text`, case-insensitive substring match.
- Classifier: TF-IDF (5,000 features, unigrams and bigrams), `LogisticRegression(class_weight="balanced")`, seed 42. CDU and CSU merged. The script uses simple lower-cased tokens without lemmatization, so its random-split result (0.496) is close to, not the same pipeline as, the notebook (0.496 with lemmas and proper nouns removed).
- Session split: `GroupShuffleSplit` on the session file name. Time split drops AfD from the test set because it has no training examples.

## Repository layout

```
notebooks/   energiewende_discourse_analysis.ipynb (executed, API key removed)
src/         energiewende_pipeline.py
reports/     results.json
docs/        static project page
data/        (raw speech cache, git-ignored)
```

Smrity Mallik, MSc AI and Data Science (Gisma, Berlin), B.Eng. Mechanical Engineering.
GitHub: github.com/malliksmrity. LinkedIn: linkedin.com/in/smritymallik.

## Process and automation

- `PROCESS.md`: backlog, definition of done and sprint rhythm for the next work (a plan, not past experience).
- `.github/workflows/ci.yml`: GitHub Actions check (written here, not yet run on GitHub).
