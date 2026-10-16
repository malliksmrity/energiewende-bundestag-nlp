# Energiewende in the Bundestag: what German MPs say about energy

**In one sentence:** this project reads 1,161 Bundestag speeches about energy (2009 to 2021), shows which energy topics came and went, and tests how far those findings can be trusted.

Live page: `docs/index.html` (GitHub Pages, folder `/docs`).

## 1. Problem

Energy policy in Germany changes fast: the 2011 nuclear phase-out, the growth of wind and solar, the 2020 coal exit, now hydrogen. Journalists, researchers and government-affairs teams have no cheap, data-driven way to see how attention in parliament shifts before it becomes law. The question: **which energy themes rose and fell in the Bundestag between 2009 and 2021, how sure are we, and can the words of a speech even tell us which party the speaker belongs to?**

Who could use it: journalists, policy researchers, government-affairs teams (plausible uses, not tested with real users).

## 2. Dataset

- **GermaParl**, plenary protocols of the Bundestag as TEI-XML, built by the PolMine project, licence **CC BY-SA**: https://github.com/PolMine/GermaParlTEI
- Legislative periods 17 to 19 (2009 to 2021), every 8th session (93 files). A speech is kept if it contains one of 16 energy keywords: **1,161 speeches** (396, 393 and 372 per period).
- The notebook also fetches 30 whole protocols of period 20 (2021 to 2025) from the Bundestag DIP API. They have no party labels and are not used in the results below. The DIP key that was in the original notebook is removed; put yours in `.env` as `DIP_API_KEY`.
- The raw speeches are downloaded by the script and are not stored in this repository.

## 3. How I solved it

1. Download the sampled sessions and keep speeches that mention an energy keyword.
2. Count how many speeches mention each theme (nuclear, renewables, coal, hydrogen, climate) per period.
3. Test whether the changes are real: resample whole sessions (speeches in one session are not independent) to get a 95 percent range.
4. Find topics with two methods: LDA (counts words) and BERTopic (sentence embeddings), and compare them with a coherence score.
5. Predict the speaker's party from the words, and test it in three stricter ways (random split, whole sessions held out, train on the past and test on period 19).
6. Test a German language model (GBERT) against the multilingual one for the topics.

## 4. Findings

Share of energy speeches mentioning each theme, with the change from period 17 to 19 (95 percent range from resampling whole sessions):

| Theme | Period 17 (2009-13) | 18 (2013-17) | 19 (2017-21) | Change 17→19, points | Clear? |
|---|---|---|---|---|---|
| Nuclear | 31.8 % | 26.7 % | 9.9 % | −21.9 (−37.1 to −6.7) | yes |
| Renewables | 44.4 % | 40.7 % | 29.8 % | −14.6 (−28.3 to −0.8) | borderline |
| Coal | 4.8 % | 15.0 % | 10.2 % | +5.4 (−0.6 to +10.9) | no |
| Hydrogen | 2.5 % | 3.1 % | 13.7 % | +11.2 (+4.7 to +18.5) | yes |
| Climate protection | 38.9 % | 47.6 % | 57.8 % | +18.9 (+0.6 to +36.7) | borderline |

- **Nuclear fading and hydrogen rising are solid findings.** The renewables fall and the climate rise are plausible but borderline. The coal change is not distinguishable from noise.
- **The party prediction is weak.** About 50 percent right against 33 to 37 percent for always guessing the largest party (1.5 times better, not the "3 times chance" in the first notebook text, which used the wrong baseline).
- **BERTopic gave 12 interpretable topics; GBERT gave 2 generic ones** (see below).

## 5. Models used

| Task | Model | Why |
|---|---|---|
| Topics | LDA (gensim), 8 topics | classic baseline |
| Topics | BERTopic with `paraphrase-multilingual-MiniLM-L12-v2`, UMAP, HDBSCAN | meaning-based topics, works for German |
| Topics (test) | BERTopic with `deepset/gbert-base` (German BERT, mean-pooled) | does a German-only model do better? |
| Party prediction | TF-IDF (1 and 2 word groups) + logistic regression, class weights | simple, explainable baseline |
| Entities | spaCy German pipeline (de_core_news_sm) + EntityRuler with the 14-term energy word list, prefix matching (src/spacy_energy_pipeline.py) | transparent, closed vocabulary, finds German compounds |

Evaluation:

| Party prediction, test set-up | Always the largest party | Model accuracy | Macro-F1 |
|---|---|---|---|
| Random split (as in the notebook) | 0.332 | 0.496 | 0.431 |
| Whole sessions held out | 0.355 | 0.522 | 0.490 |
| Train periods 17 and 18, test period 19 | 0.368 | 0.534 | 0.413 |

| Embedding for topics | Topics | Speeches without topic | c_v coherence |
|---|---|---|---|
| multilingual MiniLM (notebook's choice) | 12 | 34.5 % | 0.523 |
| GBERT (German BERT) | 2 | 0 % | 0.614 |

GBERT's higher score is misleading: its two topics are generic words ("eine", "Deutschland", "Euro") and procedural words ("Drucksache", "Ausschuss"). GBERT is a plain language model, not trained to compare sentences. A German sentence-embedding model is the fairer next test. These coherence values use a simpler tokenizer than the notebook's (LDA 0.374, BERTopic 0.397 there), so the two sets are not comparable.

**NLP tooling and what I did not use**

- **spaCy:** used for the production-style entity pipeline. It finds **3,337 energy mentions** in the 1,161 speeches, against 2,315 with the notebook's exact-lemma method, because it also catches compounds (Klimaschutzziele 82, Atomkraftwerke 79, Klimaschutzgesetz 77, Windkraftanlagen 13). The pipeline can be saved and packaged (python -m spacy package). It is a word list, so it cannot tell support from opposition.
- **Hugging Face:** used through BERTopic's sentence models and GBERT (Transformers). The text-generation serving tool TGI is for large language models and is not needed here.
- **Spark NLP:** not used. It is built for clusters on big data; 1,161 speeches fit in memory on a laptop.

## 6. Technical documentation

- Parsing as in the notebook: `sp` elements, paragraph text, case-insensitive substring match on 16 keywords.
- Trend ranges: for periods 17 and 19, whole sessions are resampled with replacement (1,000 draws, seed 42), the difference in share is recomputed each time.
- Classifier: TF-IDF 5,000 features, `LogisticRegression(class_weight="balanced")`. CDU and CSU merged, parties with fewer than 20 speeches dropped (6 parties, 1,158 speeches). Time split drops AfD from the test set (no earlier examples).
- GBERT has no `model_type` in its config, so `src/gbert_topics.py` loads it explicitly as BERT and mean-pools the first 256 tokens. BERTopic's per-topic vectorizer uses `min_df=1`.
- Reproduce: `pip install -r requirements.txt`; `python src/energiewende_pipeline.py` (needs internet, about 60 MB, writes `reports/results.json`, `docs/data.js`); GBERT test: `pip install -r requirements-gbert.txt` then `python src/gbert_topics.py` (about 3 GB with the model downloads; writes `reports/gbert_topics.json`).
- Limits: a sample of sessions; keywords decide what counts; word counting cannot tell support from opposition; AfD only 43 speeches; LDA and BERTopic of the notebook are not re-run by the script.

## 7. Output

- Static page with theme trends and their ranges, classifier results and the coherence comparison: `docs/index.html`.
- Machine-readable results: `reports/results.json`, `reports/gbert_topics.json`.
- Executed notebook with the original outputs (API key removed): `notebooks/energiewende_discourse_analysis.ipynb`.

## 8. Deployment

| What | How | Status |
|---|---|---|
| Static page | GitHub Pages, folder `/docs` | needs enabling |
| Live app | none yet (the data download takes about 60 MB at start-up) | backlog in `PROCESS.md` |
| CI | `.github/workflows/ci.yml` compiles the code on every push | written, not yet run on GitHub |

## Coursework and extensions

Coursework (module M508 Big Data Analytics, Gisma): the notebook with LDA, BERTopic, entities, party classifier. Extended afterwards: the reproducible script, session-bootstrap trend ranges, stricter splits, corrected baseline, GBERT test, static page. Your college has approved publishing coursework (results pending).

Smrity Mallik, MSc AI and Data Science (Gisma, Berlin), B.Eng. Mechanical Engineering. GitHub: github.com/malliksmrity. LinkedIn: linkedin.com/in/smritymallik.
