# Energiewende in the Bundestag: topics in energy speeches, 2009 to 2021

Text analysis of 1,161 Bundestag speeches about energy: which themes rose or fell, how certain those changes are, and how well the words of a speech predict the speaker's party.

Live demo: https://huggingface.co/spaces/malliksmrity/energiewende-speech-analyzer


## 1. Problem
Energy policy changed quickly in this period (nuclear phase-out, coal exit, hydrogen). The task is to measure how attention in parliament shifted and to say how reliable that measurement is.

## 2. Data
GermaParl, plenary protocols of the Bundestag (PolMine project, CC BY-SA), legislative periods 17 to 19, every 8th session. A speech is kept if it contains one of 16 energy keywords: 1,161 speeches (396, 393 and 372 per period). Raw text is downloaded by the script and not stored in the repository.

## 3. Approach
1. Select speeches by keyword and clean the text.
2. Count the share of speeches mentioning each theme per period.
3. Resample whole sessions (1,000 draws) for 95 % ranges, since speeches within a session are not independent.
4. Topic models: LDA and BERTopic; BERTopic also with German embeddings.
5. Entity pipeline in spaCy for 14 energy terms.
6. Party prediction with TF-IDF and logistic regression under three test set-ups.

## 4. Results
Change in the share of speeches mentioning a theme, period 17 to 19 (percentage points, 95 % range):

| Theme | Change | Clear? |
|---|---|---|
| Nuclear | −21.9 (−37.1 to −6.7) | yes |
| Hydrogen | +11.2 (+4.7 to +18.5) | yes |
| Renewables | −14.6 (−28.3 to −0.8) | borderline |
| Climate protection | +18.9 (+0.6 to +36.7) | borderline |
| Coal | +5.4 (−0.6 to +10.9) | no |

Party prediction accuracy is about 50 % against 33 to 37 % for always guessing the largest party: better than chance, not reliable.

Topic models (same clustering and tokenizer, only the embedding differs):

| Embedding | Topics | Coherence (c_v) |
|---|---|---|
| Multilingual MiniLM | 12 | 0.523 |
| GBERT | 2 | 0.614 |
| German sentence model (gbert-large-paraphrase-cosine) | 2 | see `reports/gbert_topics.json` |

The German models score higher on coherence but find only two generic topics (function words and procedural terms), so MiniLM's 12 readable topics are more useful here.

## 5. Technical notes
- spaCy EntityRuler with prefix matching finds 3,337 mentions, against 2,315 for exact-lemma matching, because it also catches compounds (e.g. Klimaschutzgesetz).
- Classifier: TF-IDF (5,000 features), logistic regression with balanced class weights; six parties, CDU and CSU merged.
- Reproduce: `pip install -r requirements.txt`, then `python src/energiewende_pipeline.py`. The German-embedding test needs `requirements-gbert.txt`.

## 6. Limitations
- A sample of sessions; keywords decide which speeches count.
- Counting words cannot tell support from opposition.
- Party labels are uneven (385 CDU/CSU speeches, 43 AfD).
- Coursework (module M508, Gisma), extended afterwards.


