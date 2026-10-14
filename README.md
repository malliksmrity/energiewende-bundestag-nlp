# How Germany's Parliament Talks About Energy

**A plain-language look at what Bundestag MPs said about the energy transition, and how the parties differ.**

*Started as a coursework project (Big Data Analytics, Gisma University of Applied Sciences) and extended afterwards. See "What I changed after submission".*

## The question

When Germany debates its energy future (wind, solar, hydrogen, coal, nuclear), what do MPs actually talk about? Do the parties talk about it differently? And did the debate change over time?

I used a computer program to read more than 1,100 parliamentary speeches on energy so that patterns can be seen without reading them all by hand.

**What this project does not do:** it shows what politicians *said*. It does not show whether they later *did* it. That would need a different kind of data (laws passed, votes, budgets).

## What I found

- **Energy debate is often about money, not just the environment.** One of the clearest themes mixes words like *Euro*, *Milliarde* (billion), *renewable* and *climate protection*.
- **The themes can be separated.** The program split the speeches into groups such as nuclear power, climate protection, and renewables.
- **Parties use somewhat different words.** Given only the text of a speech, the program guessed the right party about half the time (50%). Guessing "CDU/CSU" every time would be right about a third of the time (33%). So the parties differ, but not sharply.
- **Some parties are harder to tell apart.** FDP and DIE LINKE were hardest to identify. They have fewer speeches in the data, so the program had less to learn from.
- TODO: one finding about how the debate changed over time, from your chart.

## The data

| Years | Where it comes from | What I used |
|---|---|---|
| 2009 – 2021 | GermaParl, a public collection of Bundestag speeches | 1,161 speeches about energy, each with the speaker's party |
| From late 2021 | The Bundestag's official data service (DIP) | 30 full session records from the start of the new term; no party information |

A speech counts as "about energy" if it contains one of 16 energy words (such as *Energiewende*, *Windkraft*, *Wasserstoff*, *Kohleausstieg*).

## How it works, in four steps

1. **Collect** the speeches from the public sources.
2. **Filter** for energy-related ones and clean up the text.
3. **Find themes:** an algorithm groups speeches that use similar words.
4. **Compare parties:** I checked which themes each party stresses and tested whether the party can be guessed from the words alone.

## What it cannot tell you (limits)

- **Words are not actions.** Nothing here shows whether promises were kept.
- **The newest period is thin.** Only 30 session records, with no party labels, so conclusions about the crisis years are weak.
- **Small groups, shaky numbers.** For example, only 9 AfD speeches were in the test set.
- **Some unrelated speeches slip in,** because words like *Versorgungssicherheit* are also used in security debates.
- **The theme finder depends on its settings,** and I have not yet tested other settings.

## What I changed after submission

TODO: list only what you have actually done, for example:
- Removed a private access key from the code
- Corrected how the classifier's accuracy is compared (against the majority-guess baseline)
- Removed untested business recommendations

## For technical readers

**Pipeline:** GermaParl TEI-XML (terms 17–19) and DIP API (term 20) → keyword filter → spaCy German lemmatisation and stopword removal → exploratory analysis → dictionary-based entity tagging (14 terms) → LDA (8 topics) vs. BERTopic (GBERT embeddings) → topic emphasis by party → TF-IDF + logistic regression party classifier.

| Measure | Result |
|---|---|
| Topic coherence (c_v), LDA / BERTopic | 0.374 / 0.397 (small difference) |
| Classifier accuracy / macro-F1 | 0.50 / 0.43 (6 parties, test n=232) |
| Majority-class baseline | about 0.33 |

CDU and CSU are merged into one class. Parties with fewer than 20 speeches were dropped. The split is random and stratified (80/20, seed 42); a time-based split would be stricter. Term 20 is excluded from the classifier.

**Run it:**

```bash
git clone https://github.com/malliksmrity/energiewende-bundestag-nlp.git
cd energiewende-bundestag-nlp
pip install -r requirements.txt
cp .env.example .env     # add your own DIP API key
```

TODO: add the order in which to run the notebook.

## Sources and licence

GermaParl (PolMine), CC BY-SA. Bundestag DIP API. TODO: add full citations and check the terms of use. Code: MIT licence.

**Author:** Smrity Mallik · [LinkedIn](https://linkedin.com/in/smritymallik) · [GitHub](https://github.com/malliksmrity)
