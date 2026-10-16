"""A German spaCy pipeline that tags energy technologies and policies in Bundestag speeches.

Build:  python src/spacy_energy_pipeline.py            (writes models/energiewende_de/ and reports/spacy_entities.json)
Use:    import spacy; nlp = spacy.load("models/energiewende_de"); doc = nlp("... Windkraftanlagen ...")
Package (optional): python -m spacy package models/energiewende_de dist --name energiewende_ner --version 0.1.0 --build wheel

What it does
  - starts from spaCy's German small model (tokenizer, lemmatizer, part-of-speech), with its statistical NER and parser switched off,
  - adds an EntityRuler with the project's word list of energy technologies and policies (labels TECHNOLOGY and POLICY),
  - matches a term at the START of a word, so German compounds such as "Windkraftanlagen" or "Atomkraftwerke" are found as well.

Why: the first notebook matched only tokens that were exactly one of the 14 words after lemmatisation, which misses compounds.
This script counts how many mentions each way finds. The word list is a gazetteer, so it can only find what it already knows,
and it cannot tell "for" from "against" (see README).

Needs: spacy, de_core_news_sm, and data/energy_speeches.csv (created by src/energiewende_pipeline.py).
"""
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import spacy

ROOT = Path(__file__).resolve().parents[1]
ENTITY_DICTIONARY = {
    "windkraft": "TECHNOLOGY", "windenergie": "TECHNOLOGY", "photovoltaik": "TECHNOLOGY", "solarenergie": "TECHNOLOGY",
    "wasserstoff": "TECHNOLOGY", "braunkohle": "TECHNOLOGY", "steinkohle": "TECHNOLOGY", "atomkraft": "TECHNOLOGY", "kernenergie": "TECHNOLOGY",
    "energiewende": "POLICY", "kohleausstieg": "POLICY", "atomausstieg": "POLICY", "klimaschutz": "POLICY", "versorgungssicherheit": "POLICY",
}


def build():
    nlp = spacy.load("de_core_news_sm", disable=["ner", "parser"])
    nlp.add_pipe("sentencizer")
    ruler = nlp.add_pipe("entity_ruler", config={"overwrite_ents": True})
    ruler.add_patterns([{"label": lab, "id": term, "pattern": [{"LOWER": {"REGEX": "^" + re.escape(term) + r"\w*"}}]} for term, lab in ENTITY_DICTIONARY.items()])
    return nlp


def main():
    nlp = build()
    (ROOT / "models").mkdir(exist_ok=True)
    nlp.to_disk(ROOT / "models" / "energiewende_de")
    nlp2 = spacy.load(ROOT / "models" / "energiewende_de")  # round trip: the saved pipeline must load and give the same result
    df = pd.read_csv(ROOT / "data" / "energy_speeches.csv", dtype={"period": str})
    texts = df["text"].astype(str).tolist()
    ruler_hits, lemma_hits, substring_hits, examples = Counter(), Counter(), Counter(), {}
    per_period = {p: Counter() for p in sorted(df.period.unique())}
    compound_forms = Counter()
    for doc, text, period in zip(nlp2.pipe(texts, batch_size=40), texts, df.period):
        for ent in doc.ents:
            key = ent.ent_id_ or ent.text.lower()
            ruler_hits[key] += 1
            per_period[period][key] += 1
            if ent.text.lower() != key:
                compound_forms[ent.text.lower()] += 1
                examples.setdefault(key, ent.sent.text.strip()[:160])
        for tok in doc:
            if tok.lemma_.lower() in ENTITY_DICTIONARY:
                lemma_hits[tok.lemma_.lower()] += 1
        low = text.lower()
        for term in ENTITY_DICTIONARY:
            substring_hits[term] += low.count(term)
    out = {"speeches": len(texts), "model": "de_core_news_sm + EntityRuler (prefix match), saved to models/energiewende_de",
           "mentions_by_method": {"ruler_prefix_match": sum(ruler_hits.values()), "exact_lemma_match_like_the_notebook": sum(lemma_hits.values()), "plain_substring_count": sum(substring_hits.values())},
           "mentions_per_term_ruler": dict(ruler_hits.most_common()), "mentions_per_period_ruler": {p: dict(c.most_common()) for p, c in per_period.items()},
           "top_compound_forms_found": dict(compound_forms.most_common(15)), "example_sentences_with_compounds": examples,
           "limits": ["a gazetteer finds only listed words", "prefix matching can over-match words that merely start with a term", "no for/against distinction"]}
    (ROOT / "reports" / "spacy_entities.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("speeches", "mentions_by_method", "top_compound_forms_found")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
