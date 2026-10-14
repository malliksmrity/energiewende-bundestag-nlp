"""Reproducible core of the Energiewende project, plus stricter tests.

Steps
  1. Download every 8th plenary session of Bundestag periods 17, 18, 19 from GermaParl
     (PolMine, TEI-XML, CC BY-SA) and keep speeches that mention an energy keyword.
  2. Count how often a few energy terms appear per period (the trend chart on the page).
  3. Predict the speaker's party from the words, and compare three ways of testing:
       - random split              (what the notebook did)
       - split by session          (no session appears in both train and test)
       - split by time             (train on periods 17 and 18, test on period 19)
     with a "always the biggest party" baseline for each.

Run from the repo root:  python src/energiewende_pipeline.py
Needs internet. Raw speeches are cached in data/energy_speeches.csv (git-ignored).
Writes reports/results.json and docs/data.js.
"""
import json
import re
import urllib.request
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "energy_speeches.csv"
RAW = "https://raw.githubusercontent.com/PolMine/GermaParlTEI/main"
API = "https://api.github.com/repos/PolMine/GermaParlTEI/contents"
PERIODS, STEP = ["17", "18", "19"], 8
KEYWORDS = ["energiewende", "erneuerbare energie", "erneuerbaren energien", "windkraft", "windenergie",
            "photovoltaik", "solarenergie", "wasserstoff", "braunkohle", "steinkohle", "atomkraft",
            "kernenergie", "kohleausstieg", "atomausstieg", "klimaschutz", "versorgungssicherheit"]
TREND_TERMS = {"Nuclear (Atomkraft, Kernenergie)": ["atomkraft", "kernenergie", "atomausstieg"],
               "Renewables (erneuerbar, Windkraft, Photovoltaik)": ["erneuerbare", "windkraft", "windenergie", "photovoltaik", "solarenergie"],
               "Coal (Kohle)": ["braunkohle", "steinkohle", "kohleausstieg"],
               "Hydrogen (Wasserstoff)": ["wasserstoff"],
               "Climate protection (Klimaschutz)": ["klimaschutz"]}
PROCEDURAL = {"dass", "prozent", "drucksache", "ausschuss", "fraktion", "beratung", "kollege", "kollegin",
              "herr", "frau", "damen", "herren", "sitzung", "tagesordnung", "wort", "beifall"}


def get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "portfolio-research"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def collect():
    rows = []
    for period in PERIODS:
        names = sorted(d["name"] for d in json.loads(get(f"{API}/{period}")) if d["name"].endswith(".xml"))[::STEP]
        for name in names:
            try:
                root = ET.fromstring(get(f"{RAW}/{period}/{name}"))
            except Exception as exc:  # network or parse problem: skip the session, say so
                print("skipped", period, name, exc)
                continue
            for sp in root.iter("sp"):
                text = " ".join(p.text or "" for p in sp.findall("p"))
                if any(k in text.lower() for k in KEYWORDS):
                    rows.append({"period": period, "session": name, "party": sp.get("party", "NA"), "text": text})
    return pd.DataFrame(rows)


def clean(t):
    return " ".join(w for w in re.findall(r"[a-zäöüß]{3,}", t.lower()) if w not in PROCEDURAL)


def evaluate(train, test, name):
    vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    Xtr, Xte = vec.fit_transform(train["clean"]), vec.transform(test["clean"])
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(Xtr, train["party"])
    base = DummyClassifier(strategy="most_frequent").fit(Xtr, train["party"])
    p, b = clf.predict(Xte), base.predict(Xte)
    return {"split": name, "n_train": int(len(train)), "n_test": int(len(test)),
            "baseline_accuracy": round(float(accuracy_score(test["party"], b)), 3),
            "accuracy": round(float(accuracy_score(test["party"], p)), 3),
            "macro_f1": round(float(f1_score(test["party"], p, average="macro")), 3)}


def main():
    if CACHE.exists():
        df = pd.read_csv(CACHE, dtype={"period": str})
    else:
        df = collect()
        CACHE.parent.mkdir(exist_ok=True)
        df.to_csv(CACHE, index=False)
    df["party"] = df["party"].replace({"LINKE": "DIE LINKE", "CSU": "CDU/CSU", "CDU": "CDU/CSU"})

    low = df.text.str.lower()
    trend = {label: {p: round(float(low[df.period == p].apply(lambda t: any(w in t for w in words)).mean()), 3)
                     for p in PERIODS} for label, words in TREND_TERMS.items()}
    per_period = df.period.value_counts().sort_index().to_dict()

    d = df[~df.party.isin(["NA"])].copy()
    keep = d.party.value_counts()
    d = d[d.party.isin(keep[keep >= 20].index)]
    d["clean"] = d.text.map(clean)
    parties = d.party.value_counts().to_dict()

    res = []
    tr, te = train_test_split(d, test_size=0.2, random_state=SEED, stratify=d.party)
    res.append(evaluate(tr, te, "Random split (as in the notebook)"))
    g = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    i_tr, i_te = next(g.split(d, groups=d.session))
    res.append(evaluate(d.iloc[i_tr], d.iloc[i_te], "Whole sessions held out"))
    past, future = d[d.period.isin(["17", "18"])], d[d.period == "19"]
    future = future[future.party.isin(past.party.unique())]  # AfD enters only in period 19
    res.append(evaluate(past, future, "Train periods 17 and 18, test period 19"))

    out = {"n_speeches": int(len(df)), "per_period": per_period, "parties_used": parties,
           "classification": res, "trend_share_of_speeches": trend,
           "notebook_reported": {"lda_c_v": 0.374, "bertopic_c_v": 0.397, "party_accuracy": 0.496, "party_macro_f1": 0.434,
                                 "note": "from the executed notebook; BERTopic and LDA are not re-run by this script"}}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "data.js").write_text("window.PAGE_DATA = " + json.dumps(out) + ";\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
