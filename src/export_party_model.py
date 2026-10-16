"""Export the party-prediction model (TF-IDF + logistic regression) as plain JSON so it can run in a web page.

It trains on ALL labelled speeches (the held-out scores in the README come from the split experiments, not from this model), then writes
docs/party_model.js with the vocabulary, idf weights, coefficients and intercepts, and a few test sentences with the Python answers
so the JavaScript version can be checked against it.

Run from the repo root:  python src/export_party_model.py      (needs data/energy_speeches.csv)
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from energiewende_pipeline import PROCEDURAL, clean, SEED  # noqa: E402

df = pd.read_csv(ROOT / "data" / "energy_speeches.csv", dtype={"period": str})
df["party"] = df["party"].replace({"LINKE": "DIE LINKE", "CSU": "CDU/CSU", "CDU": "CDU/CSU"})
d = df[df.party != "NA"].copy()
keep = d.party.value_counts()
d = d[d.party.isin(keep[keep >= 20].index)]
d["clean"] = d.text.map(clean)

vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = vec.fit_transform(d["clean"])
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED).fit(X, d["party"])
terms = vec.get_feature_names_out()
classes = clf.classes_.tolist()
tops = {c: terms[np.argsort(clf.coef_[i])[-12:][::-1]].tolist() for i, c in enumerate(classes)}

tests = []
for t in d.sample(6, random_state=SEED).text.tolist() + ["Die Energiewende braucht mehr Windkraft und Photovoltaik, der Kohleausstieg muss schneller kommen.",
                                                          "Wir brauchen Versorgungssicherheit und bezahlbare Strompreise, keine Verbote."]:
    t = t[:900]
    p = clf.predict_proba(vec.transform([clean(t)]))[0]
    tests.append({"text": t, "proba": [round(float(x), 5) for x in p]})

out = {"classes": classes, "terms": terms.tolist(), "idf": [round(float(x), 5) for x in vec.idf_], "coef": [[round(float(x), 4) for x in row] for row in clf.coef_],
       "intercept": [round(float(x), 4) for x in clf.intercept_], "procedural": sorted(PROCEDURAL), "top_terms": tops, "tests": tests,
       "trained_on": int(len(d)), "note": "trained on all labelled speeches; held-out accuracy is about 0.50 (see README), so treat the output as a weak signal"}
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs" / "party_model.js").write_text("window.PARTY = " + json.dumps(out, ensure_ascii=False, separators=(",", ":")) + ";\n", encoding="utf-8")
print("classes", classes, "| vocabulary", len(terms), "| speeches", len(d), "| file KB", round((ROOT / "docs" / "party_model.js").stat().st_size / 1024))
