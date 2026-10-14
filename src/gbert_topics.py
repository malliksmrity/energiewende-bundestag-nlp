"""Compare German GBERT embeddings with a multilingual model for BERTopic topics.

Models
  paraphrase-multilingual-MiniLM-L12-v2   sentence model, 50+ languages (used in the notebook)
  deepset/gbert-base                      German BERT (Chan et al., 2020). It is a plain language model, not
                                          trained for sentence similarity: sentence-transformers adds mean pooling.
                                          It may do worse than the sentence model. That is part of the test.

Needs data/energy_speeches.csv (created by src/energiewende_pipeline.py) and requirements-gbert.txt.
Roughly 3 GB of disk (packages and two model downloads) and a few minutes on CPU.

Run from the repo root:  python src/gbert_topics.py
Writes reports/gbert_topics.json.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
MODELS = ["paraphrase-multilingual-MiniLM-L12-v2", "deepset/gbert-base"]
STOP = set("""aber alle allem allen aller alles als also am an ander andere anderen auch auf aus bei bin bis bist da damit dann der den des dem die das dass
daß du er es etwas euer eure für gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich ihr ihre ihren im in indem ins ist jede jedem jeden jeder jedes jetzt kann
kein keine keinen können könnte machen man manche mein meine mit muss musste nach nicht nichts noch nun nur ob oder ohne sehr sein seine seinen sich sie sind so solche soll sollte
sondern sonst über um und uns unser unsere unter viel vom von vor während war waren warum was weil weiter welche wenn werde werden wie wieder will wir wird wirst wo wollen
wollte würde würden zu zum zur zwar zwischen herr frau damen herren kollege kollegin präsident präsidentin beifall bundesregierung hat gibt geht gehen sagen sage ganz
schon mehr immer ja nein dies diese diesem diesen dieser dieses wurde wurden worden sein""".split())


def tokens(text):
    return [w for w in re.findall(r"[a-zäöüß]{3,}", text.lower()) if w not in STOP]


def run(model_name, docs, toks):
    from bertopic import BERTopic
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from hdbscan import HDBSCAN
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    emb = SentenceTransformer(model_name).encode(docs, show_progress_bar=False)
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom"),
        vectorizer_model=CountVectorizer(tokenizer=tokens, lowercase=False, ngram_range=(1, 1), min_df=3, token_pattern=None),
        calculate_probabilities=False)
    labels, _ = topic_model.fit_transform(docs, emb)
    labels = np.array(labels)
    topics = {t: [w for w, _ in topic_model.get_topic(t)][:8] for t in topic_model.get_topics() if t != -1}
    d = Dictionary(toks)
    cv = CoherenceModel(topics=list(topics.values()), texts=toks, dictionary=d, coherence="c_v").get_coherence() if topics else None
    return {"model": model_name, "n_topics": len(topics), "outlier_share": round(float(np.mean(labels == -1)), 3),
            "coherence_c_v": None if cv is None else round(float(cv), 3), "topics": {int(k): v for k, v in topics.items()}}


def main():
    df = pd.read_csv(ROOT / "data" / "energy_speeches.csv", dtype={"period": str})
    docs = df["text"].astype(str).tolist()
    toks = [tokens(t) for t in docs]
    results = [run(m, docs, toks) for m in MODELS]
    out = {"n_speeches": len(docs), "seed": SEED, "results": results,
           "note": "same UMAP, HDBSCAN, tokenizer and coherence measure for both models; only the embedding differs"}
    (ROOT / "reports" / "gbert_topics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    for r in results:
        print(r["model"], r["n_topics"], "topics, outliers", r["outlier_share"], "c_v", r["coherence_c_v"])


if __name__ == "__main__":
    main()
