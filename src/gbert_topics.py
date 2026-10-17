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
MODELS = ["paraphrase-multilingual-MiniLM-L12-v2", "deepset/gbert-base",
          "deutsche-telekom/gbert-large-paraphrase-cosine"]  # the last one is a German model trained for sentence similarity
STOP = set("""aber alle allem allen aller alles als also am an ander andere anderen auch auf aus bei bin bis bist da damit dann der den des dem die das dass
daÃƒÆ’Ã…Â¸ du er es etwas euer eure fÃƒÆ’Ã‚Â¼r gegen gewesen hab habe haben hat hatte hatten hier hin hinter ich ihr ihre ihren im in indem ins ist jede jedem jeden jeder jedes jetzt kann
kein keine keinen kÃƒÆ’Ã‚Â¶nnen kÃƒÆ’Ã‚Â¶nnte machen man manche mein meine mit muss musste nach nicht nichts noch nun nur ob oder ohne sehr sein seine seinen sich sie sind so solche soll sollte
sondern sonst ÃƒÆ’Ã‚Â¼ber um und uns unser unsere unter viel vom von vor wÃƒÆ’Ã‚Â¤hrend war waren warum was weil weiter welche wenn werde werden wie wieder will wir wird wirst wo wollen
wollte wÃƒÆ’Ã‚Â¼rde wÃƒÆ’Ã‚Â¼rden zu zum zur zwar zwischen herr frau damen herren kollege kollegin prÃƒÆ’Ã‚Â¤sident prÃƒÆ’Ã‚Â¤sidentin beifall bundesregierung hat gibt geht gehen sagen sage ganz
schon mehr immer ja nein dies diese diesem diesen dieser dieses wurde wurden worden sein""".split())


def tokens(text):
    return [w for w in re.findall(r"[a-zÃƒÆ’Ã‚Â¤ÃƒÆ’Ã‚Â¶ÃƒÆ’Ã‚Â¼ÃƒÆ’Ã…Â¸]{3,}", text.lower()) if w not in STOP]


def embed(model_name, docs):
    """Sentence embeddings. deepset/gbert-base has no model_type in its config, so it is loaded
    explicitly as a BERT model and mean-pooled (first 256 tokens of each speech)."""
    if not model_name.startswith("deepset/"):
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name)
        st.max_seq_length = min(256, st.max_seq_length or 256)  # speed: first 256 tokens of each speech, as for GBERT
        return st.encode(docs, show_progress_bar=False, batch_size=16)
    import torch
    from transformers import BertConfig, BertModel, BertTokenizerFast
    torch.manual_seed(SEED)
    tok = BertTokenizerFast.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, config=BertConfig.from_pretrained(model_name)).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(docs), 16):
            b = tok(docs[i:i + 16], padding=True, truncation=True, max_length=256, return_tensors="pt")
            h = model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
    return np.vstack(out)


def run(model_name, docs, toks):
    from bertopic import BERTopic
    from gensim.corpora import Dictionary
    from gensim.models import CoherenceModel
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer
    from umap import UMAP

    emb = embed(model_name, docs)
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED),
        hdbscan_model=HDBSCAN(min_cluster_size=15, metric="euclidean", cluster_selection_method="eom"),
        vectorizer_model=CountVectorizer(tokenizer=tokens, lowercase=False, ngram_range=(1, 1), min_df=1, token_pattern=None),
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
    import sys
    todo = sys.argv[1:] or MODELS  # optional: run only the named models and merge into the existing report
    results = [run(m, docs, toks) for m in todo]
    path = ROOT / "reports" / "gbert_topics.json"
    if sys.argv[1:] and path.exists():
        old = json.loads(path.read_text(encoding="utf-8"))["results"]
        results = [r for r in old if r["model"] not in todo] + results
    out = {"n_speeches": len(docs), "seed": SEED, "results": results,
           "note": "same UMAP, HDBSCAN, tokenizer and coherence measure for both models; only the embedding differs"}
    (ROOT / "reports" / "gbert_topics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2))
    for r in results:
        print(r["model"], r["n_topics"], "topics, outliers", r["outlier_share"], "c_v", r["coherence_c_v"])


if __name__ == "__main__":
    main()
