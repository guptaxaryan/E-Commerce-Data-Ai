"""Simple TF-IDF retrieval over synthesized table documents.

Each DataFrame is converted into a pseudo-document containing:
- Table name
- Column names and dtypes
- First few sample rows (serialized)

We build a corpus of these documents and expose a `retrieve` function to
get the most relevant docs for a user question. This forms a lightweight
RAG layer enriching the LLM prompt with contextual grounding.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

@dataclass
class TableDoc:
    name: str
    text: str

class Retriever:
    def __init__(self, docs: List[TableDoc]):
        self._docs = docs
        corpus = [d.text for d in docs]
        self._vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
        self._matrix = self._vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, k: int = 3) -> List[TableDoc]:
        if not self._docs:
            return []
        q_vec = self._vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self._matrix).flatten()
        ranked: List[Tuple[int, float]] = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in ranked[:k]:
            results.append(self._docs[idx])
        return results


def build_docs(dataframes: Dict[str, pd.DataFrame]) -> List[TableDoc]:
    docs: List[TableDoc] = []
    for name, df in dataframes.items():
        cols = ", ".join([f"{c}:{df[c].dtype}" for c in df.columns])
        sample = df.head(5).to_dict(orient="records")
        text = (
            f"TABLE {name}\nCOLUMNS {cols}\nSAMPLE {json.dumps(sample, default=str)}"
        )
        docs.append(TableDoc(name=name, text=text))
    return docs


def build_retriever(dataframes: Dict[str, pd.DataFrame]) -> Retriever:
    return Retriever(build_docs(dataframes))
