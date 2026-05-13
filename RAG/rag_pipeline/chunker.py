"""
rag/embeddings/chunker.py — Pandas preprocessing + sentence chunking.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Optional
import pandas as pd
try:
    from underthesea import sent_tokenize
except ImportError:
    def sent_tokenize(t): return re.split(r'[.!?]\s+', t)


@dataclass
class ReviewChunk:
    text: str; chunk_index: int; review_id: str; product_id: int
    shopee_product_id: str; rating: int; sentiment: str
    author: Optional[str] = None; reviewed_at: Optional[str] = None
    def to_dict(self): return asdict(self)


class ReviewPreprocessor:
    MIN_CHARS = 20
    def process_batch(self, reviews: list[dict]) -> pd.DataFrame:
        if not reviews: return pd.DataFrame()
        df = pd.DataFrame(reviews)
        for col in ["review_id","content","rating","sentiment","shopee_product_id"]:
            if col not in df.columns: df[col] = "" if col!="rating" else 0
        df = df.drop_duplicates(subset=["review_id"]).dropna(subset=["content"])
        df["content"] = df["content"].astype(str).str.strip().apply(self._clean)
        df = df[df["content"].str.len() >= self.MIN_CHARS]
        df["rating"] = pd.to_numeric(df["rating"],errors="coerce").fillna(0).astype(int)
        df["sentiment"] = df.apply(self._sentiment, axis=1)
        return df.reset_index(drop=True)

    @staticmethod
    def _clean(t):
        t = re.sub(r"<[^>]+>"," ",t); t = re.sub(r"https?://\S+","",t)
        return re.sub(r"\s+"," ",t).strip()

    @staticmethod
    def _sentiment(row):
        r = int(row.get("rating",0))
        if r>=4: return "positive"
        if r==3: return "neutral"
        if r in(1,2): return "negative"
        ex = str(row.get("sentiment","")).lower()
        return ex if ex in("positive","neutral","negative") else "neutral"


class ReviewChunker:
    def __init__(self, max_chars=512, overlap_sents=1):
        self.max_chars=max_chars; self.overlap_sents=overlap_sents
        self.preprocessor=ReviewPreprocessor()

    def chunk_reviews(self, reviews: list[dict], product_id: int) -> list[ReviewChunk]:
        df = self.preprocessor.process_batch(reviews)
        if df.empty: return []
        return [c for _,row in df.iterrows() for c in self._chunk_row(row,product_id)]

    def _chunk_row(self, row, product_id):
        text = str(row.get("content","")).strip()
        if not text: return []
        common = dict(review_id=str(row.get("review_id","")), product_id=product_id,
                      shopee_product_id=str(row.get("shopee_product_id","")),
                      rating=int(row.get("rating",0)), sentiment=str(row.get("sentiment","neutral")),
                      author=str(row.get("author","")), reviewed_at=str(row.get("reviewed_at","")))
        if len(text)<=self.max_chars:
            return [ReviewChunk(text=text,chunk_index=0,**common)]
        sents=sent_tokenize(text); chunks=[]; current=[]; cur_len=0; idx=0
        for sent in sents:
            if cur_len+len(sent)>self.max_chars and current:
                chunks.append(ReviewChunk(text=" ".join(current),chunk_index=idx,**common))
                idx+=1; current=current[-self.overlap_sents:] if self.overlap_sents else []; cur_len=sum(len(s) for s in current)
            current.append(sent); cur_len+=len(sent)
        if current: chunks.append(ReviewChunk(text=" ".join(current),chunk_index=idx,**common))
        return chunks
