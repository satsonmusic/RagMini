# RagMini
## Overview

This project is a minimal Retrieval-Augmented Generation (RAG) stack you can run locally.

It turns a folder of docs into something you can query with grounded answers.

## What it does

- Ingests a directory of text/markdown files
- Chunks content into passages
- Builds a lightweight index
- Retrieves top matching chunks for a query
- Produces an answer with citations (file + chunk id)

## Why it’s useful

Agents and summarizers hallucinate when they lack context.

RAG makes answers verifiable and repeatable by forcing the system to quote sources.

## Inputs

- A folder of `.txt` / `.md` docs
- Query text
- Optional: chunk size and overlap

## Outputs

- Answer text
- Citations (which file/chunk supported each part)
- Debug view: retrieved chunks + scores

## Design principles

- Start with lexical retrieval (BM25-ish) before embeddings
- Always show citations so you can audit
- Keep the index rebuild fast enough to run often

## Success metrics

- citation coverage (% of answer sentences grounded)
- retrieval precision (top-k relevance)
- time to index / query latency

## Reference implementation (copy/paste)

This is a minimal, working Python implementation (stdlib-only). It:

- chunks files
- builds a simple TF-IDF index
- retrieves top-k chunks
- prints a grounded answer with citations

Create a folder with these files:

- `rag_ministack.py` (below)
- `docs/` (put a few `.md` files in there)

### rag_[ministack.py](http://ministack.py)

```python
import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

def _lab_root() -> Path:
	return Path(__file__).resolve().parent

def _resolve(p: str) -> Path:
	path = Path(p)
	return path if path.is_absolute() else (_lab_root() / path).resolve()

def tokenize(text: str) -> List[str]:
	text = (text or "").lower()
	return re.findall(r"[a-z0-9_]{2,}", text)

@dataclass
class Chunk:
	chunk_id: str
	source: str
	text: str

def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
	words = text.split()
	if not words:
		return []
	step = max(1, chunk_size - overlap)
	chunks = []
	for i in range(0, len(words), step):
		w = words[i : i + chunk_size]
		if not w:
			break
		chunks.append(" ".join(w))
		if i + chunk_size >= len(words):
			break
	return chunks

def load_chunks(docs_dir: Path, chunk_size: int, overlap: int) -> List[Chunk]:
	chunks: List[Chunk] = []
	for p in sorted(docs_dir.rglob("*")):
		if not p.is_file():
			continue
		if p.suffix.lower() not in {".md", ".txt"}:
			continue
		text = p.read_text(encoding="utf-8", errors="replace")
		for idx, ct in enumerate(chunk_text(text, chunk_size, overlap)):
			chunks.append(Chunk(chunk_id=f"{p.name}#{idx}", source=str(p), text=ct))
	return chunks

def build_tfidf_index(chunks: List[Chunk]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
	# idf(term) and tfidf vectors per chunk
	df: Dict[str, int] = {}
	chunk_tfs: List[Dict[str, int]] = []
	for ch in chunks:
		toks = tokenize(ch.text)
		tf: Dict[str, int] = {}
		for t in toks:
			tf[t] = tf.get(t, 0) + 1
		chunk_tfs.append(tf)
		for t in set(tf.keys()):
			df[t] = df.get(t, 0) + 1

	n = max(1, len(chunks))
	idf: Dict[str, float] = {}
	for t, d in df.items():
		idf[t] = math.log((n + 1) / (d + 1)) + 1.0

	vecs: List[Dict[str, float]] = []
	for tf in chunk_tfs:
		v: Dict[str, float] = {}
		for t, c in tf.items():
			v[t] = (1.0 + math.log(c)) * idf.get(t, 0.0)
		vecs.append(v)

	return idf, vecs

def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
	# sparse cosine similarity
	if not a or not b:
		return 0.0
	# iterate smaller
	if len(a) > len(b):
		a, b = b, a
	dot = 0.0
	for t, w in a.items():
		dot += w * b.get(t, 0.0)
	na = math.sqrt(sum(w * w for w in a.values()))
	nb = math.sqrt(sum(w * w for w in b.values()))
	return dot / (na * nb) if na and nb else 0.0

def vectorize_query(q: str, idf: Dict[str, float]) -> Dict[str, float]:
	tf: Dict[str, int] = {}
	for t in tokenize(q):
		tf[t] = tf.get(t, 0) + 1
	v: Dict[str, float] = {}
	for t, c in tf.items():
		v[t] = (1.0 + math.log(c)) * idf.get(t, 0.0)
	return v

def retrieve(q: str, chunks: List[Chunk], idf: Dict[str, float], vecs: List[Dict[str, float]], k: int) -> List[Tuple[float, Chunk]]:
	qv = vectorize_query(q, idf)
	scored = []
	for ch, v in zip(chunks, vecs):
		scored.append((cosine(qv, v), ch))
	scored.sort(key=lambda x: x[0], reverse=True)
	return [(s, ch) for s, ch in scored[:k] if s > 0.0]

def answer_with_citations(q: str, retrieved: List[Tuple[float, Chunk]]) -> str:
	# Minimal “grounded” answer: stitch top chunks with citations.
	# Replace with LLM generation later.
	lines = []
	lines.append(f"Query: {q}\n")
	if not retrieved:
		return "No relevant passages found."
	lines.append("Key passages:")
	for score, ch in retrieved:
		preview = ch.text[:240].strip()
		lines.append(f"- ({score:.3f}) {ch.chunk_id} — {preview}")
	lines.append("\nDraft answer (grounded):")
	lines.append(" ".join([f"[{ch.chunk_id}] {ch.text[:180].strip()}" for _s, ch in retrieved]))
	return "\n".join(lines).rstrip() + "\n"

def main() -> None:
	p = argparse.ArgumentParser(description="RAG MiniStack (TF-IDF retrieval + citations)")
	p.add_argument("--docs", default="docs")
	p.add_argument("--query", required=True)
	p.add_argument("--chunk-size", type=int, default=180)
	p.add_argument("--overlap", type=int, default=40)
	p.add_argument("--top-k", type=int, default=5)
	args, _ = p.parse_known_args()

	docs_dir = _resolve(args.docs)
	chunks = load_chunks(docs_dir, args.chunk_size, args.overlap)
	idf, vecs = build_tfidf_index(chunks)
	retrieved = retrieve(args.query, chunks, idf, vecs, args.top_k)
	print(answer_with_citations(args.query, retrieved))

if __name__ == "__main__":
	main()
```

### Run

- Put a few docs in `docs/`
- `python rag_ministack.py --query "What are our success metrics?" --top-k 5`
