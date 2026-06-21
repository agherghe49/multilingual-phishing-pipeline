"""
rag/retriever.py

Construiește și interoghează indexul FAISS cu Qwen3-Embed 0.6B.
Suportă swap ușor de embedder pentru ablation study (modifici config.py).

La primul apel construiește indexul din KB_DIR și îl salvează pe disk.
La apelurile ulterioare îl încarcă direct (rapid).
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    KB_DIR, FAISS_INDEX_PATH,
    EMBEDDER_MODEL, TOP_K_DOCS,
)


class RAGRetriever:
    """
    Wrapper FAISS + sentence-transformers pentru retrieval contextual.

    Utilizare:
        retriever = RAGRetriever()
        docs = retriever.retrieve("account verification urgency ro-RO", k=2)
    """

    def __init__(
        self,
        embedder_model: Optional[str] = None,
        force_rebuild: bool = False,
    ):
        self.model_name = embedder_model or EMBEDDER_MODEL
        self._index     = None
        self._docs      = []       # lista de documente indexate
        self._embedder  = None

        index_file    = FAISS_INDEX_PATH / "index.faiss"
        docs_file     = FAISS_INDEX_PATH / "docs.pkl"
        meta_file     = FAISS_INDEX_PATH / "meta.json"

        needs_rebuild = force_rebuild or not index_file.exists() or not docs_file.exists()

        if not needs_rebuild and meta_file.exists():
            with open(meta_file) as f:
                meta = json.load(f)
            if meta.get("embedder_model") != self.model_name:
                print(
                    f"[RAG] Embedder schimbat "
                    f"({meta.get('embedder_model')} → {self.model_name}), "
                    f"rebuild index..."
                )
                needs_rebuild = True

        if needs_rebuild:
            self._build_index(index_file, docs_file, meta_file)
        else:
            self._load_index(index_file, docs_file)

    # ── Construire index ──────────────────────────────────────────────────

    def _build_index(self, index_file: Path, docs_file: Path, meta_file: Path) -> None:
        """Construiește indexul FAISS din toate fișierele JSON din KB_DIR."""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "Rulează: pip install faiss-cpu sentence-transformers"
            )

        print(f"[RAG] Construiesc index cu {self.model_name}...")
        self._embedder = SentenceTransformer(self.model_name)

        # Colectează toate documentele din KB
        documents = self._load_kb_documents()
        print(f"[RAG] {len(documents)} documente de indexat")

        # Embed-uiește documentele
        texts = [d["content"] for d in documents]
        embeddings = self._embedder.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,  # cosine similarity via inner product
        ).astype(np.float32)

        # Construiește indexul FAISS (inner product = cosine pe vectori normalizați)
        dim   = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Salvează pe disk
        FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(index_file))
        with open(docs_file, "wb") as f:
            pickle.dump(documents, f)
        with open(meta_file, "w") as f:
            json.dump({"embedder_model": self.model_name, "dim": dim}, f)

        self._index = index
        self._docs  = documents
        print(f"[RAG] Index salvat în {FAISS_INDEX_PATH} ({dim}D, {len(documents)} docs)")

    def _load_index(self, index_file: Path, docs_file: Path) -> None:
        """Încarcă indexul FAISS existent de pe disk."""
        try:
            import faiss
        except ImportError:
            raise ImportError(
                "Rulează: pip install faiss-cpu sentence-transformers"
            )

        self._index = faiss.read_index(str(index_file))
        with open(docs_file, "rb") as f:
            self._docs = pickle.load(f)
        print(f"[RAG] Index încărcat: {len(self._docs)} documente, {self.model_name}")
        # embedderul se încarcă lazy la primul retrieve()

    def _load_kb_documents(self) -> list[dict]:
        """
        Transformă scenariile JSON din KB în documente indexabile.
        Fiecare document are: content (text), metadata (dict).
        """
        documents = []
        for path in sorted(KB_DIR.glob("*.json")):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else [data]

            for item in items:
                # Document 1: textul din "generated text" (emailul de bază)
                if item.get("generated text"):
                    documents.append({
                        "content": item["generated text"][:800],
                        "metadata": {
                            "id":          item.get("id"),
                            "category":    item.get("category"),
                            "subcategory": item.get("subcategory"),
                            "fraud_stage": "base",
                            "round":       0,
                            "source":      path.name,
                        }
                    })

                # Document 2–5: fiecare round din multi-rounds fraud
                for rnd in item.get("multi-rounds fraud", []):
                    text = rnd.get("generated_data", "")
                    if text:
                        documents.append({
                            "content": text[:800],
                            "metadata": {
                                "id":          item.get("id"),
                                "category":    item.get("category"),
                                "subcategory": item.get("subcategory"),
                                "fraud_stage": _infer_stage(rnd.get("round", 1)),
                                "round":       rnd.get("round", 1),
                                "source":      path.name,
                            }
                        })

        return documents

    # ── Retrieval ─────────────────────────────────────────────────────────

    def retrieve(self, query: str, k: int = TOP_K_DOCS) -> list[dict]:
        """
        Returnează cele mai relevante k documente pentru query.

        Args:
            query: string de căutare (topic + fraud_stage + locale)
            k:     numărul de documente returnate

        Returns:
            listă de dict cu 'content' și 'metadata'
        """
        if self._index is None:
            return []

        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)

        q_emb = self._embedder.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)

        scores, indices = self._index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self._docs):
                doc = dict(self._docs[idx])
                doc["relevance_score"] = float(score)
                results.append(doc)

        return results


def _infer_stage(round_num: int) -> str:
    if round_num <= 2:
        return "authority"
    return "urgency"
