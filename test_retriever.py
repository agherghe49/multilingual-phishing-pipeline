import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("step 1: sys.path")
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

print("step 2: import config")
from config import KB_DIR, FAISS_INDEX_PATH, EMBEDDER_MODEL, OUTPUT_DIR
print(f"  KB_DIR={KB_DIR}")
print(f"  OUTPUT_DIR={OUTPUT_DIR}")

print("step 3: import faiss + sentence_transformers")
import faiss
from sentence_transformers import SentenceTransformer
print("  ok")

print("step 4: load model")
model = SentenceTransformer(EMBEDDER_MODEL, trust_remote_code=True)
print("  ok")

print("step 5: load KB documents")
import json
from pathlib import Path
docs = []
for path in sorted(Path(KB_DIR).glob("*.json")):
    print(f"  citesc {path.name}...")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data if isinstance(data, list) else [data]
    for item in items:
        if item.get("generated text"):
            docs.append(item["generated text"][:800])
        for rnd in item.get("multi-rounds fraud", []):
            if rnd.get("generated_data"):
                docs.append(rnd["generated_data"][:800])
print(f"  {len(docs)} documente")

print("step 6: encode")
import numpy as np
embeddings = model.encode(docs, batch_size=32, show_progress_bar=True, normalize_embeddings=True).astype(np.float32)
print(f"  shape: {embeddings.shape}")

print("step 7: build FAISS index")
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"  index cu {index.ntotal} vectori")

print("step 8: save index")
FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, str(FAISS_INDEX_PATH / "index.faiss"))
print("  salvat ok")

print("DONE - indexul FAISS e gata!")
