import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "data", "visa_rules.json")
EMBEDDINGS_CACHE = os.path.join(BASE_DIR, "data", "rules_embeddings.pkl")
FAISS_CACHE = os.path.join(BASE_DIR, "data", "rules_index.faiss")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_rules():
    with open(RULES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def build_embeddings(rules, model):
    texts = [rule["text"] for rule in rules]
    embeddings = []
    for text in tqdm(texts, desc="Embedding visa rules"):
        emb = model.encode(text)
        embeddings.append(emb)
    return np.vstack(embeddings)

def save_cache(embeddings, rules):
    with open(EMBEDDINGS_CACHE, "wb") as f:
        pickle.dump({"embeddings": embeddings, "rules": rules}, f)

def load_cache():
    if not os.path.exists(EMBEDDINGS_CACHE):
        return None
    with open(EMBEDDINGS_CACHE, "rb") as f:
        return pickle.load(f)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_CACHE)
    return index

def load_faiss_index():
    if not os.path.exists(FAISS_CACHE):
        return None
    return faiss.read_index(FAISS_CACHE)

def prepare_rag_store(force_rebuild=False):
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    rules = load_rules()
    cache = load_cache() if not force_rebuild else None
    if cache and not force_rebuild:
        embeddings = cache["embeddings"]
        rules = cache["rules"]
        index = load_faiss_index()
        if index is None:
            index = build_faiss_index(embeddings)
    else:
        embeddings = build_embeddings(rules, model)
        save_cache(embeddings, rules)
        index = build_faiss_index(embeddings)
    return rules, embeddings, index, model

def retrieve(query, rules, index, model, top_k=3):
    if not rules or index is None:
        return []
    query_emb = model.encode([query])
    D, I = index.search(np.array(query_emb).astype(np.float32), top_k)
    results = []
    for idx in I[0]:
        if 0 <= idx < len(rules):
            results.append(rules[idx])
    return results