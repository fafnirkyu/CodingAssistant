import os
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import re
import numpy as np
from numpy.linalg import norm
from sentence_transformers import SentenceTransformer
from ai.hf_client import classify_with_llm_detailed as classify_with_llm
from sentence_transformers import SentenceTransformer, util
import numpy as np, os, pandas as pd

EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
KB_PATH = os.path.join("data","bitext.csv")
kb = pd.read_csv(KB_PATH) 
DATA_PATH = os.path.join("data", "bitext.csv")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

print("🔍 Initializing RAG module...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(MODEL_NAME, device=device)

# Load dataset
if not os.path.exists("data/kb_embs.npz"):
    embs = EMBED_MODEL.encode(kb["instruction"].tolist(), convert_to_tensor=False)
    np.savez("data/kb_embs.npz", embs=embs)
else:
    embs = np.load("data/kb_embs.npz")["embs"]

kb = pd.read_csv(DATA_PATH)

# Check column names
expected_cols = {"instruction", "response"}
if not expected_cols.issubset(set(kb.columns)):
    raise ValueError(f"bitext.csv must have at least these columns: {expected_cols}")

print(f"✅ Loaded {len(kb)} knowledge entries.")

# Precompute embeddings for the "instruction" column
print("⚙️ Encoding instructions for similarity search...")
kb["embeddings"] = kb["instruction"].apply(lambda x: model.encode(str(x), convert_to_tensor=True))

CATEGORY_EXAMPLES = {
    "ACCOUNT": [
        "I forgot my password",
        "I can't log in",
        "My account was locked",
        "Change my email"
    ],
    "ORDER": [
        "Where is my order?",
        "Tracking shows delivered but not received",
        "My package is missing",
        "Order delayed"
    ],
    "BILLING": [
        "I was charged twice",
        "Refund not received",
        "Payment declined",
        "Need invoice"
    ],
    "TECHNICAL": [
        "App crashes",
        "Website not loading",
        "Login error",
        "Page freezing"
    ],
    "SUBSCRIPTION": [
        "Cancel my subscription",
        "Renew my plan",
        "Upgrade to premium",
        "Subscription paused"
    ],
}

model = SentenceTransformer("all-MiniLM-L6-v2")

CATEGORY_EMBEDDINGS = {}
for cat, examples in CATEGORY_EXAMPLES.items():
    vectors = model.encode(examples, convert_to_tensor=False)
    CATEGORY_EMBEDDINGS[cat] = np.mean(vectors, axis=0)

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    return text.strip()

def rule_based_category(message: str) -> str | None:
    msg = message.lower()
    if "subscription" in msg:
        return "SUBSCRIPTION"
    if "refund" in msg or "charged" in msg or "invoice" in msg:
        return "BILLING"
    if "order" in msg or "tracking" in msg or "#" in msg:
        return "ORDER"
    if "password" in msg or "account" in msg or "login" in msg:
        return "ACCOUNT"
    if "crash" in msg or "bug" in msg or "error" in msg:
        return "TECHNICAL"
    return None

def classify_with_embeddings(message: str):
    text = clean_text(message)
    vec = model.encode(text, convert_to_tensor=False)
    scores = {cat: np.dot(vec, proto) / (norm(vec) * norm(proto))
              for cat, proto in CATEGORY_EMBEDDINGS.items()}
    best_cat = max(scores, key=scores.get)
    return best_cat, scores[best_cat]

def classify_ticket(message: str):
    # 1. Rule-based check
    rule_cat = rule_based_category(message)
    if rule_cat:
        return rule_cat, 1.0

    # 2. Embedding similarity
    emb_cat, conf = classify_with_embeddings(message)
    if conf >= 0.55:
        return emb_cat, conf

    # 3. Fallback to AI model
    llm_cat = classify_with_llm(message)
    return llm_cat, 0.5

# Retrieve knowledge base entries similar to the query
def retrieve_knowledge(query: str, top_k: int = 3) -> list:
    query_emb = model.encode(query, convert_to_tensor=True)
    similarities = [util.cos_sim(query_emb, emb).item() for emb in kb["embeddings"]]

    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]

    results = []
    for i in top_indices:
        row = kb.iloc[i]
        results.append({
            "instruction": row.get("instruction", ""),
            "category": row.get("category", ""),
            "intent": row.get("intent", ""),
            "response": row.get("response", ""),
            "score": round(similarities[i], 3)
        })

    return results

KNOWLEDGE_BASE = [
    {"instruction": "I can't log into my account", "response": "ACCOUNT"},
    {"instruction": "My order hasn't arrived yet", "response": "ORDER"},
    {"instruction": "I was charged twice", "response": "BILLING"},
    {"instruction": "The app keeps crashing", "response": "TECHNICAL"},
    {"instruction": "I want to cancel my subscription", "response": "SUBSCRIPTION"},
    {"instruction": "Promo code invalid", "response": "BILLING"},
    {"instruction": "Website very slow", "response": "TECHNICAL"},
    {"instruction": "I forgot my password", "response": "ACCOUNT"},
]

# Retrieve top-k similar examples from the knowledge base

def retrieve_examples(message: str, top_k: int = 3):
    query_emb = model.encode(message, convert_to_tensor=True)
    sims = []
    for ex in KNOWLEDGE_BASE:
        emb = model.encode(ex["instruction"], convert_to_tensor=True)
        score = util.cos_sim(query_emb, emb).item()
        sims.append((score, ex))
    sims.sort(reverse=True, key=lambda x: x[0])
    return [ex for _, ex in sims[:top_k]]