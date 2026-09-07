import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")


def ask_deepseek(prompt):
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={
            "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
        },
        json={
            "model": "deepseek-chat",
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
    )

    return response.json()["choices"][0]["message"]["content"]

def ask_deepseek(prompt):
    response = requests.post(
        "https://api.deepseek.com/chat/completions",
        headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}"},
        json={
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt}]
        }
    )
    return response.json()["choices"][0]["message"]["content"]


def load_text(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()

def chuking(text,chunk_size=500,overlap=50):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunk=text[start:end]
        chunks.append(chunk)
        start=start+chunk_size-overlap
    return chunks


text=load_text("sample.txt")
chunks=chuking(text)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
def embed_chunks(chunks):
    return embed_model.encode(chunks)

embeddings = embed_chunks(chunks)

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, chunk_embeddings, top_k=2):
    query_embedding = embed_model.encode([query])[0]  # embed the query same way as chunks
    
    similarities = []
    for i, chunk_emb in enumerate(chunk_embeddings):
        sim = cosine_similarity(query_embedding, chunk_emb)
        similarities.append((sim, i))
    
    # sort by similarity, highest first
    similarities.sort(reverse=True)
    
    top_chunks = [chunks[i] for sim, i in similarities[:top_k]]
    top_scores = [sim for sim, i in similarities[:top_k]]
    
    return top_chunks, top_scores

query = "How did Elias find the key?"
top_chunks, top_scores = retrieve(query, chunks, embeddings)

def build_prompt(query, retrieved_chunks):
    context = "\n\n---\n\n".join(retrieved_chunks)
    
    prompt = f"""Answer the question using ONLY the context below. If the answer isn't in the context, say "I don't know based on the given context."

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def rag_answer(query, chunks, chunk_embeddings, top_k=2):
    top_chunks, top_scores = retrieve(query, chunks, chunk_embeddings, top_k)
    prompt = build_prompt(query, top_chunks)
    answer = ask_deepseek(prompt)
    return answer, top_chunks, top_scores

# test it
query = "How did Elias find the key?"
answer, used_chunks, scores = rag_answer(query, chunks, embeddings)

print("QUESTION:", query)
print("\nANSWER:", answer)
print("\n(Used", len(used_chunks), "chunks, scores:", [f"{s:.3f}" for s in scores], ")")