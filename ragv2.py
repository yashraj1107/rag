from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

def load_pdf(filepath):
    reader = PdfReader(filepath)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# test it
filepath = "meridian_rag_50_pages.pdf"  
text = load_pdf(filepath)

print("Total characters extracted:", len(text))
print("\n---- First 500 characters ----")
print(text[:500])

import re

def chunk_text_smart(text, max_chunk_size=800, overlap_sentences=1):
    # Split into sentences (basic split on '.', '!', '?' followed by space/newline)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        current_chunk.append(sentence)
        current_length += len(sentence)
        
        if current_length >= max_chunk_size:
            chunks.append(" ".join(current_chunk))
            # keep last N sentences for overlap into next chunk
            current_chunk = current_chunk[-overlap_sentences:]
            current_length = sum(len(s) for s in current_chunk)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

# test it
chunks = chunk_text_smart(text)

embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(chunks, show_progress_bar=True)
print("Shape of embeddings:", embeddings.shape)

import faiss
import numpy as np

def build_faiss_index(embeddings):
    dimension = embeddings.shape[1]  # 384
    index = faiss.IndexFlatIP(dimension)  # IP = Inner Product (works like cosine similarity if vectors are normalized)
    
    # normalize embeddings so Inner Product behaves like cosine similarity
    faiss.normalize_L2(embeddings)
    
    index.add(embeddings)
    return index

# build it
embeddings = np.array(embeddings).astype('float32')  # FAISS requires float32
index = build_faiss_index(embeddings)

print("Number of vectors in index:", index.ntotal)