import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

def split_text(text, max_length=500):
    paragraphs = text.split('\n')
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_length:
            current_chunk += " " + para
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para
    chunks.append(current_chunk.strip())
    return chunks

def create_index(text):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    chunks = split_text(text)
    embeddings = model.encode(chunks)

    index = faiss.IndexFlatL2(384)
    index.add(np.array(embeddings))

    with open("chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)
    faiss.write_index(index, "index.faiss")