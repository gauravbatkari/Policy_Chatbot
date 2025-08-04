import pickle
import faiss
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

def load_index():
    index = faiss.read_index("index.faiss")
    with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
    return index, chunks

def get_answer(question):
    index, chunks = load_index()
    query_vec = model.encode([question])
    D, I = index.search(np.array(query_vec), k=3)
    context = " ".join([chunks[i] for i in I[0]])
    result = qa_pipeline(question=question, context=context)
    return result['answer']