import streamlit as st
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# Hugging Face API setup
HUGGINGFACE_API_KEY = "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"  # 👈 Replace with your actual token
API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
HEADERS = {"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"}

# Load FAISS index and paragraphs
index = faiss.read_index("ncert_faiss.index")
with open("ncert_paragraphs.txt", "r", encoding="utf-8") as f:
    paragraphs = f.read().split("\n---\n")

# Load embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")


# Streamlit UI
st.set_page_config(page_title="NCERT AI Tutor", layout="wide")
st.title("📘 Class 9 Civics AI Tutor")
st.markdown("Ask any question from NCERT Democratic Politics — powered by AI ✨")

query = st.text_input("❓ Ask a question:")

if query:
    query_vector = embedder.encode([query]).astype("float32")
    D, I = index.search(query_vector, k=1)
    context = paragraphs[I[0][0]]

    prompt = f"""
You are an expert tutor for NCERT Class 9 Civics. A student asked: "{query}"

Use the paragraph below to answer in 3–5 clear, friendly lines:

--- Paragraph ---
{context}
"""

    with st.spinner("Thinking..."):
        response = requests.post(API_URL, headers=HEADERS, json={
            "inputs": prompt,
            "parameters": {"max_new_tokens": 150, "temperature": 0.6}
        })

    if response.status_code == 200:
        answer = response.json()[0]["generated_text"]
        final_answer = answer[len(prompt):].strip()

        st.markdown("### 🧠 AI Tutor's Answer:")
        st.success(final_answer)
    else:
        st.error("Something went wrong! 😕")
        st.text(response.text)
