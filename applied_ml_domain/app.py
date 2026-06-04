from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import chromadb
import ollama
import os

app = Flask(__name__, static_folder=".")
CORS(app)

#loading model along with database for a cold start
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_collection("ml_chunks")



def query_rag(user_query: str, n_results: int = 3):
    """
    Retrieve relevant chunks from ChromaDB --> call Ollama llama3 --> return the answer + source metadata.
    """
    query_embedding = embedding_model.encode(user_query).tolist()

    greetings = [
    "hi",
    "hello",
    "hey",
    "hii",
    "good morning",
    "good evening"
    ]

    if user_query.lower() in greetings:
        return "Hello! How can I help you today?", []

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    retrieved_chunks = results["documents"][0]
    metadatas = results["metadatas"][0]

    context = "\n".join(retrieved_chunks)

    prompt = f"""
    You are an AI assistant answering questions from a document.

    Rules:
    1. Answer only using the provided context.
    2. Give a concise answer.
    3. Maximum 150 words.can go higher is asked to elaborate.
    4. Use headings and bullet points.
    5. Mention only the most relevant information.
    6. Do not list every fact from the context.
    7. Do NOT mention what the document does or does not contain.
    8. Do NOT add notes, disclaimers, assumptions, observations, or reasoning.
    9. If information is unavailable, say:
   "Sorry, I do not have relevant information on that."
   10.don't talk about cultural norms unless asked.
   11.while a greeting is given greet back no unescessary answer.

    Context:
    {context}

    Question:
    {user_query}

    Answer:"""

    response = ollama.chat(
        model="llama3:8b",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response["message"]["content"]

    # Building sources list
    sources = []
    seen = set()#we use set function so as to avoid reoccuring metadatas or avoid duplicacy
    for meta in metadatas:
        key = (meta.get("file_name", ""), meta.get("page_number", ""))#safer than meta["file_name"] as it would crash if data not available
        if key not in seen:
            seen.add(key)
            sources.append({
                "file_name": meta.get("file_name", "Unknown"),
                "page_number": meta.get("page_number", "Unknown")
            })

    return answer, sources


# Routes 
@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    try:
        answer, sources = query_rag(user_query)
        return jsonify({
            "answer": answer,
            "sources": sources
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
