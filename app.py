from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import os
import json
import time
import threading
from werkzeug.utils import secure_filename
import traceback

import fitz
import numpy as np
import faiss
import google.generativeai as genai

import unicodedata
import re

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
VECTOR_STORE_FOLDER = os.path.join(BASE_DIR, "vector_stores")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VECTOR_STORE_FOLDER, exist_ok=True)

# ✅ Gemini config

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")  


if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

genai.configure(api_key=GEMINI_API_KEY)

# Chunking & retrieval
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", 700))
MIN_CHUNK_SIZE = int(os.environ.get("MIN_CHUNK_SIZE", 200))
TOP_K = int(os.environ.get("TOP_K", 5))

MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB
CLEANUP_AFTER_SECONDS = 24 * 60 * 60

# App setup
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*")
CORS(app, resources={r"/*": {"origins": FRONTEND_ORIGIN}})




# Text cleaning utilities
def clean_text(text: str) -> str:
    if not text:
        return ""
    replacements = {
        "Ɵ": "ti", "ﬁ": "fi", "fl": "fl", "ﬂ": "fl",
        "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\r", "\n")
    text = re.sub(r"[\u200b\u00ad]", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text

# PDF text extraction
def extract_pages_text(pdf_path: str) -> tuple[list[tuple[int, str]], int]:
    doc = fitz.open(pdf_path)
    try:
        n = doc.page_count
        out: list[tuple[int, str]] = []
        for i in range(n):
            t = doc.load_page(i).get_text() or ""
            out.append((i + 1, clean_text(t)))
        return out, n
    finally:
        doc.close()

# Chunking
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, min_chunk_size: int = MIN_CHUNK_SIZE) -> list[str]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    chunks: list[str] = []
    start = 0
    n = len(cleaned)
    while start < n:
        end = min(start + chunk_size, n)
        window = cleaned[start:end]
        if end < n:
            cut = window.rfind(". ")
            if cut == -1: cut = window.rfind("? ")
            if cut == -1: cut = window.rfind("! ")
            if cut == -1: cut = window.rfind(" ")
            if cut != -1 and cut >= min_chunk_size:
                end = start + cut + 1
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = max(end - 100, start + 1) if end < n else n
    return chunks

def chunk_pdf_by_page(pages: list[tuple[int, str]], chunk_size: int = CHUNK_SIZE, min_chunk_size: int = MIN_CHUNK_SIZE) -> list[dict]:
    items: list[dict] = []
    for page_num, raw in pages:
        cleaned = " ".join((raw or "").split())
        if not cleaned:
            continue
        for part in chunk_text(cleaned, chunk_size=chunk_size, min_chunk_size=min_chunk_size):
            items.append({"text": part, "page": page_num})
    return items


    

# ------------------------------------------------------------
# Vector store
# ------------------------------------------------------------
def get_embeddings(texts):
    vectors = []

    for text in texts:
        response = genai.embed_content(
            model="models/embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        vectors.append(response["embedding"])

    vectors = np.array(vectors).astype("float32")

    faiss.normalize_L2(vectors)

    return vectors


def create_vector_store(text_chunks: list[str]):
    embeddings = get_embeddings(text_chunks)

    dim = embeddings.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    return index, embeddings

def load_latest_vector_store():
    files = [f for f in os.listdir(VECTOR_STORE_FOLDER) if f.endswith(".faiss")]
    if not files:
        raise FileNotFoundError("No vector store found. Upload a PDF first.")
    files.sort(key=lambda name: os.path.getmtime(os.path.join(VECTOR_STORE_FOLDER, name)), reverse=True)
    base_name = os.path.splitext(files[0])[0]
    index_path = os.path.join(VECTOR_STORE_FOLDER, files[0])
    chunks_path = os.path.join(VECTOR_STORE_FOLDER, f"{base_name}.chunks.json")
    if not os.path.exists(chunks_path):
        raise FileNotFoundError("Chunks file missing for latest vector store.")
    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    chunks = payload.get("chunks") or []
    if not chunks:
        raise ValueError("Loaded vector store has no chunks.")
    return index, chunks, payload

# ------------------------------------------------------------
# Search
# ------------------------------------------------------------
def embed_question(question):
    response = genai.embed_content(
        model="models/embedding-004",
        content=question,
        task_type="retrieval_query"
    )

    vector = np.array([response["embedding"]]).astype("float32")

    faiss.normalize_L2(vector)

    return vector

def search_faiss(index, question_vector, chunks, top_k=TOP_K, *, chunk_pages=None):
    top_k = max(1, min(top_k, len(chunks)))
    scores, indices = index.search(question_vector, top_k)
    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        i = int(idx)
        page = chunk_pages[i] if chunk_pages and i < len(chunk_pages) else None
        results.append({"chunk": chunks[i], "page": page, "score": float(score)})
    return results

def format_page_range(pages: list[int]) -> str:
    if not pages:
        return ""
    pages = sorted(set(pages))
    return f"Page {pages[0]}" if len(pages) == 1 else f"Page {pages[0]}-{pages[-1]}"

# ------------------------------------------------------------
# ✅ Gemini streaming answer (replaces stream_ollama_answer)
# ------------------------------------------------------------
def stream_gemini_answer(question: str, context: str):
    max_chars = 6000  # Gemini handles much larger contexts than Ollama
    safe_context = context[:max_chars] if context else "No context available."

    prompt = (
    f"""
You are an intelligent PDF assistant.

Answer the question ONLY using the provided document context.

Rules:
1. Format the answer using Markdown.
2. Use headings (#, ##) to organize sections.
3. Use numbered lists (1., 2., 3.) and bullet points (-).
4. Break long answers into sections and sub-sections.
5. Add blank lines between sections.
6. Keep answers concise and easy to read.
7. Never return one large paragraph.
8. If the answer is not present in the document, reply exactly:
"I cannot find the answer in the document."
9. If the question is unrelated to the document, reply exactly:
"The question is unrelated to the document."
10. If the question is ambiguous, reply exactly:
"The question is ambiguous. Please clarify."

Document Context:
{safe_context}

Question:
{question}

Answer:
"""
    )

    try:
        model = genai.GenerativeModel(
            model_name=GEMINI_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
        )
        response = model.generate_content(prompt, stream=True)
        for chunk in response:
            text = chunk.text
            if text:
                yield text
    except Exception as e:
        yield f"\n[Gemini Error] {e}"

# ------------------------------------------------------------
# File cleanup
# ------------------------------------------------------------
def cleanup_old_files():
    now = time.time()
    for folder in (UPLOAD_FOLDER, VECTOR_STORE_FOLDER):
        try:
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath) and (now - os.path.getmtime(fpath)) > CLEANUP_AFTER_SECONDS:
                    os.remove(fpath)
        except OSError:
            pass

def schedule_cleanup():
    def loop():
        while True:
            time.sleep(3600)
            cleanup_old_files()
    threading.Thread(target=loop, daemon=True).start()

schedule_cleanup()

# ------------------------------------------------------------
# Routes
# ------------------------------------------------------------
@app.route("/health")
def health():
    return {"status": "ok"}

@app.route("/upload", methods=["POST"])
def upload_file():
    if "pdf" not in request.files:
        return jsonify({"error": "Missing file field 'pdf'"}), 400
    file = request.files["pdf"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "Invalid filename"}), 400
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    try:
        file.save(filepath)
    except OSError as e:
        return jsonify({"error": "Failed to save file", "details": str(e)}), 500
    try:
        with open(filepath, "rb") as f:
            if f.read(5) != b"%PDF-":
                os.remove(filepath)
                return jsonify({"error": "Invalid PDF file"}), 400
    except OSError:
        os.remove(filepath)
        return jsonify({"error": "Could not verify file"}), 500
    try:
        page_pairs, pages = extract_pages_text(filepath)
    except Exception as e:
        return jsonify({"error": "Failed to extract text from PDF", "details": str(e)}), 500
    try:
        chunk_items = chunk_pdf_by_page(page_pairs)
        chunk_texts = [x["text"] for x in chunk_items]
        chunk_pages = [x["page"] for x in chunk_items]
        index, _ = create_vector_store(chunk_texts)
        base_name = os.path.splitext(filename)[0]
        faiss.write_index(index, os.path.join(VECTOR_STORE_FOLDER, f"{base_name}.faiss"))
        with open(os.path.join(VECTOR_STORE_FOLDER, f"{base_name}.chunks.json"), "w", encoding="utf-8") as f:
            json.dump({
                "source_pdf": filename, "pages": pages, "chunk_size": CHUNK_SIZE,
                "chunks": chunk_texts, "chunk_pages": chunk_pages,
            }, f, ensure_ascii=False, indent=2)
    except Exception as e:
        traceback.print_exc()
        print("VECTOR STORE ERROR:", repr(e))
        return jsonify({"error": "Failed to build vector store","details": str(e)}),500
    return jsonify({
        "message": "PDF processed successfully", "filename": filename,
        "pages": pages, "chunks_created": len(chunk_texts),
        "text_preview": (chunk_texts[0][:500] if chunk_texts else ""),
    })

_STREAM_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Question is required."}), 400
    try:
        index, chunks, payload = load_latest_vector_store()
    except Exception as e:
        return jsonify({"error": "No vector store available", "details": str(e)}), 400
    try:
        chunk_pages = payload.get("chunk_pages")
        if not isinstance(chunk_pages, list):
            chunk_pages = None
        question_vector = embed_question(question)
        top_matches = search_faiss(index, question_vector, chunks, top_k=TOP_K, chunk_pages=chunk_pages)
        if not top_matches:
            def no_context_stream():
                yield json.dumps({"source": ""}) + "\n"
                yield "I could not find relevant information in the document."
            return Response(stream_with_context(no_context_stream()), mimetype="text/plain; charset=utf-8", headers=_STREAM_HEADERS)
        
        context = "\n\n---\n\n".join(m["chunk"] for m in top_matches)
        pages = [m.get("page") for m in top_matches if isinstance(m.get("page"), int)]
        meta_line = json.dumps({"source": format_page_range(pages)})
        def token_stream():
            yield meta_line + "\n"
            yield from stream_gemini_answer(question, context)
        return Response(stream_with_context(token_stream()), mimetype="text/plain; charset=utf-8", headers=_STREAM_HEADERS)
    except Exception as e:
        return jsonify({"error": "Failed to answer question", "details": str(e)}), 500

if __name__ == "__main__":
    import os
    
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
