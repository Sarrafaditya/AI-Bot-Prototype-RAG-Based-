"""
AI Book Prototype — RAG Backend
Embeddings: sentence-transformers (local, free)
Chat: Groq LLaMA 3.3 70B (free tier, India me kaam karta hai)
"""

import os, uuid
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue, PayloadSchemaType
)

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")
QDRANT_URL      = os.getenv("QDRANT_URL")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "book_chunks_v2"
EMBED_DIM       = 384
CHAT_MODEL      = "llama-3.3-70b-versatile"

print("⏳ Loading embedding model...")
embedder     = SentenceTransformer("all-MiniLM-L6-v2")
print("✅ Embedding model loaded!")

groq_client  = Groq(api_key=GROQ_API_KEY)
qdrant       = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=60)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="AI Book Prototype")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# ─── Helpers ──────────────────────────────────────────────────────────────────

def ensure_collection():
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name="book_id",
            field_schema=PayloadSchemaType.KEYWORD,
        )
        print(f"✅ Created collection: {COLLECTION_NAME}")
    else:
        print(f"✅ Collection ready: {COLLECTION_NAME}")


def extract_text_from_pdf(pdf_bytes: bytes) -> list[dict]:
    pages = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    for i, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def chunk_pages(pages: list[dict], book_id: str, book_name: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for page in pages:
        for idx, split in enumerate(splitter.split_text(page["text"])):
            chunks.append({
                "text":      split,
                "book_id":   book_id,
                "book_name": book_name,
                "page":      page["page"],
                "chunk_idx": idx,
            })
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedder.encode(texts, show_progress_bar=True).tolist()


def embed_query(text: str) -> list[float]:
    return embedder.encode([text])[0].tolist()


# ─── Startup ──────────────────────────────────────────────────────────────────

@app.on_event("startup")
def startup():
    ensure_collection()
    print("🚀 AI Book Prototype (Groq) ready!")


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def root():
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("<h1>Server running!</h1>")


class UploadResponse(BaseModel):
    book_id:      str
    book_name:    str
    total_pages:  int
    total_chunks: int
    message:      str


@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    pdf_bytes = await file.read()
    book_id   = str(uuid.uuid4())[:8]
    book_name = file.filename.replace(".pdf", "")

    pages = extract_text_from_pdf(pdf_bytes)
    if not pages:
        raise HTTPException(status_code=422, detail="No readable text found in PDF.")

    chunks     = chunk_pages(pages, book_id, book_name)
    texts      = [c["text"] for c in chunks]

    print(f"📄 {len(chunks)} chunks — embedding shuru...")
    embeddings = embed_texts(texts)
    print(f"✅ Embeddings done!")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text":      chunks[i]["text"],
                "book_id":   book_id,
                "book_name": book_name,
                "page":      chunks[i]["page"],
                "chunk_idx": chunks[i]["chunk_idx"],
            },
        )
        for i in range(len(chunks))
    ]
    batch_size = 100
    for i in range(0, len(points), batch_size):
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points[i:i+batch_size])
        print(f"✅ Batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} uploaded")

    return UploadResponse(
        book_id=book_id,
        book_name=book_name,
        total_pages=len(pages),
        total_chunks=len(chunks),
        message=f"✅ '{book_name}' processed! {len(chunks)} chunks indexed.",
    )


class AskRequest(BaseModel):
    question: str
    book_id:  str
    history:  list[dict] = []


@app.post("/ask")
async def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    query_vector = embed_query(req.question)

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=5,
        query_filter=Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=req.book_id))]
        ),
        with_payload=True,
    )

    if not results:
        raise HTTPException(status_code=404, detail="No relevant content found.")

    context = "\n\n---\n\n".join(
        f"[Chunk {i+1} | Page {r.payload['page']}]\n{r.payload['text']}"
        for i, r in enumerate(results)
    )

    system_prompt = f"""You are an intelligent tutor helping students understand their textbook.

STRICT RULES:
1. Answer ONLY using the context provided below.
2. If the answer is not in the context, say: "I couldn't find this in the book. Try rephrasing."
3. Be clear, concise, and student-friendly.
4. Do NOT mention page numbers in your answer.
5. Never make up information.

BOOK CONTEXT:
{context}"""

    messages = [{"role": "system", "content": system_prompt}]
    for turn in req.history[-6:]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": req.question})

    def generate():
        stream = groq_client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            stream=True,
            temperature=0.2,
            max_tokens=800,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/books")
def list_books():
    results, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=1000,
        with_payload=True,
        with_vectors=False,
    )
    books = {}
    for point in results:
        bid = point.payload.get("book_id")
        if bid and bid not in books:
            books[bid] = {
                "book_id":   bid,
                "book_name": point.payload.get("book_name", "Unknown"),
            }
    return list(books.values())


@app.get("/health")
def health():
    return {"status": "ok", "chat_model": CHAT_MODEL, "embed": "all-MiniLM-L6-v2"}
@app.delete("/books/{book_id}")
def delete_book(book_id: str):
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
        ),
    )
    return {"message": f"Book {book_id} deleted!"}