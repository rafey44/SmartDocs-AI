import os
import hashlib
from pathlib import Path

import chromadb
import streamlit as st
import fitz  # PyMuPDF
from google import genai
from google.genai import types

APP_TITLE = "📚 AI RAG Document Chat"
EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-3.5-flash-lite"
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
BASE_DIR = Path(__file__).parent
PRELOADED_DIR = BASE_DIR / "documents"

st.set_page_config(
    page_title="SmartDocs AI",
    page_icon="📚",
    layout="wide"
)

st.title("📚 SmartDocs AI")

st.markdown("""
### 🔐 Cyber Security Fundamentals

This section contains a **Cyber Security Fundamentals** document provided by SmartDocs AI.

You can ask questions related to the topics covered in this document, and the AI will answer using information from the PDF.

**💡 Example questions:**
- What is cybersecurity?
- What is phishing?
- What is malware?
- What is social engineering?
- How does encryption protect data?
- What are common cyber attacks?
""")

st.divider()

st.markdown("""
### 📤 Ask Questions From Your Own PDF

Have your own document? Upload a PDF and ask questions about its content.

Your uploaded PDF will be processed by the RAG system, and you can ask questions based on the information inside it.
""")

def get_api_key():
    try:
        key = st.secrets.get("GEMINI_API_KEY")
        if key:
            return key
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


def get_client():
    api_key = get_api_key()
    if not api_key:
        st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets or your environment variables.")
        st.stop()
    return genai.Client(api_key=api_key)


def extract_pdf_pages(pdf_bytes):
    pages = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                pages.append({"page": page_number, "text": text})
    return pages


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def build_chunks(pdf_items):
    all_chunks, all_metadata = [], []
    for pdf_item in pdf_items:
        filename = pdf_item["name"]
        try:
            pages = extract_pdf_pages(pdf_item["bytes"])
        except Exception as exc:
            st.warning(f"Could not read {filename}: {exc}")
            continue
        for page_info in pages:
            for chunk_number, chunk in enumerate(chunk_text(page_info["text"]), start=1):
                all_chunks.append(chunk)
                all_metadata.append({
                    "source": filename,
                    "page": page_info["page"],
                    "chunk": chunk_number,
                })
    return all_chunks, all_metadata


def embed_texts(client, texts, task_type):
    if not texts:
        return []
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=768,
        ),
    )
    return [embedding.values for embedding in result.embeddings]


def build_vector_database(client, pdf_items):
    chunks, metadata = build_chunks(pdf_items)
    if not chunks:
        return None, 0

    embeddings = embed_texts(client, chunks, "RETRIEVAL_DOCUMENT")
    chroma_client = chromadb.Client()
    collection = chroma_client.get_or_create_collection(name="rag_documents")

    ids = [
        hashlib.md5(
            f"{metadata[i]['source']}-{metadata[i]['page']}-{metadata[i]['chunk']}-{i}".encode()
        ).hexdigest()
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadata,
    )
    return collection, len(chunks)


def retrieve_chunks(client, collection, question, top_k=TOP_K):
    query_embedding = embed_texts(client, [question], "RETRIEVAL_QUERY")[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(documents, metadatas, distances)
    ]


def generate_answer(client, question, retrieved_chunks):
    context = "\n\n".join(
        f"""SOURCE {i}
Document: {item["metadata"]["source"]}
Page: {item["metadata"]["page"]}
Content:
{item["text"]}"""
        for i, item in enumerate(retrieved_chunks, start=1)
    )

    prompt = f"""You are a document question-answering assistant.

Answer the user's question ONLY using the provided retrieved context.

Rules:
1. Do not invent facts.
2. Do not use outside knowledge when the answer is not in the context.
3. If the context does not contain enough information, say:
"I couldn't find enough information in the provided documents."
4. Give a concise, clear answer.
5. You may combine information from multiple sources.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{context}
"""

    interaction = client.interactions.create(model=LLM_MODEL, input=prompt)
    return interaction.output_text


def load_preloaded_pdfs():
    items = []
    if not PRELOADED_DIR.exists():
        return items
    for path in sorted(PRELOADED_DIR.glob("*.pdf")):
        try:
            items.append({"name": path.name, "bytes": path.read_bytes(), "type": "preloaded"})
        except Exception:
            pass
    return items


if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []
if "knowledge_base" not in st.session_state:
    st.session_state.knowledge_base = None
if "indexed_signature" not in st.session_state:
    st.session_state.indexed_signature = None

with st.sidebar:
    st.header("📚 SmartDocs AI")

    st.subheader("🔐 Preloaded Document")

    preloaded = load_preloaded_pdfs()

    if preloaded:
        st.success("Cyber Security Fundamentals is available.")

        for item in preloaded:
            st.write(f"📘 {item['name']}")

        st.caption(
            "Ask questions about Cyber Security Fundamentals "
            "in the main chat."
        )
    else:
        st.warning("No preloaded PDF found.")

    st.divider()

    st.subheader("📤 Upload Your Own PDF")

    st.caption(
        "Upload your own PDF if you want to ask questions "
        "about a different document."
    )

    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or multiple PDF files."
    )

    if uploaded_files:
        st.session_state.uploaded_pdfs = [
            {
                "name": file.name,
                "bytes": file.getvalue(),
                "type": "uploaded"
            }
            for file in uploaded_files
        ]

    if st.session_state.uploaded_pdfs:
        st.write("📗 Your uploaded PDF(s):")

        for item in st.session_state.uploaded_pdfs:
            st.write(f"• {item['name']}")
            all_pdf_items = preloaded + st.session_state.uploaded_pdfs

if not all_pdf_items:
    st.warning("Add at least one PDF to start.")
    st.stop()

signature_source = [
    item["name"] + ":" + hashlib.md5(item["bytes"]).hexdigest()
    for item in all_pdf_items
]
document_signature = "|".join(sorted(signature_source))

client = get_client()

if st.session_state.indexed_signature != document_signature:
    with st.spinner("🔄 Processing documents and building the vector database..."):
        try:
            collection, chunk_count = build_vector_database(client, all_pdf_items)
            st.session_state.knowledge_base = collection
            st.session_state.indexed_signature = document_signature
            st.session_state.chunk_count = chunk_count
        except Exception as exc:
            st.error(f"Could not build the RAG knowledge base: {exc}")
            st.stop()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("PDFs", len(all_pdf_items))
with col2:
    st.metric("Text chunks", st.session_state.get("chunk_count", 0))
with col3:
    st.metric("Retrieved per question", TOP_K)

st.divider()
st.subheader("💬 Ask a question")
question = st.text_area(
    "Ask something about your documents:",
    placeholder="Example: What are the main concepts explained in these documents?",
    height=100,
)
ask_button = st.button("🔍 Ask AI", type="primary", use_container_width=True)

if ask_button:
    if not question.strip():
        st.warning("Please enter a question first.")
        st.stop()

    collection = st.session_state.knowledge_base
    if collection is None or collection.count() == 0:
        st.error("The vector database is empty.")
        st.stop()

    with st.spinner("🔎 Retrieving relevant information..."):
        try:
            retrieved = retrieve_chunks(client, collection, question.strip())
        except Exception as exc:
            st.error(f"Retrieval error: {exc}")
            st.stop()

    with st.spinner("🤖 Generating answer from retrieved context..."):
        try:
            answer = generate_answer(client, question.strip(), retrieved)
        except Exception as exc:
            st.error(f"LLM error: {exc}")
            st.stop()

    st.subheader("🤖 Answer")
    st.write(answer)

    st.divider()
    st.subheader("📚 Sources used")
    for index, item in enumerate(retrieved, start=1):
        metadata = item["metadata"]
        with st.expander(f"Source {index}: {metadata['source']} — Page {metadata['page']}"):
            st.write(item["text"])
            st.caption(
                f"Chunk: {metadata['chunk']} | Vector distance: {item['distance']:.4f}"
            )

with st.expander("🧠 How this RAG app works"):
    st.markdown("""
**1. Extract** → Text is extracted from each PDF page.

**2. Chunk** → Large text is divided into smaller overlapping pieces.

**3. Embed** → Each chunk is converted into a numerical vector.

**4. Store** → Chunks + embeddings + source/page metadata are stored in ChromaDB.

**5. Embed question** → Your question is converted into a vector.

**6. Retrieve** → ChromaDB finds the most similar chunks.

**7. Augment** → The retrieved chunks are placed into the LLM prompt.

**8. Generate** → Gemini answers using that retrieved context.
""")
