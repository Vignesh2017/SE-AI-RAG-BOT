import hashlib
import os
import tempfile
from pathlib import Path
from typing import List, Sequence

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PDF_FOLDER = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "faiss_index"

DATA_DIR.mkdir(parents=True, exist_ok=True)
PDF_FOLDER.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv()

st.set_page_config(page_title="Multi-PDF RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Multi-PDF RAG Assistant")
st.caption("Ask questions across multiple PDFs and verify every answer against its file and page.")

DEFAULT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

SYSTEM_PROMPT = """You are a careful document question-answering assistant.
Answer using ONLY the supplied PDF excerpts. If the excerpts do not contain enough
information, say you could not find enough evidence in the uploaded documents.
Do not make up facts, filenames, page numbers, quotes, or citations.
Cite supporting evidence inline using [Source: filename, p. N].
When sources disagree, clearly describe the disagreement and cite both sources.
Keep answers clear and useful. The source labels are included in each excerpt.
"""

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Question: {question}\n\nPDF excerpts:\n{context}\n\nAnswer with inline source citations.")
])


def get_api_key() -> str:
    return os.getenv("OPENAI_API_KEY", "").strip()


def file_fingerprint(uploaded_files) -> str:
    digest = hashlib.sha256()
    for f in sorted(uploaded_files, key=lambda x: x.name.lower()):
        digest.update(f.name.encode("utf-8"))
        digest.update(f.getvalue())
    return digest.hexdigest()


def file_fingerprint_from_paths(pdf_paths: Sequence[Path]) -> str:
    digest = hashlib.sha256()
    for pdf_path in sorted(pdf_paths, key=lambda path: path.name.lower()):
        digest.update(pdf_path.name.encode("utf-8"))
        digest.update(pdf_path.read_bytes())
    return digest.hexdigest()


def normalize_page_number(page_doc: Document) -> int:
    raw_page = page_doc.metadata.get("page", 0)
    try:
        return int(raw_page) + 1
    except (TypeError, ValueError):
        return 1


def load_pdf_documents(pdf_paths: Sequence[Path]) -> List[Document]:
    documents: List[Document] = []
    for pdf_path in pdf_paths:
        try:
            pages = PyPDFLoader(str(pdf_path)).load()
        except Exception:
            continue
        for page_doc in pages:
            page_number = normalize_page_number(page_doc)
            page_doc.metadata["source"] = pdf_path.name
            page_doc.metadata["page_number"] = page_number
            page_doc.metadata.pop("file_path", None)
            page_doc.metadata.pop("filename", None)
            documents.append(page_doc)
    return documents


def list_local_pdf_files(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return sorted(
        [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() == ".pdf"],
        key=lambda path: path.name.lower(),
    )


def load_pdf_uploads(uploaded_files) -> List[Document]:
    documents = []
    for uploaded in uploaded_files:
        suffix = Path(uploaded.name).suffix or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getvalue())
            tmp_path = tmp.name
        try:
            pages = PyPDFLoader(tmp_path).load()
            for page_doc in pages:
                page_number = normalize_page_number(page_doc)
                page_doc.metadata["source"] = uploaded.name
                page_doc.metadata["page_number"] = page_number
                page_doc.metadata.pop("file_path", None)
                page_doc.metadata.pop("filename", None)
                documents.append(page_doc)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return documents


def load_saved_vector_store():
    index_file = INDEX_DIR / "index.faiss"
    metadata_file = INDEX_DIR / "index.pkl"

    if not index_file.exists() or not metadata_file.exists():
        return None

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    return FAISS.load_local(
        str(INDEX_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )


def build_vector_store_from_documents(documents: List[Document], chunk_size: int, chunk_overlap: int):
    if not documents:
        raise ValueError("No readable PDF pages were found.")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    for chunk in chunks:
        chunk.metadata["citation"] = (
            f'{chunk.metadata.get("source", "Unknown PDF")}, '
            f'p. {chunk.metadata.get("page_number", "Unknown")}'
        )
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(INDEX_DIR))
    return vector_store, len(documents), len(chunks), sorted({document.metadata["source"] for document in documents})


def build_vector_store(uploaded_files, chunk_size: int, chunk_overlap: int):
    pages = load_pdf_uploads(uploaded_files)
    return build_vector_store_from_documents(pages, chunk_size, chunk_overlap)


def build_vector_store_from_folder(folder: Path, chunk_size: int, chunk_overlap: int):
    pdf_paths = list_local_pdf_files(folder)
    documents = load_pdf_documents(pdf_paths)
    return build_vector_store_from_documents(documents, chunk_size, chunk_overlap)


def format_context(docs: List[Document]) -> str:
    excerpts = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "Unknown PDF")
        page = doc.metadata.get("page_number", "Unknown")
        excerpts.append(
            f"[Excerpt {idx} | Source: {source}, p. {page}]\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(excerpts)


def answer_question(question: str, vector_store, model_name: str, k: int):
    matches = vector_store.similarity_search(question, k=k)
    if not matches:
        return "I couldn't find relevant information in the uploaded PDFs.", []
    context = format_context(matches)
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = PROMPT | llm
    response = chain.invoke({"question": question, "context": context})
    return response.content, matches


with st.sidebar:
    st.header("Configuration")
    model_name = st.text_input("Chat model", value=DEFAULT_MODEL, help="For example: gpt-4o-mini")
    chunk_size = st.slider("Chunk size (characters)", 500, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (characters)", 0, 400, 150, 25)
    top_k = st.slider("Retrieved chunks per question", 2, 10, 5)
    st.divider()
    st.markdown("**API key**")
    if get_api_key():
        st.success("OPENAI_API_KEY detected")
    else:
        st.warning("Set OPENAI_API_KEY in your .env file before indexing PDFs.")

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.index_fingerprint = None
    st.session_state.index_summary = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

folder_pdf_files = list_local_pdf_files(PDF_FOLDER)
initialization_error = None

if st.session_state.vector_store is None:
    if os.getenv("OPENAI_API_KEY"):
        try:
            st.session_state.vector_store = load_saved_vector_store()
            if st.session_state.vector_store is not None:
                st.session_state.index_fingerprint = (
                    file_fingerprint_from_paths(folder_pdf_files) if folder_pdf_files else None
                )
        except Exception as exc:
            initialization_error = f"Could not load saved index: {exc}"

    if st.session_state.vector_store is None and folder_pdf_files:
        try:
            store, page_count, chunk_count, names = build_vector_store_from_folder(
                PDF_FOLDER, chunk_size, chunk_overlap
            )
            st.session_state.vector_store = store
            st.session_state.index_fingerprint = file_fingerprint_from_paths(folder_pdf_files)
            st.session_state.index_summary = (
                f"Auto-loaded {len(names)} PDF(s) from local folder, {page_count} page(s), and "
                f"{chunk_count} text chunk(s)."
            )
        except Exception as exc:
            initialization_error = f"Could not load PDFs from local folder: {exc}"

selected_files = st.session_state.get("pdf_uploads", [])
fingerprint = (
    file_fingerprint(selected_files)
    if selected_files
    else (file_fingerprint_from_paths(folder_pdf_files) if folder_pdf_files else None)
)
index_mismatch = (
    st.session_state.vector_store is not None
    and st.session_state.index_fingerprint is not None
    and fingerprint != st.session_state.index_fingerprint
)

st.subheader("Ask your documents")
if st.session_state.vector_store is None:
    st.info("Upload PDFs or add PDFs to the local folder below, then build the knowledge base.")
elif index_mismatch:
    st.warning("The selected PDFs changed after indexing. Rebuild the knowledge base below before asking questions.")
else:
    for item in st.session_state.chat_history:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("sources"):
                with st.expander("Retrieved source excerpts"):
                    for source in item["sources"]:
                        st.markdown(
                            f"**{source['label']}** · similarity result\n\n"
                            f"> {source['excerpt']}"
                        )

    with st.form("question_form", clear_on_submit=True):
        question = st.text_input("Your question", placeholder="Ask a question about your PDFs...")
        ask_clicked = st.form_submit_button("Ask", type="primary")

    if ask_clicked and question.strip():
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and drafting an evidence-based answer..."):
                try:
                    answer, docs = answer_question(
                        question, st.session_state.vector_store, model_name, top_k
                    )
                    source_items = []
                    seen = set()
                    for doc in docs:
                        label = f'{doc.metadata.get("source", "Unknown PDF")}, p. {doc.metadata.get("page_number", "Unknown")}'
                        key = (label, doc.page_content[:120])
                        if key in seen:
                            continue
                        seen.add(key)
                        source_items.append({
                            "label": label,
                            "excerpt": doc.page_content[:1200] + ("…" if len(doc.page_content) > 1200 else "")
                        })
                    st.markdown(answer)
                    with st.expander("Retrieved source excerpts", expanded=True):
                        for source in source_items:
                            st.markdown(f"**{source['label']}**\n\n> {source['excerpt']}")
                    st.session_state.chat_history.append({
                        "role": "assistant", "content": answer, "sources": source_items
                    })
                except Exception as exc:
                    st.error(f"Could not answer this question: {exc}")
    elif ask_clicked:
        st.warning("Enter a question before clicking Ask.")

if st.session_state.chat_history and st.button("Clear conversation"):
    st.session_state.chat_history = []
    st.rerun()

st.divider()
st.subheader("Manage PDF documents")

if initialization_error:
    st.error(initialization_error)
elif st.session_state.vector_store is not None and not selected_files:
    st.success("Loaded the saved or local-folder FAISS index.")
if st.session_state.index_summary:
    st.info(st.session_state.index_summary)

uploaded_files = st.file_uploader(
    "Upload multiple PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select several PDFs at once. You can select multiple files in the file picker.",
    key="pdf_uploads",
)

if uploaded_files:
    st.write(f"**{len(uploaded_files)} PDF(s) selected**")
    with st.expander("Selected files", expanded=False):
        for file in uploaded_files:
            st.write(f"📄 {file.name} · {len(file.getvalue()) / 1024:.1f} KB")

if folder_pdf_files:
    st.caption(f"Found {len(folder_pdf_files)} PDF(s) in the local folder: {PDF_FOLDER}")

col1, col2 = st.columns([1, 2])
with col1:
    index_clicked = st.button(
        "Build / Rebuild knowledge base",
        type="primary",
        disabled=not uploaded_files,
    )
with col2:
    st.caption("Uploaded PDFs are indexed when you click the button. Local-folder PDFs are loaded automatically when no saved FAISS index is available.")

if index_clicked:
    if not get_api_key():
        st.error("OPENAI_API_KEY is missing. Add it to .env and restart Streamlit.")
    else:
        try:
            with st.spinner("Reading PDFs, splitting text, creating embeddings, and building FAISS index..."):
                store, page_count, chunk_count, names = build_vector_store(
                    uploaded_files, chunk_size, chunk_overlap
                )
            st.session_state.vector_store = store
            st.session_state.index_fingerprint = file_fingerprint(uploaded_files)
            st.session_state.chat_history = []
            st.session_state.index_summary = (
                f"Indexed {len(names)} PDF(s), {page_count} page(s), and "
                f"{chunk_count} text chunk(s)."
            )
            st.success(st.session_state.index_summary)
            st.rerun()
        except Exception as exc:
            st.exception(exc)
