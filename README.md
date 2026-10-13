# Multi-PDF RAG Assistant

## Chosen use case

**Enterprise Knowledge Assistant for technical and business documents.** A developer, analyst, or operations user can upload several PDFs—API specifications, architecture documents, SOPs, product manuals, and business process guides—then ask one question across the entire collection. Answers include file and page citations so the user can verify the evidence.

## Features

- Upload multiple PDFs in one session.
- Auto-load PDFs from a local `data/pdfs` folder when no FAISS index exists.
- Extract page-level text and preserve filename + human-readable page number.
- Split text into overlapping chunks.
- Create OpenAI embeddings and store all chunks in one FAISS vector index.
- Persist the FAISS index under `data/faiss_index` for reuse across app restarts.
- Retrieve the most relevant chunks for each question.
- Answer using retrieved context only, with inline `[Source: filename, p. N]` citations.
- Show retrieved excerpts for manual verification.
- Warn when the selected PDF collection changes after indexing.

## Architecture

```text
Multiple PDF uploads
        |
        v
PyPDFLoader (page-level extraction)
        |
        v
RecursiveCharacterTextSplitter
        |
        v
OpenAI Embeddings (text-embedding-3-small)
        |
        v
One FAISS vector index
        |
User question --> Similarity search (top-k chunks)
        |
        v
ChatOpenAI (gpt-4o-mini) + evidence-only prompt
        |
        v
Answer with filename/page citations + expandable excerpts
```

## Requirements

- Python 3.10+ (Python 3.11 is a sensible default)
- An OpenAI API key with API access and billing configured
- Internet access for OpenAI API calls
- Windows, macOS, or Linux

## Run locally (Windows CMD)

Open CMD in this project directory:

```bat
py -3.11 -m venv .venv
.venv\\Scripts\\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
notepad .env
```

Add your real key to `.env`:

```dotenv
OPENAI_API_KEY=sk-your-real-key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Save the file, then run:

```bat
streamlit run app.py
```

Streamlit opens the app in your browser, usually at `http://localhost:8501`.

### PowerShell activation alternative

```powershell
.\\.venv\\Scripts\\Activate.ps1
```

If PowerShell blocks activation, use Windows CMD as shown above or run `.venv\\Scripts\\python.exe -m streamlit run app.py`.

## How to use

1. Place PDFs in the local `data/pdfs` folder if you want the app to auto-load them when no FAISS index exists. The app will scan this folder for `.pdf` files automatically.
2. Alternatively, upload multiple PDF files using the file picker. Select several files in one operation if needed.
3. Choose chunk size, overlap, and number of retrieved chunks in the sidebar.
4. Click **Build / Rebuild knowledge base** and wait for embedding/index creation to finish.
5. Ask questions in the chat box.
6. Check each inline citation and open **Retrieved source excerpts** to inspect the text that was supplied to the model.
7. If you change the selected PDFs or the local folder contents, rebuild the index before asking more questions.

## Source citation design

`PyPDFLoader` typically stores the page index as zero-based metadata. This app converts it to a human-readable page number by adding 1 and stores it as `page_number`. Every split chunk inherits the original PDF's filename and page number. Citations refer to the PDF's physical page order as extracted by the loader; if a PDF has printed page labels that differ from its physical page order, the app currently displays physical page order.

## Important limitations

- **Scanned PDFs:** PDFs containing only page images may need OCR before text can be retrieved. This starter app does not run OCR.
- **Citation accuracy:** The app provides page-level citations from retrieved chunks, not guaranteed sentence-level attribution. Always inspect the displayed excerpts for high-stakes decisions.
- **PDF extraction:** Tables, multi-column layouts, headers, and footers may extract imperfectly.
- **Persistence:** The FAISS index is stored under `data/faiss_index` and can be reused on later app runs.
- **Cost:** Embedding documents and asking questions call OpenAI APIs and may incur charges.
- **Privacy:** Uploaded document text is sent to the configured OpenAI API for embeddings and question answering. Do not upload confidential material unless your organization's policies permit it.
- **No answer found:** The model is instructed to say when the retrieved excerpts are insufficient, but generative systems can still make mistakes.

## Troubleshooting

### `OPENAI_API_KEY is missing`
Make sure `.env` is named exactly `.env`, is in the same folder as `app.py`, and contains a valid key. Restart Streamlit after changing it.

### OpenAI 401 / 403 errors
Check the key, project permissions, organization policy, API billing, and model access. A ChatGPT subscription is not the same as OpenAI API billing.

### FAISS / package installation problems on Windows
Use a supported 64-bit Python version and a fresh virtual environment. If `faiss-cpu` has no compatible wheel for your Python version, use Python 3.11 or 3.12 rather than an experimental/newer Python version.

### No useful answers
Try smaller chunks (e.g. 800–1000 characters), overlap around 100–200 characters, and top-k around 4–6. Confirm the PDFs contain selectable text and that the relevant information is actually present.

## Suggested next steps for a production version

- Persist the FAISS index and add document collection management.
- Add OCR for scanned documents.
- Add reranking and hybrid keyword + vector search.
- Add user authentication, per-user document isolation, audit logs, and rate limits.
- Add page previews and downloadable source reports.
- Add automated tests for page metadata, chunk metadata, retrieval, and citations.
