from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from tqdm import tqdm
import shutil
import os

from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

# NLTK for sentence tokenization
nltk.download('punkt_tab')

# Paths
UPLOAD_DIR = "uploads"
CHROMA_PATH = "chroma"
STATIC_DIR = "static"

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def get_embedding_function():
    return OllamaEmbeddings(model="nomic-embed-text")


# Semantic Chunking Function
def semantic_chunk(text, embed_model, threshold=0.75):
    sentences = sent_tokenize(text)
    if len(sentences) < 2:
        return [Document(page_content=text)]

    embeddings = embed_model.embed_documents(sentences)
    chunks = []
    current_chunk = sentences[0]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(
            [embeddings[i]], [embeddings[i-1]]
        )[0][0]
        if sim < threshold:
            chunks.append(Document(page_content=current_chunk))
            current_chunk = sentences[i]
        else:
            current_chunk += " " + sentences[i]

    chunks.append(Document(page_content=current_chunk))
    return chunks


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_path = Path(STATIC_DIR) / "index.html"
    return index_path.read_text()


@app.post("/upload")
async def upload_pdf(files: list[UploadFile] = File(...)):
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
    embedder = get_embedding_function()
    all_uploaded = []

    for file in files:
        file_path = Path(UPLOAD_DIR) / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print(f"\nProcessing file: {file.filename}")
        loader = PyPDFLoader(str(file_path))
        pages = loader.load()

        text = "\n".join([p.page_content for p in pages])
        new_chunks = semantic_chunk(text, embedder, threshold=0.75)

        if new_chunks:
            print(f"Adding new semantically split chunks: {len(new_chunks)}")
            new_chunk_ids = [f"{file.filename}:{i}" for i in range(len(new_chunks))]

            for idx, chunk in enumerate(new_chunks):
                chunk.metadata["id"] = new_chunk_ids[idx]

            for i in tqdm(range(0, len(new_chunks), 20), desc=f"Batches for {file.filename}"):
                batch_chunks = new_chunks[i:i+20]
                batch_ids = new_chunk_ids[i:i+20]
                db.add_documents(batch_chunks, ids=batch_ids)

            final_docs = db.get()
            print(f"Final document count in ChromaDB: {len(final_docs['ids'])}")
        else:
            print("No new documents to add")

        all_uploaded.append(file.filename)

    return {"message": "Files uploaded and processed.", "uploaded": all_uploaded}


@app.post("/query")
async def query_api(request: Request):
    data = await request.json()
    query = data.get("question", "")
    print(f"Received query: {query}")

    # Initialize the ChromaDB with embeddings function
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())

    # Perform similarity search to find the most relevant chunks
    results = db.similarity_search_with_score(query, k=5)
    print(f"Found {len(results)} matching results in ChromaDB.")

    # Extract the content of the top results
    context = "\n\n---\n\n".join([doc.page_content for doc, _ in results])

    # Construct the prompt for the LLM (Ollama) to generate a response
    prompt = f"""
    You're a helpful assistant. Read the information below and try to answer the user's question as clearly and
    naturally as possible. If the answer isn't fully clear from the content, give your best possible response based on
    what's there.

    Content:
    {context}

    Question: {query}

    Answer:
    """

    # Initialize the LLM (Ollama) to generate the answer
    model = OllamaLLM(model="mistral")
    response = model.invoke(prompt)
    print(f"Generated response: {response}")

    # Extract the source information from document metadata
    sources = [doc.metadata.get("id") for doc, _ in results]

    # You can map document IDs to more descriptive names if needed
    document_info = {source: {"document_name": source.split(":")[0], "chunk_id": source.split(":")[1]} for source in
                     sources}

    source = [f"{info['document_name']} (Chunk {info['chunk_id']})" for info in document_info.values()]
    print(f"Document Info: {source}")
    # Return the answer along with the document source information
    # Extract unique PDF names
    unique_pdf_names = list(set(info['document_name'] for info in document_info.values()))

# Return only the response and PDF names
    return JSONResponse(content={"response": response, "sources": unique_pdf_names})

    #return JSONResponse(content={"response": response, "sources": source})
