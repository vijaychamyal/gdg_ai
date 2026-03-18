import os
import re
import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

collection_name = "pdf_knowledge_indu"
vector_size     = 384
batch_size      = 32
chunk_size      = 1000
chunk_overlap   = 200

# symbol removal + text cleaning 
def clean_text(text):
    #remove special characters
    #null bytes remove
    text = re.sub(r'\x00', ' ', text)

    #special symbols remove (U+2500 to U+27FF)
    text = re.sub(r'[\u2500-\u27FF]', ' ', text)

    #general punctuation special chars
    text = re.sub(r'[\u2000-\u206F]', ' ', text)

    #only printable ascii, rest all space
    text = re.sub(r'[^\x20-\x7E]', ' ', text)

    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r' +', ' ', text)
    return text.strip()

#garbage detect after cleaning
def is_garbage_text(text):
    # after removing symbols check whether real words or not
    #first cleann
    cleaned = re.sub(r'[^\x20-\x7E]', ' ', text)
    cleaned = re.sub(r' +', ' ', cleaned).strip()
    words = cleaned.split()

    #skip if less words
    if len(words) < 10:
        return True
    avg_len = sum(len(w) for w in words) / len(words)
    if avg_len < 2.5:
        return True
    return False


def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"pdf not found: {pdf_path}")

    filename = os.path.basename(pdf_path)
    doc      = fitz.open(pdf_path)
    pages    = []
    skipped  = 0
    print(f"Opening '{filename}' — {len(doc)} total pages")

    for page_num in range(len(doc)):
        raw_text = doc[page_num].get_text()
        #empty page skip
        if len(raw_text.strip()) < 30:
            skipped += 1
            continue

        if is_garbage_text(raw_text):
            skipped += 1
            continue

        # clean after removiing symbols
        cleaned = clean_text(raw_text)
        #is it meaningful after cleaning
        if len(cleaned) < 50:
            skipped += 1
            continue

        pages.append({
            "text"    : cleaned,
            "page_num": page_num + 1,
            "source"  : filename
        })
    doc.close()
    print(f"Loaded  : {len(pages)} good pages")
    print(f"Skipped : {skipped} pages")
    return pages

#recursive chunking 
def make_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size     =chunk_size,
        chunk_overlap  =chunk_overlap,
        separators     =["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )

    all_chunks = []
    chunk_id   = 0

    for page in pages:
        chunks = splitter.split_text(page["text"])

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) < 80:
                continue
            all_chunks.append({
                "chunk_text": chunk,
                "page_num"  : page["page_num"],
                "source"    : page["source"],
                "chunk_id"  : chunk_id
            })
            chunk_id += 1

    print(f"Total chunks: {len(all_chunks)}")
    return all_chunks

#model loading 
def load_model():
    print("loading model")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("model ready")
    return model

#batch embeddings
def embed_chunks(chunks, model):
    texts = [chunk["chunk_text"] for chunk in chunks]
    print(f"Embedding {len(texts)} chunks")

    vectors = model.encode(
        texts,
        batch_size          =batch_size,
        show_progress_bar   =True,
        convert_to_numpy    =True,
        normalize_embeddings=True
    )
    print(f"Embeddings done — shape: {vectors.shape}")
    return vectors

#qdrant setup
def setup_qdrant():
    try:
        client = QdrantClient(host="localhost", port=6333)
        client.get_collections()
        print("Qdrant connected (Docker @ localhost:6333)")
        return client
    except Exception as e:
        print("Qdrant is not connected")
        print("First run: docker run -p 6333:6333 qdrant/qdrant")
        raise e

def create_collection(client):
    existing = [c.name for c in client.get_collections().collections]

    if collection_name in existing:
        print(f"Collection '{collection_name}' already exists — skipping.")
        return False
    client.create_collection(
        collection_name=collection_name,
        vectors_config =VectorParams(
            size    =vector_size,
            distance=Distance.COSINE
        )
    )
    print(f"Collection '{collection_name}' created!")
    return True

#points insert
def insert_to_qdrant(chunks, vectors, client):
    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(PointStruct(
            id     = i,
            vector = vector.tolist(),
            payload = {
                "chunk_text": chunk["chunk_text"],
                "page_num"  : chunk["page_num"],
                "source"    : chunk["source"],
                "chunk_id"  : chunk["chunk_id"]
            }
        ))
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    print(f"Inserted {len(points)} points!")

#verify
def verify_insert(client):
    info   = client.get_collection(collection_name)
    sample = client.retrieve(
        collection_name=collection_name,
        ids            =[0],
        with_payload   =True,
        with_vectors   =False
    )
    print(f"\nverification")
    print(f"  points stored : {info.points_count}")
    print(f"  vector size   : {info.config.params.vectors.size}")
    print(f"  chunk_text    : {sample[0].payload['chunk_text'][:200]}")
    print(f"  page_num      : {sample[0].payload['page_num']}")
    print(f"  source        : {sample[0].payload['source']}")

#main pipeline 
def build_knowledge_base(pdf_path):
    client  = setup_qdrant()
    created = create_collection(client)

    if not created:
        print("Already exists delete collection from dashboard:")
        print("http://localhost:6333/dashboard")
        model = load_model()
        return client, model

    pages   = load_pdf(pdf_path)
    chunks  = make_chunks(pages)
    model   = load_model()
    vectors = embed_chunks(chunks, model)
    insert_to_qdrant(chunks, vectors, client)
    verify_insert(client)
    print(f"  pdf     : {os.path.basename(pdf_path)}")
    print(f"  pages   : {len(pages)}")
    print(f"  chunks  : {len(chunks)}")
    print(f"  vectors : {vectors.shape}")

    return client, model


if __name__ == "__main__":
    pdf_path = "GDG Inductions 2026.pdf"
    client, model = build_knowledge_base(pdf_path)
