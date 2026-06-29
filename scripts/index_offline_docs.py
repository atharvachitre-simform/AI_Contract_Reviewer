import os
import sys
import hashlib
from pathlib import Path
import fitz  # PyMuPDF
from dotenv import load_dotenv

# Load env variables
load_dotenv()

# Add root folder to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import config
from ai_service.services.azure_clients import AzureClientFactory
from qdrant_client.models import VectorParams, Distance, PointStruct


def index_folder(folder_path: str, collection_name: str, factory: AzureClientFactory):
    path = Path(folder_path)
    if not path.exists():
        print(f"Directory {folder_path} does not exist. Skipping.")
        return

    print(f"Indexing folder: {folder_path} into collection: {collection_name}")
    client = factory.qdrant_client
    if not client:
        print("Qdrant client not initialized. Exiting.")
        return

    # Ensure collection exists
    try:
        client.get_collection(collection_name)
        print(f"Collection '{collection_name}' already exists.")
    except Exception:
        print(f"Creating collection '{collection_name}'...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=config.QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
        )

    # Get embedding client
    embedding_client = factory.get_openai_client(factory.embedding_deployment)
    if not embedding_client:
        print("Embedding client not available. Exiting.")
        return

    pdf_files = list(path.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}.")
        return

    points = []
    point_count = 0

    for pdf in pdf_files:
        print(f"Processing {pdf.name}...")
        try:
            doc = fitz.open(pdf)
            for page_num, page in enumerate(doc, start=1):
                text = page.get_text("text").strip()
                if not text:
                    continue

                # Generate embedding
                vector = embedding_client.get_embedding(text)
                
                # Create point ID
                doc_hash = hashlib.md5(f"{pdf.name}_{page_num}".encode("utf-8")).hexdigest()
                
                payload = {
                    "content": text,
                    "text": text,
                    "file_name": pdf.name,
                    "page_number": page_num,
                    "source_category": collection_name
                }

                points.append(
                    PointStruct(
                        id=doc_hash,
                        vector=vector,
                        payload=payload
                    )
                )
                point_count += 1

                # Batch upsert every 20 points
                if len(points) >= 20:
                    client.upsert(collection_name=collection_name, points=points)
                    points = []
            doc.close()
        except Exception as e:
            print(f"Failed to process {pdf.name}: {e}")

    # Upsert remaining points
    if points:
        client.upsert(collection_name=collection_name, points=points)

    print(f"Finished indexing {folder_path}. Total points upserted: {point_count}")


def main():
    factory = AzureClientFactory()
    
    # Re-create collections fresh to clear any parent-child items
    client = factory.qdrant_client
    if client:
        for col in ["legal_standards", "red_flag_library"]:
            try:
                client.delete_collection(col)
                print(f"Re-creating fresh collection: {col}")
            except Exception:
                pass

    index_folder("data/legal_standards", "legal_standards", factory)
    index_folder("data/red_flag_library", "red_flag_library", factory)


if __name__ == "__main__":
    main()
