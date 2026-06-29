import base64
import logging
import os
from typing import Any
import fitz
import uuid
from qdrant_client.models import PointStruct, VectorParams, Distance, FilterSelector, Filter, FieldCondition, MatchValue
from app import config

logger = logging.getLogger(__name__)


def index_pdf_pages_in_qdrant(azure_factory: Any, contract_id: str, pdf_bytes: bytes) -> None:
    """Render PDF pages to images, query LLM vision for captions, embed captions, and save to Qdrant."""
    if not azure_factory.qdrant_client:
        return

    vision_deployment = (
        os.getenv("AZURE_OPENAI_DEPLOYMENT_VISION")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_CHAT")
        or os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        or azure_factory.openai_deployment_name
        or "GPT-4o"
    )
    vision_client = azure_factory.get_openai_client(vision_deployment)
    embedding_client = azure_factory.get_openai_client(azure_factory.embedding_deployment)

    if not vision_client or not embedding_client:
        logger.warning("Vision or embedding client not available for multimodal page indexing.")
        return

    client = azure_factory.qdrant_client
    collection_name = "contracts-pages"

    # Ensure collection exists
    try:
        client.get_collection(collection_name)
    except Exception:
        try:
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=config.QDRANT_VECTOR_SIZE, distance=Distance.COSINE),
            )
        except Exception as create_err:
            logger.warning(f"Failed to create Qdrant collection '{collection_name}': {create_err}")
            return

    # Purge old page vectors for this contract before re-indexing (staleness prevention)
    try:
        client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="contract_id", match=MatchValue(value=contract_id)
                        )
                    ]
                )
            ),
        )
    except Exception as e:
        logger.warning(f"Purging old page vectors failed (continuing): {e}")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        points = []

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            # Render page to PNG at 150 DPI for captioning efficiency
            pix = page.get_pixmap(matrix=fitz.Matrix(150 / 72, 150 / 72), alpha=False)
            img_bytes = pix.tobytes("png")
            b64_image = base64.b64encode(img_bytes).decode("utf-8")

            # Vision Prompt
            system_prompt = (
                "You are an expert contract review vision assistant. "
                "Describe the key legal clauses, headers, tables, signatures, or terms visible on this contract page. "
                "Summarize it in a clear, concise paragraph that describes what the page contains."
            )
            user_content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                {"type": "text", "text": "Describe the content of this contract page."}
            ]
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]

            try:
                caption = vision_client.chat_complete_multimodal(
                    messages=messages, max_tokens=300, temperature=0.1
                )
                if not caption or not caption.strip():
                    continue

                # Embed caption
                vector = embedding_client.get_embedding(caption)
                point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{contract_id}_page_{page_idx}"))

                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload={
                            "contract_id": contract_id,
                            "source_page": page_idx + 1,
                            "text": caption,
                            "modality": "image",
                            "agent_id": "multimodal_indexer"
                        }
                    )
                )
            except Exception as page_err:
                logger.warning(f"Failed to index page {page_idx + 1}: {page_err}")

        if points:
            client.upsert(collection_name=collection_name, points=points)
            logger.info(f"Indexed {len(points)} page images in Qdrant '{collection_name}' collection.")
        doc.close()
    except Exception as doc_err:
        logger.error(f"Failed to open PDF document: {doc_err}")
