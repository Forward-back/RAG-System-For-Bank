from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List, Optional
import logging
import os
import chromadb

from src.infra.model_registry import SharedEmbeddings

logger = logging.getLogger(__name__)


class EmbeddingStore:

    def __init__(self):
        self.embeddings = SharedEmbeddings()
        logger.info("Embedding model ready (shared instance)")


    @staticmethod
    def validate_chunks(chunks: List[Document]) -> List[Document]:
        valid = []

        for doc in chunks:
            if not isinstance(doc.page_content, str):
                continue
            if not doc.page_content.strip():
                continue
            valid.append(doc)

        return valid


    def create_or_load_db(
        self,
        chunks: List[Document],
        persist_directory: str = "./chroma_db",
        collection_name: str = "rag_collection",
        rebuild: bool = False,
    ) -> Chroma:

        chunks = self.validate_chunks(chunks)
        db_exists = os.path.exists(persist_directory) and os.listdir(persist_directory)

        if rebuild and db_exists:
            logger.info("Rebuilding vector database...")
            client = chromadb.PersistentClient(path=persist_directory)
            try:
                client.delete_collection(name=collection_name)
            except Exception:
                pass
            db_exists = False

        if db_exists:
            logger.info("Loading existing Chroma database...")
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )

        if len(chunks) == 0:
            logger.info("Creating empty Chroma database (no documents to index)...")
            client = chromadb.PersistentClient(path=persist_directory)
            client.get_or_create_collection(name=collection_name)
            return Chroma(
                persist_directory=persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )

        logger.info("Creating new Chroma database with %d chunks...", len(chunks))
        BATCH_SIZE = 5000
        vectordb = None

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i : i + BATCH_SIZE]
            logger.debug("Inserting batch %d (Size: %d chunks)...", i // BATCH_SIZE + 1, len(batch))

            if vectordb is None:
                vectordb = Chroma.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    persist_directory=persist_directory,
                    collection_name=collection_name,
                )
            else:
                vectordb.add_documents(documents=batch)

        return vectordb
