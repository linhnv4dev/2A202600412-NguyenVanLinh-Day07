from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot, compute_similarity
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    """
    A vector store for text chunks.

    Tries to use ChromaDB if available; falls back to an in-memory store.
    The embedding_fn parameter allows injection of mock embeddings for tests.
    """

    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
        reset: bool = True,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._use_chroma = False
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0

        try:
            import chromadb  # noqa: F401
            import os

            # TODO: initialize chromadb client + collection
            # [FIXED] Use PersistentClient to save data to disk
            persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_data")
            client = chromadb.PersistentClient(path=persist_dir)
            # [FIXED] Delete collection if reset=True (for tests), otherwise keep data
            if reset:
                try:
                    client.delete_collection(name=collection_name)
                except Exception:
                    pass
            self._collection = client.get_or_create_collection(name=collection_name)
            
            self._use_chroma = True
        except Exception:
            self._use_chroma = False
            self._collection = None

    def _make_record(self, doc: Document) -> dict[str, Any]:
        # TODO: build a normalized stored record for one document
        embedding = self._embedding_fn(doc.content)

        # [FIXED] Ensure metadata is not empty for ChromaDB and includes doc_id
        metadata = doc.metadata.copy() if doc.metadata else {}
        if not metadata.get('doc_id'):
            metadata['doc_id'] = doc.id
        
        return {
            "id": f"{doc.id}_{self._next_index}",
            "text": doc.content,
            "embedding": embedding,
            "metadata": metadata,
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        # TODO: run in-memory similarity search over provided records
        query_embedding = self._embedding_fn(query)

        scored = []
        for r in records:
            score = compute_similarity(query_embedding, r["embedding"])
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        # [FIXED] Return results with 'content' and 'score' keys
        return [
            {"content": r["text"], "score": score, "metadata": r["metadata"]}
            for score, r in scored[:top_k]
        ]

    def add_documents(self, docs: list[Document]) -> None:
        """
        Embed each document's content and store it.

        For ChromaDB: use collection.add(ids=[...], documents=[...], embeddings=[...])
        For in-memory: append dicts to self._store
        """
        # TODO: embed each doc and add to store
        for doc in docs:
            record = self._make_record(doc)

            if self._use_chroma:
                self._collection.add(
                    ids=[record["id"]],
                    documents=[record["text"]],
                    embeddings=[record["embedding"]],
                    metadatas=[record["metadata"]],
                )
            else:
                self._store.append(record)

            self._next_index += 1

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        """
        Find the top_k most similar documents to query.

        For in-memory: compute dot product of query embedding vs all stored embeddings.
        """
        # TODO: embed query, compute similarities, return top_k
        if self._use_chroma:
            results = self._collection.query(
                query_embeddings=[self._embedding_fn(query)],
                n_results=top_k,
                include=["embeddings", "documents", "metadatas", "distances"]
            )

            # [FIXED] Convert distances to scores (1 - distance for cosine)
            output = []
            for i in range(len(results["documents"][0])):
                distance = results.get("distances", [[]])[0][i] if results.get("distances") else 0
                # Convert distance to similarity score (for cosine distance: similarity = 1 - distance)
                score = 1 - distance if distance is not None else 0
                output.append(
                    {
                        "content": results["documents"][0][i],
                        "score": score,
                        "metadata": results["metadatas"][0][i],
                    }
                )
            return output

        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        """Return the total number of stored chunks."""
        # TODO
        if self._use_chroma:
            return self._collection.count()
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        """
        Search with optional metadata pre-filtering.

        First filter stored chunks by metadata_filter, then run similarity search.
        """
        # TODO: filter by metadata, then search among filtered chunks
        if not metadata_filter:
            return self.search(query, top_k)

        filtered = []

        for record in self._store:
            match = True
            for k, v in metadata_filter.items():
                if record["metadata"].get(k) != v:
                    match = False
                    break
            if match:
                filtered.append(record)

        return self._search_records(query, filtered, top_k)

    def delete_document(self, doc_id: str) -> bool:
        """
        Remove all chunks belonging to a document.

        Returns True if any chunks were removed, False otherwise.
        """
        # TODO: remove all stored chunks where metadata['doc_id'] == doc_id
        if self._use_chroma:
            # [FIXED] Delete from ChromaDB
            # First get all documents with this doc_id
            all_items = self._collection.get()
            ids_to_delete = []
            for i, metadata in enumerate(all_items.get("metadatas", [])):
                if metadata.get("doc_id") == doc_id:
                    ids_to_delete.append(all_items["ids"][i])
            
            if ids_to_delete:
                self._collection.delete(ids=ids_to_delete)
                return True
            return False
        
        original_size = len(self._store)

        self._store = [
            r for r in self._store
            if r["metadata"].get("doc_id") != doc_id
        ]

        return len(self._store) != original_size
