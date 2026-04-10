from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from src.agent import KnowledgeBaseAgent
from src.chunking import SentenceChunker
from src.embeddings import (
    EMBEDDING_PROVIDER_ENV,
    LOCAL_EMBEDDING_MODEL,
    OPENAI_EMBEDDING_MODEL,
    LocalEmbedder,
    OpenAIEmbedder,
    _mock_embed,
)
from src.models import Document
from src.store import EmbeddingStore

SAMPLE_FILES = [
    "data/2019-09-02 Giao trinh Triet hoc (Khong chuyen).docx.md"
]


def load_documents_from_files(file_paths: list[str]) -> list[Document]:
    """Load documents from file paths for the manual demo."""
    allowed_extensions = {".md", ".txt"}
    documents: list[Document] = []

    for raw_path in file_paths:
        path = Path(raw_path)

        if path.suffix.lower() not in allowed_extensions:
            print(f"Skipping unsupported file type: {path} (allowed: .md, .txt)")
            continue

        if not path.exists() or not path.is_file():
            print(f"Skipping missing file: {path}")
            continue

        content = path.read_text(encoding="utf-8")
        documents.append(
            Document(
                id=path.stem,
                content=content,
                metadata={"source": str(path), "extension": path.suffix.lower()},
            )
        )

    return documents


def demo_llm(prompt: str) -> str:
    """A simple mock LLM for manual RAG testing."""
    preview = prompt[:400].replace("\n", " ")
    return f"[DEMO LLM] Generated answer from prompt preview: {preview}..."


def openai_llm(prompt: str) -> str:
    """Use OpenAI gpt-4o-mini for RAG."""
    try:
        import os
        from openai import OpenAI
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return "[ERROR] OPENAI_API_KEY not set in .env"
        
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"[ERROR] OpenAI API error: {str(e)}"


def run_manual_demo(question: str | None = None, sample_files: list[str] | None = None) -> int:
    files = sample_files or SAMPLE_FILES
    query = question or "Summarize the key information from the loaded files."

    print("=== Manual File Test ===")
    print("Accepted file types: .md, .txt")
    print("Input file list:")
    for file_path in files:
        print(f"  - {file_path}")

    docs = load_documents_from_files(files)
    if not docs:
        print("\nNo valid input files were loaded.")
        print("Create files matching the sample paths above, then rerun:")
        print("  python3 main.py")
        return 1

    print(f"\nLoaded {len(docs)} documents")
    for doc in docs:
        print(f"  - {doc.id}: {doc.metadata['source']}")

    load_dotenv(override=False)
    provider = os.getenv(EMBEDDING_PROVIDER_ENV, "mock").strip().lower()
    if provider == "local":
        try:
            embedder = LocalEmbedder(model_name=os.getenv("LOCAL_EMBEDDING_MODEL", LOCAL_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    elif provider == "openai":
        try:
            embedder = OpenAIEmbedder(model_name=os.getenv("OPENAI_EMBEDDING_MODEL", OPENAI_EMBEDDING_MODEL))
        except Exception:
            embedder = _mock_embed
    else:
        embedder = _mock_embed

    print(f"\nEmbedding backend: {getattr(embedder, '_backend_name', embedder.__class__.__name__)}")

    # [FIXED] Use different collection name for different embedding providers
    provider_suffix = "mock" if provider == "mock" else getattr(embedder, '_backend_name', provider)
    collection_name = f"manual_test_store_{provider_suffix}"
    
    store = EmbeddingStore(collection_name=collection_name, embedding_fn=embedder)
    
    # [SKIP CHUNKING] Check if data already exists
    collection_size = store.get_collection_size()
    if collection_size == 0:
        print("\n=== Chunking Documents (SentenceChunker) ===")
        chunker = SentenceChunker(max_sentences_per_chunk=3)
        chunked_docs = []
        for doc in docs:
            chunks = chunker.chunk(doc.content)
            print(f"  - {doc.id}: {len(chunks)} chunks")
            for i, chunk in enumerate(chunks):
                chunked_docs.append(
                    Document(
                        id=f"{doc.id}_chunk_{i}",
                        content=chunk,
                        metadata={**doc.metadata, "chunk_index": i, "doc_id": doc.id}
                    )
                )

        print(f"Total chunks created: {len(chunked_docs)}")
        store.add_documents(chunked_docs)
        print(f"Stored {store.get_collection_size()} chunks in EmbeddingStore")
    else:
        print(f"\n[SKIP CHUNKING] Found {collection_size} chunks in existing collection")
        print("Reusing cached data from previous run")
    print("\n=== EmbeddingStore Search Test ===")
    print(f"Query: {query}")
    search_results = store.search(query, top_k=3)
    for index, result in enumerate(search_results, start=1):
        print(f"{index}. score={result['score']:.3f} source={result['metadata'].get('source')} chunk={result['metadata'].get('chunk_index')}")
        print(f"   content preview: {result['content'][:120].replace(chr(10), ' ')}...")

    print("\n=== KnowledgeBaseAgent Test ===")
    # [CHANGED] Use openai_llm instead of demo_llm
    agent = KnowledgeBaseAgent(store=store, llm_fn=openai_llm)
    print(f"Question: {query}")
    print("Agent answer (using gpt-4o-mini):")
    print(agent.answer(query, top_k=3))
    return 0


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else None
    return run_manual_demo(question=question)


if __name__ == "__main__":
    raise SystemExit(main())
