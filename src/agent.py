from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        # TODO: store references to store and llm_fn
        self._store = store
        self._llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        # TODO: retrieve chunks, build prompt, call llm_fn
        # 1. retrieve chunks
        results = self._store.search(question, top_k=top_k)

        # 2. build context
        # [FIXED] Use 'content' instead of 'text' to match search results
        context_parts = [r["content"] for r in results]
        context = "\n\n".join(context_parts)

        # 3. build prompt
        prompt = f"""You are a helpful assistant.
Use the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:"""

        # 4. call llm
        return self._llm_fn(prompt)
