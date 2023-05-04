from dataclasses import dataclass
import tiktoken

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter


@dataclass
class EstimatedCost:
    word_count: int
    token_count: int
    estimated_cost: int


def estimate_embedding_cost(
    documents: list[Document],
    separator: str = "\n\n",
    chunk_size: int = 1000,
    chunk_overlap: int = 20,
    encoding: str = "gpt-4",
    cost_per_token: float = 0.0004,
) -> EstimatedCost:
    text_splitter = CharacterTextSplitter(
        separator=separator,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splitted_docs = text_splitter.split_documents(documents=documents)

    enc = tiktoken.encoding_for_model(encoding)

    total_word_count = sum(len(doc.page_content.split()) for doc in splitted_docs)
    total_token_count = sum(len(enc.encode(doc.page_content)) for doc in splitted_docs)

    estimated_cost = total_token_count * cost_per_token / 1000

    return EstimatedCost(
        word_count=total_word_count,
        token_count=total_token_count,
        estimated_cost=estimated_cost,
    )
