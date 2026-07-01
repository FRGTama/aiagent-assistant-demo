"""Chunking strategies for splitting documents into pieces for embedding and retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import tiktoken

from backend.rag.models import Document

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    chunk_size: int = 512
    chunk_overlap: int = 64  # ponytail: accepted but not used by _split_text — wire in sliding-window merge when retrieval demands it
    separators: list[str] | None = None
    max_chunks: int | None = None
    encoding_name: str = "cl100k_base"
    similarity_threshold: float = 0.7


def _token_len(text: str, encoding: tiktoken.Encoding) -> int:
    return len(encoding.encode(text, disallowed_special=()))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _split_text(
    text: str,
    separators: list[str],
    chunk_size: int,
    encoding: tiktoken.Encoding,
) -> list[str]:
    if not separators or _token_len(text, encoding) <= chunk_size:
        return [text] if text else []

    sep, *rest = separators
    parts = list(text) if not sep else text.split(sep)

    chunks: list[str] = []
    buf = ""
    for part in parts:
        if not part:
            continue
        candidate = f"{buf}{sep}{part}" if buf else part
        if _token_len(candidate, encoding) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
                buf = ""
            if _token_len(part, encoding) > chunk_size:
                chunks.extend(_split_text(part, rest, chunk_size, encoding))
            else:
                buf = part
    if buf:
        chunks.append(buf)

    return chunks


def _split_at_indices(items: list[int], indices: list[int]) -> list[list[int]]:
    if not indices:
        return [items]
    result: list[list[int]] = []
    start = 0
    for idx in sorted(indices):
        if start < idx:
            result.append(items[start:idx])
        start = idx
    if start < len(items):
        result.append(items[start:])
    return result


def recursive_character_split(
    docs: list[Document],
    config: ChunkingConfig,
) -> list[Document]:
    separators = config.separators or ["\n\n", "\n", ".", " "]
    encoding = tiktoken.get_encoding(config.encoding_name)
    chunks: list[Document] = []

    for doc in docs:
        parts = _split_text(doc.page_content, separators, config.chunk_size, encoding)
        for i, text in enumerate(parts):
            if config.max_chunks is not None and len(chunks) >= config.max_chunks:
                break
            chunks.append(Document(
                page_content=text,
                metadata={**doc.metadata, "chunk_index": i, "total_chunks": len(parts)},
            ))

    return chunks


def semantic_splitter(
    docs: list[Document],
    config: ChunkingConfig,
    embedding_fn: Callable[[list[str]], list[list[float]]],
) -> list[Document]:
    encoding = tiktoken.get_encoding(config.encoding_name)
    all_chunks: list[Document] = []

    for doc in docs:
        base_config = ChunkingConfig(
            chunk_size=max(config.chunk_size // 2, 1),
            separators=config.separators,
            encoding_name=config.encoding_name,
        )
        base_chunks = recursive_character_split([doc], base_config)
        if len(base_chunks) < 2:
            all_chunks.extend(base_chunks)
            continue

        texts = [c.page_content for c in base_chunks]
        embeddings = embedding_fn(texts)
        sims = [_cosine_sim(embeddings[i], embeddings[i + 1]) for i in range(len(embeddings) - 1)]

        boundaries = [i + 1 for i, s in enumerate(sims) if s <= config.similarity_threshold]
        seg_indices = _split_at_indices(list(range(len(base_chunks))), boundaries)

        for idx_group in seg_indices:
            merged = "\n\n".join(texts[i] for i in idx_group)
            meta = {
                k: v
                for k, v in base_chunks[idx_group[0]].metadata.items()
                if k not in ("chunk_index", "total_chunks")
            }

            if _token_len(merged, encoding) > config.chunk_size:
                fallback = recursive_character_split(
                    [Document(page_content=merged, metadata=meta)],
                    ChunkingConfig(
                        chunk_size=config.chunk_size,
                        separators=config.separators,
                        encoding_name=config.encoding_name,
                        max_chunks=config.max_chunks,
                    ),
                )
                all_chunks.extend(fallback)
            else:
                all_chunks.append(Document(page_content=merged, metadata=meta))

    for i, chunk in enumerate(all_chunks):
        chunk.metadata["chunk_index"] = i
        chunk.metadata["total_chunks"] = len(all_chunks)

    return all_chunks


CHUNKER_REGISTRY: dict[str, Any] = {
    "recursive": recursive_character_split,
    "semantic": semantic_splitter,
}


def chunk_documents(
    docs: list[Document],
    strategy: str,
    config: ChunkingConfig,
    **kwargs: Any,
) -> list[Document]:
    loader = CHUNKER_REGISTRY.get(strategy)
    if loader is None:
        raise ValueError(
            f"Unsupported chunking strategy '{strategy}'. "
            f"Supported: {', '.join(CHUNKER_REGISTRY)}"
        )
    if strategy == "semantic":
        embedding_fn = kwargs.get("embedding_fn")
        if embedding_fn is None:
            raise ValueError("'semantic' strategy requires 'embedding_fn' in kwargs")
        return loader(docs, config, embedding_fn)
    return loader(docs, config)
