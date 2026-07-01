"""
Embedding service abstraction for generating vector representations of text.

Supports multiple providers (Ollama, OpenAI, Gemini) via a common interface,
with caching, batching, and retry logic.
"""


class EmbeddingConfig:
    """
    Configuration for the embedding service.
    Fields:
      - provider: "ollama" | "openai" | "gemini"
      - model: model name (e.g., "mxbai-embed-large", "text-embedding-3-small")
      - dimensions: output vector dimensions (if configurable by model)
      - batch_size: texts per batch (default 32)
      - endpoint: base URL for local providers (Ollama)
      - api_key: API key for cloud providers
    """


def embed_texts(texts: list, config: EmbeddingConfig) -> list:
    """
    Generate embeddings for a list of texts.
    Logic:
      1. Check embedding cache (LRU) for existing vectors by text hash.
      2. Collect uncached texts into batches of config.batch_size.
      3. Dispatch batch to the appropriate provider:
         - Ollama: POST to /api/embed with model & texts.
         - OpenAI: client.embeddings.create().
         - Gemini: embed_content().
      4. Apply retry with exponential backoff on transient failures.
      5. Store results in cache, return list of embedding vectors.
    """


def embed_query(query: str, config: EmbeddingConfig) -> list:
    """
    Generate embedding for a single query string.
    Logic:
      1. Optionally apply query prefix (instruct models need instruction prepended).
      2. Call embed_texts([processed_query], config).
      3. Return single vector.
    """


class EmbeddingCache:
    """
    LRU cache for embedding vectors keyed by text hash.
    Logic:
      1. dict with max_size eviction.
      2. Key = hashlib.sha256(text.encode()).hexdigest().
      3. Thread-safe with a lock.
    """
