# AI Agent

A modular AI agent system with long-term memory, RAG, OCR, translation, and web search capabilities.

## Features

- **Conversational AI** — Local inference via Ollama (`mannix/jan-nano`) with LangChain/LangGraph orchestration
- **Long-Term Memory** — Episodic memory with reflection-based summarization, stored in ChromaDB for semantic retrieval
- **RAG Pipeline** — Vector database (ChromaDB + Ollama embeddings) for retrieving relevant context from stored documents
- **OCR** — Extract text from images and image-based PDFs using Gemini or PaddleOCR
- **Translation** — Translate content to Vietnamese via Google Gemini
- **Web Search** — Real-time web lookup via Tavily API and Perplexity
- **LangGraph Agent** — Persistent stateful agent with tool-calling, memory load/store, and conditional routing
- **Gradio UI** — Chat interface supporting image and PDF file uploads

## Project Structure

```
aiagent/
├── main.py                   # Simple Ollama conversation loop
├── models.py                 # Model factory (Ollama, Gemini, Perplexity)
├── memory.py                 # ChromaDB memory management
├── agent_tools.py            # Tool definitions (vector DB, OCR, translation, search, PDF)
├── agent_rag.py              # LangChain agent (ReAct pattern)
├── agent_rag_langgraph.py    # LangGraph agent with persistent memory
├── mem_sentinel.py           # Episodic memory reflection system
├── utils.py                  # Conversation formatting utilities
├── frontend/
│   └── gradio_ui.py          # Gradio chat interface
├── docs/                     # PDF/image test files
├── chroma_db/                # Vector store persistence
├── output/                   # OCR output directory
└── requirements.txt
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com/) running locally with models like `mannix/jan-nano`, `llama3.2`, `mxbai-embed-large`

## Setup

1. Clone the repo and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure `.env`:
   ```
   HOST=192.168.1.5
   OLLAMA_PORT=11434
   SERVER_ENDPOINT=192.168.1.5:11434
   LOCAL_ENDPOINT=localhost:11434
   EMBEDDING_MODEL=mxbai-embed-large
   CHROMA_PORT=8002
   TAVILY_API_KEY=<your_key>
   GOOGLE_API_KEY=<your_key>
   ```

## Usage

### CLI conversation loop
```bash
python main.py
```

### LangGraph agent (persistent memory)
```bash
python agent_rag_langgraph.py
```

### Gradio UI
```bash
python frontend/gradio_ui.py
```

## Tools

| Tool | Description |
|------|-------------|
| `query_vector_db` | Search the vector store for relevant memories |
| `store_vector_db` | Store documents in the vector database |
| `ocr` | Extract text from an image using Gemini |
| `extract_text_from_pdf_images` | OCR each page of a scanned/image-based PDF |
| `extract_text_from_pdf` | Extract text from a text-based PDF |
| `translate_to_vn` | Translate text to Vietnamese |
| `search_the_web` | Web search via Tavily |
| `write_to_file` | Write content to disk |
