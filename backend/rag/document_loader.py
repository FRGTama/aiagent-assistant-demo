from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import chardet
import yaml
from bs4 import BeautifulSoup
from PIL import Image
from playwright.sync_api import sync_playwright

from backend.rag.models import Document

logger = logging.getLogger(__name__)

EXTENSION_MAP: dict[str, str] = {
    ".pdf": "pdf",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".gif": "image",
    ".bmp": "image",
    ".tiff": "image",
    ".tif": "image",
    ".webp": "image",
    ".md": "markdown",
    ".markdown": "markdown",
    ".html": "html",
    ".htm": "html",
    ".txt": "text",
    ".text": "text",
    ".json": "json",
    ".docx": "docx",
}

_OCR_TEXT_THRESHOLD = 50


def _detect_encoding(raw_bytes: bytes) -> str:
    result = chardet.detect(raw_bytes)
    return result["encoding"] if result["encoding"] else "utf-8"


def _decode_text(raw_bytes: bytes, encoding: str | None = None) -> str:
    if encoding is None:
        encoding = _detect_encoding(raw_bytes)
    try:
        return raw_bytes.decode(encoding)
    except (UnicodeDecodeError, LookupError):
        return raw_bytes.decode("utf-8", errors="replace")


def _make_metadata(file_path: str, file_type: str, **extra: Any) -> dict[str, Any]:
    return {
        "source": str(Path(file_path).resolve()),
        "file_type": file_type,
        **extra,
    }

def _try_read_bytes(file_path: str) -> bytes | None:
    try:
        return Path(file_path).read_bytes()
    except OSError as exc:
        logger.warning("Failed to read %s: %s", file_path, exc)
        return None

def classify_document(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()
    doc_type = EXTENSION_MAP.get(ext)

    if doc_type == "pdf":
        try:
            with open(file_path, "rb") as f:
                header = f.read(4)
            return "pdf" if header == b"%PDF" else "text"
        except OSError:
            return "text"

    if doc_type in ("html", "htm"):
        try:
            with open(file_path, "rb") as f:
                header = f.read(200)
            text = header.decode("utf-8", errors="ignore").strip().lower()
            if text.startswith(("<!doctype html", "<html", "<?xml")):
                return "html"
            return "text"
        except OSError:
            return "text"

    if doc_type in ("docx",):
        try:
            with open(file_path, "rb") as f:
                header = f.read(2)
            return "docx" if header == b"PK" else "text"
        except OSError:
            return "text"

    if doc_type is not None:
        return doc_type
    # magic byte fallback
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)

        if header[:4] == b"%PDF":
            return "pdf"
        if header[:4] == b"\x89PNG":
            return "image"
        if header[:2] in (b"\xff\xd8",):
            return "image"
        if header[:2] == b"PK":
            return "docx"
        if header[:4] in (b"GIF8",):
            return "image"

        text_start = header.decode("utf-8", errors="ignore").strip().lower()
        if text_start.startswith(("<!doctype", "<html")):
            return "html"

        return "text"
    except OSError:
        return "text"


def load_pdf(file_path: str) -> list[Document]:
    try:
        import pymupdf
    except ImportError:
        raise ImportError("PyMuPDF (pymupdf) is required to load PDFs") from None

    try:
        doc = pymupdf.open(file_path)
    except Exception as exc:
        logger.warning("Failed to open PDF %s: %s", file_path, exc)
        return []

    documents: list[Document] = []
    try:
        for page_num in range(len(doc)):
            try:
                page = doc.load_page(page_num)
                text = page.get_text().strip()

                if len(text) < _OCR_TEXT_THRESHOLD:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                    try:
                        import pytesseract
                        ocr_text = pytesseract.image_to_string(img).strip()
                        if ocr_text:
                            text = ocr_text
                    except ImportError:
                        logger.warning("pytesseract not installed – skipping OCR for page %s", page_num + 1)
                    except Exception as exc:
                        logger.warning("OCR failed for page %s of %s: %s", page_num + 1, file_path, exc)

                documents.append(Document(
                    page_content=text,
                    metadata=_make_metadata(file_path, "pdf", page_number=page_num + 1),
                ))
            except Exception as exc:
                logger.warning("Failed to process page %s of %s: %s", page_num + 1, file_path, exc)
    finally:
        doc.close()

    return documents


def load_image(file_path: str) -> list[Document]:
    raise NotImplementedError("OCR not yet implemented")


def load_markdown(file_path: str, strip_frontmatter: bool = True) -> list[Document]:

    raw_bytes = _try_read_bytes(file_path)
    if raw_bytes is None:
        return []
    
    text = _decode_text(raw_bytes)
    metadata: dict[str, Any] = _make_metadata(file_path, "markdown")

    if strip_frontmatter and text.startswith("---"):
        match = re.match(r"^---\n(.*?)\n(?:---|\.\.\.)\n", text, re.DOTALL)
        if match:
            frontmatter_text = match.group(1)
            try:
                frontmatter = yaml.safe_load(frontmatter_text)
                if isinstance(frontmatter, dict):
                    for k, v in frontmatter.items():
                        metadata[f"frontmatter_{k}"] = v
            except yaml.YAMLError:
                pass
            text = text[match.end():]

    return [Document(page_content=text, metadata=metadata)]


def _fetch_page_text(file_path: str) -> str | None:

    raw_bytes = _try_read_bytes(file_path)
    if raw_bytes is None:
        return None
    return _decode_text(raw_bytes)

def _text_from_soup(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "lxml")
    title = soup.title.string.strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, title


def _load_html_static(file_path: str) -> tuple[str, str]:
    html = _fetch_page_text(file_path)
    if html is None:
        return "", ""
    return _text_from_soup(html)


def _load_html_js(file_path: str) -> tuple[str, str]:
    path = Path(file_path).resolve()
    url = path.as_uri()

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, wait_until="networkidle")
            page.wait_for_timeout(2000)
            html = page.content()
            browser.close()
    except Exception as exc:
        logger.warning("Playwright rendering failed for %s: %s", file_path, exc)
        return "", ""

    return _text_from_soup(html)

def load_html(file_path: str) -> list[Document]:
    
    strategies = [_load_html_static, _load_html_js]

    text = ""
    title = ""

    for loader_fn in strategies:
        text, title = loader_fn(file_path)
        if text:
            break

    metadata: dict[str, Any] = _make_metadata(file_path, "html")
    if title:
        metadata["title"] = title

    return [Document(page_content=text, metadata=metadata)]


def load_text(file_path: str) -> list[Document]:

    raw_bytes = _try_read_bytes(file_path)
    if raw_bytes is None:
        return []
    text = _decode_text(raw_bytes)
    return [Document(
        page_content=text,
        metadata=_make_metadata(file_path, "text"),
    )]


def load_json(file_path: str, text_fields: list[str] | None = None) -> list[Document]:
    if text_fields is None:
        text_fields = ["text", "content", "body"]

    raw_bytes = _try_read_bytes(file_path)
    if raw_bytes is None:
        return []

    content_type = raw_bytes[:1]
    try:
        if content_type == b"{":
            data = json.loads(raw_bytes)
            records = [data]
        elif content_type == b"[":
            data = json.loads(raw_bytes)
            records = list(data) if isinstance(data, list) else [data]
        else:
            data = json.loads(raw_bytes)
            records = data if isinstance(data, list) else [data]
    except json.JSONDecodeError as exc:
        logger.warning("Invalid JSON in %s: %s", file_path, exc)
        return []

    documents: list[Document] = []
    for idx, record in enumerate(records):
        if isinstance(record, dict):
            parts: list[str] = []
            for field in text_fields:
                value = record.get(field)
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(value, list):
                    parts.extend(str(item) for item in value if isinstance(item, str))
            page_content = "\n".join(parts)
        elif isinstance(record, str):
            page_content = record
        else:
            page_content = str(record) if record is not None else ""

        documents.append(Document(
            page_content=page_content,
            metadata=_make_metadata(file_path, "json", record_index=idx),
        ))

    return documents


def load_docx(file_path: str) -> list[Document]:
    try:
        with zipfile.ZipFile(file_path) as z:
            xml_content = z.read("word/document.xml")
    except Exception as exc:
        logger.warning("Failed to read DOCX %s: %s", file_path, exc)
        return []

    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        root = ET.fromstring(xml_content)
    except ET.ParseError as exc:
        logger.warning("Failed to parse DOCX XML in %s: %s", file_path, exc)
        return []

    text_parts: list[str] = []
    for t_elem in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t"):
        if t_elem.text:
            text_parts.append(t_elem.text)

    return [Document(
        page_content="\n".join(text_parts),
        metadata=_make_metadata(file_path, "docx"),
    )]


LOADER_REGISTRY: dict[str, Any] = {
    "pdf": load_pdf,
    "image": load_image,
    "markdown": load_markdown,
    "html": load_html,
    "text": load_text,
    "json": load_json,
    "docx": load_docx,
}


def load_document(file_path: str) -> list[Document]:
    doc_type = classify_document(file_path)
    loader = LOADER_REGISTRY.get(doc_type)

    if loader is None:
        raise ValueError(
            f"Unsupported document type '{doc_type}' for file: {file_path}. "
            f"Supported types: {', '.join(LOADER_REGISTRY)}"
        )

    logger.info("Loading %s as type '%s'", file_path, doc_type)
    try:
        return loader(file_path)
    except Exception as exc:
        logger.error("Loader for '%s' failed on %s: %s", doc_type, file_path, exc)
        return []
