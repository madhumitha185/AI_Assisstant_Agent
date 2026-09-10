"""
RAG pipeline: document loading, chunking, embeddings, and vector store management.
"""
import os
from typing import List

from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

import config


def _get_loader(file_path: str):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        return PyPDFLoader(file_path)
    elif ext in (".txt", ".md"):
        return TextLoader(file_path, encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def load_documents(file_paths: List[str]) -> List[Document]:
    """Load raw documents from a list of file paths (pdf, txt, md)."""
    docs: List[Document] = []
    for path in file_paths:
        loader = _get_loader(path)
        loaded = loader.load()
        for d in loaded:
            d.metadata["source"] = os.path.basename(path)
        docs.extend(loaded)
    return docs


def split_documents(documents: List[Document]) -> List[Document]:
    """Chunk documents using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


_embeddings = None


def get_embeddings() -> HuggingFaceEmbeddings:
    """Singleton embedding model. Free, runs locally on CPU, downloaded once."""
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def load_or_create_vectorstore(persist_dir: str = None) -> Chroma:
    """Load an existing Chroma store from disk, or create a fresh (empty) one."""
    persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
    embeddings = get_embeddings()
    vectorstore = Chroma(
        collection_name="study_assistant",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vectorstore


def add_files_to_vectorstore(vectorstore: Chroma, file_paths: List[str]) -> int:
    """Load, chunk, and add new files to an existing vector store. Returns #chunks added."""
    raw_docs = load_documents(file_paths)
    chunks = split_documents(raw_docs)
    if chunks:
        vectorstore.add_documents(chunks)
    return len(chunks)


def get_retriever(vectorstore: Chroma, k: int = None):
    """Build a similarity-search retriever over the vector store."""
    k = k or config.RETRIEVER_K
    return vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": k})
