import os
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from typing import List
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import markdown

def load_pdf_files(data):
    data_path = os.path.abspath(f"../{data}")
    # data_path = os.path.abspath(f"{data}")
    loader = DirectoryLoader(
        data_path,
        glob = "*.pdf",
        loader_cls = PyPDFLoader
    )
    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content = doc.page_content,
                metadata = {"source": src}
            )
        )
    return minimal_docs

def clean_minimal_documents(docs):
    cleaned_docs = []
    for doc in docs:
        cleaned_doc = doc.page_content.replace('\t', ' ')
        src = doc.metadata.get("source")
        metadata = {"source": src}
        cleaned_docs.append(
            Document(page_content=cleaned_doc, 
            metadata = {"source": src})
        )
    return cleaned_docs




def text_split(cleaned_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(cleaned_docs)
    return text_chunks



def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(
        model_name = model_name,
    )
    return embeddings


def format_newlines_to_html(text):
    """Replaces newlines with <br> tags and strips Markdown bolding."""
    
    # 1. Handle Newlines: Replace \n with <br>
    # Note: Using <br><br> for a single \n is a common trick to ensure
    # paragraphs get proper spacing.
    text_with_br = text.replace('\n', '<br>')
    
    # 2. Handle Bolding: Replace **text** with <strong>text</strong>
    # This requires a more robust approach, often using a library (see below).
    # A simple regex can work for basic cases but is not recommended for complex Markdown.
    
    return text_with_br

def format_text_to_markdown(text):
    # This single line converts all Markdown (**bold**, *italic*, newlines, lists)
    # to their corresponding HTML tags (<strong>, <em>, <br>, <ul>).
    return markdown.markdown(text)