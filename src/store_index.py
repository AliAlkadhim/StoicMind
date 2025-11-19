from dotenv import load_dotenv
from pinecone import Pinecone
import os
from src.helper import *
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore



load_dotenv()

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY


extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
clean_minimal_docs=clean_minimal_documents(minimal_docs)
text_chunks = text_split(clean_minimal_docs)

############# Embedding ##############
embedding = download_embeddings()
pinecone_client = Pinecone(api_key = PINECONE_API_KEY)

############ Pinecone ################
index_name = "stoicmind"
if not pinecone_client.has_index(index_name):
    pinecone_client.create_index(
        name = index_name,
        dimension = 384, #dimension of the embeddings
        metric = "cosine", #cosine similarity
        spec = ServerlessSpec(cloud = "aws", region = "us-east-1")
    )
index = pinecone_client.Index(index_name)

docsearch = PineconeVectorStore.from_documents(
    documents = text_chunks,
    embedding = embedding,
    index_name = index_name
)


# if __name__=="__main__":
#     extracted_data = load_pdf_files("data")
#     print(extracted_data)