from dotenv import load_dotenv
from pinecone import Pinecone
import os
from typing import List
import torch
from src.helper import download_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import *
from flask import Flask, render_template, jsonify, request
from src.helper import format_newlines_to_html, format_text_to_markdown

app = Flask(__name__)

load_dotenv(override=True)

PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embedding = download_embeddings()

index_name = "stoicmind"
docsearch = PineconeVectorStore.from_existing_index(
    embedding = embedding,
    index_name = index_name
)

retriever = docsearch.as_retriever(search_type="mmr", search_kwargs={"k":10})
llm_gemini = ChatGoogleGenerativeAI(
    model= "gemini-2.5-flash",  # or "gemini-1.5-pro"
    temperature=0.2,
    api_key = GEMINI_API_KEY
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

question_answer_chain = create_stuff_documents_chain(llm_gemini, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

@app.route("/")
def index():
    return render_template("chat.html")


@app.route("/chat", methods=["GET", "POST"])
def chat():
    user_msg = request.form["msg"]
    input = user_msg
    response = rag_chain.invoke({"input": input})
    # response_str = str(response["answer"])
    ai_response = str(response["answer"])
    ai_response = format_newlines_to_html(ai_response)
    ai_response = format_text_to_markdown(ai_response)
    # rendered_html_response = render_template("chat.html", latest_response=ai_response)
    return ai_response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)