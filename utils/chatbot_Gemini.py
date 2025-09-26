import torch
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env antes de qualquer coisa
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Responda à pergunta com base apenas no contexto fornecido. "
     "Não invente informações. Responda em português do Brasil."),
    ("human",
     "Contexto:\n{context}\n\nPergunta: {question}\n\nResposta:")
])


def load_markdown_chunks(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n##", "\n#", "\n\n", "\n", " "]
    )
    chunks = splitter.split_text(content)
    return chunks
@st.cache_resource 
def load_vectorStore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE}
    )
    vectorStore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorStore

def create_conversation_chain(vectorStore):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(search_kwargs={"k": 8}),
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
        output_key="answer"
    )
    return conversation_chain