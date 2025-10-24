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
from langchain.chains import RetrievalQA
from langchain_google_genai import HarmBlockThreshold, HarmCategory


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    safety_settings={
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    },
)

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
      "Você é um assistente universitário factual. Use *estritamente* os trechos de contexto fornecidos para responder à pergunta. "
      "Não faça suposições nem use conhecimento externo. "
      "Se a informação não estiver no contexto, responda exatamente: 'Não foi possível encontrar a informação no contexto fornecido.' "
      "Responda em português do Brasil."),
    ("human", 
      "Contexto:\n{context}\n\nPergunta: {question}\n\nResposta Factual:")
])

def load_vectorStore():

    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": DEVICE}
    )
    vectorStore = FAISS.load_local(
        "faiss_index_reduzido_mapeado",
        embeddings,
        allow_dangerous_deserialization=True
    )
    return vectorStore

def create_conversation_chain(vectorStore):
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorStore.as_retriever(search_kwargs={"k": 8}),
        chain_type_kwargs={"prompt": QA_PROMPT},
        return_source_documents=True,
    )
    return qa_chain