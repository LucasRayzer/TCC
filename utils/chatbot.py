import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline

# Configurações 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto" if DEVICE=="cuda" else None)

# Cria pipeline de geração de texto
text_gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if DEVICE=="cuda" else -1,
)

# Wrap para LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipe)

# Funções de embeddings e vetor store
def create_vectorStore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="WhereIsAI/UAE-Large-V1", 
        model_kwargs={"device": DEVICE}
    )
    vectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorStore

def create_conversation_chain(vectorStore):
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(),
        memory=memory
    )

    return conversation_chain
