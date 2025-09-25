import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Configuração de device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo e tokenizer
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
    device_map="auto"
)

# Cria pipeline de geração de texto
text_gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False,
    repetition_penalty=1.2,
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)

# Wrap para LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipe)

# Prompt estruturado
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "Responda à pergunta com base apenas no contexto fornecido. "
     "Não invente informações. Responda em português do Brasil."),
    ("human", 
     "Contexto:\n{context}\n\nPergunta: {question}\n\nResposta:")
])

# Ler arquivos em .md (mantido caso queira reprocessar algum doc isolado)
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

# Carregar índice FAISS existente
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