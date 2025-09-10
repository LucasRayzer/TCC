import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate

# Configuração de device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo e tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-7b-it",
    dtype=torch.float16,
    device_map="auto"   # deixa HuggingFace distribuir automaticamente
)

# Cria pipeline de geração de texto
text_gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,   # aumenta a janela de resposta
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
    repetition_penalty=1.1
)

# Wrap para LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipe)

# Prompt estruturado
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "Você é um assistente inteligente especializado em responder perguntas institucionais da UDESC. "
     "Use os documentos como referência principal. "
     "Se não encontrar a resposta neles, use seu conhecimento pré-treinado para dar a melhor resposta possível."),
    ("human", 
     "Contexto dos documentos:\n{context}\n\n"
     "Pergunta: {question}\nResposta:")
])


# Cria vetor store
def create_vectorStore(chunks):
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE}
    )
    vectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorStore


# Cria cadeia de conversação
def create_conversation_chain(vectorStore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversation_chain
