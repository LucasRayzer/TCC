import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate

# Configuração de device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Carrega modelo e tokenizer
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# Cria pipeline de geração de texto
text_gen_pipe = pipeline(
     "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.1,
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id
)

# Wrap para LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipe)

# Prompt estruturado
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "Você é um assistente especializado em responder perguntas institucionais da UDESC. "
     "Responda SEMPRE em português do Brasil, de forma clara e objetiva. "
     "Baseie-se apenas nos documentos fornecidos. "
     "Se a resposta não estiver nos documentos, diga: "
     "\"Não encontrei informações nos documentos para responder a essa pergunta.\" "
     "Nunca invente respostas e nunca use conhecimento pré-treinado."),
    ("human", 
     "Contexto dos documentos:\n{context}\n\n"
     "Pergunta: {question}\n\nResposta:")
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
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )
    return conversation_chain
