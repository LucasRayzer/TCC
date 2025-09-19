import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
import re
from langchain.prompts import PromptTemplate
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
    repetition_penalty=1.2,  # aumenta penalização
    return_full_text=False,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,  # evita loops
)

# Wrap para LangChain
llm = HuggingFacePipeline(pipeline=text_gen_pipe)

# Prompt estruturado
REFINE_QUESTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "Responda a pergunta com base apenas no contexto fornecido. "
     "Não invente informações. Responda em português do Brasil."),
    ("human", 
     "Contexto:\n{context_str}\n\nPergunta: {question}\n\nResposta:")
])

REFINE_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "Você recebeu uma resposta inicial e novos documentos. "
     "Melhore a resposta somente se os novos documentos trouxerem informação adicional. "
     "Se não trouxerem nada novo, repita a resposta anterior. "
     "Sempre responda em português do Brasil."),
    ("human", 
     "Resposta inicial: {existing_answer}\n\n"
     "Novos documentos:\n{context_str}\n\n"
     "Pergunta: {question}\n\nResposta refinada:")
])
from langchain.prompts import PromptTemplate

MAP_PROMPT = PromptTemplate(
    template=(
        "Você é um assistente especializado em responder perguntas com base em documentos. "
        "Responda somente com informações que aparecem no documento fornecido. "
        "Se o documento não for relevante, apenas diga: 'Sem informação relevante'. "
        "Responda sempre em português do Brasil.\n\n"
        "Documento:\n{context}\n\n"
        "Pergunta: {question}\n\n"
        "Resposta parcial:"
    ),
    input_variables=["context", "question"],
)

REDUCE_PROMPT = PromptTemplate(
    template=(
        "Você receberá várias respostas parciais vindas de diferentes documentos. "
        "Sua tarefa é combinar essas respostas em uma única resposta coesa. "
        "Não invente informações, só use o que aparecer nas respostas parciais. "
        "Responda sempre em português do Brasil.\n\n"
        "Respostas parciais:\n{summaries}\n\n"
        "Pergunta: {question}\n\n"
        "Resposta final:"
    ),
    input_variables=["summaries", "question"],
)

# Ler arquivos em .md
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
        retriever=vectorStore.as_retriever(search_kwargs={"k": 5}),  # pode ajustar k
        memory=memory,
        chain_type="map_reduce",
        combine_docs_chain_kwargs={
        "question_prompt": MAP_PROMPT,   # prompt aplicado a cada doc
        "combine_prompt": REDUCE_PROMPT, # prompt de combinação final
        },
        return_source_documents=False,   # não precisa
    )
    return conversation_chain