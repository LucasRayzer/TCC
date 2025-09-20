import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_chroma import Chroma

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

# Cria cadeia de conversação
def create_conversation_chain(vectorStore):
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorStore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        chain_type="refine",
        combine_docs_chain_kwargs={
            "question_prompt": REFINE_QUESTION_PROMPT,
            "refine_prompt": REFINE_PROMPT,
        },
        return_source_documents=False,
    )
    return conversation_chain

# Diretório persistente do Chroma
PERSIST_DIR = "C:/Users/11941578900/Documents/GitHub/TCC/TCC_TrataDocumentos/ChromaDB"

# Carrega banco vetorial persistente
def load_vectorStore():
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": DEVICE}
    )
    vectorStore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorStore
