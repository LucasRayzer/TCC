import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA


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

llm = HuggingFacePipeline(pipeline=text_gen_pipe)

# Prompt estruturado
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
      "Você é um assistente universitário factual. Use *estritamente* os trechos de contexto fornecidos para responder à pergunta. "
      "Não faça suposições nem use conhecimento externo. "
      "Se a informação não estiver no contexto, responda exatamente: 'Não foi possível encontrar a informação no contexto fornecido.' "
      "Responda em português do Brasil."),
    ("human", 
      "Contexto:\n{context}\n\nPergunta: {question}\n\nResposta Factual:")
])


# Carregar índice FAISS existente
def load_vectorStore():
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": DEVICE}
    )
    vectorStore = FAISS.load_local(
        "faiss_index_reduzido_mapeado2",
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