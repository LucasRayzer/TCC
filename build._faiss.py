import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Caminho base onde estão os .md
base_path = Path(r"C:\Users\11941578900\Documents\GitHub\TCC\TCC_TrataDocumentos\Resoluções-Reduzido")

# Device
DEVICE = "cuda"

# Modelo de embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base",
    model_kwargs={"device": DEVICE}
)

# Splitter para dividir o texto dos .md
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n##", "\n#", "\n\n", "\n", " "]
)

# Armazena textos e metadados
all_texts = []
metadatas = []

for md_file in base_path.rglob("*.md"):
    print(f"Processando: {md_file}")
    
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Divide o conteúdo do .md em chunks
    chunks = splitter.split_text(content)
    all_texts.extend(chunks)
    
    # Captura o nome base do documento original
    doc_name = md_file.stem.replace(".md", ".pdf")
    
    # Adiciona metadado para cada chunk
    metadatas.extend([{"document_id": doc_name}] * len(chunks))

print(f"Total de chunks: {len(all_texts)}")

# Cria FAISS e salva com metadados
vectorstore = FAISS.from_texts(
    texts=all_texts,
    embedding=embeddings,
    metadatas=metadatas
)

# Salva o índice FAISS localmente
faiss_path = "faiss_index_reduzido_mapeado"
vectorstore.save_local(faiss_path)

# Salva também os textos e metadados juntos em pickle
with open("chunks.pkl", "wb") as f:
    pickle.dump({"texts": all_texts, "metadatas": metadatas}, f)

print(f"Índice salvo em {faiss_path}/ e chunks.pkl criado.")
