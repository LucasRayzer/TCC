import os
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Caminho base onde estão os .md
base_path = Path(r"C:\Users\11941578900\Documents\GitHub\TCC\TCC_TrataDocumentos")

# Device
DEVICE = "cuda"

# Modelo de embeddings
embeddings = HuggingFaceInstructEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={"device": DEVICE}
)

# Carregar e processar todos os .md
all_texts = []
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n##", "\n#", "\n\n", "\n", " "]
)

for md_file in base_path.rglob("*.md"):
    print(f"Processando: {md_file}")
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    chunks = splitter.split_text(content)
    all_texts.extend(chunks)

print(f"Total de chunks: {len(all_texts)}")

# Criar FAISS e salvar
vectorstore = FAISS.from_texts(texts=all_texts, embedding=embeddings)

# Salva o índice FAISS localmente
faiss_path = "faiss_index"
vectorstore.save_local(faiss_path)

# (Opcional) salvar também os textos/chunks em pickle
with open("chunks.pkl", "wb") as f:
    pickle.dump(all_texts, f)

print(f"Índice salvo em {faiss_path}/")
