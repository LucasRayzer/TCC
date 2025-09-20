from pathlib import Path
import shutil
from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from utils import chatbot

# Caminho raiz dos MDs
md_root = Path(r"C:/Users/11941578900/Documents/GitHub/TCC/TCC_TrataDocumentos/Documentos")

# Pasta onde o Chroma vai salvar os dados persistentes
persist_dir = Path(r"C:/Users/11941578900/Documents/GitHub/TCC/TCC_TrataDocumentos/ChromaDB")

# Limpa o diretório para evitar conflitos
if persist_dir.exists():
    shutil.rmtree(persist_dir)
persist_dir.mkdir(parents=True, exist_ok=True)

# Inicializa embeddings
embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Lista para todos os chunks e metadados
all_chunks = []
metadata_list = []

# Percorre todos os arquivos Markdown
for md_path in md_root.rglob("*.md"):
    chunks = chatbot.load_markdown_chunks(md_path)
    print(f"Arquivo {md_path.name} -> {len(chunks)} chunks")

    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        metadata_list.append({
            "arquivo": md_path.name,
            "ano": md_path.parent.name.replace("-md", ""),
            "orgao": md_path.parent.parent.name,
            "chunk_id": i + 1
        })

# Cria o banco vetorial persistente sem passar client_settings como dict
vectorstore = Chroma.from_texts(
    texts=all_chunks,
    embedding=embeddings,
    metadatas=metadata_list,
    persist_directory=str(persist_dir)
)

# Salva no disco
vectorstore.persist()

# Verifica os primeiros documentos
docs = vectorstore.similarity_search("teste", k=5)
for doc in docs:
    print(doc.page_content)
    print(doc.metadata)
