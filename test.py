from langchain.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInstructEmbeddings

persist_dir = "C:/Users/11941578900/Documents/GitHub/TCC/TCC_TrataDocumentos/ChromaDB"

embeddings = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Carrega o Chroma persistido
vectorstore = Chroma(
    persist_directory=persist_dir,
    embedding_function=embeddings
)

# Quantos documentos estão salvos
all_docs = vectorstore.get(include=["documents", "metadatas"])
print("Total de documentos salvos:", len(all_docs["documents"]))

# Ver os primeiros documentos e metadados
for i, (doc, meta) in enumerate(zip(all_docs["documents"], all_docs["metadatas"])):
    print(f"\nDocumento {i+1}")
    print("Conteúdo:", doc[:200], "...")  # só primeiras 200 chars
    print("Metadados:", meta)
    if i >= 4:  # mostrar só 5 exemplos
        break
