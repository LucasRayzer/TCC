from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import numpy as np
import torch
import os

MODEL_NAME = "intfloat/multilingual-e5-base"
INDEX_PATH = "faiss_index_reduzido_mapeado" 
QUERY = "Quantas vagas do vestibular da UDESC são reservadas para quem estudou todo o ensino médio em escola pública?"
TOP_K = 15

print("\nTeste de consistência dos embeddings")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
emb = HuggingFaceEmbeddings(model_name=MODEL_NAME, model_kwargs={"device": DEVICE})

# Gera embedding de exemplo
query_emb = emb.embed_query(QUERY)
print("Dimensão do embedding de consulta:", len(query_emb))
print("Primeiras 6 dimensões:", np.array(query_emb)[:6])

# Verifica se existe índice FAISS
if not os.path.exists(INDEX_PATH):
    raise FileNotFoundError(f"Índice FAISS não encontrado em: {INDEX_PATH}")

# Carrega o índice FAISS existente
vectorStore = FAISS.load_local(INDEX_PATH, emb, allow_dangerous_deserialization=True)

# Pega a dimensão interna do índice
faiss_index = vectorStore.index
index_dim = faiss_index.d
print("Dimensão esperada pelo FAISS:", index_dim)

if len(query_emb) != index_dim:
    print("Dimensão inconsistente.")
else:
    print("Dimensões consistentes entre query e índice.")


print("\nTeste de Recuperação e Similaridade")

retriever = vectorStore.as_retriever(search_kwargs={"k": TOP_K})
docs = retriever.get_relevant_documents(QUERY)

if not docs:
    print("Nenhum documento retornado pelo retriever.")
else:
    print(f"Top-{TOP_K} documentos retornados:\n")
    for i, d in enumerate(docs):
        doc_name = d.metadata.get("document_id", "Desconhecido")
        print(f"Doc {i+1} | {doc_name} (len {len(d.page_content)}):")
        print(d.page_content[:400].replace("\n", " "))
        print()

    # Calcula similaridade coseno entre query e cada doc
    q_vec = np.array(query_emb)
    from numpy.linalg import norm
    print("Similaridades coseno:")
    for i, doc in enumerate(docs):
        d_vec = np.array(emb.embed_documents([doc.page_content])[0])
        cos_sim = float(np.dot(q_vec, d_vec) / (norm(q_vec) * norm(d_vec)))
        doc_name = doc.metadata.get("document_id", "Desconhecido")
        print(f"Doc {i+1} | {doc_name}: cos = {cos_sim:.4f}")

    print("\nInterpretação:")
    print("Similaridade < 0.4  -> provavelmente irrelevante")
    print("0.4–0.6 -> possivelmente relacionada")
    print("> 0.6 -> fortemente relevante ao contexto")
