import re
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Caminho do documento
pdf_path = Path("C:/Users/11941578900/Documents/GitHub/TCC/TCC/Documentos/Resoluções UDESC/resol - consuni/2009/029-2009-cni.pdf")

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def get_alteracoes(text):
    # captura diferentes formatos de (Resolução n° XXX/AAAA)
    pattern = r"\(Alterada pela Resolução\s*(?:n[º°]\s*)?(\d+)[/\-](\d{4}).*?\)"
    alteracoes = re.findall(pattern, text)

    arquivos_relacionados = []
    for num, ano in alteracoes:
        arquivos_relacionados.append(f"{num.zfill(3)}-{ano}-cni.pdf")
    return arquivos_relacionados

def normalize_text(text):
    # Remove cabeçalho de alterações
    text = re.sub(r"\(Alterada pela Resolução.*?\)\s*", "", text)
    # Remove vcárias quebras de linha
    text = re.sub(r"\n+", "\n", text)
    # Junta quebras no meio das frases
    text = re.sub(r"(?<!\.)\n(?![A-Z])", " ", text)
    return text.strip()

def chunk_by_artigo(text):
    # Divide por artigos para usar como chunks
    chunks = re.split(r"(Art\.\s*\d+)", text)
    result = []
    #preâmbulo para chunck 0
    if chunks and chunks[0].strip():
        result.append(chunks[0].strip())
    for i in range(1, len(chunks), 2):
        artigo = chunks[i].strip()
        conteudo = chunks[i+1].strip() if i+1 < len(chunks) else ""
        result.append(f"{artigo} {conteudo}")
    return result

def embed_and_store(chunks, index, model, metadata):
    vectors = model.encode(chunks, convert_to_numpy=True)
    index.add(vectors.astype(np.float32))
    return index
def save_chunks_markdown(chunks, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Chunks do arquivo {pdf_path.name}\n\n")
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"## Chunk {i}\n")
            f.write("```\n")
            f.write(chunk.strip())
            f.write("\n```\n\n")

# execuções
text = extract_text(pdf_path)
alteracoes = get_alteracoes(text)
print("Arquivos a ignorar (já consolidados):", alteracoes)

normalized = normalize_text(text)
chunks = chunk_by_artigo(normalized)
# Salva em markdown para validar
output_md = pdf_path.with_suffix(".md")  # mesmo nome do pdf
save_chunks_markdown(chunks, output_md)
print(f"Chunks salvos em: {output_md}")
# Inicializa FAISS + modelo
model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

metadata = {"source": pdf_path.name}
index = embed_and_store(chunks, index, model, metadata)

print(f"Processado {pdf_path.name}, {len(chunks)} chunks enviados para FAISS")
