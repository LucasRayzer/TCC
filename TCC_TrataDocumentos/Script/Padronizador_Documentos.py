import re
import pdfplumber
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd

# Caminho base
base_path = Path(r"C:\Users\11941578900\Documents\GitHub\TCC\TCC_TrataDocumentos\Documentos")

def extract_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
    return text

def get_alteracoes(text):
    pattern = r"\(Alterada pela Resolução\s*(?:n[º°]\s*)?(\d+)[/\-](\d{4}).*?\)"
    alteracoes = re.findall(pattern, text)
    return [f"{num.zfill(3)}-{ano}-cni.pdf" for num, ano in alteracoes]

def normalize_text(text):
    text = re.sub(r"\(Alterada pela Resolução.*?\)\s*", "", text)
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"(?<!\.)\n(?![A-Z])", " ", text)
    return text.strip()

def chunk_by_artigo(text):
    padrao = r"(CAP[IÍ]TULO\s+[IVXLC]+.*?(?=CAP[IÍ]TULO\s+[IVXLC]+|SEÇÃO\s+[IVXLC]+|$))"
    secoes = r"(SEÇÃO\s+[IVXLC]+.*?(?=CAP[IÍ]TULO\s+[IVXLC]+|SEÇÃO\s+[IVXLC]+|$))"
    artigos = r"(Art\.\s*\d+[ºo]?(?:.*?)(?=Art\.\s*\d+[ºo]?|CAP[IÍ]TULO\s+[IVXLC]+|SEÇÃO\s+[IVXLC]+|$))"
    pattern = f"{padrao}|{secoes}|{artigos}"
    matches = re.findall(pattern, text, flags=re.S)

    chunks = []
    buffer = ""
    for grupo in matches:
        chunk = next(filter(None, grupo)).strip()
        if re.match(r"CAP[IÍ]TULO|SEÇÃO", chunk):
            if buffer.strip():
                chunks.append(buffer.strip())
                buffer = ""
            chunks.append(chunk)
        else:
            buffer += "\n\n" + chunk
    if buffer.strip():
        chunks.append(buffer.strip())
    return chunks

def save_chunks_markdown(chunks, output_path, pdf_name):
    output_path.parent.mkdir(parents=True, exist_ok=True)  # cria diretórios necessários
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Chunks do arquivo {pdf_name}\n\n")
        for i, chunk in enumerate(chunks, start=1):
            f.write(f"## Chunk {i}\n")
            f.write("```\n")
            f.write(chunk.strip())
            f.write("\n```\n\n")

def embed_and_store(chunks, index, model):
    if not chunks:  # se não houver chunks, não faz nada
        return index
    vectors = model.encode(chunks, convert_to_numpy=True)
    vectors = np.atleast_2d(vectors)  # garante shape (n, d)
    index.add(vectors.astype(np.float32))
    return index


# Inicializa FAISS + modelo
model = SentenceTransformer("all-MiniLM-L6-v2")
dimension = model.get_sentence_embedding_dimension()
index = faiss.IndexFlatL2(dimension)

# Percorre todos os PDFs
for pdf_path in base_path.rglob("*.pdf"):
    print(f"Processando: {pdf_path}")

    text = extract_text(pdf_path)
    alteracoes = get_alteracoes(text)
    if alteracoes:
        print(f" - Ignorado (já consolidado): {alteracoes}")
        continue

    normalized = normalize_text(text)
    chunks = chunk_by_artigo(normalized)

    # Cria pasta paralela com sufixo -md
    parent_dir = pdf_path.parent
    md_dir = parent_dir.parent / (parent_dir.name + "-md")   
    output_md = md_dir / pdf_path.with_suffix(".md").name

    save_chunks_markdown(chunks, output_md, pdf_path.name)

    # Embeddings
    index = embed_and_store(chunks, index, model)
    print(f" - {len(chunks)} chunks salvos em {output_md}")

# Exporta todos os vetores para CSV único
xb = index.reconstruct_n(0, index.ntotal)
df = pd.DataFrame(xb)
df.to_csv("vetores.csv", index=False)
print(f"\nProcessados todos os PDFs. Total de chunks no FAISS: {index.ntotal}")
