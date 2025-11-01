import re
import pdfplumber
import pdfplumber.utils 
from pathlib import Path
import numpy as np

# Caminho base
base_path = Path(r"C:\Users\11941578900\Documents\GitHub\TCC\TCC_TrataDocumentos\Resoluções-Reduzido")

def find_strikethroughs(page):
    """
    Encontra todos os objetos line e rect na página que
    parecem ser linhas de "strikethrough" (riscado).
    """
    strikethroughs = []
    
    # Encontra linhas horizontais finas
    for line in page.lines:
        if (line['height'] == 0 and 
            line['linewidth'] < 2 and 
            line['linewidth'] > 0.5):
            strikethroughs.append(line)

    for rect in page.rects:
        if (rect['height'] < 2 and rect['height'] > 0.5 and 
            rect['width'] > 2 and (rect['fill'] or rect['non_stroking_color'] is not None)):
            
            strikethroughs.append({
                'x0': rect['x0'],
                'top': rect['top'],
                'x1': rect['x1'],
                'bottom': rect['bottom'], 
                'linewidth': rect['height']
            })
    return strikethroughs

def is_char_struck(char, strikethroughs):
    """
    Verifica se um objeto 'char' se sobrepõe a alguma
    das linhas/retângulos de strikethrough.
    """
  
    for s in strikethroughs:

        ho = (char['x0'] < s['x1'] and char['x1'] > s['x0'])
        
        s_mid_y = (s['top'] + s['bottom']) / 2
        vo = (s_mid_y > char['top'] and s_mid_y < char['bottom'])
        
        if ho and vo:
            return True
    return False


def extract_text(pdf_path):
    """
    Extrai texto do PDF, filtrando qualquer texto
    que esteja coberto por um "strikethrough".
    """
    final_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if not page.chars: 
                continue

            # Encontra todos os elementos gráficos de "risco" na página
            strikethroughs = find_strikethroughs(page)
            
            #se não houver riscos, usa o método padrão rápido
            if not strikethroughs:
                page_text = page.extract_text()
                if page_text:
                    final_text += page_text + "\n"
                continue
            
            # Obtém todos os caracteres da página
            all_chars = page.chars
            
            # Filtra a lista, mantendo apenas os caracteres não riscados
            filtered_chars = [
                c for c in all_chars 
                if not is_char_struck(c, strikethroughs)
            ]
            
            page_text = pdfplumber.utils.extract_text(
                filtered_chars, 
                x_tolerance=3, 
                y_tolerance=3  
            )
            
            if page_text:
                final_text += page_text + "\n"

    return final_text.strip()


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
    if not chunks:  
        return index
    vectors = model.encode(chunks, convert_to_numpy=True)
    vectors = np.atleast_2d(vectors)  # garante corpo (n, d)
    index.add(vectors.astype(np.float32))
    return index


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

print("Processamento concluído.")
