import re
import pdfplumber
import pdfplumber.utils 
from pathlib import Path
import numpy as np

# Caminho base
base_path = Path(r"C:\Users\11941578900\Documents\GitHub\TCC\TCC_TrataDocumentos\Resoluções-Reduzido")

def find_strikethroughs(page, min_length=20, max_linewidth=4.0):
    """
    Detecta objetos na página que provavelmente são strikethroughs.
    Retorna lista de dicionários com x0,x1,top,bottom,linewidth.
    Mais robusto: detecta tanto 'lines' (com y0/y1) quanto 'rects' (com height).
    """
    strikes = []

    # 1) linhas (page.lines) — podem ter y0 ~= y1 ou small height
    for line in getattr(page, "lines", []):
        # pdfplumber line dict normalmente tem: x0,x1,y0,y1,width (width = linewidth)
        x0 = line.get("x0", line.get("x0", 0))
        x1 = line.get("x1", line.get("x1", 0))
        y0 = line.get("y0", line.get("y0", 0))
        y1 = line.get("y1", line.get("y1", 0))
        linewidth = line.get("width", line.get("linewidth", 0.0))

        # horizontal? y difference small
        if abs(y1 - y0) <= 2.5 and (x1 - x0) >= min_length and 0.1 < linewidth <= max_linewidth:
            top = min(y0, y1)
            bottom = max(y0, y1)
            strikes.append({
                "x0": float(x0),
                "x1": float(x1),
                "top": float(top),
                "bottom": float(bottom),
                "linewidth": float(linewidth)
            })

    # 2) rects — às vezes strikethroughs são retângulos finos ou traços desenhados
    for rect in getattr(page, "rects", []):
        x0 = rect.get("x0", 0)
        x1 = rect.get("x1", 0)
        top = rect.get("top", rect.get("y0", 0))
        bottom = rect.get("bottom", rect.get("y1", top))
        height = rect.get("height", abs(bottom - top) if bottom and top else rect.get("height", 0))
        width = rect.get("width", abs(x1 - x0) if x1 and x0 else rect.get("width", 0))

        # Verifique cores/stroking info: algumas versões não têm 'fill' verdadeiro
        fill = rect.get("fill", False)
        non_stroking = rect.get("non_stroking_color", None)
        stroking = rect.get("stroking_color", None)

        if height is None:
            # fallback: calc pela top/bottom
            height = abs(bottom - top)

        # é um retângulo muito fino e longo (possível strikethrough)
        if 0.3 < height <= 4.0 and width >= min_length:
            # se tiver alguma cor ou stroke, provável que seja riscado
            if fill or non_stroking is not None or stroking is not None or height <= 3.0:
                strikes.append({
                    "x0": float(x0),
                    "x1": float(x1),
                    "top": float(min(top, bottom)),
                    "bottom": float(max(top, bottom)),
                    "linewidth": float(height)
                })

    # 3) às vezes objetos 'lines' também aparecem em page.objects ou page.edges — verificar se presente
    # (mantive simples por enquanto)

    # Opcional: mesclar strikes muito próximos (para evitar múltiplos strikes quase idênticos)
    if strikes:
        strikes = _merge_close_strikes(strikes)

    return strikes

def _merge_close_strikes(strikes, y_tol=1.5, x_tol=1.5):
    """
    Mescla strikes que se sobrepõem ou estão muito próximos (evita duplicatas).
    """
    merged = []
    strikes = sorted(strikes, key=lambda s: (s['top'], s['x0']))
    for s in strikes:
        if not merged:
            merged.append(dict(s))
            continue
        last = merged[-1]
        # se overlap vertical e horizontal ou muito próximo, expande last
        vert_overlap = not (s['bottom'] < last['top'] - y_tol or s['top'] > last['bottom'] + y_tol)
        horiz_overlap = not (s['x1'] < last['x0'] - x_tol or s['x0'] > last['x1'] + x_tol)
        if vert_overlap and horiz_overlap:
            last['x0'] = min(last['x0'], s['x0'])
            last['x1'] = max(last['x1'], s['x1'])
            last['top'] = min(last['top'], s['top'])
            last['bottom'] = max(last['bottom'], s['bottom'])
            last['linewidth'] = max(last['linewidth'], s.get('linewidth', 0))
        else:
            merged.append(dict(s))
    return merged

def is_char_struck(char, strikethroughs, x_pad=1.0, y_pad=1.0):
    """
    Retorna True se o char (dicionário com x0,x1,top,bottom) for coberto por algum strikethrough.
    Usa margens horizontais e verticais (x_pad, y_pad) para tolerância.
    """
    cx0 = float(char.get("x0", 0))
    cx1 = float(char.get("x1", 0))
    ctop = float(char.get("top", char.get("y0", 0)))
    cbottom = float(char.get("bottom", char.get("y1", 0)))

    for s in strikethroughs:
        # sobreposição horizontal
        ho = (cx1 + x_pad > s['x0'] and cx0 - x_pad < s['x1'])
        # considere a linha média vertical do strike
        s_mid = (s['top'] + s['bottom']) / 2.0
        # caractere é verticalmente atravessado pela linha (com tolerância)
        vo = (s_mid + y_pad > ctop and s_mid - y_pad < cbottom)

        if ho and vo:
            return True
    return False


def extract_text(pdf_path):
    """
    Extrai texto do PDF, removendo caracteres cobertos por strikes detectados.
    """
    final_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            if not page.chars:
                continue

            # Detecta strikes
            strikethroughs = find_strikethroughs(page)

            if not strikethroughs:
                page_text = page.extract_text()
                if page_text:
                    final_text += page_text + "\n"
                continue

            all_chars = page.chars

            # Filtra caracteres riscados
            filtered_chars = [c for c in all_chars if not is_char_struck(c, strikethroughs)]

            # extrai texto a partir dos chars filtrados
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
