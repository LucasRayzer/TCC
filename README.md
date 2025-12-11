# 🤖 Assistente Universitário RAG (Retrieval-Augmented Generation)

Este repositório contém o código-fonte para um sistema de Perguntas e Respostas (QA) factual baseado na arquitetura **Retrieval-Augmented Generation (RAG)**. O objetivo é fornecer respostas precisas a perguntas sobre documentos regulatórios/universitários (resoluções), utilizando o conteúdo **estritamente** indexado de arquivos PDF pré-processados.

O projeto utiliza a biblioteca `langchain` para orquestração, suportando dois _backends_ de Large Language Models (LLMs): **Google Gemini (via API)** e **Meta Llama-3.1 (HuggingFace local)**.

## 🌟 Visão Geral

O _workflow_ do projeto é dividido em três etapas principais:

1.  **Pré-processamento e Padronização:** Extração de texto de PDFs (removendo texto riscado, normalizando formatação) e divisão em _chunks_ por artigo/capítulo.
2.  **Indexação (Build FAISS):** Criação de um índice vetorial FAISS a partir dos _chunks_ textuais usando embeddings multilingual-e5-base.
3.  **Chatbot (RAG):** Implementação de uma _chain_ de QA que recupera informações do índice FAISS e as utiliza como contexto para o LLM responder de forma factual.

## 📦 Estrutura do Repositório

| Arquivo                                | Descrição                                                                                                                                                              |
|:---------------------------------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Padronizador_Documentos_Corrigido.py` | Pipeline de extração, limpeza (remoção de texto riscado/alterado), normalização e _chunking_ de PDFs em arquivos Markdown por Artigo/Capítulo.                         |
| `build._faiss.py`                      | Script para criar o índice vetorial FAISS a partir dos arquivos Markdown gerados pelo padronizador, utilizando o modelo de embeddings `intfloat/multilingual-e5-base`. |
| `chatbot_Gemini.py`                    | Implementação da _chain_ RAG usando o modelo **Gemini 2.5 Pro** via API do Google.                                                                                     |
| `chatbot.py`                           | Implementação da _chain_ RAG usando o modelo **Llama 3.1 8B Instruct** rodando localmente (ou em GPU via `device_map="auto"`).                                         |
### Dados e Resultados

| Diretório | Descrição |
|:---|:---|
| `Resultados_Gemini/` | Contém os arquivos CSV gerados com as métricas de avaliação do RAGAS utilizando o modelo Gemini. |
| `Resultados_Llama_8B/` | Contém os arquivos CSV gerados com as métricas de avaliação do RAGAS utilizando o modelo Llama 3.1 8B. |
| `Graficos_Comparativos/` | Pasta de saída onde são salvos os gráficos comparativos (PNG) gerados pelos scripts de visualização. |