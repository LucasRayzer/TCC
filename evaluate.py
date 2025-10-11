import os
import logging
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)

# --- MUDANÇA PRINCIPAL AQUI ---
# 1. Importamos o chatbot Llama para GERAR as respostas (o "aluno")
from utils.chatbot import load_vectorStore, create_conversation_chain
# 2. Importamos o LLM do Gemini para AVALIAR as respostas (o "juiz")
from utils.chatbot_Gemini import llm as judge_llm
# --------------------------------

# Configuração do Logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)

# Seu Dataset (sem alterações)
questions = [
    # ... (sua lista de perguntas completa aqui) ...
    "Qual a finalidade da Comissão Própria de Avaliação (CPA) da UDESC, segundo a Resolução Nº 008/2009?",
    "Como é constituída a Comissão Própria de Avaliação (CPA)?",
    "Qual a duração do mandato dos membros da CPA e das Comissões Setoriais de Avaliação (CSAs)?",
    "Com que frequência a CPA deve se reunir?",
    "Quais são as atribuições das Comissões Setoriais de Avaliação (CSAs)?",
    "Qual o prazo para solicitar a segunda chamada de uma avaliação na UDESC?",
    "Em quais situações um aluno pode solicitar a segunda chamada de uma prova?",
    "O que acontece se o requerimento de segunda chamada for deferido?",
    "Quais conteúdos são exigidos na avaliação de segunda chamada?",
    "O que acontece com os casos não previstos na resolução sobre segunda chamada?",
    "Quais são os regimes de trabalho para professores efetivos na UDESC?",
    "Qual a carga horária mínima de aulas para um professor efetivo com regime de 40 horas semanais?",
    "Em que situações a carga horária de um professor de 40 horas semanais pode ser reduzida para 8 horas semanais de aulas?",
    "Qual é o número máximo de orientações de trabalho de conclusão de curso que um docente pode ter?",
    "Qual a carga horária máxima que um docente pode alocar para projetos de ensino, pesquisa ou ações de extensão?",
]
ground_truths = [
    # ... (sua lista de respostas completas aqui) ...
    ["A CPA tem por finalidade a implementação, coordenação, condução e sistematização do processo de avaliação institucional da UDESC."],
    ["A CPA é constituída pelo Coordenador da Coordenadoria de Avaliação Institucional (como Presidente), quatro representantes docentes, três representantes técnico-administrativos, dois representantes do corpo discente e um representante da sociedade civil organizada."],
    ["Os membros da CPA e das CSAs, com exceção do presidente da CPA, terão um mandato de dois anos, sendo permitida a recondução."],
    ["A CPA deve realizar reuniões ordinárias mensais e reuniões extraordinárias a qualquer tempo, convocadas com no mínimo 48 horas de antecedência."],
    ["As CSAs têm como atribuições: sensibilizar a comunidade acadêmica do seu Centro, desenvolver a auto-avaliação no respectivo Centro, sistematizar informações e analisar resultados, elaborar um relatório da auto-avaliação e encaminhá-lo à CPA, e prestar informações solicitadas pela CPA ou COAI."],
    ["O acadêmico tem o prazo de 5 (cinco) dias úteis, contados a partir da data de realização da avaliação, para solicitar a segunda chamada."],
    ["As situações incluem: problema de saúde do aluno ou parente de 1º grau, ter sido vítima de ação involuntária, manobras militares, luto por parentes, convocação judicial, atividades autorizadas pela chefia, direitos por lei, coincidência de horário com outras avaliações, convocação para competições oficiais ou convocação pelo chefe imediato no trabalho."],
    ["Caso o requerimento seja deferido, a Secretaria de Ensino de Graduação e/ou Secretaria do Departamento encaminhará um expediente ao professor da disciplina para informá-lo sobre o deferimento e os demais procedimentos."],
    ["Nas avaliações de segunda chamada, serão exigidos somente os conteúdos referentes à avaliação em questão."],
    ["Os casos omissos na resolução serão resolvidos pelo Chefe do Departamento."],
    ["Os regimes de trabalho para professores efetivos são: tempo integral (40 horas semanais) e tempo parcial (30, 20 ou 10 horas semanais)."],
    ["Um professor efetivo com regime de trabalho de 40 (quarenta) horas semanais deve alocar uma carga horária mínima de 12 (doze) horas semanais de aulas."],
    ["Permite-se alocar uma carga horária mínima de 8 (oito) horas semanais de aulas somente para professores do corpo docente de programas de pós-graduação stricto sensu recomendados pela CAPES."],
    ["O docente pode ter no máximo 5 (cinco) orientações de trabalho de conclusão de curso."],
    ["A alocação de carga horária para projetos de ensino, de pesquisa e/ou ações de extensão, por docente, não poderá exceder, por semestre, a 50% (cinquenta por cento) da carga horária de seu regime de trabalho."],
]

# Geração do Dataset (usando o chatbot Llama)
print("Preparando o chatbot Llama para gerar o dataset de avaliação...")
vector_store = load_vectorStore()
conversation_chain = create_conversation_chain(vector_store)

evaluation_data = []
# ... (o resto do seu código de geração de dados permanece o mesmo) ...
print("Gerando respostas para as perguntas de teste...")
for i, question in enumerate(questions):
    try:
        response = conversation_chain.invoke({"question": question})
        answer = response.get("answer", "")
        contexts = [doc.page_content for doc in response.get("source_documents", [])]

        if not answer or not contexts:
            LOGGER.warning(f"Resposta ou contexto vazio para a pergunta: '{question}'. Pulando.")
            continue

        data_point = { "question": question, "answer": answer, "contexts": contexts, "ground_truth": ground_truths[i][0] }
        evaluation_data.append(data_point)
        print(f"  - Pergunta {i+1} processada.")
    except Exception as e:
        LOGGER.error(f"Erro ao gerar resposta para a pergunta '{question}': {e}")

if not evaluation_data:
    raise ValueError("Nenhum dado de avaliação foi gerado. Verifique os logs.")

ragas_dataset = Dataset.from_list(evaluation_data)

# Configuração da Avaliação
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]

from langchain_community.embeddings import HuggingFaceEmbeddings
hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Execução da Avaliação (usando o Gemini como juiz)
print("\nExecutando a avaliação com RAGAS usando Gemini como 'juiz'...")
try:
    result = evaluate(
        dataset=ragas_dataset,
        metrics=metrics,
        llm=judge_llm,  # <-- Usamos o modelo Gemini aqui
        embeddings=hf_embeddings,
        raise_exceptions=True
    )

    print("Avaliação concluída!")
    df_results = result.to_pandas()
    print("\n--- RESULTADOS DA AVALIAÇÃO ---")
    print(df_results)
    df_results.to_csv("ragas_evaluation_results_70B.csv", index=False)
    print("\nResultados salvos em 'ragas_evaluation_results.csv'")

except Exception as e:
    LOGGER.error(f"A AVALIAÇÃO FALHOU! Ocorreu um erro durante 'ragas.evaluate':")
    LOGGER.error(e)