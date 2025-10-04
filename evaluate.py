import os
from datasets import Dataset
import pandas as pd
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness, # Métrica que precisa de "ground_truth"
)

from utils.chatbot_Gemini import load_vectorStore, create_conversation_chain, llm as chatbot_llm


questions = [
    # Perguntas do doc 008-2009-cni.pdf (Criação da CPA)
    "Qual a finalidade da Comissão Própria de Avaliação (CPA) da UDESC, segundo a Resolução Nº 008/2009?",
    "Como é constituída a Comissão Própria de Avaliação (CPA)?",
    "Qual a duração do mandato dos membros da CPA e das Comissões Setoriais de Avaliação (CSAs)?",
    "Com que frequência a CPA deve se reunir?",
    "Quais são as atribuições das Comissões Setoriais de Avaliação (CSAs)?",

    # Perguntas do doc 039-2015-cpe.pdf (Avaliação em Segunda Chamada)
    "Qual o prazo para solicitar a segunda chamada de uma avaliação na UDESC?",
    "Em quais situações um aluno pode solicitar a segunda chamada de uma prova?",
    "O que acontece se o requerimento de segunda chamada for deferido?",
    "Quais conteúdos são exigidos na avaliação de segunda chamada?",
    "O que acontece com os casos não previstos na resolução sobre segunda chamada?",

    # Perguntas do doc 029-2009-cni.pdf (Normas de Ocupação Docente)
    "Quais são os regimes de trabalho para professores efetivos na UDESC?",
    "Qual a carga horária mínima de aulas para um professor efetivo com regime de 40 horas semanais?",
    "Em que situações a carga horária de um professor de 40 horas semanais pode ser reduzida para 8 horas semanais de aulas?",
    "Qual é o número máximo de orientações de trabalho de conclusão de curso que um docente pode ter?",
    "Qual a carga horária máxima que um docente pode alocar para projetos de ensino, pesquisa ou ações de extensão?",
]

# Isso é necessário para a métrica 'answer_correctness'.
ground_truths = [
    # Respostas do doc 008-2009-cni.pdf
    ["A CPA tem por finalidade a implementação, coordenação, condução e sistematização do processo de avaliação institucional da UDESC."],
    ["A CPA é constituída pelo Coordenador da Coordenadoria de Avaliação Institucional (como Presidente), quatro representantes docentes, três representantes técnico-administrativos, dois representantes do corpo discente e um representante da sociedade civil organizada."],
    ["Os membros da CPA e das CSAs, com exceção do presidente da CPA, terão um mandato de dois anos, sendo permitida a recondução."],
    ["A CPA deve realizar reuniões ordinárias mensais e reuniões extraordinárias a qualquer tempo, convocadas com no mínimo 48 horas de antecedência."],
    ["As CSAs têm como atribuições: sensibilizar a comunidade acadêmica do seu Centro, desenvolver a auto-avaliação no respectivo Centro, sistematizar informações e analisar resultados, elaborar um relatório da auto-avaliação e encaminhá-lo à CPA, e prestar informações solicitadas pela CPA ou COAI."],

    # Respostas do doc 039-2015-cpe.pdf
    ["O acadêmico tem o prazo de 5 (cinco) dias úteis, contados a partir da data de realização da avaliação, para solicitar a segunda chamada."],
    ["As situações incluem: problema de saúde do aluno ou parente de 1º grau, ter sido vítima de ação involuntária, manobras militares, luto por parentes, convocação judicial, atividades autorizadas pela chefia, direitos por lei, coincidência de horário com outras avaliações, convocação para competições oficiais ou convocação pelo chefe imediato no trabalho."],
    ["Caso o requerimento seja deferido, a Secretaria de Ensino de Graduação e/ou Secretaria do Departamento encaminhará um expediente ao professor da disciplina para informá-lo sobre o deferimento e os demais procedimentos."],
    ["Nas avaliações de segunda chamada, serão exigidos somente os conteúdos referentes à avaliação em questão."],
    ["Os casos omissos na resolução serão resolvidos pelo Chefe do Departamento."],

    # Respostas do doc 029-2009-cni.pdf
    ["Os regimes de trabalho para professores efetivos são: tempo integral (40 horas semanais) e tempo parcial (30, 20 ou 10 horas semanais)."],
    ["Um professor efetivo com regime de trabalho de 40 (quarenta) horas semanais deve alocar uma carga horária mínima de 12 (doze) horas semanais de aulas."],
    ["Permite-se alocar uma carga horária mínima de 8 (oito) horas semanais de aulas somente para professores do corpo docente de programas de pós-graduação stricto sensu recomendados pela CAPES."],
    ["O docente pode ter no máximo 5 (cinco) orientações de trabalho de conclusão de curso."],
    ["A alocação de carga horária para projetos de ensino, de pesquisa e/ou ações de extensão, por docente, não poderá exceder, por semestre, a 50% (cinquenta por cento) da carga horária de seu regime de trabalho."],
]

# gera as respostas e contextos usando seu chatbot
print("Preparando o chatbot para gerar o dataset de avaliação...")
vector_store = load_vectorStore()
conversation_chain = create_conversation_chain(vector_store)

# Lista para armazenar os resultados
evaluation_data = []

print("Gerando respostas para as perguntas de teste...")
for i, question in enumerate(questions):
    # Usa .invoke() para obter a resposta e os documentos fonte
    response = conversation_chain.invoke({"question": question})
    
    # Extraindo os dados necessários para o RAGAS
    answer = response["answer"]
    contexts = [doc.page_content for doc in response["source_documents"]]
    
    # Monta o dicionário com os dados
    data_point = {
        "question": question,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truths[i][0] # Pega o gabarito correspondente
    }
    evaluation_data.append(data_point)
    print(f"  - Pergunta {i+1} processada.")

# Converte a lista de dicionários para um Dataset do Hugging Face
ragas_dataset = Dataset.from_list(evaluation_data)

print("\nConfigurando as métricas do RAGAS...")

#Importa o wrapper do local correto, conforme a documentação oficial.
from ragas.llms.langchain import LangchainLLM

# Cria uma instância do wrapper "embrulhando" o seu modelo Gemini.
ragas_llm = LangchainLLM(llm=chatbot_llm)

# Lista as métricas que queremos usar.
metrics = [
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
]


print("Executando a avaliação com RAGAS... (Isso pode levar alguns minutos)")

# 4. Passamos o dataset, as métricas e o LLM "embrulhado" para a função evaluate.
result = evaluate(
    dataset=ragas_dataset,
    metrics=metrics,
    llm=ragas_llm,
)

print("Avaliação concluída!")

# Converte o resultado para um DataFrame do Pandas para melhor visualização
df_results = result.to_pandas()
print("\n--- RESULTADOS DA AVALIAÇÃO ---")
print(df_results)

# Salva os resultados em um arquivo CSV
df_results.to_csv("ragas_evaluation_results.csv", index=False)
print("\nResultados salvos em 'ragas_evaluation_results.csv'")