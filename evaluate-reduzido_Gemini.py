import sys
import asyncio
import grpc
import traceback
import platform

if platform.system() == "Windows":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def ignore_grpc_shutdown_error(loop, context):
    msg = context.get("message", "")
    if "POLLER" not in msg:
        loop.default_exception_handler(context)

try:
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(ignore_grpc_shutdown_error)
except Exception:
    pass

import os
import logging
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    answer_correctness,
)
from ragas.run_config import RunConfig

# o chatbot gemini para GERAR as respostas (o "aluno")
from utils.chatbot_Gemini import load_vectorStore, create_conversation_chain
#  o LLM do Gemini para AVALIAR as respostas (o "juiz")
from utils.chatbot_Gemini import llm as judge_llm


# Configuração do Logging
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


questions = [
    "Quem precisa assinar a autodeclaração dizendo que pertence ao grupo racial negro no vestibular da UDESC?",
    "Quem pode pedir o regime especial de atendimento domiciliar na UDESC?",
    "Que tipos de atividades podem contar como Atividades Complementares na UDESC?",
    "Como o aluno valida uma atividade de extensão na UDESC?",
    "Qual é a média mínima para um aluno ser aprovado na UDESC?",
    "Qual é o valor de um crédito nas disciplinas da graduação?",
    "Quantas disciplinas isoladas podem ser cursadas por semestre na UDESC?",
    "Quais disciplinas podem ser oferecidas em caráter de Estudo Dirigido?",
    "Quais condições o aluno deve atender para se inscrever no Exame de Suficiência?",
    "Como funciona a avaliação do extraordinário aproveitamento?",
    "É permitido ao aluno concluir o curso em menos tempo que o mínimo previsto?",
    "É permitido ao aluno da UDESC matricular-se em dois ou mais cursos de graduação ao mesmo tempo?",
    "É possível desmatricular-se de disciplinas fora do período do calendário acadêmico?",
    "O que acontece com as disciplinas do currículo em extinção que não têm equivalência na nova matriz curricular?",
    "Quem pode solicitar a inclusão do nome social nos registros acadêmicos da UDESC?",
    "Quem está habilitado a participar da solenidade de Outorga de Grau?",
    "Quantas avaliações mínimas o professor deve aplicar por disciplina?",
    "Quem pode solicitar a segunda chamada de uma avaliação?",
    "Como funciona o abono de faltas por crença religiosa?",
    "Quais normas a UDESC adota nos processos de revalidação e reconhecimento de diplomas estrangeiros?",
    "Qual é o prazo para solicitar a revisão de nota após a divulgação do resultado da avaliação?",
    "Quantos alunos são necessários para a abertura de uma turma de disciplina optativa?",
    "Quais são as modalidades de ingresso para ocupação de vagas ociosas nos cursos de graduação da UDESC?",
    "Qual é o critério de equivalência mínima para que a disciplina seja validada?"
]

ground_truths = [
    ["Os candidatos classificados nas vagas destinadas a pessoas negras precisam assinar a autodeclaração."],
    ["Podem solicitar estudantes em licença maternidade, em situações excepcionais com atestado médico, ou acadêmicos com condições de saúde que impeçam a frequência às aulas, como tratamentos, infecções, traumatismos ou outras afecções comprovadas por profissional de saúde."],
    ["Podem ser atividades de ensino, pesquisa, extensão, administração universitária ou mistas que integrem teoria e prática, conforme previsto na regulamentação."],
    ["Entregando o formulário assinado pelo coordenador da ação à Coordenação de UCE, que registra os créditos."],
    ["Média 7,0 e frequência mínima de 75%."],
    ["Cada crédito corresponde a 18 horas de atividades."],
    ["Cada solicitante pode se matricular em no máximo duas disciplinas isoladas por semestre, salvo exceções previstas, como disciplinas orientadas ou casos de alunos estrangeiros que necessitam validar o diploma."],
    ["Podem ser oferecidas disciplinas da matriz curricular da UDESC que foram extintas, estão em extinção ou possuem equivalência parcial com as novas matrizes."],
    ["O aluno deve cumprir pré-requisitos, não ter sido reprovado, não ter feito exame antes, não ter faltado previamente e apresentar documentos que comprovem conhecimento ou habilidade."],
    ["A avaliação é feita por Banca Examinadora de 3 professores, com provas sobre todo o conteúdo da disciplina; a nota é a média aritmética, e mínimo 9,0 garante aprovação sem exame final."],
    ["Não, a conclusão do curso não pode ser inferior ao prazo mínimo estabelecido para integralização do currículo."],
    ["Não, é vedada a matrícula e a frequência simultânea em dois ou mais cursos."],
    ["Sim, até 15 dias antes do encerramento do período letivo, mediante solicitação justificada e comprovada pelo Sistema Acadêmico, a ser analisada pela Chefia do Departamento."],
    ["Essas disciplinas devem permanecer no histórico escolar do(a) acadêmico(a) e podem ser oferecidas conforme as normativas da universidade, se necessário."],
    ["Pessoas transgênero, travestis e transexuais podem requerer a inclusão do nome social, sendo que acadêmicos maiores de 18 anos podem solicitar a qualquer tempo, enquanto menores precisam de autorização por escrito dos pais ou responsáveis legais."],
    ["Somente o aluno que concluiu o currículo completo do curso, incluindo estágios e/ou Trabalho de Conclusão de Curso (TCC), conforme aprovação do colegiado. Alunos que concluíram apenas uma nova habilitação do curso não participam da solenidade."],
    ["No mínimo duas avaliações por semestre."],
    ["O acadêmico regularmente matriculado que deixou de comparecer à avaliação nas datas fixadas pelo professor, mediante requerimento assinado e entregue na Secretaria de Ensino de Graduação ou Secretaria do Departamento."],
    ["Acadêmicos que, por força de crença religiosa, deixarem de comparecer às aulas sextas-feiras após 18h até o pôr do sol de sábado, têm essas faltas registradas como dispensa. O acadêmico deve comprovar a participação na congregação através de atestados com firma reconhecida. Essas faltas não contam para o cálculo de"],
    ["A UDESC adota a Resolução nº 3, de 22 de junho de 2016, da Câmara de Educação Superior do CNE/MEC, e a Portaria Normativa nº 22, de 13 de dezembro de 2016, do Ministro de Estado da Educação."],
    ["O aluno deve apresentar a solicitação à Secretaria Acadêmica do Centro em até 10 dias após a publicação do resultado da avaliação."],
    ["O número mínimo de acadêmicos necessários para a realização de cada turma/disciplina optativa é igual a 10 (dez)."],
    ["As modalidades de ingresso para ocupação de vagas ociosas são: Transferência Interna, Transferência Externa, Reingresso após Abandono, Reingresso após Cancelamento de Matrícula pelo(a) estudante, e Retorno ao Portador de Diploma de Graduação."],
    ["O programa da disciplina cursada deve corresponder a, no mínimo, 75% do conteúdo e da carga horária da disciplina que o(a) acadêmico(a) deveria cumprir na UDESC."]
]

async def main():
    print("Preparando o chatbot GEMINI para gerar o dataset de avaliação...")
    vector_store = load_vectorStore()
    conversation_chain = create_conversation_chain(vector_store)

    evaluation_data = []

    print("Gerando respostas para as perguntas de teste (em paralelo)...")

    tasks = []
    for question in questions:
        tasks.append(conversation_chain.ainvoke({"query": question}))

    try:
        responses = await asyncio.gather(*tasks)
        print("Respostas geradas. Processando...")

        for i, response in enumerate(responses):
            question = questions[i] 
            try: 
                answer = response.get("result", "") 
                source_docs = response.get("source_documents", [])
                contexts = [doc.page_content for doc in response.get("source_documents", [])]

                if not answer or not contexts:
                    LOGGER.warning(f"Resposta ou contexto vazio para a pergunta: '{question}'. Pulando.")
                    continue

                # Coleta todos os documentos consultados
                doc_names = []
                for doc in source_docs:
                    doc_id = doc.metadata.get("document_id")
                    if doc_id and doc_id not in doc_names:
                        doc_names.append(doc_id)

                doc_names_str = ", ".join(doc_names) if doc_names else "Desconhecido"

                data_point = {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "ground_truth": ground_truths[i][0], 
                    "document_id": doc_names_str,
                }
                evaluation_data.append(data_point)
                print(f"   - Pergunta {i+1} processada.")
            except Exception as e:
                LOGGER.error(f"Erro ao processar resposta para a pergunta '{question}': {e}")
                
    except Exception as e:
        LOGGER.error(f"Erro crítico durante a geração de respostas com asyncio.gather: {e}")
        return 

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
    hf_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    # Execução da avaliação usando o Gemini como juiz
    print("\nExecutando a avaliação com RAGAS usando Gemini como juiz")
    try:
       
        # Limita o RAGAS a 5 chamadas paralelas de cada vez para evitar Rate Limit
        config = RunConfig(max_workers=5)
        try:
            result = evaluate(
                dataset=ragas_dataset,
                metrics=metrics,
                llm=judge_llm,
                embeddings=hf_embeddings,
                raise_exceptions=False,   # mudar para False durante debug para não interromper tudo
                run_config=config
            )
        except Exception as e_eval:
            LOGGER.error("evaluate() lançou exceção direta: %s", e_eval)
            LOGGER.error(traceback.format_exc())
            raise
    
        print("Avaliação concluída!")
        df_results = result.to_pandas()
        
        df_results["document_id"] = [d["document_id"] for d in evaluation_data]

        print("\nRESULTADOS DA AVALIAÇÃO")
        print(df_results) 
        
        output_folder = "Resultados_Gemini"
        
        # Garante que a pasta exista
        os.makedirs(output_folder, exist_ok=True) 
        
        output_filename = "ragas_evaluation_results_gemini_5k.csv"
        output_path = os.path.join(output_folder, output_filename)
        df_results.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"\nResultados salvos em '{output_path}'")
    except Exception as e:
        LOGGER.error(f"A AVALIAÇÃO FALHOu! Ocorreu um erro durante 'ragas.evaluate':")
        LOGGER.error(e)
        LOGGER.error(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())