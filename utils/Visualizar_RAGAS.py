import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import webbrowser
import os

file_path = 'C:/Users/11941578900/Documents/GitHub/TCC/ragas_evaluation_results_8B_2k.csv'

try:
    print(f"Tentando ler o arquivo: {file_path}")
    df = pd.read_csv(file_path)
    print("Arquivo lido com sucesso!")

    #tabela html

    print("Gerando tabela HTML estilizada...")
    numeric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']

    # or doc vermelho ao verde
    cmap = sns.diverging_palette(10, 130, as_cmap=True)

    styled_df = df.style.background_gradient(
        cmap=cmap,
        subset=numeric_cols,
        low=0,
        high=1
    ).format("{:.2f}", subset=numeric_cols).set_caption(
        "Visualização dos Resultados da Avaliação RAGAS"
    ).set_properties(**{'text-align': 'left', 'border': '1px solid black'})

    html = styled_df.to_html()

    html_file_path = 'visualizacao_resultados_8B_4k.html'
    with open(html_file_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"--> Tabela HTML salva como '{html_file_path}'")

    #grafico

    print("Gerando gráfico de barras...")
    average_scores = df[numeric_cols].mean()

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.barh(average_scores.index, average_scores.values, color=sns.color_palette("viridis_r", len(average_scores)))

    ax.set_title('Pontuação Média das Métricas RAGAS', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pontuação Média', fontsize=12)
    ax.set_xlim(0, 1.1)

    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.2f}', va='center', ha='left', fontsize=11, color='black')
    
    plt.tight_layout()
    
    chart_file_path = 'grafico_scores_ragas_8B_2k.png'
    plt.savefig(chart_file_path)
    
    print(f"--> Gráfico de barras salvo como '{chart_file_path}'")

    #tabela no navegador
    print("\nAbrindo a tabela de resultados no seu navegador...")
    webbrowser.open('file://' + os.path.realpath(html_file_path))

except FileNotFoundError:
    print(f"ERRO: O arquivo '{file_path}' não foi encontrado neste diretório. Verifique se o nome está correto.")
except Exception as e:
    print(f"Ocorreu um erro inesperado: {e}")