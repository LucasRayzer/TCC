import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuração dos caminhos dos arquivos para cada k
configuracao_arquivos = {
    2: {
        'Gemini': 'C:/Users/Lucas Rayzer/Desktop/TCC/Resultados_Gemini/ragas_evaluation_results_gemini_2k.csv',
        'Llama 8B': 'C:/Users/Lucas Rayzer/Desktop/TCC/Resultados_Llama_8B/ragas_evaluation_results_8B_2k.csv'
    },
    5: {
        'Gemini': 'C:/Users/Lucas Rayzer/Desktop/TCC/Resultados_Gemini/ragas_evaluation_results_gemini_5k.csv',
        'Llama 8B': 'C:/Users/Lucas Rayzer/Desktop/TCC/Resultados_Llama_8B/ragas_evaluation_results_8B_5k.csv'
    },
    8: {
        'Gemini': 'C:/Users/Lucas Rayzer/Desktop/TCC/Resultados_Gemini/ragas_evaluation_results_gemini_8k_original.csv',
        'Llama 8B': 'C:/Users/Lucas Rayzer/Desktop/TCC/Resultados_Llama_8B/ragas_evaluation_results_8B_8k.csv'
    }
}

# Colunas numéricas a serem consideradas
numeric_cols = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
output_dir = 'C:/Users/Lucas Rayzer/Desktop/TCC/Graficos_Comparativos/'

# Cria o diretório de saída se não existir
os.makedirs(output_dir, exist_ok=True)


# Função para gerar gráfico comparativo
def gerar_grafico_comparativo(df_melted, k_valor):
    """Gera e salva um gráfico de barras agrupadas comparando modelos para um k."""

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Usando seaborn para gráfico de barras agrupadas
    sns.barplot(
        data=df_melted,
        x='Pontuação Média',
        y='Métrica',
        hue='Modelo',  # Cria a comparação lado a lado
        palette='viridis',  # Paleta de cores distinta para os modelos
        ax=ax,
        orient='h'
    )

    ax.set_title(f'Comparação de Métricas RAGAS (Top-k = {k_valor})', fontsize=16, fontweight='bold')
    ax.set_xlabel('Pontuação Média', fontsize=12)
    ax.set_ylabel('Métrica RAGAS', fontsize=12)
    ax.set_xlim(0, 1.1)
    ax.legend(title='Modelo', loc='center right', bbox_to_anchor=(1.15, 0.5))  # Move a legenda para fora

    # Adicionando os valores nas barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3, fontsize=10)

    plt.tight_layout()

    chart_file_path = os.path.join(output_dir, f'comparativo_ragas_k{k_valor}.png')
    plt.savefig(chart_file_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico comparativo para k={k_valor} salvo em: '{chart_file_path}'")
    plt.close(fig)


# Lógica principal para processar os dados e gerar gráficos
print("Iniciando geração de gráficos comparativos...")

# Iteração sobre cada configuração de k (2, 5, 8) definido no dicionário
for k, modelos_paths in configuracao_arquivos.items():
    print(f"\nProcessando dados para k={k}...")

    dfs_medias = []
    arquivos_encontrados = True

    # Leitura dos arquivos e cálculo das médias
    for nome_modelo, caminho_arquivo in modelos_paths.items():
        try:
            print(f"  Lendo: {caminho_arquivo}")
            df_temp = pd.read_csv(caminho_arquivo)

            # Calcula a média e transforma em um DataFrame com o nome do modelo como coluna
            media_scores = df_temp[numeric_cols].mean().to_frame(name=nome_modelo)
            dfs_medias.append(media_scores)

        except FileNotFoundError:
            print(f"  ERRO: O arquivo para {nome_modelo} (k={k}) não foi encontrado: {caminho_arquivo}")
            arquivos_encontrados = False
            break
        except Exception as e:
            print(f"  ERRO inesperado ao ler {caminho_arquivo}: {e}")
            arquivos_encontrados = False
            break

    # Se todos os arquivos foram lidos com sucesso  gera o gráfico
    if arquivos_encontrados and len(dfs_medias) > 1:
        # Junta as médias dos modelos lado a lado
        df_combinado = pd.concat(dfs_medias, axis=1)

        df_melted = df_combinado.reset_index().melt(
            id_vars='index',
            var_name='Modelo',
            value_name='Pontuação Média'
        )
        df_melted.rename(columns={'index': 'Métrica'}, inplace=True)

        # Chama a função de plotagem
        gerar_grafico_comparativo(df_melted, k)
    else:
        print(f"Pulo a geração do gráfico para k={k} devido a erros na leitura dos arquivos.")

print("\nProcessamento concluído.")