##################################################
## Generates a LaTeX file containing the historical plots of each selected CHEMICAL,
## showing the evolution of its semantic relationship with the target disease.
##################################################

# IMPORTS:
import os
from pathlib import Path

import jinja2
import re
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from tikzplotlib import get_tikz_code

from target_disease import target_disease, folder_name


def select_top_n_chemicals(model_type, combination, metric, top_n=20):
    """
    Lê uma lista de arquivos CSV (um por CHEMICAL), seleciona os 'top_n' mais
    relevantes com base no valor máximo de uma métrica e salva essa lista em um arquivo.

    Args:
        model_type (str): O tipo de modelo ('w2v' ou 'ft').
        combination (str): O número da combinação do modelo (ex: '15' para w2v).
        top_n (int): O número de CHEMICALS a serem selecionados.
        metric (str): A coluna a ser usada para o ranqueamento (e.g., o produto escalar máximo).

    Returns:
        list: Uma lista dos 'top_n' caminhos de arquivos CSV mais relevantes.
    """
    # 1. Constrói o caminho para a pasta de validação com base no tipo de modelo
    validation_folder = f'./data/{folder_name}/validation/per_compound/{model_type}/'

    # 2. Encontra todos os arquivos CSV relevantes na pasta
    try:
        csv_files = [
            os.path.join(validation_folder, f)
            for f in os.listdir(validation_folder)
            if f.endswith(f'_comb{combination}.csv')
        ]
        if not csv_files:
            print(f"Aviso: Nenhum arquivo CSV encontrado em '{validation_folder}' para a combinação '{combination}'.")
            return []
    except FileNotFoundError:
        print(f"Erro: O diretório de validação não foi encontrado: '{validation_folder}'")
        return []

    # 3. Lê os arquivos, calcula os scores e coleta os nomes
    scores = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty and metric in df.columns:
                # Extrai o nome do CHEMICAL do caminho do arquivo
                # (Mantive sua lógica original que era robusta)
                chemical_name = os.path.basename(file_path).split(f'_comb{combination}')[0]
                max_score = df[metric].max()
                scores.append((max_score, file_path, chemical_name))
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"Aviso: Não foi possível ler ou o arquivo está vazio: {file_path}. Erro: {e}")
            continue

    # 4. Ordena a lista de scores em ordem decrescente
    scores.sort(key=lambda x: x[0], reverse=True)

    # 5. Prepara os dados para salvar e imprimir
    # Pega os 'top_n' resultados ou todos se houver menos que 'top_n'
    top_scores = scores[:top_n]

    # Imprime os top N selecionados para verificação
    print(f"\n--- Top {len(top_scores)} CHEMICALS selecionados para '{model_type.upper()}' pela métrica '{metric}' ---")
    for score, _, name in top_scores:
        print(f"{name}: {score:.4f}")
    print("----------------------------------------------------------------------\n")

    # 6. Salva os nomes dos CHEMICALS selecionados em um arquivo CSV
    # Constrói o caminho do arquivo de saída
    output_dir = f'./data/{folder_name}/{model_type}/'
    os.makedirs(output_dir, exist_ok=True)  # Garante que o diretório exista
    output_filename = os.path.join(output_dir, f'top_{top_n}_{metric}.csv')

    # Cria um DataFrame com os nomes e salva
    top_chemicals_names = [name for _, _, name in top_scores]
    # Usamos o nome original (com underscores) que foi extraído
    top_df = pd.DataFrame({'chemical_name': top_chemicals_names})
    top_df.to_csv(output_filename, index=False)
    print(f"Lista dos top CHEMICALS salva em: '{output_filename}'")

    # 7. Retorna apenas os caminhos dos arquivos dos top N para uso posterior (plotagem)
    return [file_path for _, file_path, _ in top_scores]

def generate_historical_plots(csv_files_to_plot, column_to_plot, year_range):
    """
    Gera um conjunto de gráficos históricos e retorna o código Ti*k*Z correspondente.
    Esta função foi refatorada para ser mais clara e remover a lógica da 'timeline'.

    Args:
        csv_files_to_plot (list): Lista de caminhos para os arquivos CSV a serem plotados.
        column_to_plot (str): O nome da coluna a ser usada para o eixo Y dos gráficos.
        year_range (tuple): O intervalo de anos (início, fim) para o eixo X.

    Returns:
        str: Uma string contendo o código Ti*k*Z para os gráficos.
    """
    # 1. PREPARAÇÃO DOS DADOS PARA PLOTAGEM
    all_plots_data = []
    for file_path in csv_files_to_plot:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                # Extrai o nome do CHEMICAL (já sanitizado) do nome do arquivo
                # Isso é mais robusto do que dividir por '/'
                chemical_name = os.path.basename(file_path).split('_comb')[0].replace('_', ' ')
                all_plots_data.append({'name': chemical_name, 'data': df})
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"Aviso ao plotar: Não foi possível ler ou o arquivo está vazio: {file_path}. Erro: {e}")

    if not all_plots_data:
        print("Nenhum dado válido para plotar. Retornando string vazia.")
        return ""

    # 2. CRIAÇÃO DOS SUBPLOTS COM MATPLOTLIB
    num_plots = len(all_plots_data)
    num_cols = 2
    # Calcula o número de linhas necessário, garantindo que seja um inteiro
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, sharex='all', figsize=(20, 5 * num_rows))
    # Achatamos a matriz de eixos para facilitar a iteração, caso haja mais de uma linha
    axs = axs.flatten()

    for i, plot_info in enumerate(all_plots_data):
        ax = axs[i]
        df = plot_info['data']

        ax.plot(df['year'], df[column_to_plot])

        ax.set_title(plot_info['name'].capitalize())
        ax.grid(visible=True)
        ax.set_xlim(year_range[0] - 1, year_range[1] + 1)
        ax.tick_params(axis='x', labelrotation=45)

        # Define os ticks do eixo X a cada 5 anos para não poluir
        ax.set_xticks([year for year in range(year_range[0], year_range[1] + 1, 5)])

    # Oculta eixos extras que não foram usados
    for i in range(num_plots, len(axs)):
        axs[i].set_axis_off()

    # 3. CONFIGURAÇÃO GERAL E CONVERSÃO PARA TIKZ
    fig.tight_layout(pad=3.0)
    fig.supxlabel('Ano de Publicação', y=0.01, fontsize=24)
    y_label = ' '.join(column_to_plot.split('_')).capitalize()
    fig.supylabel(f"Relação com '{target_disease.capitalize()}': {y_label}", x=0.01, fontsize=24)

    plt.close(fig)  # Fecha a figura para não exibi-la no notebook/script

    # Parâmetros para o Ti*k*Z
    # Convertemos a tupla de ticks para uma string formatada para Ti*k*Z
    xtick_str = ','.join(map(str, range(year_range[0], year_range[1] + 1, 5)))

    latex_string = get_tikz_code(
        fig,
        axis_width='\\textwidth/2.2',  # Largura relativa para caber duas colunas
        axis_height='125',
        extra_axis_parameters=[
            f'xtick={{{xtick_str}}}',
            'x tick label style={/pgf/number format/1000 sep=}',
            'y tick label style={/pgf/number format/fixed, /pgf/number format/precision=2, scaled y ticks=false}'
        ]
    )

    return latex_string


# PROGRAMA PRINCIPAL
if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    print('Iniciando a geração do relatório de plots...')

    # --- Configuração do Jinja2 ---
    latex_jinja_env = jinja2.Environment(
        block_start_string='\BLOCK{', block_end_string='}',
        variable_start_string='\VAR{', variable_end_string='}',
        comment_start_string='\#{', comment_end_string='}',
        line_statement_prefix='%%', line_comment_prefix='%#',
        trim_blocks=True, autoescape=False,
        loader=jinja2.FileSystemLoader(os.path.abspath('.'))
    )
    template = latex_jinja_env.get_template('./latent_knowledge_template.tex')

    plots_data = {}  # Dicionário para armazenar todos os códigos Ti*k*Z

    metrics_to_plot = [
        'dot_product_result_absolute',
        'softmax',
        'softmax_normalization',
        'softmax_standardization'
    ]

    # Pega os nomes de todos os arquivos que vieram do crawler.
    aggregated_files = sorted(list(map(str, Path(f'./data/{folder_name}/aggregated_results').glob('*.txt'))))

    year_range = int(Path(aggregated_files[0]).stem[-9:-5]), int(Path(aggregated_files[-1]).stem[-9:-5])

    # --- Geração para Word2Vec ---
    print("\n--- Processando Word2Vec (combinação 15) ---")
    for metric in metrics_to_plot:
        print(f"\nSelecionando e plotando para a métrica: '{metric}'")

        # 1. SELECIONA os Top N arquivos BASEADO na métrica atual
        top_w2v_files = select_top_n_chemicals('w2v', '15', metric, 20)

        # 2. GERA o plot USANDO a mesma métrica
        # A chave do dicionário de plots deve ser única para cada modelo e métrica
        plot_key = f"plot_w2v_comb15_{metric}"

        # Verifica se a seleção retornou algum arquivo para evitar erros
        if top_w2v_files:
            plots_data[plot_key] = generate_historical_plots(top_w2v_files, metric, year_range)
        else:
            plots_data[plot_key] = "\\textit{Nenhum dado encontrado para esta métrica.}"

    # --- Geração para FastText ---
    print("\n--- Processando FastText (combinação 16) ---")
    for metric in metrics_to_plot:
        print(f"\nSelecionando e plotando para a métrica: '{metric}'")

        # 1. SELECIONA os Top N arquivos BASEADO na métrica atual
        top_ft_files = select_top_n_chemicals('ft', '16', metric, 20)

        # 2. GERA o plot USANDO a mesma métrica
        plot_key = f"plot_ft_comb16_{metric}"

        if top_ft_files:
            plots_data[plot_key] = generate_historical_plots(top_ft_files, metric, year_range)
        else:
            plots_data[plot_key] = "\\textit{Nenhum dado encontrado para esta métrica.}"

    # --- Renderiza o template LaTeX ---
    print('\nGerando arquivo .tex do relatório...')
    report_latex = template.render(
        target_disease_name=target_disease.replace('_', ' ').title(),
        **plots_data  # Desempacota o dicionário de plots no template
    )

    # --- Salva o arquivo final ---
    dat = date.today().strftime("%d_%m_%Y")
    output_filename = f'./latent_knowledge_report_{folder_name}_{dat}.tex'

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(report_latex)

    print(f'\nRelatório salvo com sucesso em: {output_filename}')
    print('END!')