##################################################
## Generates a LaTeX file containing the historical plots of each selected CHEMICAL,
## showing the evolution of its semantic relationship with the target disease.
##################################################

# IMPORTS:
import os
import jinja2
import re
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from tikzplotlib import get_tikz_code

from target_disease import target_disease, folder_name

def select_top_n_chemicals(csv_files, top_n=20, metric='dot_product_result_absolute'):
    """
    Lê uma lista de arquivos CSV (um por CHEMICAL) e seleciona os 'top_n' mais
    relevantes com base no valor máximo de uma métrica.

    Args:
        csv_files (list): Lista de caminhos para os arquivos CSV.
        top_n (int): O número de CHEMICALS a serem selecionados.
        metric (str): A coluna a ser usada para o ranqueamento (e.g., o produto escalar máximo).

    Returns:
        list: Uma lista dos 'top_n' caminhos de arquivos CSV mais relevantes.
    """
    scores = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            if not df.empty and metric in df.columns:
                # Extrai o nome do CHEMICAL do caminho do arquivo
                chemical_name = os.path.basename(file_path).split('_comb')[0]
                max_score = df[metric].max()
                scores.append((max_score, file_path, chemical_name))
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"Aviso: Não foi possível ler ou o arquivo está vazio: {file_path}. Erro: {e}")
            continue

    # Ordena a lista de scores em ordem decrescente
    scores.sort(key=lambda x: x[0], reverse=True)

    # Imprime os top N selecionados para verificação
    print(f"\n--- Top {top_n} CHEMICALS selecionados pela métrica '{metric}' (valor máximo) ---")
    for score, _, name in scores[:top_n]:
        print(f"{name}: {score:.4f}")
    print("----------------------------------------------------------------------\n")

    # Retorna apenas os caminhos dos arquivos dos top N
    return [file_path for _, file_path, _ in scores[:top_n]]


def generate_historical_plots(csv_files_to_plot, column_to_plot, year_range=(1963, 2023)):
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

    # --- Encontra os arquivos CSV gerados pelo script anterior ---
    w2v_folder = f'./data/{folder_name}/validation/per_compound/w2v/'
    ft_folder = f'./data/{folder_name}/validation/per_compound/ft/'

    csv_files_w2v = [os.path.join(w2v_folder, f) for f in os.listdir(w2v_folder) if f.endswith('_comb15.csv')]
    csv_files_ft = [os.path.join(ft_folder, f) for f in os.listdir(ft_folder) if f.endswith('_comb16.csv')]

    # --- Seleciona os Top N CHEMICALS para plotagem ---
    top_n_to_plot = 20  # Você pode ajustar este número
    top_w2v_files = select_top_n_chemicals(csv_files_w2v, top_n=top_n_to_plot)
    top_ft_files = select_top_n_chemicals(csv_files_ft, top_n=top_n_to_plot)

    # --- Gera os plots para cada modelo e métrica ---
    metrics_to_plot = [
        'dot_product_result_absolute',
        'softmax',
        'softmax_normalization',
        'softmax_standardization'
    ]

    plots_data = {}  # Dicionário para armazenar todos os códigos Ti*k*Z

    # Geração para Word2Vec
    print("\n--- Gerando plots para Word2Vec (combinação 15) ---")
    for metric in metrics_to_plot:
        key = f"plot_comb15_{metric}"
        print(f"Gerando para a métrica: {metric}...")
        plots_data[key] = generate_historical_plots(top_w2v_files, column_to_plot=metric)

    # Geração para FastText
    print("\n--- Gerando plots para FastText (combinação 16) ---")
    for metric in metrics_to_plot:
        key = f"plot_comb16_{metric}"
        print(f"Gerando para a métrica: {metric}...")
        plots_data[key] = generate_historical_plots(top_ft_files, column_to_plot=metric)

    # --- Renderiza o template LaTeX ---
    print('\nGerando arquivo .tex do relatório...')
    report_latex = template.render(
        target_disease_name=target_disease.replace('_', ' ').title(),
        **plots_data  # Desempacota o dicionário de plots no template
    )

    # --- Salva o arquivo final ---
    dat = date.today().strftime("%d_%m")
    output_filename = f'./latent_knowledge_report_{folder_name}_{dat}.tex'

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(report_latex)

    print(f'\nRelatório salvo com sucesso em: {output_filename}')
    print('END!')