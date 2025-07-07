##################################################
## Generates a LaTeX file containing the historical plots of each selected CHEMICAL,
## showing the evolution of its semantic relationship with the target disease.
##
## --- VERSÃO CORRIGIDA ---
## - Itera por cada ano para encontrar os melhores compostos daquele ano.
## - Salva um CSV com os top N compostos para cada ano em 'data/{disease}/{model}/{ano}/'.
## - Gera os gráficos para o LaTeX com base nos melhores compostos do ÚLTIMO ano.
##################################################

# IMPORTS:
import os
from pathlib import Path
import jinja2
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from tikzplotlib import get_tikz_code

# Assumindo que este arquivo existe e define as variáveis
from target_disease import target_disease, normalized_target_disease

def select_top_n_chemicals_per_year(model_type, combination, metric, year, top_n=20):
    """
    Lê os arquivos CSV de cada composto, seleciona os 'top_n' mais relevantes
    com base no valor de uma métrica EM UM ANO ESPECÍFICO e salva essa lista em um arquivo.

    Args:
        model_type (str): O tipo de modelo ('w2v' ou 'ft').
        combination (str): O número da combinação do modelo (ex: '15').
        metric (str): A coluna a ser usada para o ranqueamento.
        year (int): O ano específico para avaliar a métrica.
        top_n (int): O número de compostos a serem selecionados.

    Returns:
        list: Uma lista dos caminhos de arquivos CSV dos 'top_n' compostos.
    """
    # 1. Caminho para a pasta com os dados históricos de cada composto
    validation_folder = f'./data/{normalized_target_disease}/validation/per_compound/{model_type}/'

    # 2. Lista todos os arquivos CSV relevantes para a combinação
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

    # 3. Lê cada arquivo, encontra o score para o ano específico e coleta os nomes
    scores_for_year = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Filtra o DataFrame para encontrar a linha correspondente ao ano
            year_data = df[df['year'] == year]

            if not year_data.empty:
                # Extrai o score da métrica para aquele ano
                score = year_data[metric].iloc[0]
                chemical_name = os.path.basename(file_path).split(f'_comb{combination}')[0]
                scores_for_year.append((score, file_path, chemical_name))
        except (pd.errors.EmptyDataError, FileNotFoundError, KeyError) as e:
            # Ignora arquivos vazios, não encontrados ou sem a coluna da métrica
            continue

    # 4. Ordena a lista de scores em ordem decrescente
    scores_for_year.sort(key=lambda x: x[0], reverse=True)

    # 5. Seleciona os 'top_n' melhores
    top_scores = scores_for_year[:top_n]

    # Imprime os top N selecionados para o ano atual
    print(f"--- Top {len(top_scores)} compostos para o ano {year} (Métrica: '{metric}') ---")
    for score, _, name in top_scores:
        print(f"  {name}: {score:.4f}")

    # 6. Salva os nomes dos compostos selecionados em um CSV específico para o ano
    output_dir = f'./data/{normalized_target_disease}/{model_type}/{year}/'
    os.makedirs(output_dir, exist_ok=True)  # Garante que o diretório do ano exista
    output_filename = os.path.join(output_dir, f'top_{top_n}_{metric}.csv')

    top_chemicals_names = [name for _, _, name in top_scores]
    top_df = pd.DataFrame({'chemical_name': top_chemicals_names})
    top_df.to_csv(output_filename, index=False)
    print(f"-> Lista salva em: '{output_filename}'")

    # 7. Retorna os caminhos dos arquivos dos top N para uso posterior (plotagem)
    return [file_path for _, file_path, _ in top_scores]


def generate_historical_plots(csv_files_to_plot, column_to_plot, year_range):
    """
    Gera um conjunto de gráficos históricos e retorna o código Ti*k*Z correspondente.
    (Esta função não precisou de alterações, pois ela já plota a série temporal completa
    dos arquivos que recebe, o que é o comportamento desejado).

    Args:
        csv_files_to_plot (list): Lista de caminhos para os arquivos CSV a serem plotados.
        column_to_plot (str): O nome da coluna a ser usada para o eixo Y dos gráficos.
        year_range (tuple): O intervalo de anos (início, fim) para o eixo X.

    Returns:
        str: Uma string contendo o código Ti*k*Z para os gráficos.
    """
    all_plots_data = []
    for file_path in csv_files_to_plot:
        try:
            df = pd.read_csv(file_path)
            if not df.empty:
                chemical_name = os.path.basename(file_path).split('_comb')[0].replace('_', ' ')
                all_plots_data.append({'name': chemical_name, 'data': df})
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"Aviso ao plotar: Não foi possível ler ou o arquivo está vazio: {file_path}. Erro: {e}")

    if not all_plots_data:
        print("Nenhum dado válido para plotar. Retornando string vazia.")
        return ""

    num_plots = len(all_plots_data)
    num_cols = 2
    num_rows = (num_plots + num_cols - 1) // num_cols

    fig, axs = plt.subplots(num_rows, num_cols, sharex='all', figsize=(20, 5 * num_rows))
    axs = axs.flatten() if num_plots > 1 else [axs]

    for i, plot_info in enumerate(all_plots_data):
        ax = axs[i]
        df = plot_info['data']
        ax.plot(df['year'], df[column_to_plot])
        ax.set_title(plot_info['name'].capitalize())
        ax.grid(visible=True)
        ax.set_xlim(year_range[0] - 1, year_range[1] + 1)
        ax.tick_params(axis='x', labelrotation=45)
        ax.set_xticks([year for year in range(year_range[0], year_range[1] + 1, 5)])

    for i in range(num_plots, len(axs)):
        axs[i].set_axis_off()

    fig.tight_layout(pad=3.0)
    fig.supxlabel('Ano de Publicação', y=0.01, fontsize=24)
    y_label = ' '.join(column_to_plot.split('_')).capitalize()
    fig.supylabel(f"Relação com '{target_disease.capitalize()}': {y_label}", x=0.01, fontsize=24)

    plt.close(fig)

    xtick_str = ','.join(map(str, range(year_range[0], year_range[1] + 1, 5)))
    latex_string = get_tikz_code(
        fig,
        axis_width='\\textwidth/2.2',
        axis_height='125',
        extra_axis_parameters=[
            f'xtick={{{xtick_str}}}',
            'x tick label style={/pgf/number format/1000 sep=}',
            'y tick label style={/pgf/number format/fixed, /pgf/number format/precision=2, scaled y ticks=false}'
        ]
    )
    return latex_string


if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    print('Iniciando a geração do relatório de plots...')

    latex_jinja_env = jinja2.Environment(
        block_start_string='\BLOCK{', block_end_string='}',
        variable_start_string='\VAR{', variable_end_string='}',
        comment_start_string='\#{', comment_end_string='}',
        line_statement_prefix='%%', line_comment_prefix='%#',
        trim_blocks=True, autoescape=False,
        loader=jinja2.FileSystemLoader(os.path.abspath('.'))
    )
    template = latex_jinja_env.get_template('./latent_knowledge_template.tex')

    plots_data = {}

    metrics_to_plot = [
        'dot_product_result_absolute',
        'softmax',
        'softmax_normalization',
        'softmax_standardization'
    ]

    # Pega os nomes de todos os arquivos que vieram do crawler.
    aggregated_files = sorted(list(map(str, Path(f'./data/{normalized_target_disease}/aggregated_results').glob('*.txt'))))

    # Define a faixa de anos a ser processada
    year_range = int(Path(aggregated_files[0]).stem[-4:]), int(Path(aggregated_files[-1]).stem[-4:])
    start_year = year_range[0]
    end_year = year_range[1]

    print(f"\nProcessando dados para o intervalo de anos: {year_range[0]} a {year_range[1]}")

    # --- Processamento para Word2Vec ---
    print("\n" + "="*50)
    print("--- Processando Word2Vec (combinação 15) ---")
    print("="*50)
    for metric in metrics_to_plot:
        top_w2v_files_for_plotting = []  # Armazena os melhores do ÚLTIMO ano para plotar

        # 1. GERA os CSVs com os TOP N para CADA ANO
        print(f"\n>>> Gerando rankings anuais para a métrica: '{metric}'")
        for year in range(start_year, end_year + 1):
            top_files_this_year = select_top_n_chemicals_per_year(
                model_type='w2v',
                combination='15',
                metric=metric,
                year=year,
                top_n=20
            )
            # Se este for o último ano, guarda a lista de arquivos para gerar o gráfico
            if year == end_year:
                top_w2v_files_for_plotting = top_files_this_year

        # 2. GERA o gráfico para o LaTeX com base nos melhores do ÚLTIMO ANO
        print(f"\n>>> Gerando gráfico para '{metric}' com base nos melhores de {end_year}...")
        plot_key = f"plot_w2v_comb15_{metric}"
        if top_w2v_files_for_plotting:
            plots_data[plot_key] = generate_historical_plots(top_w2v_files_for_plotting, metric, year_range)
            print("Gráfico gerado com sucesso.")
        else:
            plots_data[plot_key] = f"\\textit{{Nenhum dado encontrado para a métrica '{metric}' no ano de {end_year}.}}"
            print("Nenhum dado encontrado para gerar o gráfico.")

    # --- Processamento para FastText ---
    print("\n" + "="*50)
    print("--- Processando FastText (combinação 16) ---")
    print("="*50)
    for metric in metrics_to_plot:
        top_ft_files_for_plotting = [] # Armazena os melhores do ÚLTIMO ano para plotar

        # 1. GERA os CSVs com os TOP N para CADA ANO
        print(f"\n>>> Gerando rankings anuais para a métrica: '{metric}'")
        for year in range(start_year, end_year + 1):
            top_files_this_year = select_top_n_chemicals_per_year(
                model_type='ft',
                combination='16',
                metric=metric,
                year=year,
                top_n=20
            )
            # Se este for o último ano, guarda a lista de arquivos para gerar o gráfico
            if year == end_year:
                top_ft_files_for_plotting = top_files_this_year

        # 2. GERA o gráfico para o LaTeX com base nos melhores do ÚLTIMO ANO
        print(f"\n>>> Gerando gráfico para '{metric}' com base nos melhores de {end_year}...")
        plot_key = f"plot_ft_comb16_{metric}"
        if top_ft_files_for_plotting:
            plots_data[plot_key] = generate_historical_plots(top_ft_files_for_plotting, metric, year_range)
            print("Gráfico gerado com sucesso.")
        else:
            plots_data[plot_key] = f"\\textit{{Nenhum dado encontrado para a métrica '{metric}' no ano de {end_year}.}}"
            print("Nenhum dado encontrado para gerar o gráfico.")


    # --- Renderiza o template LaTeX ---
    print('\nGerando arquivo .tex do relatório...')
    report_latex = template.render(
        target_disease_name=target_disease.replace('_', ' ').title(),
        **plots_data  # Desempacota o dicionário de plots no template
    )

    # --- Salva o arquivo final ---
    dat = date.today().strftime("%d_%m_%Y")
    output_filename = f'./latent_knowledge_report_{normalized_target_disease}_{dat}.tex'

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write(report_latex)

    print(f'\nRelatório salvo com sucesso em: {output_filename}')
    print('END!')