import os
import jinja2
import pandas as pd
from matplotlib import pyplot as plt
from datetime import date
from tikzplotlib import get_tikz_code

from src.utils import *

os.chdir(Path(__file__).resolve().parent.parent)


def select_top_n_chemicals_per_year(model_type, normalized_target_disease, combination, metric, year, top_n=20):
    """
    Goes through the compounds' history and selects the top N compounds for a given year using a given metric.
    Args:
        model_type: Either 'w2v' or 'ft'.
        normalized_target_disease: The target disease normalized the same way as its data folder.
        combination: The parameter combination used to train the model.
        metric: One of the metrics in the compounds' history.
        year: The specific year to analyze the compounds' history.
        top_n: How many of the best compounds to select.

    Returns:
        list: A list of file paths for the top N compounds' history for the given year.
    """
    validation_folder = f'./data/{normalized_target_disease}/validation/{model_type}/compound_history/'

    # Gets all compound history files.
    try:
        csv_files = [
            os.path.join(validation_folder, f)
            for f in os.listdir(validation_folder)
            if f.endswith(f'_comb{combination}.csv')
        ]
        if not csv_files:
            return []
    except FileNotFoundError:
        return []

    # Gets all (score, file_path, chemical_name) tuples for the given year.
    scores_for_year = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            year_data = df[df['year'] == year]

            if year_data.empty: continue

            score = year_data[metric].iloc[0]
            chemical_name = os.path.basename(file_path).split(f'_comb{combination}')[0]
            scores_for_year.append((score, file_path, chemical_name))
        except (pd.errors.EmptyDataError, FileNotFoundError, KeyError):
            continue

    # Sorts the scores in descending order by the specified metric.
    scores_for_year.sort(key=lambda x: x[0], reverse=True)

    # Selects the top N scores for the year.
    top_scores = scores_for_year[:top_n]

    # Saves the list as a CSV file in the corresponding year folder.
    output_dir = f'./data/{normalized_target_disease}/validation/{model_type}/top_n_compounds/{year}/'
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'top_{top_n}_{metric}.csv')

    top_chemicals_names = [name for _, _, name in top_scores]
    top_df = pd.DataFrame({'chemical_name': top_chemicals_names})
    top_df.to_csv(output_filename, index=False)

    return [file_path for _, file_path, _ in top_scores]


def generate_historical_plots(csv_files_to_plot, target_disease, column_to_plot, year_range):
    """
    Generates a set of historical plots and returns the corresponding TikZ code. (This function did not need changes, as it already plots the complete time series of the files it receives, which is the desired behavior).

    Args:
        target_disease: The target disease for which the plots are generated.
        csv_files_to_plot: List of paths to the CSV files to be plotted.
        column_to_plot: The name of the column to be used for the Y axis of the plots.
        year_range: The range of years (start, end) for the X axis.

    Returns:
        str: A string containing the TikZ code for the plots.
    """
    all_plots_data = []
    for file_path in csv_files_to_plot:
        df = pd.read_csv(file_path)

        if df.empty: continue

        chemical_name = os.path.basename(file_path).split('_comb')[0].replace('_', ' ')
        all_plots_data.append({'name': chemical_name, 'data': df})

    if not all_plots_data:
        print('No data to plot.')
        return ''

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
    fig.supxlabel('Year published', y=0.01, fontsize=24)
    y_label = ' '.join(column_to_plot.split('_')).capitalize()
    fig.supylabel(f'Relationship with {target_disease.capitalize()}: {y_label}', x=0.01, fontsize=24)

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


def main():
    target_disease = get_target_disease()
    normalized_target_disease = get_normalized_target_disease()

    pd.options.mode.chained_assignment = None

    plots_data = {}

    metrics_to_plot = [
        'normalized_dot_product',
        'delta_normalized_dot_product',
        'euclidian_distance',
    ]

    start_year, end_year = get_corpus_year_range(normalized_target_disease)
    year_range = (start_year, end_year)

    # Computes the top N compounds for each year and generates plots for Word2Vec.
    for metric in metrics_to_plot:
        top_w2v_files_for_plotting = []

        # Selects and saves the top N compounds for each year using the metric. Saves the last list for plotting.
        print(f'Getting compounds with the best {metric} in each year...')
        for year in range(start_year, end_year + 1):
            top_files_this_year = select_top_n_chemicals_per_year(
                model_type='w2v',
                normalized_target_disease=normalized_target_disease,
                combination='15',
                metric=metric,
                year=year,
                top_n=20
            )

            if year == end_year:
                top_w2v_files_for_plotting = top_files_this_year

        print(f'Plotting {metric} for the best compounds from {end_year}...')
        plot_key = f'plot_w2v_comb15_{metric}'
        if top_w2v_files_for_plotting:
            plots_data[plot_key] = generate_historical_plots(top_w2v_files_for_plotting, target_disease, metric, year_range)
        else:
            plots_data[plot_key] = f'\\textit{{{metric} values not found until {end_year}.}}'
            print('No data to plot.')

    # Same thing for FastText.
    for metric in metrics_to_plot:
        top_ft_files_for_plotting = []

        print(f'Getting compounds with the best {metric} in each year...')
        for year in range(start_year, end_year + 1):
            top_files_this_year = select_top_n_chemicals_per_year(
                model_type='ft',
                normalized_target_disease=normalized_target_disease,
                combination='16',
                metric=metric,
                year=year,
                top_n=20
            )

            if year == end_year:
                top_ft_files_for_plotting = top_files_this_year

        print(f'Plotting {metric} for the best compounds from {end_year}...')
        plot_key = f'plot_ft_comb16_{metric}'
        if top_ft_files_for_plotting:
            plots_data[plot_key] = generate_historical_plots(top_ft_files_for_plotting, target_disease, metric, year_range)
        else:
            plots_data[plot_key] = f'\\textit{{{metric} values not found until {end_year}.}}'
            print('No data to plot.')

    # latex_jinja_env = jinja2.Environment(
    #     block_start_string='\BLOCK{', block_end_string='}',
    #     variable_start_string='\VAR{', variable_end_string='}',
    #     comment_start_string='\#{', comment_end_string='}',
    #     line_statement_prefix='%%', line_comment_prefix='%#',
    #     trim_blocks=True, autoescape=False,
    #     loader=jinja2.FileSystemLoader(os.path.abspath('.'))
    # )
    # template = latex_jinja_env.get_template('./data/latent_knowledge_template.tex')
    #
    # print('Generating LaTeX report file...')
    # report_latex = template.render(
    #     target_disease_name=target_disease.replace('_', ' ').title(),
    #     **plots_data
    # )
    #
    # dat = date.today().strftime('%d_%m_%Y')
    # output_file = f'./data/{normalized_target_disease}/reports/latent_knowledge_report_{dat}.tex'
    #
    # Path(f'./data/{normalized_target_disease}/reports/').mkdir(parents=True, exist_ok=True)
    # with open(output_file, 'w', encoding='utf-8') as f:
    #     f.write(report_latex)

    print('End :)')


if __name__ == '__main__':
    main()