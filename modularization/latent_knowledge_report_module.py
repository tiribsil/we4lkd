import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import date

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display

try:
    from tikzplotlib import get_tikz_code
    TIKZ_AVAILABLE = True
except ImportError:
    TIKZ_AVAILABLE = False
    print("Warning: tikzplotlib not available. Plots will not be generated.")

try:
    import jinja2
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False
    print("Warning: jinja2 not available. LaTeX report will not be generated.")

from utils import setup_logger, normalize_disease_name


class LatentKnowledgeReportGenerator:
    """
    Gerador de relatórios de conhecimento latente para análise de compostos químicos
    e sua relação com doenças ao longo do tempo.
    """
    
    DEFAULT_METRICS = [
        'normalized_dot_product',
        'delta_normalized_dot_product',
        'euclidean_distance',
        'score'
    ]
    
    def __init__(
        self,
        disease_name: str,
        model_type: str,
        top_n_compounds: int = 10,
        delta_threshold: float = 0.001,
        metrics_to_plot: Optional[List[str]] = None):
        """
        Inicializa o gerador de relatórios.
        
        Args:
            disease_name: Nome da doença
            top_n_compounds: Número de compostos top a selecionar
            delta_threshold: Threshold para delta_normalized_dot_product
            metrics_to_plot: Lista de métricas para plotar
            base_dir: Diretório base do projeto (se None, usa diretório atual)
        """
        self.logger = setup_logger("report_generator", log_to_file=False)
        
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)

        self.model_type = model_type
        self.top_n_compounds = top_n_compounds
        self.delta_threshold = delta_threshold
        self.metrics_to_plot = metrics_to_plot or self.DEFAULT_METRICS
        
        # Configurar caminhos ABSOLUTOS
        self.base_dir = Path('./')
        self.data_root = self.base_dir / 'data'
        self.base_path = self.data_root / self.normalized_disease_name
        self.validation_path = self.base_path / 'validation'
        self.corpus_path = self.base_path / 'corpus'
        self.reports_path = self.base_path / 'reports'
        
        # Criar diretórios necessários
        self._create_directories()
        
        # Cache
        self._year_range_cache = None
        self._compound_files_cache = {}
        
        # Configurações de plot
        self.plot_config = {
            'figsize': (20, 5),
            'dpi': 100,
            'grid': True,
            'rotation': 45
        }
        
        self.logger.info(f"ReportGenerator initialized for '{self.disease_name}'")
        self.logger.info(f"Metrics to analyze: {', '.join(self.metrics_to_plot)}")

    def _create_directories(self):
        """Cria estrutura de diretórios necessária."""
        try:
            directories = [
                Path(f'{self.validation_path}/w2v/top_n_compounds'),
                Path(f'{self.validation_path}/ft/top_n_compounds'),
                Path(self.reports_path)
            ]
            
            for directory in directories:
                directory.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created/verified directory: {directory}")
            
        except Exception as e:
            self.logger.error(f"Error creating directories: {e}")

    @property
    def year_range(self) -> Tuple[int, int]:
        """Retorna range de anos do corpus (cached)."""
        if self._year_range_cache is None:
            self._year_range_cache = self._get_corpus_year_range()
        return self._year_range_cache

    def _get_corpus_year_range(self) -> Tuple[int, int]:
        """Extrai range de anos dos arquivos agregados."""
        aggregated_path = self.corpus_path / 'aggregated_abstracts'
        
        self.logger.info(f"Looking for year files in: {aggregated_path}")
        
        if not aggregated_path.exists():
            self.logger.warning(f"Aggregated path not found: {aggregated_path}")
            return (2000, 2025)
        
        year_files = sorted(aggregated_path.glob('results_file_*.txt'))
        
        if not year_files:
            self.logger.warning(f"No year files (results_file_*.txt) found in {aggregated_path}")
            # Tentar listar o que existe
            all_files = list(aggregated_path.glob('*.txt'))
            if all_files:
                self.logger.info(f"Found {len(all_files)} .txt files in directory")
                for f in all_files[:5]:
                    self.logger.debug(f"  Example file: {f.name}")
            return (2000, 2025)
        
        years = []
        for file in year_files:
            try:
                # Extrair ano do nome results_file_YYYY.txt
                year_str = file.stem.split('_')[-1]
                year = int(year_str)
                years.append(year)
                self.logger.debug(f"Found year file: {file.name} -> year {year}")
            except (ValueError, IndexError) as e:
                self.logger.warning(f"Could not extract year from {file.name}: {e}")
                continue
        
        if not years:
            self.logger.warning("No valid years extracted from files")
            return (2000, 2025)
        
        start_year, end_year = min(years), max(years)
        self.logger.info(f"Corpus year range determined: {start_year}-{end_year}")
        
        return (start_year, end_year)

    def _get_compound_history_files(self) -> List[Path]:
        """
        Retorna lista de arquivos de histórico de compostos (cached).
        Returns:
            Lista de caminhos de arquivos
        """
        cache_key = f"{self.model_type}"
        
        if cache_key in self._compound_files_cache:
            return self._compound_files_cache[cache_key]
        
        history_folder = Path(f'{self.validation_path}/{self.model_type}/compound_history')
        
        self.logger.debug(f"Looking for compound files in: {history_folder}")
        
        if not history_folder.exists():
            self.logger.warning(f"History folder not found: {history_folder}")
            self._compound_files_cache[cache_key] = []
            return []
        
        # Buscar arquivos com o padrão correto
        pattern = f'*_{self.model_type}.csv'
        files = sorted(history_folder.glob(pattern))
        
        if not files:
            # Listar alguns arquivos para debug
            all_files = list(history_folder.glob('*.csv'))
            self.logger.warning(f"No files matching pattern '{pattern}' in {history_folder}")
            if all_files:
                self.logger.info(f"Found {len(all_files)} CSV files with different patterns:")
                for f in all_files[:5]:
                    self.logger.debug(f"  Example: {f.name}")
        else:
            self.logger.info(f"Found {len(files)} compound history files for {self.model_type}")
        
        self._compound_files_cache[cache_key] = files
        return files

    def _extract_chemical_name(self, file_path: Path) -> str:
        """Extrai nome químico do caminho do arquivo."""
        try:
            name = file_path.stem.replace(f'_{self.model_type}', '')
            # Substitui underscores por espaços
            return name.replace('_', ' ').strip()
        except Exception as e:
            self.logger.debug(f"Error extracting chemical name from {file_path}: {e}")
            return file_path.stem

    def select_top_compounds_for_year(
        self,
        metric: str,
        year: int
    ) -> List[Tuple[float, Path, str]]:
        """
        Seleciona os top N compostos para um ano específico.
        
        Args:
            metric: Nome da métrica
            year: Ano alvo
            
        Returns:
            Lista de tuplas (score, file_path, chemical_name)
        """
        csv_files = self._get_compound_history_files()
        
        if not csv_files:
            return []
        
        scores_for_year = []
        valid_files = 0
        
        for file_path in csv_files:
            try:
                df = pd.read_csv(file_path)
                # Verificar colunas necessárias
                if 'year' not in df.columns:
                    self.logger.debug(f"No 'year' column in {file_path.name}")
                    continue
                
                if metric not in df.columns:
                    self.logger.debug(f"Metric '{metric}' not in {file_path.name}")
                    continue
                
                # Filtrar por ano
                year_data = df[df['year'] == year]
                
                if year_data.empty:
                    continue
                
                score = year_data[metric].iloc[0]
                
                # Validar score
                if pd.isna(score) or not np.isfinite(score):
                    continue
                
                chemical_name = self._extract_chemical_name(file_path)
                scores_for_year.append((score, file_path, chemical_name))
                valid_files += 1
                
            except Exception as e:
                self.logger.debug(f"Error reading {file_path.name}: {e}")
                continue
        
        if not scores_for_year:
            self.logger.warning(
                f"No valid data found for {self.model_type} year {year} metric {metric} "
                f"(checked {len(csv_files)} files, {valid_files} had valid data)"
            )
            return []
        
        # Ordenar scores
        reverse_order = metric != 'euclidean_distance'
        scores_for_year.sort(key=lambda x: x[0], reverse=reverse_order)
        
        # Selecionar top N
        top_scores = scores_for_year[:self.top_n_compounds]
        
        # Filtrar por threshold se necessário
        if metric == 'delta_normalized_dot_product':
            before_filter = len(top_scores)
            top_scores = [s for s in top_scores if s[0] > self.delta_threshold]
            if len(top_scores) < before_filter:
                self.logger.debug(
                    f"Filtered {before_filter - len(top_scores)} compounds "
                    f"below threshold {self.delta_threshold}"
                )
        
        self.logger.info(
            f"Selected {len(top_scores)}/{len(scores_for_year)} top compounds "
            f"for year {year} using {metric} ({self.model_type})"
        )
        
        return top_scores

    def save_top_compounds_list(
        self,
        top_compounds: List[Tuple[float, Path, str]],
        metric: str,
        year: int
    ) -> Optional[Path]:
        """
        Salva lista de top compostos em CSV.
        
        Args:
            top_compounds: Lista de tuplas (score, file_path, chemical_name)
            metric: Nome da métrica
            year: Ano
            
        Returns:
            Caminho do arquivo salvo ou None se erro
        """
        try:
            output_dir = Path(f'{self.validation_path}/{self.model_type}/top_n_compounds/{str(year)}')
            output_dir.mkdir(parents=True, exist_ok=True)
            
            output_file = Path(f'{output_dir}/top_{self.top_n_compounds}_{metric}.csv')
            
            # Criar DataFrame
            data = [
                {'chemical_name': name, metric: score} 
                for score, _, name in top_compounds
            ]
            
            df = pd.DataFrame(data)
            df.to_csv(output_file, index=False)
            
            self.logger.debug(f"Saved top compounds list: {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error saving top compounds list: {e}")
            return None

    def save_top_compounds_for_latest_year(
        self,
        metric: str = 'score',
        header_name: str = 'chemical_name',
        top_n: Optional[int] = None
    ) -> Optional[Path]:
        """
        Salva os top N compostos do último ano em um arquivo .txt.

        Args:
            metric: Métrica a ser usada para ordenação (default 'score')
            header_name: Nome da coluna contendo o nome químico
            top_n: Número de compostos a salvar (default self.top_n_compounds)

        Returns:
            Caminho do arquivo .txt salvo ou None em caso de erro
        """
        top_n = top_n or self.top_n_compounds
        start_year, end_year = self.year_range

        # Caminho do diretório do último ano
        top_compounds_path = Path(f'{self.validation_path}/{self.model_type}/top_n_compounds/{str(end_year)}')
        score_files = list(top_compounds_path.glob(f'top_20_{metric}.csv'))

        if not score_files:
            self.logger.warning(
                f"Score file not found in '{top_compounds_path.resolve()}'. "
                f"Ensure latent knowledge report has been generated with metric '{metric}'."
            )
            return None

        score_file = score_files[0]

        try:
            df = pd.read_csv(score_file)
            if header_name not in df.columns:
                self.logger.warning(f"Column '{header_name}' not found in {score_file}")
                return None
            potential_treatments = df[header_name].head(top_n).tolist()
        except Exception as e:
            self.logger.error(f"Error reading score file {score_file}: {e}")
            return None

        # Salvar arquivo txt
        output_path = Path(f'{self.base_path}/potential_treatments.txt')
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for treatment in potential_treatments:
                    f.write(treatment + '\n')

            self.logger.info(f"Top {len(potential_treatments)} potential treatments from {end_year} saved to {output_path}")

            # Printar no console
            self.logger.info(f"{'Compound'}")
            self.logger.info("-" * 60)
            for treatment in potential_treatments:
                self.logger.info(treatment)

            return output_path

        except Exception as e:
            self.logger.error(f"Error saving potential treatments to {output_path}: {e}")
            return None

    def feedback_new_topics(
        self,
        max_new_topics: int = 4,
        max_topics: int = 4
    ) -> None:
        """
        Adiciona novos compostos do arquivo 'potential_treatments.txt' ao
        arquivo 'topics_of_interest.txt', respeitando limites de quantidade.

        Args:
            max_new_topics: Máximo de novos tópicos a serem adicionados neste run
            max_topics: Limite total de tópicos no arquivo de interesse
        """
        potential_treatments_file = Path(f'{self.base_path}/potential_treatments.txt')
        topics_file = Path(f'{self.base_path}/topics_of_interest.txt')

        if not potential_treatments_file.exists():
            self.logger.warning("'potential_treatments.txt' not found. Skipping feedback loop.")
            return

        try:
            # Ler tópicos existentes
            if topics_file.exists():
                with open(topics_file, 'r', encoding='utf-8') as f:
                    existing_topics = set(line.strip() for line in f if line.strip())
            else:
                existing_topics = set()

            # Ler novos tratamentos potenciais
            with open(potential_treatments_file, 'r', encoding='utf-8') as f:
                potential_new_treatments = [line.strip() for line in f if line.strip()]

            new_topics_added_count = 0
            for topic in potential_new_treatments:
                if topic in existing_topics:
                    continue
                existing_topics.add(topic)
                new_topics_added_count += 1
                if new_topics_added_count >= max_new_topics:
                    break

            # Escrever arquivo atualizado
            with open(topics_file, 'w', encoding='utf-8') as f:
                for topic in sorted(existing_topics):
                    f.write(f"{topic}\n")

            self.logger.info(
                f"Feedback complete: {new_topics_added_count} new potential treatments added. "
                f"Total topics now: {len(existing_topics)}."
            )

        except IOError as e:
            self.logger.error(f"Error during feedback loop: {e}")

    def generate_historical_plots(self, df: pd.DataFrame, metrics: list, output_dir: Path):
        """
        Gera gráficos históricos para as métricas especificadas.

        Melhorias:
        - Corrige erro de multi-dimensional indexing (usa .to_numpy().ravel()).
        - Remove valores NaN e converte tipos automaticamente.
        - Ajusta espaçamento de ticks do eixo X conforme número de anos.
        - Aplica estilo consistente e legível.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Garantir que a coluna 'year' seja numérica
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year'])
        df = df.sort_values('year')

        for metric in metrics:
            if metric not in df.columns:
                self.logger.warning(f"⚠️ Métrica '{metric}' não encontrada no DataFrame. Pulando...")
                continue

            # Remover linhas com NaN para essa métrica
            metric_df = df[['year', metric]].dropna()

            if metric_df.empty:
                self.logger.warning(f"⚠️ Nenhum dado disponível para a métrica '{metric}'. Pulando...")
                continue

            # Converter para arrays 1D
            years = metric_df['year'].to_numpy().ravel()
            values = metric_df[metric].to_numpy().ravel()

            # Criar gráfico
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(years, values, linewidth=2, marker='o', markersize=4)
            ax.set_title(f"{metric} ao longo do tempo", fontsize=12, fontweight='bold')
            ax.set_xlabel("Ano", fontsize=10)
            ax.set_ylabel(metric, fontsize=10)

            # Ajuste dinâmico dos ticks do eixo X
            unique_years = np.unique(years)
            num_years = len(unique_years)
            if num_years > 15:
                step = max(1, num_years // 15)
                ax.set_xticks(unique_years[::step])
            else:
                ax.set_xticks(unique_years)

            ax.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()

            # Salvar arquivo
            output_path = output_dir / f"{metric}_historico.png"
            fig.savefig(output_path, dpi=150)
            plt.close(fig)

            self.logger.info(f"✅ Gráfico salvo: {output_path}")


    def process_model_type(self) -> Dict[str, str]:
        """ Processa todas as métricas para um tipo de modelo.
        Retorna: Dicionário com plots (TikZ ou placeholders) para cada métrica.
        """
        self.logger.info("="*60)
        self.logger.info(f"Processing {self.model_type.upper()}")
        self.logger.info("="*60)

        start_year, end_year = self.year_range
        plots_data = {}

        for metric in self.metrics_to_plot:
            self.logger.info(f"--- Processing metric: {metric} ---")
            top_compounds_last_year = []

            # Processar cada ano
            for year in range(start_year, end_year + 1):
                top_compounds = self.select_top_compounds_for_year(metric, year)
                if top_compounds:
                    self.save_top_compounds_list(top_compounds, metric, year)

                    # Guardar compostos do último ano para plotar
                    if year == end_year:
                        top_compounds_last_year = top_compounds

            # Gerar plots históricos para os top compounds do último ano
            if top_compounds_last_year:
                self.logger.info(
                    f"Generating plots for {len(top_compounds_last_year)} "
                    f"top compounds from {end_year}"
                )
                df_list = []
                for _, file_path, chemical_name in top_compounds_last_year:
                    try:
                        df_compound = pd.read_csv(file_path)
                        if 'year' not in df_compound.columns or metric not in df_compound.columns:
                            continue
                        df_temp = df_compound[['year', metric]].copy()
                        df_temp['chemical_name'] = chemical_name
                        df_list.append(df_temp)
                    except Exception as e:
                        self.logger.debug(f"Error reading {file_path.name}: {e}")
                        continue

                if df_list:
                    df_plot = pd.concat(df_list, ignore_index=True)
                    output_dir = self.base_path / 'plots' / self.model_type / metric
                    self.generate_historical_plots(df_plot, metrics=[metric], output_dir=output_dir)
                    plot_key = f'plot_{self.model_type}_{metric}'
                    plots_data[plot_key] = f'\\textit{{Plot saved for {metric}}}'
                else:
                    self.logger.warning(f"No valid data to plot for {metric}")
                    plots_data[f'plot_{self.model_type}_{metric}'] = (
                        f'\\textit{{No data available for {metric} until {end_year}.}}'
                    )
            else:
                self.logger.warning(f"No compounds found for {metric} in {end_year}")
                plots_data[f'plot_{self.model_type}_{metric}'] = (
                    f'\\textit{{No data available for {metric} until {end_year}.}}'
                )

        return plots_data

    def generate_latex_report(self, plots_data: Dict[str, str]) -> Optional[Path]:
        """
        Gera relatório LaTeX usando template Jinja2.
        
        Args:
            plots_data: Dicionário com códigos TikZ dos plots
            
        Returns:
            Caminho do arquivo LaTeX gerado
        """
        if not JINJA2_AVAILABLE:
            self.logger.warning("jinja2 not available, skipping LaTeX report generation")
            return None
        
        template_path = self.data_root / 'latent_knowledge_template.tex'
        
        self.logger.info(f"Looking for template at: {template_path}")
        
        if not template_path.exists():
            self.logger.warning(f"Template not found at {template_path}")
            return None
        
        try:
            # Configurar Jinja2 para LaTeX
            latex_jinja_env = jinja2.Environment(
                block_start_string='\\BLOCK{',
                block_end_string='}',
                variable_start_string='\\VAR{',
                variable_end_string='}',
                comment_start_string='\\#{',
                comment_end_string='}',
                line_statement_prefix='%%',
                line_comment_prefix='%#',
                trim_blocks=True,
                autoescape=False,
                loader=jinja2.FileSystemLoader(str(self.data_root))
            )
            
            template = latex_jinja_env.get_template('latent_knowledge_template.tex')
            
            self.logger.info("Rendering LaTeX template...")
            
            report_latex = template.render(
                target_disease_name=self.disease_name.replace('_', ' ').title(),
                **plots_data
            )
            
            # Salvar relatório
            today = date.today().strftime('%Y_%m_%d')
            output_file = self.reports_path / f'latent_knowledge_report_{today}.tex'
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_latex)
            
            self.logger.info(f"LaTeX report saved to {output_file}")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Error generating LaTeX report: {e}")
            self.logger.exception(e)
            return None

    def generate_summary_statistics(self) -> Dict[str, any]:
        """
        Gera estatísticas resumidas do processo.
        
        Returns:
            Dicionário com estatísticas
        """
        stats = {
            'disease': self.disease_name,
            'year_range': self.year_range,
            'metrics_analyzed': len(self.metrics_to_plot),
            'top_n_compounds': self.top_n_compounds,
            'files': len(self._get_compound_history_files()),
        }
        
        return stats

    def run(self, generate_latex: bool = False) -> bool:
        """
        Executa pipeline completo de geração de relatórios.

        Args:
            generate_latex: Se True, gera relatório LaTeX

        Returns:
            True se sucesso, False caso contrário
        """
        try:
            self.logger.info("="*70)
            self.logger.info("Starting Latent Knowledge Report Generation")
            self.logger.info("="*70)
            
            # Processar modelo (top compounds + plots)
            plots = self.process_model_type()

            # Salvar top compounds do último ano em potential_treatments.txt
            self.logger.info("Saving top compounds for the latest year...")
            top_txt_file = self.save_top_compounds_for_latest_year(metric='score')
            if top_txt_file:
                self.logger.info(f"Potential treatments saved to {top_txt_file}")
                self.logger.info("Running feedback loop to update topics_of_interest.txt...")
                self.feedback_new_topics(max_new_topics=4)
            else:
                self.logger.warning("Could not generate potential_treatments.txt")

            # Gerar relatório LaTeX se solicitado
            if generate_latex and plots:
                latex_file = self.generate_latex_report(plots)
                if latex_file:
                    self.logger.info(f"LaTeX report generated: {latex_file}")
            
            # Estatísticas finais
            stats = self.generate_summary_statistics()

            self.logger.info("Generation Complete - Summary Statistics")
            self.logger.info("="*70)
            self.logger.info(f"Disease: {stats['disease']}")
            self.logger.info(f"Year range: {stats['year_range'][0]}-{stats['year_range'][1]}")
            self.logger.info(f"Metrics analyzed: {stats['metrics_analyzed']}")
            self.logger.info(f"Top compounds per year: {stats['top_n_compounds']}")
            self.logger.info(f"Compounds analyzed: {stats['files']}")
            self.logger.info("="*70)
            
            return True
            
        except Exception as e:
            self.logger.exception(f"Error in report generation pipeline: {e}")
            return False


if __name__ == '__main__':

    generator = LatentKnowledgeReportGenerator(
        disease_name="acute myeloid leukemia",
        model_type = "w2v",
        top_n_compounds=20,
        delta_threshold=0.001
    )
    
    # Executar pipeline
    success = generator.run(generate_latex=True)
    
    exit(0 if success else 1)
