import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import date

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from utils import get_logger, normalize_disease_name, LoggerFactory

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from utils import get_logger, normalize_disease_name, LoggerFactory

class LatentKnowledgeReportGenerator:
    """
    Gerador de relatórios visuais baseados nos dados processados pelo ValidationModule.
    Responsável por visualizar os resultados e alimentar o ciclo de feedback (topics_of_interest).
    """
    
    DEFAULT_METRICS = [
        'normalized_dot_product',
        'delta_normalized_dot_product',
        'score'
    ]
    
    def __init__(
        self,
        disease_name: str,
        model_subfolder: str, # Ex: "w2v_fixed"
        start_year: int,      # Ano inicial do período de teste
        target_year: int,     # Ano foco do relatório (geralmente o último)
        top_n_to_plot: int = 10,
        metrics_to_plot: Optional[List[str]] = None,
        ):
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.model_subfolder = model_subfolder
        self.start_year = start_year
        self.target_year = target_year
        self.top_n_to_plot = top_n_to_plot
        self.metrics_to_plot = metrics_to_plot or self.DEFAULT_METRICS
        
        # --- Configurar caminhos ---
        self.base_dir = Path('./')
        self.data_root = self.base_dir / 'data'
        self.base_path = self.data_root / self.normalized_disease_name
        
        # Caminhos específicos do subfolder do modelo
        self.validation_path = self.base_path / 'validation' / self.model_subfolder
        self.top_n_path = self.validation_path / 'top_n_compounds'
        self.history_path = self.validation_path / 'compound_history'
        self.model_directory = Path(f'{self.base_path}/models/{self.model_subfolder}')
        
        self.reports_path = self.base_path / 'reports'
        self.plots_path = self.base_path / 'plots' / self.model_subfolder
        
        # Criar diretórios necessários
        self.reports_path.mkdir(parents=True, exist_ok=True)
        self.plots_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"ReportGenerator initialized for '{self.disease_name}' / '{self.model_subfolder}'")

    def _get_top_compounds_from_file(self, metric: str, year: int) -> List[Tuple[float, str]]:
        """
        Lê o arquivo top_N gerado pelo ValidationModule na pasta do ano específico.
        """
        # Tenta encontrar o arquivo independentemente do "Top X" usado na geração (50, 20, 100...)
        search_dir = self.top_n_path / str(year)
        if not search_dir.exists():
            self.logger.warning(f"Directory not found: {search_dir}")
            return []
            
        candidates = list(search_dir.glob(f'top_*_{metric}.csv'))
        
        if not candidates:
            self.logger.warning(f"No ranking file found for metric '{metric}' in year {year}")
            return []
            
        # Pega o primeiro que encontrar (geralmente só tem um por métrica)
        file_path = candidates[0]

        try:
            df = pd.read_csv(file_path)
            # Pega os top N desejados para o plot
            top_df = df.head(self.top_n_to_plot)
            
            results = []
            for _, row in top_df.iterrows():
                results.append((row[metric], row['chemical_name']))
            return results
            
        except Exception as e:
            self.logger.error(f"Error reading ranking file {file_path}: {e}")
            return []

    def _sanitize_filename(self, name: str) -> str:
        return re.sub(r'[\\/*?:"<>|]', '_', name)[:50]

    def generate_historical_plots(self, df: pd.DataFrame, metric: str):
        """
        Gera gráfico histórico para os compostos no DataFrame.
        """
        metric_plot_dir = self.plots_path / metric
        metric_plot_dir.mkdir(parents=True, exist_ok=True)

        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df = df.dropna(subset=['year']).sort_values('year')
        
        # Plot combinado (todos os compostos no mesmo gráfico)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        compounds = df['chemical_name'].unique()
        
        for compound in compounds:
            subset = df[df['chemical_name'] == compound].copy()

            # Converter explicitamente para numpy arrays (evita indexing multi-dimensional em pandas)
            x = pd.to_numeric(subset['year'], errors='coerce').to_numpy()
            y = pd.to_numeric(subset.get(metric, pd.Series([])), errors='coerce').to_numpy()

            # Filtrar valores inválidos
            if x.size == 0 or y.size == 0:
                continue
            mask = ~np.isnan(x) & ~np.isnan(y)
            if not mask.any():
                continue

            ax.plot(x[mask], y[mask], marker='o', markersize=4, label=compound)

        ax.set_title(f"Top {len(compounds)} Compounds: {metric} ({self.target_year})", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel(metric)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path = metric_plot_dir / f"combined_top_{metric}_{self.target_year}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Plot saved: {output_path}")

    def feedback_new_topics(self, max_new_topics: int = 8, max_total_topics: int = 10) -> None:
        """
        Lê potential_treatments.txt (gerado nesta run) e adiciona novos termos
        ao topics_of_interest.txt (global da doença) para futuras buscas.
        """
        potential_file = self.base_path / 'potential_treatments.txt'
        topics_file = self.base_path / 'topics_of_interest.txt'

        if not potential_file.exists():
            self.logger.warning("'potential_treatments.txt' not found. Skipping feedback loop.")
            return

        try:
            # 1. Ler tópicos existentes
            existing_topics = set()
            if topics_file.exists():
                with open(topics_file, 'r', encoding='utf-8') as f:
                    existing_topics = {line.strip() for line in f if line.strip()}
            
            # 2. Ler novos candidatos
            with open(potential_file, 'r', encoding='utf-8') as f:
                candidates = [line.strip() for line in f if line.strip()]

            # 3. Adicionar novos (respeitando limites)
            added_count = 0
            for cand in candidates:
                if len(existing_topics) >= max_total_topics:
                    self.logger.info(f"Max topics limit ({max_total_topics}) reached.")
                    break
                
                if cand not in existing_topics:
                    existing_topics.add(cand)
                    added_count += 1
                    if added_count >= max_new_topics or len(existing_topics) >= max_total_topics:
                        break
            
            # 4. Salvar
            if added_count > 0:
                # Ordenar para manter arquivo limpo
                sorted_topics = sorted(list(existing_topics))
                with open(topics_file, 'w', encoding='utf-8') as f:
                    for topic in sorted_topics:
                        f.write(f"{topic}\n")
                
                self.logger.info(f"Feedback loop: Added {added_count} new topics to interest list.")
            else:
                self.logger.info("Feedback loop: No new topics added (duplicates or limit reached).")

        except Exception as e:
            self.logger.error(f"Error in feedback loop: {e}")

    def process_plots(self) -> Dict[str, str]:
        """
        Gera os plots para cada métrica baseada nos top compostos do ano alvo.
        """
        plots_data = {}

        for metric in self.metrics_to_plot:
            self.logger.info(f"Processing plots for {metric}...")
            
            top_list = self._get_top_compounds_from_file(metric, self.target_year)
            
            if not top_list:
                plots_data[f'plot_{metric}'] = "No data available."
                continue

            dfs = []
            for _, name in top_list:
                fname = self._sanitize_filename(name) + ".csv"
                fpath = self.history_path / fname
                
                if fpath.exists():
                    try:
                        hist_df = pd.read_csv(fpath)
                        hist_df['chemical_name'] = name
                        dfs.append(hist_df)
                    except Exception:
                        pass
            
            if dfs:
                full_df = pd.concat(dfs, ignore_index=True)
                self.generate_historical_plots(full_df, metric)
                plots_data[f'plot_{metric}'] = f"Plot generated for {metric}"
            else:
                plots_data[f'plot_{metric}'] = "Could not load histories."

        return plots_data

    def _load_model(self, year: int) -> Optional[object]:
        """Loads the model for a specific year."""
        # Pattern: *_{year}.model or *_{start}_{year}.model
        # We try to find a file ending in _{year}.model in the subfolder
        if not self.model_directory.exists():
            return None
            
        candidates = list(self.model_directory.glob(f'*_{year}.model'))
        if not candidates:
            # Try searching with Regex for more flexibility
            pattern = re.compile(rf'.*_(\d+)_{year}\.model$')
            for model_file in self.model_directory.glob('*.model'):
                if pattern.match(model_file.name):
                    candidates.append(model_file)
                    break
        
        if candidates:
            model_file = candidates[0]
            try:
                return Word2Vec.load(str(model_file))
            except Exception:
                try:
                    return KeyedVectors.load(str(model_file))
                except Exception:
                    self.logger.error(f"Failed to load model {model_file}")
        return None

    def generate_pca_trajectory_plot(self):
        """
        Gera um gráfico PCA 2D mostrando a trajetória dos compostos recomendados
        relativa à doença (que fica na origem).
        """
        self.logger.info("Generating refined PCA trajectory plot...")
        # 1. Obter top compostos recomendados para o ano alvo (por 'score')
        top_compounds_data = self._get_top_compounds_from_file('score', self.target_year)
        if not top_compounds_data:
            self.logger.warning("No top compounds found for PCA plot.")
            return
        
        compounds = [name for _, name in top_compounds_data]
        
        # 2. Coletar embeddings para cada ano no período [start_year, target_year]
        years = range(self.start_year, self.target_year + 1)
        
        # Estrutura: data_by_year[year] = { label: vector }
        data_by_year = {}
        
        for year in years:
            model = self._load_model(year)
            if not model:
                continue
            
            vocab = model.wv if hasattr(model, 'wv') else model
            year_data = {}
            
            # Embedding da doença
            dis_vec = None
            for variant in [self.disease_name, self.disease_name.lower(), self.disease_name.replace(' ', '_'), self.disease_name.lower().replace(' ', '_')]:
                if variant in vocab:
                    dis_vec = vocab[variant]
                    break
            
            if dis_vec is None:
                continue
                
            year_data["DISEASE"] = dis_vec
            
            # Embeddings dos compostos (RELATIVOS à doença)
            for compound in compounds:
                for variant in [compound, compound.lower(), compound.replace(' ', '_'), compound.lower().replace(' ', '_')]:
                    if variant in vocab:
                        year_data[compound] = vocab[variant] - dis_vec
                        break
            
            # Ajustar a própria doença para ser a origem (opcional, mas conceitualmente correto aqui)
            year_data["DISEASE"] = dis_vec - dis_vec # [0, 0, ... 0]
            
            data_by_year[year] = year_data
        
        if not data_by_year:
            self.logger.warning("No consistent embeddings found for PCA plot.")
            return

        # 3. Preparar entrada para PCA
        # Queremos o PCA fitado em todos os vetores relativos para manter a escala consistente
        all_rel_vectors = []
        labels_years = [] # (year, label)
        
        for year, year_data in data_by_year.items():
            for label, vec in year_data.items():
                all_rel_vectors.append(vec)
                labels_years.append((year, label))
        
        if not all_rel_vectors:
            return

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(all_rel_vectors))
        
        # Reorganizar trajetórias
        from collections import defaultdict
        trajectories = defaultdict(list)
        for i, (year, label) in enumerate(labels_years):
            trajectories[label].append((year, reduced[i]))
            
        # 4. Plotagem
        fig, ax = plt.subplots(figsize=(12, 9))
        
        cmap = plt.get_cmap('tab20')
        colors = {compound: cmap(i % 20) for i, compound in enumerate(compounds)}
        colors["DISEASE"] = "black"
        
        for label, points in trajectories.items():
            points.sort(key=lambda x: x[0])
            coords = np.array([p[1] for p in points])
            color = colors.get(label, "grey")
            
            # Trajetória (Dotted line, 80% opacity)
            if len(coords) > 1:
                ax.plot(coords[:, 0], coords[:, 1], linestyle=':', color=color, alpha=0.8, linewidth=1.2)

            # Ponto Final
            last_vec = coords[-1]
            if label == "DISEASE":
                ax.scatter(last_vec[0], last_vec[1], color='black', marker='X', s=150, label="Target Disease (Origin)", zorder=5)
            else:
                ax.scatter(last_vec[0], last_vec[1], color=color, marker='o', s=60, label=label, alpha=1.0)
            
            # REMOVIDO: ax.annotate(...) conforme solicitado pelo usuário

        ax.set_title(f"PCA Trajectory of Recommendations relative to '{self.disease_name}'\nPeriod: {self.start_year}-{self.target_year}", fontsize=14)
        ax.set_xlabel("PCA 1")
        ax.set_ylabel("PCA 2")
        
        # Colocar a legenda fora do plot se for muito grande
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        ax.axvline(x=0, color='k', linestyle='-', alpha=0.2)
        
        plt.tight_layout()
        
        output_path = self.plots_path / f"pca_trajectory_relative_{self.target_year}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        self.logger.info(f"Refined PCA plot saved: {output_path}")

    def run(self, max_new_topics: int = 8, max_total_topics: int = 10) -> bool:
        """Executa a geração do relatório e feedback loop."""
        self.logger.info("=== Starting Report Generation ===")
        
        # 1. Gerar Plots Históricos
        self.process_plots()

        # 2. Gerar Plot PCA de Trajetória
        self.generate_pca_trajectory_plot()
        
        # 3. Gerar lista de Tratamentos Potenciais
        top_score = self._get_top_compounds_from_file('score', self.target_year)
        if top_score:
            pt_path = self.base_path / 'potential_treatments.txt'
            try:
                with open(pt_path, 'w', encoding='utf-8') as f:
                    for _, name in top_score:
                        f.write(f"{name}\n")
                self.logger.info(f"Potential treatments list updated: {pt_path}")
                
                # 4. Executar Feedback Loop
                self.logger.info("Running feedback loop...")
                self.feedback_new_topics(max_total_topics=max_total_topics, max_new_topics=max_new_topics)
                
            except Exception as e:
                self.logger.error(f"Error saving potential treatments: {e}")
        else:
            self.logger.warning("No top compounds found by score. Skipping feedback loop.")

        self.logger.info("=== Report Generation Complete ===")
        return True

if __name__ == '__main__':
    # Exemplo de uso
    generator = LatentKnowledgeReportGenerator(
        disease_name="acute myeloid leukemia",
        model_subfolder="w2v_fixed",
        target_year=1967
    )
    generator.run()
