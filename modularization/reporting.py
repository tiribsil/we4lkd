import os
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import date

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from utils import get_logger, normalize_disease_name, LoggerFactory

# Global Plotting Config
plt.style.use('seaborn-muted') # Try to use a nice style
rcParams.update({
    'figure.dpi': 600,
    'savefig.dpi': 600,
    'font.size': 14,
    'axes.titlesize': 18,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

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
        ground_truth: Optional[Dict[str, int]] = None,
        corpus_start_year: Optional[int] = None
        ):
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.model_subfolder = model_subfolder
        self.start_year = start_year
        self.target_year = target_year
        self.top_n_to_plot = top_n_to_plot
        self.metrics_to_plot = metrics_to_plot or self.DEFAULT_METRICS
        self.ground_truth = ground_truth
        self.corpus_start_year = corpus_start_year or start_year
        
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
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        compounds = df['chemical_name'].unique()
        
        for compound in compounds:
            subset = df[df['chemical_name'] == compound].copy()
            x = pd.to_numeric(subset['year'], errors='coerce').to_numpy()
            y = pd.to_numeric(subset.get(metric, pd.Series([])), errors='coerce').to_numpy()

            if x.size == 0 or y.size == 0:
                continue
            mask = ~np.isnan(x) & ~np.isnan(y)
            if not mask.any():
                continue

            ax.plot(x[mask], y[mask], marker='o', markersize=4, label=compound, linewidth=1.5)

        self._apply_aesthetic_style(
            ax, 
            f"Historical Context: {metric.replace('_', ' ').title()}", 
            "Year", 
            metric.replace('_', ' ').title()
        )
        plt.tight_layout()

        output_path = metric_plot_dir / f"combined_top_{metric}_{self.target_year}.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        self.logger.info(f"Plot saved: {output_path}")

    def generate_ranking_convergence_plot(self):
        """Plots the ranking position of confirmed drugs over time leading to report."""
        if not self.ground_truth:
            self.logger.warning("No ground truth for ranking convergence plot.")
            return

        self.logger.info("Generating Ranking Convergence plot...")
        top_list = self._get_top_compounds_from_file('score', self.target_year)
        compounds_of_interest = [name for _, name in top_list if name in self.ground_truth]
        
        if not compounds_of_interest:
            self.logger.warning("No confirmed drugs in top recommendations.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        
        for name in compounds_of_interest:
            report_year = self.ground_truth[name]
            fname = self._sanitize_filename(name) + ".csv"
            fpath = self.history_path / fname
            if not fpath.exists(): continue
            
            hist_df = pd.read_csv(fpath)
            ranks = []
            years_rel = []
            for _, row in hist_df.iterrows():
                yr = int(row['year'])
                rank_dir = self.top_n_path / str(yr)
                rank_file = list(rank_dir.glob("top_*_score.csv"))
                if rank_file:
                    rdf = pd.read_csv(rank_file[0])
                    col = 'chemical_name' if 'chemical_name' in rdf.columns else 'compound_name'
                    if name in rdf[col].values:
                        idx = rdf[rdf[col] == name].index[0] + 1
                        ranks.append(idx)
                        years_rel.append(yr - report_year)

            if ranks:
                ax.plot(years_rel, ranks, marker='s', markersize=5, label=f"{name} (Report: {report_year})", alpha=0.8)

        self._apply_aesthetic_style(ax, "Ranking Convergence to Literature Report", "Years relative to Report", "Rank Position")
        ax.set_yscale('log')
        ax.invert_yaxis()
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        plt.tight_layout()
        
        output_path = self.plots_path / f"ranking_convergence_{self.target_year}.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        self.logger.info(f"Ranking convergence plot saved: {output_path}")

    def generate_lead_time_scatter_plot(self):
        """Correlates lead time with rank in the target year."""
        if not self.ground_truth: return
        
        self.logger.info("Generating Lead-Time Analysis plot...")
        data = []
        for name, report_year in self.ground_truth.items():
            first_seen_year = None
            for yr in range(self.start_year, self.target_year + 1):
                rank_dir = self.top_n_path / str(yr)
                rank_file = list(rank_dir.glob("top_*_score.csv"))
                if rank_file:
                    rdf = pd.read_csv(rank_file[0])
                    col = 'chemical_name' if 'chemical_name' in rdf.columns else 'compound_name'
                    if name in rdf[col].values:
                        first_seen_year = yr
                        break
            
            if first_seen_year and first_seen_year <= report_year:
                lead_time = report_year - first_seen_year
                rank_dir = self.top_n_path / str(self.target_year)
                rank_file = list(rank_dir.glob("top_*_score.csv"))
                if rank_file:
                   rdf = pd.read_csv(rank_file[0])
                   col = 'chemical_name' if 'chemical_name' in rdf.columns else 'compound_name'
                   if name in rdf[col].values:
                       target_rank = rdf[rdf[col] == name].index[0] + 1
                       data.append({'name': name, 'lead_time': lead_time, 'rank': target_rank})

        if not data: 
            self.logger.warning("No data points for lead-time analysis.")
            return

        df = pd.DataFrame(data)
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(df['lead_time'], df['rank'], alpha=0.7, edgecolors='w', s=120, zorder=3)
        
        if len(df) > 2:
            z = np.polyfit(df['lead_time'], df['rank'], 1)
            p = np.poly1d(z)
            x_range = np.linspace(df['lead_time'].min(), df['lead_time'].max(), 100)
            ax.plot(x_range, p(x_range), "r--", alpha=0.8, linewidth=1.5, label=f'Trendline', zorder=2)

        self._apply_aesthetic_style(ax, "Discovery Lead-Time Performance", "Lead Time (Years Early)", "Target Year Rank")
        ax.invert_yaxis()
        plt.tight_layout()
        
        output_path = self.plots_path / f"lead_time_analysis_{self.target_year}.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        self.logger.info(f"Lead-time analysis plot saved: {output_path}")

    def generate_discovery_timeline_plot(self, max_compounds: int = 5):
        """
        Plots the ranking evolution over absolute years for high-lead-time compounds.
        Focuses on discoveries made during model development/selection (up to selection_end_year).
        """
        if not self.ground_truth: return
        
        self.logger.info("Generating Discovery Timeline plot...")
        
        # 1. Identify high-lead-time candidates
        # Align logic with ModelEvaluator._calculate_final_performance
        candidates = []
        for name, report_year in self.ground_truth.items():
            # Filter out compounds reported BEFORE this period (already known)
            if report_year < self.start_year:
                continue

            first_rank_year = None
            # Search from the CORPUS start to see the TRUE first discovery
            for yr in range(self.corpus_start_year, self.target_year + 1):
                rank_dir = self.top_n_path / str(yr)
                rank_file = list(rank_dir.glob("top_*_score.csv"))
                if rank_file:
                    rdf = pd.read_csv(rank_file[0])
                    col = 'chemical_name' if 'chemical_name' in rdf.columns else 'compound_name'
                    if name in rdf[col].values:
                        first_rank_year = yr
                        break
            
            if first_rank_year:
                # Calculate lead time (years early)
                lead_time = report_year - first_rank_year
                candidates.append({
                    'name': name, 
                    'lead_time': lead_time, 
                    'report_year': report_year,
                    'first_year': first_rank_year
                })

        # Sort by lead time (descending) like Top 5 Biggest Anticipations
        candidates = sorted(candidates, key=lambda x: x['lead_time'], reverse=True)[:max_compounds]
        
        if not candidates:
            self.logger.warning("No suitable candidates for discovery timeline plot.")
            return

        fig, ax = plt.subplots(figsize=(12, 7))
        
        for cand in candidates:
            name = cand['name']
            report_year = cand['report_year']
            
            # Fetch ranking history
            fname = self._sanitize_filename(name) + ".csv"
            fpath = self.history_path / fname
            if not fpath.exists(): continue
            
            # Instead of using the history CSV (which might not have ranks), 
            # we iterate through the top_n folders to get consistent ranking data
            ranks = []
            plot_years = []
            
            for yr in range(self.start_year, self.target_year + 1):
                rank_dir = self.top_n_path / str(yr)
                rank_file = list(rank_dir.glob("top_*_score.csv"))
                if rank_file:
                    rdf = pd.read_csv(rank_file[0])
                    col = 'chemical_name' if 'chemical_name' in rdf.columns else 'compound_name'
                    if name in rdf[col].values:
                        idx = rdf[rdf[col] == name].index[0] + 1
                        ranks.append(idx)
                        plot_years.append(yr)

            if plot_years:
                line, = ax.plot(plot_years, ranks, marker='o', markersize=4, label=f"{name}", alpha=0.9, linewidth=2)
                
                # Add "Report Year" marker
                if report_year <= self.target_year:
                    # Case 1: Reported within the plotting period
                    ax.scatter([report_year], [ranks[plot_years.index(report_year)] if report_year in plot_years else ranks[-1]], 
                               color=line.get_color(), marker='*', s=200, edgecolors='black', zorder=5)
                    ax.annotate(f"Reported {report_year}", (report_year, ranks[plot_years.index(report_year)] if report_year in plot_years else ranks[-1]), 
                                textcoords="offset points", xytext=(0,10), ha='center', 
                                fontsize=9, fontweight='bold', color=line.get_color())
                else:
                    # Case 2: Reported after the plotting period
                    last_year = plot_years[-1]
                    last_rank = ranks[-1]
                    ax.annotate(f"→ Reported {report_year}", (last_year, last_rank), 
                                textcoords="offset points", xytext=(10,0), va='center', ha='left',
                                fontsize=9, fontweight='bold', color=line.get_color(),
                                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=line.get_color(), alpha=0.8))

        # Aesthetics
        self._apply_aesthetic_style(
            ax, 
            "Discovery Timeline: Latent Knowledge to Literature Report", 
            "Year", 
            "Ranking Position"
        )
        ax.set_yscale('log')
        ax.invert_yaxis()
        from matplotlib.ticker import ScalarFormatter
        ax.yaxis.set_major_formatter(ScalarFormatter())
        
        # Add a "Selection Period" span if we have the info (hardcoded or passed)
        # Based on user-provided cat: 2005-2016
        ax.axvspan(2005, 2016, color='gray', alpha=0.1, label='Model Selection Period')
        
        plt.tight_layout()
        
        output_path = self.plots_path / f"discovery_timeline_{self.target_year}.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        self.logger.info(f"Discovery timeline plot saved: {output_path}")

    def _apply_aesthetic_style(self, ax, title, xlabel, ylabel, legend=True):
        """Aplica padrões de estética científica ao plot."""
        ax.set_title(title, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, labelpad=10)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(True, linestyle='--', alpha=0.3)
        if legend:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False, fontsize='medium')

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

        # Novo: Gráfico de comparação de modelos (se disponível)
        self.generate_model_comparison_plot()

        return plots_data

    def generate_model_comparison_plot(self):
        """Gera um gráfico comparando a performance anual de todos os candidatos."""
        perf_csv = self.reports_path / "selection_performance.csv"
        if not perf_csv.exists():
            self.logger.warning(f"Selection performance file not found: {perf_csv}")
            return

        self.logger.info("Generating Model Comparison plot...")
        try:
            df = pd.read_csv(perf_csv)
            if df.empty: return

            fig, ax = plt.subplots(figsize=(10, 6))
            
            for model_name in df['model_name'].unique():
                subset = df[df['model_name'] == model_name].sort_values('year')
                # Ensure we pass numpy arrays to avoid indexing errors if pandas versions are tricky
                yrs = subset['year'].to_numpy()
                vals = subset['years_early'].to_numpy()
                ax.plot(yrs, vals, marker='o', label=model_name, linewidth=2)

            self._apply_aesthetic_style(
                ax, 
                "Model Candidate Comparison (Annual Performance)", 
                "Validation Year", 
                "Average Years Early"
            )
            plt.tight_layout()

            output_path = self.plots_path / "model_comparison_annual.pdf"
            fig.savefig(output_path)
            plt.close(fig)
            self.logger.info(f"Model comparison plot saved: {output_path}")

        except Exception as e:
            self.logger.error(f"Error generating model comparison plot: {e}")

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

    def _align_vectors(self, base_model, target_model) -> np.ndarray:
        """
        Calcula a matriz de rotação (Orthogonal Procrustes) para alinhar o model base ao target.
        """
        base_wv = base_model.wv if hasattr(base_model, 'wv') else base_model
        target_wv = target_model.wv if hasattr(target_model, 'wv') else target_model
        
        common_vocab = set(base_wv.key_to_index.keys()) & set(target_wv.key_to_index.keys())
        if len(common_vocab) < 10:
            self.logger.warning("Not enough common vocabulary for alignment.")
            return np.eye(base_wv.vector_size)
            
        common_vocab = sorted(list(common_vocab))
        A = np.array([base_wv[w] for w in common_vocab])
        B = np.array([target_wv[w] for w in common_vocab])
        
        # Orthogonal Procrustes: minimize ||A*Q - B||^2 sub Q^T*Q = I
        M = B.T @ A
        U, S, Vt = np.linalg.svd(M)
        W = Vt.T @ U.T
        return W # Matriz de transformação: Base_vector @ W ~ Target_vector

    def generate_pca_trajectory_plot(self):
        """
        Gera um gráfico PCA 2D mostrando a trajetória dos compostos recomendados
        relativa à doença, com alinhamento Procrustes para suavizar os saltos.
        """
        self.logger.info("Generating Aligned PCA trajectory plot...")
        
        top_compounds_data = self._get_top_compounds_from_file('score', self.target_year)
        if not top_compounds_data:
            self.logger.warning("No top compounds found for PCA plot.")
            return
        
        compounds = [name for _, name in top_compounds_data]
        years = sorted(list(range(self.start_year, self.target_year + 1)))
        
        # 1. Carregar todos os modelos
        models_by_year = {}
        for year in years:
            m = self._load_model(year)
            if m: models_by_year[year] = m
            
        if not models_by_year or self.target_year not in models_by_year:
            self.logger.error("Target year model or other models not found.")
            return
            
        target_model = models_by_year[self.target_year]
        
        # 2. Coletar e Alinhar dados
        data_by_year = {}
        for year, model in models_by_year.items():
            vocab = model.wv if hasattr(model, 'wv') else model
            
            # Matriz de alinhamento para este ano (exceto o próprio alvo)
            W = self._align_vectors(model, target_model) if year != self.target_year else np.eye(vocab.vector_size)
            
            # Embedding da doença
            dis_vec = None
            for variant in [self.disease_name, self.disease_name.lower(), self.disease_name.replace(' ', '_'), self.disease_name.lower().replace(' ', '_')]:
                if variant in vocab:
                    dis_vec = vocab[variant] @ W # Aplica rotação
                    break
            
            if dis_vec is None: continue
            
            year_data = {}
            # Embeddings dos compostos (ROTACIONADOS e então RELATIVOS à doença)
            for compound in compounds:
                for variant in [compound, compound.lower(), compound.replace(' ', '_'), compound.lower().replace(' ', '_')]:
                    if variant in vocab:
                        aligned_vec = vocab[variant] @ W
                        year_data[compound] = aligned_vec - dis_vec
                        break
            
            year_data["DISEASE"] = dis_vec - dis_vec # Origem (0,0)
            data_by_year[year] = year_data

        # 3. PCA e Plotagem
        all_rel_vectors = []
        labels_years = []
        for year, year_data in data_by_year.items():
            for label, vec in year_data.items():
                all_rel_vectors.append(vec)
                labels_years.append((year, label))
        
        if not all_rel_vectors: return

        pca = PCA(n_components=2)
        reduced = pca.fit_transform(np.array(all_rel_vectors))
        
        # Correção da Origem (visto que PCA centraliza)
        origin_proj = pca.transform(np.zeros((1, np.array(all_rel_vectors).shape[1])))[0]
        reduced = reduced - origin_proj
        
        from collections import defaultdict
        trajectories = defaultdict(list)
        for i, (year, label) in enumerate(labels_years):
            trajectories[label].append((year, reduced[i]))
            
        fig, ax = plt.subplots(figsize=(16, 9))
        cmap = plt.get_cmap('tab20')
        colors = {compound: cmap(i % 20) for i, compound in enumerate(compounds)}
        colors["DISEASE"] = "black"
        
        for label, points in trajectories.items():
            points.sort(key=lambda x: x[0])
            coords = np.array([p[1] for p in points])
            color = colors.get(label, "grey")
            
            if len(coords) > 1:
                ax.plot(coords[:, 0], coords[:, 1], linestyle=':', color=color, alpha=0.5, linewidth=1.5, zorder=2)

            last_vec = coords[-1]
            if label == "DISEASE":
                ax.scatter(last_vec[0], last_vec[1], color='black', marker='X', s=200, label="Target Disease (Origin)", zorder=5)
            else:
                ax.scatter(last_vec[0], last_vec[1], color=color, marker='o', s=150, label=label, alpha=1.0, zorder=4, edgecolors='w')

        self._apply_aesthetic_style(
            ax, 
            f"Aligned Semantic Trajectory relative to '{self.disease_name}'", 
            "PCA Component 1", 
            "PCA Component 2"
        )
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.2, linewidth=0.8)
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.2, linewidth=0.8)
        plt.tight_layout()
        
        output_path = self.plots_path / f"pca_trajectory_relative_{self.target_year}.pdf"
        fig.savefig(output_path)
        plt.close(fig)
        self.logger.info(f"Aligned PCA plot saved: {output_path}")

    def run(self, max_new_topics: int = 8, max_total_topics: int = 10) -> bool:
        """Executa a geração do relatório e feedback loop."""
        self.logger.info("=== Starting Report Generation ===")
        
        # 1. Gerar Plots Históricos
        self.process_plots()

        # 2. Gerar Plot PCA de Trajetória
        self.generate_pca_trajectory_plot()

        # 3. Gerar Novos Plots Científicos
        if self.ground_truth:
            self.generate_ranking_convergence_plot()
            self.generate_lead_time_scatter_plot()
            self.generate_discovery_timeline_plot()
        
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
