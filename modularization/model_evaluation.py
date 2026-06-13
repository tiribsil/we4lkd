import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from embedding_training import CandidateModelTraining
from metric_generation import ValidationModule
from reporting import LatentKnowledgeReportGenerator
from extract_report_years import GroundTruthGenerator
from utils import get_logger, normalize_disease_name, _load_checkpoint

class ModelEvaluator:
    """
    Módulo responsável pela Fase 4 (Final Test / Report).
    Executa o treinamento incremental do MELHOR modelo selecionado no período de teste,
    avalia sua performance final e gera o relatório visual.
    """

    def __init__(
        self, 
        disease_name: str, 
        model_name: str, 
        corpus_start_year: int, 
        test_start_year: int, 
        test_end_year: int,
        ground_truth: Optional[Dict[str, int]] = None,
        models: Optional[Dict[str, List]] = None
    ):
        self.disease_name = disease_name
        self.normalized_disease_name = normalize_disease_name(disease_name)
        self.model_name = model_name
        
        self.corpus_start_year = corpus_start_year
        self.test_start_year = test_start_year
        self.test_end_year = test_end_year
        self.ground_truth = ground_truth
        
        self.logger = get_logger(self.__class__.__name__)
        
        self.base_path = Path(f"data/{self.normalized_disease_name}")
        self.top_n_base_path = self.base_path / "validation" / self.model_name / "top_n_compounds"

        # Instancia o treinador para ter acesso aos hiperparâmetros e métodos de treino
        self.trainer = CandidateModelTraining(
            disease_name=disease_name,
            start_year=corpus_start_year,
            end_year=test_end_year
        )

        # Carrega combinações de modelos
        if models:
            self.trainer.model_combinations.update(models)
        else:
            # Fallback: tentar carregar do checkpoint se não for fornecido
            checkpoint = _load_checkpoint(self.normalized_disease_name)
            if checkpoint and "trained_models_info" in checkpoint:
                self.trainer.model_combinations.update(checkpoint["trained_models_info"])
                self.logger.info(f"Loaded {len(checkpoint['trained_models_info'])} models from checkpoint.")

    def _get_model_params(self) -> List[any]:
        """Recupera os hiperparâmetros do modelo selecionado (ex: w2v_comb2)."""
        if self.model_name in self.trainer.model_combinations:
            return self.trainer.model_combinations[self.model_name]
        else:
            raise ValueError(f"Model '{self.model_name}' not found in defined combinations.")

    def _get_rank_at_year(self, compound: str, year: int) -> Optional[int]:
        """
        Returns the 1-based rank of `compound` in the top_n_score.csv for `year`.
        Returns None if the file is missing or the compound is not listed.
        """
        year_path = self.top_n_base_path / str(year)
        if not year_path.exists():
            return None
        csv_files = list(year_path.glob("top_*_score.csv"))
        if not csv_files:
            return None
        try:
            df = pd.read_csv(csv_files[0])
            col = 'chemical_name' if 'chemical_name' in df.columns else 'compound_name'
            if col not in df.columns:
                return None
            names = df[col].tolist()
            if compound in names:
                return names.index(compound) + 1  # 1-based rank
        except Exception as e:
            self.logger.debug(f"Could not read rank for {compound} at {year}: {e}")
        return None

    def _compute_extended_metrics(
        self,
        details_df: pd.DataFrame,
        ground_truth: Dict[str, int],
    ) -> Dict[str, Any]:
        """
        Computes all applicable metrics from metric_ideas.tex.
        Returns a flat dict of metric_name -> scalar value.
        """
        t0 = self.corpus_start_year
        years = list(range(self.test_start_year, self.test_end_year + 1))
        n_years = len(years)
        results: Dict[str, Any] = {}

        # ── TDG (Time-to-Discovery Gain) ──────────────────────────────────────
        # Already encoded in details_df['years_early']
        tdg_values = details_df['years_early'].tolist()
        results['tdg_mean'] = float(np.mean(tdg_values)) if tdg_values else 0.0
        results['tdg_min']  = float(np.min(tdg_values))  if tdg_values else 0.0
        results['tdg_max']  = float(np.max(tdg_values))  if tdg_values else 0.0

        # ── NDG (Normalized Discovery Gain) ───────────────────────────────────
        # NDG(c) = (t*(c) - t_pred(c)) / (t*(c) - t0)
        ndg_values = []
        for _, row in details_df.iterrows():
            t_star = row['literature_report_year']
            t_pred = row['recommendation_year']
            denom  = t_star - t0
            if denom > 0:
                ndg_values.append((t_star - t_pred) / denom)
        results['ndg_mean'] = float(np.mean(ndg_values)) if ndg_values else 0.0
        results['ndg_min']  = float(np.min(ndg_values))  if ndg_values else 0.0
        results['ndg_max']  = float(np.max(ndg_values))  if ndg_values else 0.0

        # ── LKD (mean TDG over compounds the model correctly anticipated) ─────
        detected_tdg = [v for v in tdg_values if v > 0]
        results['lkd']   = float(np.mean(detected_tdg)) if detected_tdg else 0.0
        results['lkd_n'] = len(detected_tdg)

        # ── Weighted LKD ──────────────────────────────────────────────────────
        # LKD_w = mean( TDG(c) * (1 / rank_at_t_pred(c)) ) over all matched compounds
        weighted_vals = []
        for _, row in details_df.iterrows():
            compound = row['compound']
            rec_year = int(row['recommendation_year'])
            tdg      = row['years_early']
            rank     = self._get_rank_at_year(compound, rec_year)
            if rank is not None and rank > 0:
                weighted_vals.append(tdg * (1.0 / rank))
        results['weighted_lkd'] = float(np.mean(weighted_vals)) if weighted_vals else 0.0

        # ── Temporal Hit@K ────────────────────────────────────────────────────
        # For each year: fraction of not-yet-discovered GT compounds in top-K
        Ks = [5, 10, 20]
        hitk_lists: Dict[int, List[float]] = {k: [] for k in Ks}
        for year in years:
            year_path  = self.top_n_base_path / str(year)
            if not year_path.exists():
                continue
            csv_files = list(year_path.glob("top_*_score.csv"))
            if not csv_files:
                continue
            try:
                df_rank = pd.read_csv(csv_files[0])
                col = 'chemical_name' if 'chemical_name' in df_rank.columns else 'compound_name'
                if col not in df_rank.columns:
                    continue
                ranked = df_rank[col].tolist()
            except Exception:
                continue
            # Compounds NOT yet discovered at this year (future discoveries)
            future_gt = {c for c, yr in ground_truth.items() if yr > year}
            n_future = len(future_gt)
            if n_future == 0:
                continue
            for K in Ks:
                top_k_set = set(ranked[:K])
                hits = len(top_k_set & future_gt)
                hitk_lists[K].append(hits / n_future)
        for K in Ks:
            vals = hitk_lists[K]
            results[f'hit_at_{K}'] = float(np.mean(vals)) if vals else 0.0

        # ── AUC Gain vs. random baseline ──────────────────────────────────────
        # Model curve: cumulative fraction of GT compounds recommended by year t
        # Random baseline: linear ramp from 0 to 1 (uniform discovery rate)
        gt_compounds = set(ground_truth.keys())
        first_rec_year: Dict[str, int] = {}
        for year in years:
            year_path = self.top_n_base_path / str(year)
            if not year_path.exists():
                continue
            csv_files = list(year_path.glob("top_*_score.csv"))
            if not csv_files:
                continue
            try:
                df_r = pd.read_csv(csv_files[0])
                col  = 'chemical_name' if 'chemical_name' in df_r.columns else 'compound_name'
                if col not in df_r.columns:
                    continue
                for c in df_r[col]:
                    if c in gt_compounds and c not in first_rec_year:
                        first_rec_year[c] = year
            except Exception:
                continue

        n_gt = len(gt_compounds)
        if n_gt > 0 and n_years > 1:
            model_curve  = []
            random_curve = []
            for i, year in enumerate(years):
                discovered = sum(1 for yr in first_rec_year.values() if yr <= year)
                model_curve.append(discovered / n_gt)
                random_curve.append(i / (n_years - 1))   # 0 → 1 linearly
            x_norm = [i / (n_years - 1) for i in range(n_years)]
            auc_model  = float(np.trapz(model_curve,  x_norm))
            auc_random = float(np.trapz(random_curve, x_norm))  # always ≈ 0.5
            results['auc_model']  = auc_model
            results['auc_random'] = auc_random
            results['auc_gain']   = auc_model - auc_random
        else:
            results['auc_model']  = 0.0
            results['auc_random'] = 0.0
            results['auc_gain']   = 0.0

        # ── Emergence Score ───────────────────────────────────────────────────
        # Max delta_normalized_dot_product per detected compound (from history CSVs)
        history_path       = self.base_path / "validation" / self.model_name / "compound_history"
        detected_compounds = set(details_df['compound'].tolist())
        emergence_scores   = []
        if history_path.exists():
            for compound in detected_compounds:
                fname     = re.sub(r'[\\/*?:"<>|]', '_', compound)[:50]
                hist_file = history_path / f"{fname}.csv"
                if not hist_file.exists():
                    continue
                try:
                    hdf = pd.read_csv(hist_file)
                    if 'delta_normalized_dot_product' in hdf.columns:
                        max_delta = hdf['delta_normalized_dot_product'].max()
                        if pd.notna(max_delta):
                            emergence_scores.append(float(max_delta))
                except Exception:
                    pass
        results['emergence_mean'] = float(np.mean(emergence_scores)) if emergence_scores else 0.0
        results['emergence_max']  = float(np.max(emergence_scores))  if emergence_scores else 0.0

        # ── LKD-Composite (α = 0.8 β = 0.2) ──────────────────────────────────
        results['lkd_composite'] = (
            0.8 * results['ndg_mean'] + 0.2 * results['hit_at_10']
        ) / 1.0

        return results

    def compute_metrics(self) -> Tuple[float, Dict[int, float]]:
        """Alias para _calculate_final_performance, usado pelo ModelSelector."""
        return self._calculate_final_performance()

    def _calculate_final_performance(self) -> Tuple[float, Dict[int, float]]:
        """
        Calcula a métrica 'Mean Years Early' e gera estatísticas detalhadas.
        """
        if self.ground_truth:
            ground_truth = self.ground_truth
        else:
            self.logger.info("Generating Ground Truth for evaluation...")
            gt_gen = GroundTruthGenerator(self.disease_name, self.logger)
            ground_truth = gt_gen.generate_ground_truth(threshold=3)
            self.ground_truth = ground_truth
        
        if not ground_truth:
            self.logger.warning("No ground truth generated. Score will be 0.")
            return 0.0, {}

        # Encontrar primeira recomendação no período de teste
        first_recommendation = {}
        all_first_recommendation = {}
        
        for year in range(self.test_start_year, self.test_end_year + 1):
            year_path = self.top_n_base_path / str(year)
            if not year_path.exists(): continue
                
            csv_files = list(year_path.glob("top_*_score.csv"))
            if not csv_files: continue
            
            try:
                df = pd.read_csv(csv_files[0])
                col = 'chemical_name' if 'chemical_name' in df.columns else 'compound_name'
                
                if col in df.columns:
                    for compound in df[col].values:
                        if compound not in all_first_recommendation:
                            all_first_recommendation[compound] = year
                        if compound not in first_recommendation and compound in ground_truth:
                            first_recommendation[compound] = year
            except Exception as e:
                self.logger.error(f"Error reading ranking file for year {year}: {e}")

        # Calculate hit percentage based on user's rules
        hits = 0
        all_compounds = set(all_first_recommendation.keys()).union(set(ground_truth.keys()))
        for c in all_compounds:
            rec_year = all_first_recommendation.get(c)
            rep_year = ground_truth.get(c)
            
            if rec_year is not None and rep_year is None:
                hits += 1  # first recommendation but no report year
            elif rec_year is not None and rep_year is not None and rec_year < rep_year:
                hits += 1  # first recommendation before report year

        hit_percentage = (hits / len(all_compounds) * 100) if all_compounds else 0.0

        # Preparar dados
        details = []
        for compound, rec_year in first_recommendation.items():
            report_year = ground_truth[compound]
            if report_year < self.test_start_year:
                continue
            how_early = report_year - rec_year
            details.append((compound, rec_year, report_year, how_early))

        if not details:
            self.logger.warning("No intersection between recommendations and ground truth in test period.")
            return 0.0, {}

        # Criar DataFrame
        details_df = pd.DataFrame(details, columns=['compound', 'recommendation_year', 'literature_report_year', 'years_early'])
        
        # Cálculos Estatísticos
        mean_early = details_df['years_early'].mean()
        median_early = details_df['years_early'].median()
        std_dev = details_df['years_early'].std() if len(details_df) > 1 else 0.0
        mode = details_df['years_early'].mode().tolist()
        
        # Logar Detalhes (Formato similar ao script antigo)
        self.logger.info(f"{'='*40}")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"  Test Period: {self.test_start_year}-{self.test_end_year}")
        self.logger.info(f"  Anticipation Mean: {mean_early:.2f}")
        self.logger.info(f"  Anticipation Median: {median_early}")
        self.logger.info(f"  Anticipation Std Dev: {std_dev:.2f}")
        self.logger.info(f"  Anticipation Mode: {mode}")
        self.logger.info(f"  Hit Percentage: {hit_percentage:.2f}%")
        
        self.logger.info("Top 5 Biggest Anticipations:")
        top_5_biggest = details_df.nlargest(5, 'years_early')
        for _, row in top_5_biggest.iterrows():
            self.logger.info(f"  {row['years_early']} years - {row['compound']}")

        self.logger.info("Top 5 Smallest (Delays):")
        top_5_smallest = details_df.nsmallest(5, 'years_early')
        for _, row in top_5_smallest.iterrows():
            self.logger.info(f"  {row['years_early']} years - {row['compound']}")
        # ── Extended Metrics (metric_ideas.tex) ──────────────────────────────
        ext = self._compute_extended_metrics(details_df, ground_truth)

        self.logger.info(f"{'='*40}")
        self.logger.info("  Extended Metrics (metric_ideas.tex)")
        self.logger.info(f"  {'─'*36}")
        self.logger.info(
            f"  TDG  mean={ext['tdg_mean']:+.2f} yr  "
            f"min={ext['tdg_min']:+.0f}  max={ext['tdg_max']:+.0f}"
        )
        self.logger.info(
            f"  NDG  mean={ext['ndg_mean']:.3f}  "
            f"min={ext['ndg_min']:.3f}  max={ext['ndg_max']:.3f}"
        )
        self.logger.info(
            f"  LKD (anticipations only, n={ext['lkd_n']}): {ext['lkd']:.2f} yr"
        )
        self.logger.info(f"  Weighted LKD: {ext['weighted_lkd']:.4f}")
        self.logger.info(f"  {'─'*36}")
        self.logger.info(f"  Temporal Hit@K (avg over years):")
        self.logger.info(f"    Hit@5  = {ext['hit_at_5']:.4f}")
        self.logger.info(f"    Hit@10 = {ext['hit_at_10']:.4f}")
        self.logger.info(f"    Hit@20 = {ext['hit_at_20']:.4f}")
        self.logger.info(f"  {'─'*36}")
        self.logger.info(
            f"  AUC Gain vs. random: {ext['auc_gain']:+.4f}  "
            f"(model={ext['auc_model']:.4f}, random={ext['auc_random']:.4f})"
        )
        self.logger.info(f"  {'─'*36}")
        self.logger.info(f"  Emergence Score (detected compounds):")
        self.logger.info(
            f"    mean={ext['emergence_mean']:.4f}  max={ext['emergence_max']:.4f}"
        )
        self.logger.info(f"  {'─'*36}")
        self.logger.info(f"  LKD-Composite (α=0.8 β=0.2): {ext['lkd_composite']:.4f}")
        self.logger.info(f"{'='*40}")
        
        # Salvar CSV
        output_csv = self.base_path / "reports" / "final_model_validation.csv"
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        details_df.to_csv(output_csv, index=False)
        self.logger.info(f"Full validation details saved to {output_csv}")

        # Cálculos Anuais
        annual_scores = {}
        for year in range(self.test_start_year, self.test_end_year + 1):
            year_data = details_df[details_df['recommendation_year'] == year]
            if not year_data.empty:
                annual_scores[year] = year_data['years_early'].mean()
            else:
                annual_scores[year] = 0.0

        return ext['lkd_composite'], annual_scores

    def run(self) -> bool:
        """
        Executa o pipeline de avaliação final.
        """
        self.logger.info(f"=== Starting Final Evaluation for '{self.model_name}' ===")
        
        try:
            # 1. Recuperar Parâmetros
            params = self._get_model_params()
            self.logger.info(f"Hyperparameters: {params}")

            # 2. Treinamento Incremental (Ano a Ano no período de teste)
            self.logger.info("--- Step 1: Incremental Training ---")
            for year in range(self.test_start_year, self.test_end_year + 1):
                # O método train_specific_model verifica se o modelo já existe antes de treinar
                success = self.trainer.train_specific_model(self.model_name, params, year)
                if not success:
                    self.logger.error(f"Failed to train model for year {year}. Aborting.")
                    return False

            # 3. Geração de Métricas e Rankings
            self.logger.info("--- Step 2: Generating Metrics & Rankings ---")
            # ValidationModule é inteligente o suficiente para processar o range de uma vez
            validator = ValidationModule(
                disease_name=self.disease_name,
                model_subfolder=self.model_name,
                start_year=self.test_start_year,
                end_year=self.test_end_year,
                use_chembl=True,
                top_n_to_save=20
            )
            if not validator.run():
                self.logger.error("Validation module failed.")
                return False

            # 4. Cálculo de Performance (Score Final)
            self.logger.info("--- Step 3: Calculating Final Performance ---")
            # Atualiza self.ground_truth se for gerado internamente
            if not self.ground_truth:
                self._calculate_final_performance()
            else:
                self._calculate_final_performance()

            # 5. Geração do Relatório Visual
            self.logger.info("--- Step 4: Generating Visual Report ---")
            # O ReportGenerator vai pegar os dados gerados pelo ValidationModule
            # e criar os gráficos (incluindo a trajetória PCA).
            reporter = LatentKnowledgeReportGenerator(
                disease_name=self.disease_name,
                model_subfolder=self.model_name,
                start_year=self.test_start_year,
                target_year=self.test_end_year, # Foca o relatório no último ano
                top_n_to_plot=15,
                ground_truth=self.ground_truth,
                corpus_start_year=self.corpus_start_year
            )
            reporter.run()

            self.logger.info("=== Final Evaluation Completed Successfully ===")
            return True

        except Exception as e:
            self.logger.exception(f"Fatal error in ModelEvaluator: {e}")
            return False

if __name__ == "__main__":
    # Exemplo de uso manual
    evaluator = ModelEvaluator(
        disease_name="acute myeloid leukemia",
        model_name="w2v_comb1",
        corpus_start_year=1990,
        test_start_year=2021,
        test_end_year=2024
    )
    evaluator.run()
