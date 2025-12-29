import datetime
from pathlib import Path
import pandas as pd

from topic_expansion import IterativeTopicExpansion
from data_collection import DataCollection
from preprocessing import Preprocessing
from embedding_training import CandidateModelTraining
from metric_generation import ValidationModule
from model_selection import ModelSelector
from model_evaluation import ModelEvaluator
from contextualization import ContextualizationModule
from utils import LoggerFactory, get_logger, _load_checkpoint, _save_checkpoint, normalize_disease_name

def run_full_pipeline(disease_name: str, max_topics: int, max_new_topics: int, train_val_test_split: list[float]):
    if len(train_val_test_split) != 3:
        raise ValueError("The train_val_test_split list must have 3 values!")

    # Initialize logging for the entire project
    LoggerFactory.setup(log_to_file=True, log_file='logs/main_pipeline.log')
    logger = get_logger(__name__)
    logger.info(f"Start: {disease_name}")

    normalized_disease_name = normalize_disease_name(disease_name)
    checkpoint_data = _load_checkpoint(normalized_disease_name)
    
    last_expansion_year = checkpoint_data.get("last_expansion_year")
    model_dev_end_year = checkpoint_data.get("model_dev_end_year")
    model_selection_end_year = checkpoint_data.get("model_selection_end_year")
    best_model = checkpoint_data.get("best_model_name")

    logger.info(">>> Phase 1 (Expansion)")

    # Checkpoint 1: Topic Expansion
    if not checkpoint_data["phase_1_topic_expansion_completed"]:
        topic_expander = IterativeTopicExpansion(
            disease_name=disease_name,
            max_topics=max_topics,
            max_new_topics=max_new_topics
        )
        last_expansion_year = topic_expander.run()
        logger.info(f"Expansion done: {last_expansion_year}")
        
        checkpoint_data["last_expansion_year"] = last_expansion_year
        checkpoint_data["phase_1_topic_expansion_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info(f"Phase 1 skipped ({last_expansion_year})")

    logger.info(">>> Phase 2 (Dev)")

    model_dev_start_year = last_expansion_year + 1
    today_year = datetime.datetime.now().year
    year_range = today_year - model_dev_start_year
    
    if model_dev_end_year is None:
        model_dev_end_year = int(model_dev_start_year + train_val_test_split[0] * year_range)
        checkpoint_data["model_dev_end_year"] = model_dev_end_year
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    
    logger.info(f"Model dev: {model_dev_start_year}-{int(model_dev_end_year)}")

    # Checkpoint 2: Full Data Collection
    if not checkpoint_data["phase_2_data_collection_completed"]:
        logger.info("Collecting data...")
        for year in range(model_dev_start_year, today_year + 1):
            logger.info(f"Year {year}...")
            dc = DataCollection(disease_name=disease_name, target_year=year)
            dc.run()
        logger.info("Full data collection complete.")
        checkpoint_data["phase_2_data_collection_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info("Data collection skipped")

    # Checkpoint 3: Full Data Preprocessing
    if not checkpoint_data["phase_3_preprocessing_completed"]:
        logger.info("Preprocessing data...")
        preprocessor = Preprocessing(disease_name=disease_name, target_year=today_year, incremental=False)
        preprocessor.run(force_full=True)
        logger.info("Full preprocessing complete.")
        checkpoint_data["phase_3_preprocessing_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info("Preprocessing skipped")

    logger.info("Training candidate models...")

    # Checkpoint 4: Candidate Model Training
    if not checkpoint_data["phase_4_automl_training_completed"]:
        cmt = CandidateModelTraining(
            disease_name=disease_name,
            start_year=model_dev_start_year,
            end_year=model_dev_end_year
        )
        models = cmt.run()
        
        checkpoint_data["trained_models_info"] = models
        checkpoint_data["phase_4_automl_training_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info("Model training skipped")
    
    models = checkpoint_data.get("trained_models_info")

    logger.info("Candidate model training complete.")

    # Checkpoint 5: Metric Generation
    if not checkpoint_data["phase_5_metric_generation_completed"]:
        logger.info("Generating metrics for all compounds for each model...")
        
        if not models:
            logger.error("No models found for metric generation. Skipping.")
            checkpoint_data["phase_5_metric_generation_completed"] = False # Re-attempt in future
            _save_checkpoint(normalized_disease_name, checkpoint_data)
            return
            
        for model_name, model_info in models.items():
            logger.info(f"Generating metrics for model: {model_name}")
            validator = ValidationModule(
                disease_name=disease_name,
                model_subfolder=model_name,
                start_year=model_dev_end_year,
                end_year=model_dev_end_year
            )
            validator.run()
            
        logger.info("Metric generation complete for all models.")

        checkpoint_data["phase_5_metric_generation_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info("Metric gen skipped")

    logger.info(">>> Phase 3 (Selection)")

    model_selection_start_year = model_dev_end_year + 1
    
    if model_selection_end_year is None:
        model_selection_end_year = int(model_selection_start_year + train_val_test_split[1] * year_range)
        checkpoint_data["model_selection_end_year"] = model_selection_end_year
        _save_checkpoint(normalized_disease_name, checkpoint_data)

    # Checkpoint 6: Model Selection
    if not checkpoint_data["phase_6_model_selection_completed"]:
        logger.info("Selecting best model...")
        
        selector = ModelSelector(
            disease_name=disease_name,
            models=models, # <--- dicionário retornado na etapa anterior
            corpus_start_year=model_dev_start_year,
            start_year=model_selection_start_year,
            end_year=model_selection_end_year
        )

        # --------------------------------------------------------------------------------
        #
        # Esse método vai ler os top_n de cada modelos, rodar o método de pontuação,
        # e escolher o que retornou melhor pontuação.
        #
        # Para rodar o método de pontuação, precisamos ter os top_n_compounds de cada ano.
        # Para isso, precisamos rodar cada modelo ITERATIVAMENTE de model_selection_start_year
        # até model_selection_end_year.
        #
        # O nome do modelo vai ser a chave dele no dicionário e o nome da pasta dele.
        #
        # - Treina todos os modelos do dicionário com artigos até start_year, start_year + 1,
        #   ..., end_year - 1, end_year.
        # - Gera as métricas dos compostos para cada um dos modelos treinados e top_n
        # - Latent knowledge score para cada um dos modelos
        # - Escolhe melhor
        #
        # --------------------------------------------------------------------------------
        
        best_model = selector.select_best_model()
        
        logger.info(f"Best model selected: {best_model}")

        checkpoint_data["best_model_name"] = best_model
        checkpoint_data["phase_6_model_selection_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info("Model selection skipped")
        best_model = checkpoint_data.get("best_model_name", "None")


    logger.info(">>> Phase 4 (Evaluation)")

    test_start_year = model_selection_end_year + 1
    test_end_year = today_year

    # --------------------------------------------------------------------------------
    # Checkpoint 7: Final Report Generation
    # --------------------------------------------------------------------------------
    if not checkpoint_data["phase_7_final_report_completed"]:
        logger.info(f"Generating final report using the selected model: '{best_model}'...")
        
        me = ModelEvaluator(
            disease_name, 
            best_model, 
            model_dev_start_year, 
            test_start_year, 
            test_end_year
        )
        
        # --------------------------------------------------------------------------------
        #
        # Esse aqui vai só rodar iterativamente para cada ano de test_start_year até
        # test_end_year, calcular a pontuação de conhecimento latente dnv e pronto.
        #
        # - Treina o melhor modelo com artigos até start_year, start_year + 1,
        #   ..., end_year - 1, end_year.
        # - Gera as métricas dos compostos e top_n de cada ano
        # - Latent knowledge score para avaliar
        # - Avaliação manual dos top_n
        #
        # --------------------------------------------------------------------------------
        
        me.run()
        
        logger.info("Final report generation complete.")
        checkpoint_data["phase_7_final_report_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info("Final report skipped")

    logger.info(">>> Phase 5 (Summary)")

    # Checkpoint 8: Contextualization
    if not checkpoint_data.get("phase_8_contextualization_completed"):
        logger.info(f"Contextualizing {test_end_year}...")
        
        # Path construction: data/{disease}/validation/{best_model}/top_n_compounds/{today_year}/top_{n}_score.csv
        # Assuming n=20 as standard or derived from configuration
        n = 20
        top_compounds_csv = Path(f'./data/{normalized_disease_name}/validation/{best_model}/top_n_compounds/{test_end_year}/top_{n}_score.csv')
        
        if top_compounds_csv.exists():
            try:
                logger.info(f"Reading: {top_compounds_csv}")
                df = pd.read_csv(top_compounds_csv)
                
                if 'chemical_name' in df.columns:
                    compounds = df['chemical_name'].tolist()
                    logger.info(f"Found {len(compounds)} compounds to analyze.")
                    
                    # Initialize module
                    logger.info("Initializing BioMistral model...")
                    context_module = ContextualizationModule(disease=disease_name)
                    
                    # Run analysis
                    results = context_module.analyze_batch(compounds)
                    
                    # Export results
                    output_file = Path(f'./data/{normalized_disease_name}/contextualization_results_{test_end_year}.json')
                    context_module.export_json(results, str(output_file))
                    
                    checkpoint_data["phase_8_contextualization_completed"] = True
                    _save_checkpoint(normalized_disease_name, checkpoint_data)
                    
                else:
                    logger.error(f"Column 'chemical_name' not found in {top_compounds_csv}")
                    
            except Exception as e:
                logger.error(f"Error during contextualization step: {e}")
        else:
            logger.warning(f"Top compounds file not found at {top_compounds_csv}. Skipping contextualization.")
            
    else:
        logger.info("Contextualization skipped")

    logger.info("FINISHED")


if __name__ == '__main__':
    run_full_pipeline(
        disease_name='acute myeloid leukemia',
        max_topics=9,
        max_new_topics=9,
        train_val_test_split=[0.6, 0.2, 0.2]
    )
