import datetime
from pathlib import Path

from iterative_topic_expansion import IterativeTopicExpansion
from data_collection_module import DataCollection
from preprocessing_module import Preprocessing
from embeddings_training_automl import SequentialEmbeddingTrainingAutoML, ModelType
from candidate_model_training_module import CandidateModelTraining
from dotproduct_generation_module import ValidationModule
from model_selector_module import ModelSelector
from model_evaluation_module import ModelEvaluator
from latent_knowledge_report_module import LatentKnowledgeReportGenerator
from utils import LoggerFactory, _load_checkpoint, _save_checkpoint, normalize_disease_name

def run_full_pipeline(disease_name: str, max_topics: int, max_new_topics: int, train_val_test_split: list[float]):
    if len(train_val_test_split) != 3:
        raise ValueError("The train_val_test_split list must have 3 values!")

    logger = LoggerFactory.setup_logger("MainPipeline", log_to_file=True, log_file='main_pipeline.log')
    logger.info(f"Starting full pipeline for disease: {disease_name}")

    normalized_disease_name = normalize_disease_name(disease_name)
    checkpoint_data = _load_checkpoint(normalized_disease_name)
    
    last_expansion_year = checkpoint_data.get("last_expansion_year")
    model_dev_end_year = checkpoint_data.get("model_dev_end_year")
    model_selection_end_year = checkpoint_data.get("model_selection_end_year")
    best_model = checkpoint_data.get("best_model_name")

    # ===================================================================================
    # PHASE 1: Iterative Topic Expansion
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 1: Iterative Topic Expansion")
    logger.info("="*80)

    # Checkpoint 1: Topic Expansion
    if not checkpoint_data["phase_1_topic_expansion_completed"]:
        topic_expander = IterativeTopicExpansion(
            disease_name=disease_name,
            max_topics=max_topics,
            max_new_topics=max_new_topics
        )
        last_expansion_year = topic_expander.run()
        logger.info(f"Topic expansion phase concluded. The last processed year was {last_expansion_year}.")
        
        checkpoint_data["last_expansion_year"] = last_expansion_year
        checkpoint_data["phase_1_topic_expansion_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info(f"Skipping Phase 1: Iterative Topic Expansion (completed in a previous run). Last processed year: {last_expansion_year}")

    # ===================================================================================
    # PHASE 2: Models Development
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 2: Models Development")
    logger.info("="*80)

    model_dev_start_year = last_expansion_year + 1
    today_year = datetime.datetime.now().year
    year_range = today_year - model_dev_start_year
    
    if model_dev_end_year is None:
        model_dev_end_year = int(model_dev_start_year + train_val_test_split[0] * year_range)
        checkpoint_data["model_dev_end_year"] = model_dev_end_year
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    
    logger.info(f"Model development phase will run from {model_dev_start_year} to {int(model_dev_end_year)}.")

    # Checkpoint 2: Full Data Collection
    if not checkpoint_data["phase_2_data_collection_completed"]:
        logger.info("Running full data collection...")
        for year in range(model_dev_start_year, today_year + 1):
            logger.info(f"Collecting data for year {year}...")
            dc = DataCollection(disease_name=disease_name, target_year=year)
            dc.run()
        logger.info("Full data collection complete.")
        checkpoint_data["phase_2_data_collection_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info(f"Skipping Step 2: Data Collection (completed in a previous run).")

    # Checkpoint 3: Full Data Preprocessing
    if not checkpoint_data["phase_3_preprocessing_completed"]:
        logger.info("Running full preprocessing...")
        preprocessor = Preprocessing(disease_name=disease_name, target_year=today_year, incremental=False)
        preprocessor.run(force_full=True)
        logger.info("Full preprocessing complete.")
        checkpoint_data["phase_3_preprocessing_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info(f"Skipping Step 3: Preprocessing (completed in a previous run).")

    logger.info("Starting candidate model training...")

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
        logger.info(f"Skipping Step 4: Candidate Model Training (completed in a previous run).")
    
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
        logger.info(f"Skipping Step 5: Metric Generation (completed in a previous run).")

    # ===================================================================================
    # PHASE 3: Selection
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 3: Model Selection")
    logger.info("="*80)

    model_selection_start_year = model_dev_end_year + 1
    
    if model_selection_end_year is None:
        model_selection_end_year = int(model_selection_start_year + train_val_test_split[1] * year_range)
        checkpoint_data["model_selection_end_year"] = model_selection_end_year
        _save_checkpoint(normalized_disease_name, checkpoint_data)

    # Checkpoint 6: Model Selection
    if not checkpoint_data["phase_6_model_selection_completed"]:
        logger.info("Selecting the best model based on latent knowledge metrics...")
        
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
        logger.info(f"Skipping Step 6: Model Selection (completed in a previous run).")
        best_model = checkpoint_data.get("best_model_name", "None")


    # ===================================================================================
    # PHASE 4: Final Test / Report
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 4: Final Report Generation")
    logger.info("="*80)

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
        
        me.run(generate_latex=True)
        
        logger.info("Final report generation complete.")
        checkpoint_data["phase_7_final_report_completed"] = True
        _save_checkpoint(normalized_disease_name, checkpoint_data)
    else:
        logger.info(f"Skipping Step 7: Final Report Generation (completed in a previous run).")


    logger.info("="*80)
    logger.info("FULL PIPELINE FINISHED")
    logger.info("="*80)


if __name__ == '__main__':
    run_full_pipeline(
        disease_name='acute myeloid leukemia',
        max_topics=9,
        max_new_topics=9,
        train_val_test_split=[0.6, 0.2, 0.2]
    )
