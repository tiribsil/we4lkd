import datetime
from pathlib import Path

from iterative_topic_expansion import IterativeTopicExpansion
from data_collection_module import DataCollection
from preprocessing_module import Preprocessing
from embeddings_training_automl import SequentialEmbeddingTrainingAutoML, ModelType
from dotproduct_generation_module import ValidationModule
# from selection_module import ModelSelector # To be created
from latent_knowledge_report_module import LatentKnowledgeReportGenerator
from utils import LoggerFactory

def run_full_pipeline(disease_name: str, max_topics: int, max_new_topics: int, train_val_test_split: list[float]):
    if len(train_val_test_split) != 3:
        raise ValueError("The train_val_test_split list must have 3 values!")

    logger = LoggerFactory.setup_logger("MainPipeline", log_to_file=True, log_file='main_pipeline.log')
    logger.info(f"Starting full pipeline for disease: {disease_name}")

    # ===================================================================================
    # PHASE 1: Iterative Topic Expansion
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 1: Iterative Topic Expansion")
    logger.info("="*80)

    topic_expander = IterativeTopicExpansion(
        disease_name=disease_name,
        max_topics=max_topics,
        max_new_topics=max_new_topics
    )
    last_expansion_year = topic_expander.run()
    logger.info(f"Topic expansion phase concluded. The last processed year was {last_expansion_year}.")

    # ===================================================================================
    # PHASE 2: Models Development
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 2: Models Development")
    logger.info("="*80)

    model_dev_start_year = last_expansion_year + 1
    today_year = datetime.datetime.now().year
    year_range = today_year - model_dev_start_year
    
    model_dev_end_year = model_dev_start_year + train_val_test_split[0] * year_range

    logger.info(f"Model development phase will run from {model_dev_start_year} to {model_dev_end_year}.")
    logger.info("Running full data collection...")
    
    for year in range(model_dev_start_year, today_year + 1):
        logger.info(f"Collecting data for year {year}...")
        dc = DataCollection(disease_name=disease_name, target_year=year)
        dc.run()

    logger.info("Full data collection complete.")
    logger.info("Running full preprocessing...")
    
    preprocessor = Preprocessing(disease_name=disease_name, target_year=today_year, incremental=False)
    preprocessor.run(force_full=True)

    logger.info("Full preprocessing complete.")
    logger.info("Starting AutoML training for all candidate models...")

    # ---------------------------------------------------------------------
    #
    # Aqui vem o treinamento de todos os modelos até model_dev_end_year.
    # Dicionário com todos os modelos?
    #
    # ---------------------------------------------------------------------

    logger.info("AutoML training complete.")

    logger.info("Generating metrics for all compounds for each model...")
    validator = ValidationModule(
        disease_name=disease_name,
        start_year=model_dev_start_year,
        end_year=model_dev_end_year
    )
    validator.run()
    logger.info("Metric generation complete for all models.")


    # ===================================================================================
    # PHASE 3: Selection
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 3: Model Selection")
    logger.info("="*80)

    model_selection_start_year = model_dev_end_year + 1
    model_selection_end_year = model_selection_start_year + train_val_test_split[1] * year_range

    # --- TO BE IMPLEMENTED ---
    # 
    # logger.info("Selecting the best model based on latent knowledge metrics...")
    #
    # selector = ModelSelector(
    #     disease_name=disease_name,
    #     start_year=model_selection_start_year,
    #     end_year=model_selection_end_year
    # )
    #
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
    # --------------------------------------------------------------------------------
    #
    # best_model = selector.select_best_model()
    #
    # logger.info(f"Best model selected: {best_model}")

    # ===================================================================================
    # PHASE 4: Final Test / Report
    # ===================================================================================
    logger.info("="*80)
    logger.info("PHASE 4: Final Report Generation")
    logger.info("="*80)

    test_start_year = model_selection_end_year + 1
    test_end_year = today_year

    logger.info(f"Generating final report using the selected model: '{best_model}'...")
    #
    # final_reporter = LatentKnowledgeReportGenerator(
    #     disease_name=disease_name,
    #     model_type=best_model,
    #     target_year=today_year,
    # )
    #
    # --------------------------------------------------------------------------------
    #
    # Esse aqui vai só rodar iterativamente para cada ano de test_start_year até
    # test_end_year, calcular a pontuação de conhecimento latente dnv e pronto.
    #
    # --------------------------------------------------------------------------------
    #
    # final_reporter.run(generate_latex=True)
    #
    logger.info("Final report generation complete.")


    logger.info("="*80)
    logger.info("FULL PIPELINE FINISHED")
    logger.info("="*80)


if __name__ == '__main__':
    run_full_pipeline(
        disease_name='acute myeloid leukemia',
        max_topics=9,
        max_new_topics=9,
        train_val_test_split=[0.8, 0.1, 0.1]
    )
