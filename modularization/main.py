from utils import *
from data_collection_module import DataCollection
from preprocessing_module import Preprocessing
from embeddings_training_automl import SequentialEmbeddingTrainingAutoML, ModelType
#from dotproduct_generation_module import ValidationModule
from latent_knowledge_report_module import LatentKnowledgeReportGenerator


if __name__ == '__main__':
    start_year = 1970
    end_year = 1980

    disease = 'acute myeloid leukemia'
    model_type = 'w2v'
    optuna_trials = 5 #quantas vezes o optuna vai rodar o modelo para encontrar os melhores hiperparâmetros -quanto maior o valor, mais tempo demora, mas melhor fica o modelo final

    for current_year in range(start_year, end_year + 1):
        logger = LoggerFactory.setup_logger("we4lkd", target_year=str(current_year), log_to_file=True, log_file=f'logs/{current_year}.log')
        logger.info(f"{'='*20} Processing year: {current_year} {'='*20}")

        data_collection_module = DataCollection(
            disease_name="acute myeloid leukemia",
            target_year=current_year,
            expand_synonyms=True,
            filter_synonyms=True
        )

        data_collection_module.run()

        preprocessing_module = Preprocessing(
            target_year=current_year,
            disease_name=disease,
            incremental=True
        )
            
        success = preprocessing_module.run(force_full=False)
        if not success:
            logger.error(f"Preprocessing failed for year {current_year}. Skipping to next year.")
            continue

        embedding_trainer = SequentialEmbeddingTrainingAutoML(
            disease_name=disease,
            start_year=start_year,
            end_year=current_year,
            automl_config={
                'candidate_models': [
                    ModelType.WORD2VEC,
                    ModelType.FASTTEXT,
                    ModelType.GLOVE,
                    ModelType.LSA,
                    ModelType.BIOBERT,
                    ModelType.PUBMEDBERT,
                    ModelType.SCIBERT,
                    ModelType.SBERT,
                    ModelType.BIOCLINICALBERT
                ],
                'use_pca_variants': True,
                'model_selection_time_budget': 1800,
                'hyperopt_trials': 5,
                'hyperopt_timeout': 900,
            },
        )
        
    
        success = embedding_trainer.run_automl()

        """validator = ValidationModule(
            disease_name=disease,
            start_year=start_year,
            end_year=current_year,
            use_chembl=True  # Will try ChEMBL, fallback to PubChem only
        )
        
        validator.run()

        report_generator = LatentKnowledgeReportGenerator(
            disease_name=disease,
            model_type = model_type,
            top_n_compounds=20,
            delta_threshold=0.001,
            target_year=current_year
        )
        
        # Executar pipeline
        success = report_generator.run(generate_latex=True)
 """
