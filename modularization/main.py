from data_collection_module import DataCollection
from preprocessing_module import Preprocessing
from embeddings_training_module import EmbeddingTraining
from dotproduct_generation_module import ValidationModule
from latent_knowledge_report_module import LatentKnowledgeReportGenerator


if __name__ == '__main__':
    start_year = 2020
    end_year = 2025

    disease = 'acute myeloid leukemia'
    model_type = 'w2v'
    optuna_trials = 2 #quantas vezes o optuna vai rodar o modelo para encontrar os melhores hiperparâmetros -quanto maior o valor, mais tempo demora, mas melhor fica o modelo final

    for current_year in range(start_year, end_year + 1):
        print(f"{'='*20} Processing year: {current_year} {'='*20}")

        data_collection_module = DataCollection(
        disease_name="acute myeloid leukemia",
        target_year=current_year,
        expand_synonyms=True,
        filter_synonyms=True)

        data_collection_module.run()

        preprocessing_module = Preprocessing(
                disease_name=disease,
                incremental=True
            )
            
        preprocessing_module.run(force_full=False)

        embedding_trainer = EmbeddingTraining(
            disease_name=disease,
            start_year=start_year,
            end_year=current_year,
            model_type=model_type,
            use_optuna=True,
            optuna_trials=optuna_trials,
            optuna_timeout=3600
        )
        
        embedding_trainer.run(year_over_year=False)


        validator = ValidationModule(disease_name=disease,
                                    start_year=start_year,
                                    end_year = current_year)
        
        validator.run()

        report_generator = LatentKnowledgeReportGenerator(
            disease_name=disease,
            model_type = model_type,
            top_n_compounds=20,
            delta_threshold=0.001
        )
        
        # Executar pipeline
        success = report_generator.run(generate_latex=True)
 