# run_pipeline.py

import os
import sys
import importlib
from pathlib import Path

# Adiciona o diretório 'src' ao path para que possamos importar os scripts
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Tenta importar as funções de 'target_disease.py'
from src.target_disease import set_target_disease, get_normalized_target_disease

# Definição das etapas do pipeline
# Cada entrada contém: o nome do módulo, a função principal, uma descrição e o caminho do resultado.
PIPELINE_STEPS = [
    {
        "module": "1_crawler", "main_func": "main",
        "desc": "Crawl PubMed for relevant paper abstracts based on the target disease.",
        "output": "data/{disease}/corpus/raw_abstracts/"
    },
    {
        "module": "2_merge_txt", "main_func": "main",
        "desc": "Merge individual abstract files into aggregated yearly files.",
        "output": "data/{disease}/corpus/aggregated_abstracts/"
    },
    {
        "module": "3_ner_table_generator", "main_func": "main",
        "desc": "Generate a Named Entity Recognition (NER) table from aggregated abstracts.",
        "output": "data/{disease}/corpus/ner_table.csv"
    },
    {
        "module": "4_clean_summaries", "main_func": "main",
        "desc": "Clean and preprocess abstracts using PySpark, normalizing synonyms.",
        "output": "data/{disease}/corpus/clean_abstracts/"
    },
    {
        "module": "5_train_yoy", "main_func": "main",
        "desc": "Train Word2Vec/FastText models on a year-over-year basis.",
        "output": "data/{disease}/models/"
    },
    {
        "module": "6_generate_dotproducts_csv", "main_func": "main",
        "desc": "Generate dot product scores between compounds and the target disease.",
        "output": "data/{disease}/validation/{model_type}/compound_history/"
    },
    {
        "module": "7_latent_knowledge_report", "main_func": "main",
        "desc": "Generate LaTeX report with historical plots of top compounds.",
        "output": "data/{disease}/reports/"
    },
    {
        "module": "8_xai", "main_func": "main",
        "desc": "Generate explanations (XAI) for the top compound-disease relationships.",
        "output": "data/{disease}/validation/{model_type}/xai/"
    },
]


def print_menu():
    """Prints the main menu of pipeline steps."""
    print("\n--- Latent Knowledge Discovery Pipeline ---")
    print("Please choose the range of steps to execute.")
    for i, step in enumerate(PIPELINE_STEPS):
        print(f"  {i + 1}. {step['module'].split('_', 1)[1].replace('_', ' ').title()}")
    print("-----------------------------------------")


def get_user_choice():
    """Gets and validates the user's choice for start and end steps."""
    while True:
        try:
            start_str = input(f"Enter the start step (1-{len(PIPELINE_STEPS)}): ")
            start_step = int(start_str)
            if 1 <= start_step <= len(PIPELINE_STEPS):
                break
            else:
                print(f"Invalid input. Please enter a number between 1 and {len(PIPELINE_STEPS)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    while True:
        try:
            end_str = input(f"Enter the end step ({start_step}-{len(PIPELINE_STEPS)}): ")
            end_step = int(end_str)
            if start_step <= end_step <= len(PIPELINE_STEPS):
                return start_step, end_step
            else:
                print(f"Invalid input. End step must be between {start_step} and {len(PIPELINE_STEPS)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def run_step(step_number, normalized_disease_name):
    """Imports and runs a single step from the pipeline."""
    step_info = PIPELINE_STEPS[step_number - 1]
    module_name = step_info["module"]
    main_func_name = step_info["main_func"]

    print(f"\n--- Running Step {step_number}: {step_info['desc']} ---")

    try:
        # Dynamically import the module and run its main function
        module = importlib.import_module(module_name)
        main_function = getattr(module, main_func_name)
        main_function()

        # Format the output path with the actual disease name
        # A bit of a hack for steps 6-8 which have a model_type in the path
        output_path = step_info["output"].format(disease=normalized_disease_name, model_type="w2v/ft")

        print(f"--- Step {step_number} completed successfully! ---")
        print(f"--> Output can be found at: ./{output_path}")
        return True

    except Exception as e:
        print(f"\n!!!!!! An error occurred during Step {step_number} ({module_name}) !!!!!!")
        print(f"Error Type: {type(e).__name__}")
        print(f"Error Details: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    normalized_disease = get_normalized_target_disease()

    # This script must be run from the project root
    os.chdir(Path(__file__).resolve().parent)

    # Display menu and get user's choice
    print_menu()
    start, end = get_user_choice()

    # Execute the selected range of steps
    success = False
    for i in range(start, end + 1):
        success = run_step(i, normalized_disease)
        if not success:
            print("\nPipeline execution halted due to an error.")
            break

    if success:
        print("\nAll selected pipeline steps completed successfully!")


if __name__ == "__main__":
    main()