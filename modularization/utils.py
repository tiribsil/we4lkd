import os
import string
import logging
import json
from pathlib import Path
from functools import reduce
from logging.handlers import RotatingFileHandler

class CompactFormatter(logging.Formatter):
    """Custom formatter for ultra-compact logs."""
    _NAME_MAP = {
        "CandidateModelTraining": "MDev",
        "ModelSelector": "Selector",
        "DataCollection": "DC",
        "Preprocessing": "PP",
        "ValidationModule": "Validator",
        "IterativeTopicExpansion": "Expansion",
        "ModelEvaluator": "Evaluator",
        "ContextualizationModule": "Context",
        "GroundTruthGenerator": "GT",
        "LatentKnowledgeReportGenerator": "Reporter"
    }

    def __init__(self):
        super().__init__("%(asctime)s | %(levelname).1s | %(name)s: %(message)s", datefmt="%m-%d %H:%M")

    def format(self, record):
        # Strip the 'modularization.' prefix and apply mapping
        name = record.name
        if name.startswith("modularization."):
            name = name[len("modularization."):]
        
        # Apply name mapping for brevity
        record.name = self._NAME_MAP.get(name, name)
        
        if record.name == "modularization":
            record.name = "root"
            
        return super().format(record)


class LoggerFactory:
    """Centralized logging factory for the project."""
    
    _CONFIGURED = False

    @staticmethod
    def setup(
        log_level: int = logging.INFO,
        log_to_file: bool = False,
        log_file: str = "logs/pipeline.log",
        max_bytes: int = 10 * 1024 * 1024,
        backup_count: int = 5,
    ):
        """Configure the root logger for the 'modularization' hierarchy."""
        root_logger = logging.getLogger("modularization")
        
        if LoggerFactory._CONFIGURED:
            return root_logger
            
        root_logger.setLevel(log_level)
        formatter = CompactFormatter()

        if root_logger.hasHandlers():
            root_logger.handlers.clear()

        # Console
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

        # File
        if log_to_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            root_logger.addHandler(file_handler)

        LoggerFactory._CONFIGURED = True
        root_logger.info("Logging initialized")
        return root_logger

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Get a logger within the project hierarchy."""
        # Ensure name starts with modularization if it doesn't
        if not name.startswith("modularization"):
            if name == "__main__":
                name = "modularization.main"
            else:
                name = f"modularization.{name}"
        return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """Utility helper to get a project logger."""
    return LoggerFactory.get_logger(name)


def normalize_disease_name(disease_name: str) -> str:
        return disease_name.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')

def _get_checkpoint_path(disease_name: str) -> Path:
    return Path(f"artifacts/{disease_name}_pipeline_checkpoint.json")

def _load_checkpoint(disease_name: str) -> dict:
    checkpoint_path = _get_checkpoint_path(disease_name)
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as f:
            return json.load(f)
    return {
        "disease_name": disease_name,
        "last_expansion_year": None,
        "phase_1_topic_expansion_completed": False,
        "phase_2_data_collection_completed": False,
        "phase_3_preprocessing_completed": False,
        "phase_4_automl_training_completed": False,
        "phase_5_metric_generation_completed": False,
        "phase_6_model_selection_completed": False,
        "phase_7_final_report_completed": False,
        "phase_8_contextualization_completed": False,
        "model_dev_end_year": None,
        "model_selection_end_year": None,
        "trained_models_info": {},
        "best_model_name": None
    }

def _save_checkpoint(disease_name: str, checkpoint_data: dict):
    checkpoint_path = _get_checkpoint_path(disease_name)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as f:
        json.dump(checkpoint_data, f, indent=4)

@staticmethod
def default_typo_corrections():
        return {
            'mol-ecule': 'molecule',
            '‑': '-',
            '‒': '-',
            '–': '-',
            '—': '-',
            '¯': '-',
            'à': 'a',
            'á': 'a',
            'â': 'a',
            'ã': 'a',
            'ä': 'a',
            'å': 'a',
            'ç': 'c',
            'è': 'e',
            'é': 'e',
            'ê': 'e',
            'ë': 'e',
            'í': 'i',
            'î': 'i',
            'ï': 'i',
            'ñ': 'n',
            'ò': 'o',
            'ó': 'o',
            'ô': 'o',
            'ö': 'o',
            '×': 'x',
            'ø': 'o',
            'ú': 'u',
            'ü': 'u',
            'č': 'c',
            'ğ': 'g',
            'ł': 'l',
            'ń': 'n',
            'ş': 's',
            'ŭ': 'u',
            'і': 'i',
            'ј': 'j',
            'а': 'a',
            'в': 'b',
            'н': 'h',
            'о': 'o',
            'р': 'p',
            'с': 'c',
            'т': 't',
            'ӧ': 'o',
            '⁰': '0',
            '⁴': '4',
            '⁵': '5',
            '⁶': '6',
            '⁷': '7',
            '⁸': '8',
            '⁹': '9',
            '₀': '0',
            '₁': '1',
            '₂': '2',
            '₃': '3',
            '₅': '5',
            '₇': '7',
            '₉': '9',
        }

@staticmethod
def default_units_and_symbols():
    return [
        '/μm', '/mol', '°c', '≥', '≤', '<', '>', '±', '%', '/mumol',
        'day', 'month', 'year', '·', 'week', 'days',
        'weeks', 'years', '/µl', 'μg', 'u/mg',
        'mg/m', 'g/m', 'mumol/kg', '/week', '/day', 'm²', '/kg', '®',
        'ﬀ', 'ﬃ', 'ﬁ', 'ﬂ', '£', '¥', '©', '«', '¬', '®', '°', '±', '²', '³',
        '´', '·', '¹', '»', '½', '¿',
        '׳', 'ᇞ​', '‘', '’', '“', '”', '•', '˂', '˙', '˚', '˜' , '…', '‰', '′',
        '″', '‴', '€',
        '™', 'ⅰ', '↑', '→', '↓', '∗', '∙', '∝', '∞', '∼', '≈', '≠', '≤', '≥', '≦', '≫', '⊘',
        '⊣', '⊿', '⋅', '═', '■', '▵', '⟶', '⩽', '⩾', '、', '气', '益', '粒', '肾', '补',
        '颗', '', '', '', '', '，'
    ]
