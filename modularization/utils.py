import os
import string
import logging
from pathlib import Path
from functools import reduce
from logging.handlers import RotatingFileHandler

class TargetYearFilter(logging.Filter):
    def __init__(self, target_year: str):
        super().__init__()
        self.target_year = target_year

    def filter(self, record):
        record.target_year = self.target_year
        return True


class LoggerFactory:
    @staticmethod
    def setup_logger(name: str = "logger", target_year: str = "0000", log_level: int = logging.INFO,
                     log_to_file: bool = False, log_file: str = "app.log",
                     max_bytes: int = 5 * 1024 * 1024, backup_count: int = 3) -> logging.Logger:

        logger = logging.getLogger(name)
        if logger.handlers:
            return logger

        logger.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(target_year)s - %(levelname)s - %(message)s"
        )

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.addFilter(TargetYearFilter(target_year))
        logger.addHandler(console_handler)

        if log_to_file:
            os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
            file_handler = RotatingFileHandler(
                log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            file_handler.addFilter(TargetYearFilter(target_year))
            logger.addHandler(file_handler)

        return logger


def normalize_disease_name(disease_name: str) -> str:
        return disease_name.lower().translate(str.maketrans('', '', string.punctuation)).replace(' ', '_')

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