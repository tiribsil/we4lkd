import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from gensim.models import Word2Vec, FastText, KeyedVectors
import numpy as np
import shutil
from scipy.stats import qmc
import concurrent.futures
import os

from utils import LoggerFactory, normalize_disease_name

from embeddings_training_automl import ModelType, EmbeddingConfig, ModelFactory, GloVeModel as MittensGloVeModel

class CandidateModelTraining:
    def __init__(
        self,
        disease_name: str,
        start_year: int,
        end_year: int,
    ):
        self.logger = LoggerFactory.setup_logger(
            "candidate_model_training", target_year=str(start_year), log_to_file=False
        )
        self.disease_name = normalize_disease_name(disease_name)
        self.start_year = start_year
        self.end_year = end_year
        
        self.model_combinations: Dict[str, List[Any]] = {}
        self.model_combinations.update({
            "w2v_comb1": [200, 5, 2, 1, 15, 0.025, 15, 4], 
            "w2v_comb2": [200, 10, 3, 0, 5, 0.03, 20, 4],  
            "ft_comb1": [100, 5, 2, 1, 10, 0.025, 15, 4, 3, 6],   
            "glove_comb1": [300, 8, 30, 0.05, 5, 100] 
        })

        self.base_path = Path(f'./data/{self.disease_name}')
        self.corpus_path = Path(f'{self.base_path}/corpus/clean_abstracts/clean_abstracts.csv')
        self.models_base_path = Path(f'{self.base_path}/models')
        self.models_base_path.mkdir(parents=True, exist_ok=True)

        self._corpus_df = None

    @property
    def corpus_df(self) -> pd.DataFrame:
        if self._corpus_df is None:
            self._corpus_df = self._load_corpus()
        return self._corpus_df

    def _load_corpus(self) -> Optional[pd.DataFrame]:
        if not self.corpus_path.exists():
            self.logger.error(f"Corpus path not found at {self.corpus_path}")
            return None
        
        try:
            if self.corpus_path.is_dir():
                csv_files = list(self.corpus_path.glob('*.csv'))
                if not csv_files: return None
                df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
            else:
                df = pd.read_csv(self.corpus_path)
            
            if 'summary' not in df.columns:
                self.logger.error("Column 'summary' not found in corpus")
                return None
            
            if 'year_extracted' not in df.columns:
                if 'year' in df.columns:
                    df['year_extracted'] = df['year']
                else:
                    df['year_extracted'] = self.end_year # Fallback
            
            return df
        except Exception as e:
            self.logger.error(f"Error loading corpus: {e}")
            return None

    def _prepare_sentences(self, start_year: int, target_end_year: int) -> List[List[str]]:
        """
        Prepara sentenças filtrando até o ano alvo especificado.
        """
        df = self.corpus_df
        if df is None or df.empty: return []
        
        # Filtro temporal
        df_filtered = df[(df['year_extracted'] >= start_year) & (df['year_extracted'] <= target_end_year)]
        
        abstracts = df_filtered['summary'].dropna().tolist()
        self.logger.info(f"Prepared {len(abstracts)} abstracts ({start_year}-{target_end_year})")
        
        sentences = [str(abstract).split() for abstract in abstracts if abstract]
        return [s for s in sentences if len(s) > 0]
    
    def _create_config_from_params(self, architecture: str, params: List[Any], end_year: int) -> Optional[EmbeddingConfig]:
        """Helper para criar config baseado na lista de parâmetros."""
        if architecture == 'w2v':
            model_type = ModelType.WORD2VEC
            custom_params = {
                'vector_size': params[0], 
                'window': params[1], 
                'min_count': params[2],
                'sg': params[3], 
                'negative': params[4], 
                'alpha': params[5],
                'epochs': params[6], 
                'workers': params[7]
            }
        elif architecture == 'ft':
            model_type = ModelType.FASTTEXT
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'min_count': params[2],
                'sg': params[3], 'negative': params[4], 'alpha': params[5],
                'epochs': params[6], 'workers': params[7], 'min_n': params[8], 'max_n': params[9]
            }
        elif architecture == 'glove':
            model_type = ModelType.GLOVE
            custom_params = {
                'vector_size': params[0], 'window': params[1], 'max_iter': params[2],
                'learning_rate': params[3], 'min_count': params[4], 'x_max': params[5],
            }
        else:
            return None

        return EmbeddingConfig(
            model_type=model_type,
            use_pca=False,
            vector_size=params[0],
            custom_params=custom_params,
            end_year=end_year
        )

    def train_specific_model(self, model_name: str, params: List[Any], target_year: int) -> bool:
        """
        Método público para treinar um modelo específico até um ano alvo.
        Usado pelo ModelSelector.
        """
        # Determinar arquitetura pelo nome (ex: w2v_comb1 -> w2v)
        architecture = model_name.split('_')[0]
        
        # Verificar se modelo já existe para economizar tempo
        model_output_dir = self.models_base_path / model_name
        model_filename = f"{model_name}_{self.start_year}_{target_year}.model"
        model_path = model_output_dir / model_filename
        
        if model_path.exists():
            self.logger.info(f"Model {model_filename} already exists. Skipping training.")
            return True

        # Preparar dados
        sentences = self._prepare_sentences(self.start_year, target_year)
        if not sentences:
            self.logger.error(f"No data found for {self.start_year}-{target_year}")
            return False

        # Configurar e Treinar
        config = self._create_config_from_params(architecture, params, target_year)
        if not config:
            self.logger.error(f"Unknown architecture for {model_name}")
            return False

        try:
            self.logger.info(f"Training {model_name} up to {target_year}...")
            model_instance = ModelFactory.create_model(config)
            model_instance.train(sentences)
            
            # Salvar
            self._save_trained_model(model_instance, model_name, self.start_year, target_year)
            return True
        except Exception as e:
            self.logger.error(f"Failed to train {model_name}: {e}")
            return False

    def _save_trained_model(self, model_instance, model_key: str, start_year: int, end_year: int):
        model_output_dir = self.models_base_path / model_key
        model_output_dir.mkdir(parents=True, exist_ok=True)
        
        model_filename = f"{model_key}_{start_year}_{end_year}.model"
        model_path = model_output_dir / model_filename

        if hasattr(model_instance.model, 'save'):
            model_instance.model.save(str(model_path))
        elif isinstance(model_instance, MittensGloVeModel):
             if model_instance.word_vectors_keyed_vectors:
                model_instance.word_vectors_keyed_vectors.save_word2vec_format(str(model_path), binary=True)
        elif isinstance(model_instance.model, KeyedVectors):
            model_instance.model.save_word2vec_format(str(model_path), binary=True)
        else:
            import pickle
            with open(model_path, 'wb') as f:
                pickle.dump(model_instance.model, f)
        
        self.logger.info(f"Saved: {model_path}")

    # Inseridas: funções LHS e run como métodos da classe (substituem as versões globais)
    def _scale_and_cast(self, sample: np.ndarray, low: float, high: float, dtype: str):
        """Scale sample in [0,1) to [low, high] and cast according to dtype ('int'|'float'|'bin')."""
        val = low + sample * (high - low)
        if dtype == 'int':
            v = int(np.round(val))
            v = max(int(low), min(int(high), v))
            return v
        elif dtype == 'bin':
            # threshold at 0.5
            return int(val >= 0.5)
        else:
            return float(val)

    def _generate_lhs_samples(self, param_specs: List[tuple], n_samples: int) -> List[Dict[str, Any]]:
        """
        param_specs: list of tuples (name, low, high, dtype)
        dtype: 'int', 'float', 'bin'
        returns list of dicts mapping param name to sampled value
        """
        d = len(param_specs)
        sampler = qmc.LatinHypercube(d=d, seed=None)
        raw = sampler.random(n=n_samples)  # shape (n_samples, d)
        out = []
        for i in range(n_samples):
            row = {}
            for j, (name, low, high, dtype) in enumerate(param_specs):
                row[name] = self._scale_and_cast(raw[i, j], low, high, dtype)
            out.append(row)
        return out

    def _dict_to_list_params(self, architecture: str, hp: Dict[str, Any]) -> List[Any]:
        """Converte o dicionário do LHS para a lista ordenada esperada pelo _create_config_from_params."""
        if architecture == 'w2v':
            # Ordem: [vector_size, window, min_count, sg, negative, alpha, epochs, workers]
            return [
                hp['vector_size'], hp['window'], hp['min_count'], hp['sg'], 
                hp['negative'], hp['alpha'], hp['epochs'], hp['workers']
            ]
        elif architecture == 'ft':
            # Ordem: [vector_size, window, min_count, sg, negative, alpha, epochs, workers, min_n, max_n]
            return [
                hp['vector_size'], hp['window'], hp['min_count'], hp['sg'], 
                hp['negative'], hp['alpha'], hp['epochs'], hp['workers'],
                hp['min_n'], hp['max_n']
            ]
        elif architecture == 'glove':
            # Ordem: [vector_size, window, max_iter, learning_rate, min_count, x_max]
            return [
                hp['vector_size'], hp['window'], hp['max_iter'], 
                hp['learning_rate'], hp['min_count'], hp['x_max']
            ]
        return []

    def run(self) -> Dict[str, List[Any]]:
        """
        Executa LHS (Latin Hypercube Sampling) e treina modelos em paralelo.
        Salva os modelos em disco e popula self.model_combinations para uso futuro.
        """
        self.logger.info("=== Starting Candidate Training (LHS + Parallel) ===")
        
        # 1. Preparar sentenças internamente
        sentences = self._prepare_sentences(self.start_year, self.end_year)
        if not sentences:
            self.logger.error("No sentences found.")
            return {}

        n_sets = 7 # Amostras por família

        # 2. Definição dos Espaços de Parâmetros
        w2v_specs = [
            ('vector_size', 100, 300, 'int'), ('window', 2, 10, 'int'),
            ('min_count', 1, 5, 'int'), ('sg', 0, 1, 'bin'),
            ('negative', 5, 15, 'int'), ('alpha', 0.01, 0.05, 'float'),
            ('epochs', 5, 30, 'int'), ('workers', 1, 4, 'int'),
        ]

        ft_specs = w2v_specs + [ # Herda specs do w2v e adiciona específicos
            ('min_n', 2, 3, 'int'), ('max_n', 4, 6, 'int'),
        ]

        glove_specs = [
            ('vector_size', 50, 300, 'int'), ('window', 2, 8, 'int'),
            ('max_iter', 10, 50, 'int'), ('learning_rate', 0.01, 0.2, 'float'),
            ('min_count', 1, 5, 'int'), ('x_max', 10, 100, 'int'),
        ]

        # 3. Gerar Amostras LHS
        w2v_sets = self._generate_lhs_samples(w2v_specs, n_sets)
        ft_sets = self._generate_lhs_samples(ft_specs, n_sets)
        glove_sets = self._generate_lhs_samples(glove_specs, n_sets)

        # 4. Construir lista de tarefas
        tasks = []
        
        # Helper interno para preparar tarefa
        def add_tasks(arch, param_sets):
            for idx, hp in enumerate(param_sets, start=1):
                if arch == 'ft' and hp['max_n'] < hp['min_n']:
                    hp['max_n'], hp['min_n'] = hp['min_n'], hp['max_n']
                
                # Converte dict para lista ordenada usada pelo sistema
                params_list = self._dict_to_list_params(arch, hp)
                key = f"{arch}_lhs{idx}"
                tasks.append((key, arch, params_list))

        add_tasks('w2v', w2v_sets)
        add_tasks('ft', ft_sets)
        add_tasks('glove', glove_sets)

        # 5. Worker para execução paralela
        def _worker(task_data):
            model_key, arch, params = task_data
            try:
                # Usa a infraestrutura existente para criar config compatível
                config = self._create_config_from_params(arch, params, self.end_year)
                if not config:
                    return None

                # Treina usando ModelFactory (garante compatibilidade de Wrapper)
                model_instance = ModelFactory.create_model(config)
                model_instance.train(sentences)
                return (model_key, params, model_instance)
            except Exception as e:
                self.logger.error(f"Error training {model_key}: {e}")
                return None

        # 6. Executar e Salvar
        # Limita workers para evitar OOM (Out of Memory) já que modelos são pesados
        max_workers = min((os.cpu_count() or 4), 4)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_key = {executor.submit(_worker, t): t[0] for t in tasks}
            
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                result = future.result()
                
                if result:
                    m_key, m_params, m_instance = result
                    # Salva no disco (CRUCIAL para o ValidationModule encontrar depois)
                    self._save_trained_model(m_instance, m_key, self.start_year, self.end_year)
                    
                    # Atualiza o dicionário de combinações (CRUCIAL para o ModelSelector)
                    self.model_combinations[m_key] = m_params
                    self.logger.info(f"Finished & Saved: {m_key}")
                else:
                    self.logger.warning(f"Failed task: {key}")

        return self.model_combinations

if __name__ == '__main__':
    # Teste
    t = CandidateModelTraining("acute myeloid leukemia", 1990, 2000)
    # Exemplo de treino específico
    #t.train_specific_model("w2v_test", [100, 5, 2, 1, 5, 0.025, 5, 4], 1995)
    trained_models = t.run(sentences, trained_models)
