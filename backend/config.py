"""
Configuration centralisée pour le backend NLP
Élimine tous les éléments hardcodés
"""

import os
from typing import Dict, Any

class Config:
    """Configuration centralisée pour le backend"""
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', 'localhost')
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', 'http://localhost:3000,http://localhost:5173,http://localhost:5174,http://localhost:5175').split(',')
    
    # Autoencoder Configuration
    AUTOENCODER_CONFIG = {
        'input_dim': int(os.getenv('AUTOENCODER_INPUT_DIM', 1000)),
        'encoding_dim': int(os.getenv('AUTOENCODER_ENCODING_DIM', 128)),
        'hidden_layers': [512, 128],  # Peut être configuré via env si nécessaire
        'activation': os.getenv('AUTOENCODER_ACTIVATION', 'relu'),
        'optimizer': os.getenv('AUTOENCODER_OPTIMIZER', 'adam'),
        'learning_rate': float(os.getenv('AUTOENCODER_LEARNING_RATE', 0.0005)),
        'epochs': int(os.getenv('AUTOENCODER_EPOCHS', 100)),
        'batch_size': int(os.getenv('AUTOENCODER_BATCH_SIZE', 16)),
        'validation_split': float(os.getenv('AUTOENCODER_VALIDATION_SPLIT', 0.2)),
        'patience': int(os.getenv('AUTOENCODER_PATIENCE', 10))
    }
    
    # TF-IDF Configuration
    TFIDF_CONFIG = {
        'max_features': int(os.getenv('TFIDF_MAX_FEATURES', 1000)),
        'stop_words': os.getenv('TFIDF_STOP_WORDS', 'english'),
        'ngram_range': (1, 2),  # Peut être configuré si nécessaire
        'min_df': int(os.getenv('TFIDF_MIN_DF', 2)),
        'max_df': float(os.getenv('TFIDF_MAX_DF', 0.8)),
        'sublinear_tf': os.getenv('TFIDF_SUBLINEAR_TF', 'True').lower() == 'true'
    }
    
    # BERT Configuration
    BERT_CONFIG = {
        'available_models': [
            'distilbert-base-uncased',
            'bert-base-uncased', 
            'roberta-base'
        ],
        'default_model': os.getenv('BERT_DEFAULT_MODEL', 'distilbert-base-uncased'),
        'default_epochs': int(os.getenv('BERT_DEFAULT_EPOCHS', 3)),
        'default_batch_size': int(os.getenv('BERT_DEFAULT_BATCH_SIZE', 16)),
        'default_learning_rate': float(os.getenv('BERT_DEFAULT_LEARNING_RATE', 2e-5)),
        'max_length': int(os.getenv('BERT_MAX_LENGTH', 512))
    }
    
    # Pipeline Configuration
    PIPELINE_CONFIG = {
        'default_min_token_length': int(os.getenv('PIPELINE_MIN_TOKEN_LENGTH', 2)),
        'default_max_tokens': int(os.getenv('PIPELINE_MAX_TOKENS', 1000)),
        'default_stop_words': ['ok', 'well', 'um', 'uh'],
        'case_sensitive': os.getenv('PIPELINE_CASE_SENSITIVE', 'False').lower() == 'true',
        'preserve_numbers': os.getenv('PIPELINE_PRESERVE_NUMBERS', 'False').lower() == 'true',
        'preserve_punctuation': os.getenv('PIPELINE_PRESERVE_PUNCTUATION', 'False').lower() == 'true'
    }
    
    # Dataset Configuration
    DATASET_CONFIG = {
        'default_size': int(os.getenv('DATASET_DEFAULT_SIZE', 1000)),
        'max_size': int(os.getenv('DATASET_MAX_SIZE', 10000)),
        'min_text_length': int(os.getenv('DATASET_MIN_TEXT_LENGTH', 10)),
        'cache_enabled': os.getenv('DATASET_CACHE_ENABLED', 'True').lower() == 'true'
    }
    
    # File Paths
    MODELS_DIR = os.getenv('MODELS_DIR', './models')
    EMBEDDINGS_DIR = os.path.join(MODELS_DIR, 'embeddings')
    AUTOENCODER_DIR = os.path.join(MODELS_DIR, 'autoencoder')
    BERT_DIR = os.path.join(MODELS_DIR, 'bert')
    
    # Search Configuration
    SEARCH_CONFIG = {
        'default_top_k': int(os.getenv('SEARCH_DEFAULT_TOP_K', 5)),
        'max_top_k': int(os.getenv('SEARCH_MAX_TOP_K', 50)),
        'similarity_threshold': float(os.getenv('SEARCH_SIMILARITY_THRESHOLD', 0.1))
    }
    
    # Validation Configuration
    VALIDATION_CONFIG = {
        'min_training_texts': int(os.getenv('VALIDATION_MIN_TRAINING_TEXTS', 5)),
        'max_training_texts': int(os.getenv('VALIDATION_MAX_TRAINING_TEXTS', 1000)),
        'min_test_size': float(os.getenv('VALIDATION_MIN_TEST_SIZE', 0.1)),
        'max_test_size': float(os.getenv('VALIDATION_MAX_TEST_SIZE', 0.5))
    }

    @classmethod
    def get_autoencoder_config(cls) -> Dict[str, Any]:
        """Retourne la configuration de l'autoencoder"""
        return cls.AUTOENCODER_CONFIG.copy()
    
    @classmethod
    def get_tfidf_config(cls) -> Dict[str, Any]:
        """Retourne la configuration TF-IDF"""
        return cls.TFIDF_CONFIG.copy()
    
    @classmethod
    def get_bert_config(cls) -> Dict[str, Any]:
        """Retourne la configuration BERT"""
        return cls.BERT_CONFIG.copy()
    
    @classmethod
    def get_pipeline_config(cls) -> Dict[str, Any]:
        """Retourne la configuration de pipeline"""
        return cls.PIPELINE_CONFIG.copy()
    
    @classmethod
    def get_dataset_config(cls) -> Dict[str, Any]:
        """Retourne la configuration du dataset"""
        return cls.DATASET_CONFIG.copy()
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Retourne la configuration de recherche"""
        return cls.SEARCH_CONFIG.copy()
    
    @classmethod
    def validate_autoencoder_config(cls, config: Dict[str, Any]) -> bool:
        """Valide la configuration de l'autoencoder"""
        required_keys = ['input_dim', 'encoding_dim', 'learning_rate', 'epochs', 'batch_size']
        
        # Vérifier que toutes les clés requises sont présentes
        if not all(key in config for key in required_keys):
            return False
        
        # Valider les valeurs
        if config['input_dim'] <= 0 or config['input_dim'] > 10000:
            return False
        if config['encoding_dim'] <= 0 or config['encoding_dim'] > config['input_dim']:
            return False
        if config['learning_rate'] <= 0 or config['learning_rate'] >= 1:
            return False
        if config['epochs'] <= 0 or config['epochs'] > 1000:
            return False
        if config['batch_size'] <= 0 or config['batch_size'] > 1000:
            return False
        
        return True
    
    @classmethod
    def validate_search_config(cls, config: Dict[str, Any]) -> bool:
        """Valide la configuration de recherche"""
        if 'top_k' in config:
            top_k = config['top_k']
            if not isinstance(top_k, int) or top_k <= 0 or top_k > cls.SEARCH_CONFIG['max_top_k']:
                return False
        
        return True
    
    @classmethod
    def create_directories(cls):
        """Crée les répertoires nécessaires"""
        import os
        os.makedirs(cls.MODELS_DIR, exist_ok=True)
        os.makedirs(cls.EMBEDDINGS_DIR, exist_ok=True)
        os.makedirs(cls.AUTOENCODER_DIR, exist_ok=True)
        os.makedirs(cls.BERT_DIR, exist_ok=True)

# Configuration par défaut pour le développement
class DevelopmentConfig(Config):
    """Configuration pour le développement"""
    API_DEBUG = True
    
class ProductionConfig(Config):
    """Configuration pour la production"""
    API_DEBUG = False
    
    # Sécurité renforcée en production
    AUTOENCODER_CONFIG = Config.AUTOENCODER_CONFIG.copy()
    AUTOENCODER_CONFIG.update({
        'epochs': 50,  # Moins d'époques en production pour la vitesse
        'batch_size': 32  # Batch size plus grand pour l'efficacité
    })

# Sélection de la configuration selon l'environnement
def get_config():
    """Retourne la configuration appropriée selon l'environnement"""
    env = os.getenv('FLASK_ENV', 'development')
    
    if env == 'production':
        return ProductionConfig
    else:
        return DevelopmentConfig

# Configuration active
config = get_config() 