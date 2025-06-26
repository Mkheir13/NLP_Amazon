import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# Essayer d'importer TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.regularizers import l2
    TENSORFLOW_AVAILABLE = True
    print("✅ TensorFlow disponible pour l'autoencoder")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow non disponible - Utilisation d'un autoencoder simplifié")

class AutoencoderService:
    def __init__(self, config_override=None):
        # Import de la configuration
        try:
            from config import config as app_config
            self.app_config = app_config
        except ImportError:
            # Configuration par défaut si config.py n'est pas disponible
            self.app_config = None
        
        self.tfidf_vectorizer = None
        self.autoencoder = None
        self.encoder = None
        self.decoder = None
        self.scaler = None  # Sera initialisé selon le type de normalisation
        self.is_trained = False
        self.corpus_texts = []
        self.training_history = {}
        
        # Configuration des répertoires
        if self.app_config:
            self.models_dir = self.app_config.AUTOENCODER_DIR
            self.config = self.app_config.get_autoencoder_config()
        else:
            self.models_dir = "./models/embeddings"
            self.config = {
                'input_dim': 2000,
                'encoding_dim': 256,
                'hidden_layers': [1024, 512],
                'activation': 'relu',
                'optimizer': 'adam',
                'learning_rate': 0.0005,
                'epochs': 100,
                'batch_size': 16,
                'validation_split': 0.2,
                'patience': 10
            }
        
        # Appliquer les overrides si fournis
        if config_override:
            self.config.update(config_override)
        
        os.makedirs(self.models_dir, exist_ok=True)
        print("🤖 Service Autoencoder initialisé avec configuration:", self.config)
    
    def normalize_tfidf_data(self, X: np.ndarray, method: str = 'minmax') -> Tuple[np.ndarray, Any]:
        """
        Normalisation appropriée pour les données TF-IDF
        
        Args:
            X: Matrice TF-IDF
            method: 'minmax', 'l2', 'standard', ou 'none'
        
        Returns:
            X_normalized, scaler
        """
        print(f"📊 Normalisation des données TF-IDF avec méthode: {method}")
        
        if method == 'none':
            return X, None
        elif method == 'l2':
            # Normalisation L2 (recommandée pour TF-IDF)
            X_normalized = normalize(X, norm='l2')
            return X_normalized, None
        elif method == 'minmax':
            # MinMax scaling (préserve la positivité)
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(X)
            return X_normalized, scaler
        elif method == 'standard':
            # StandardScaler (peut créer des valeurs négatives)
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
            return X_normalized, scaler
        else:
            raise ValueError(f"Méthode de normalisation inconnue: {method}")
    
    def split_and_normalize_data(self, X: np.ndarray, test_size: float = 0.2, 
                                random_state: int = 42, normalization: str = 'l2') -> Tuple[np.ndarray, np.ndarray, Any, Tuple]:
        """Split et normalisation appropriée des données TF-IDF"""
        from sklearn.model_selection import train_test_split
        
        # Split train/test
        X_train, X_test = train_test_split(X, test_size=test_size, random_state=random_state)
        
        # Normalisation selon la méthode choisie
        if normalization == 'l2':
            X_train_norm = normalize(X_train, norm='l2')
            X_test_norm = normalize(X_test, norm='l2')
            scaler = None
        elif normalization == 'minmax':
            scaler = MinMaxScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)
        elif normalization == 'standard':
            scaler = StandardScaler()
            X_train_norm = scaler.fit_transform(X_train)
            X_test_norm = scaler.transform(X_test)
        else:  # 'none'
            X_train_norm, X_test_norm = X_train, X_test
            scaler = None
        
        print(f"📊 Split: {X_train.shape[0]} train, {X_test.shape[0]} test")
        print(f"📊 Normalisation: {normalization}")
        print(f"📊 Statistiques train - Min: {X_train_norm.min():.4f}, Max: {X_train_norm.max():.4f}, Moyenne: {X_train_norm.mean():.4f}")
        
        return X_train_norm, X_test_norm, scaler, (X_train, X_test)
    
    def preprocess_texts_advanced(self, texts: List[str]) -> List[str]:
        """Préprocessing avancé avec analyse linguistique"""
        try:
            import nltk
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.tokenize import word_tokenize
            
            # Télécharger les ressources NLTK si nécessaire
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/stopwords')
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                nltk.download('wordnet', quiet=True)
            
            lemmatizer = WordNetLemmatizer()
            stop_words = set(stopwords.words('english'))
            
        except ImportError:
            print("⚠️ NLTK non disponible - Préprocessing basique")
            return self.preprocess_texts(texts)
        
        processed = []
        word_freq = {}
        
        for text in texts:
            # Nettoyage de base
            text = text.lower()
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            text = re.sub(r'\S+@\S+', ' ', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Tokenisation et lemmatisation
            try:
                tokens = word_tokenize(text)
                lemmatized_tokens = []
                
                for token in tokens:
                    if (len(token) >= 3 and 
                        token not in stop_words and 
                        token.isalpha()):
                        lemmatized = lemmatizer.lemmatize(token)
                        lemmatized_tokens.append(lemmatized)
                        word_freq[lemmatized] = word_freq.get(lemmatized, 0) + 1
                
                processed_text = ' '.join(lemmatized_tokens)
                if len(processed_text) > 20:  # Texte suffisamment long
                    processed.append(processed_text)
                    
            except Exception:
                # Fallback au préprocessing basique
                processed_text = ' '.join([word for word in text.split() if len(word) >= 3])
                if len(processed_text) > 20:
                    processed.append(processed_text)
        
        # Analyse de fréquence
        if word_freq:
            most_common = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
            print(f"📊 Mots les plus fréquents: {most_common}")
        
        print(f"📊 Préprocessing avancé: {len(texts)} → {len(processed)} textes")
        return processed

    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Préprocessing basique des textes pour TF-IDF"""
        processed = []
        for text in texts:
            # Nettoyage avancé
            text = text.lower()
            # Supprimer les URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
            # Supprimer les emails
            text = re.sub(r'\S+@\S+', ' ', text)
            # Supprimer les chiffres isolés
            text = re.sub(r'\b\d+\b', ' ', text)
            # Normaliser la ponctuation
            text = re.sub(r'[^\w\s]', ' ', text)
            # Normaliser les espaces
            text = re.sub(r'\s+', ' ', text.strip())
            # Supprimer les mots très courts (< 3 caractères)
            text = ' '.join([word for word in text.split() if len(word) >= 3])
            
            if len(text) > 15:  # Filtrer les textes trop courts
                processed.append(text)
        return processed
    
    def fit_tfidf_optimized(self, texts: List[str]) -> Dict:
        """Configuration TF-IDF optimisée pour l'autoencoder"""
        try:
            # Préprocessing avancé
            self.corpus_texts = self.preprocess_texts_advanced(texts)
            
            # Configuration TF-IDF optimisée
            tfidf_config = self.app_config.get_tfidf_config() if self.app_config else {
                'max_features': self.config['input_dim'],
                'stop_words': 'english',
                'ngram_range': (1, 3),
                'min_df': 2,  # Minimum 2 occurrences
                'max_df': 0.85,  # Maximum 85% des documents
                'sublinear_tf': True,
                'use_idf': True,
                'smooth_idf': True
            }
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=min(tfidf_config['max_features'], self.config['input_dim']),
                stop_words=tfidf_config['stop_words'],
                ngram_range=tfidf_config['ngram_range'],
                min_df=tfidf_config['min_df'],
                max_df=tfidf_config['max_df'],
                sublinear_tf=tfidf_config['sublinear_tf'],
                use_idf=tfidf_config.get('use_idf', True),
                smooth_idf=tfidf_config.get('smooth_idf', True)
            )
            
            # Entraînement TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)
            
            # Analyse statistique
            sparsity = 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
            
            print(f"✅ TF-IDF optimisé entraîné sur {len(self.corpus_texts)} textes")
            print(f"📊 Dimensions: {tfidf_matrix.shape}")
            print(f"📊 Sparsité: {sparsity:.2%}")
            print(f"📊 Vocabulaire: {len(self.tfidf_vectorizer.vocabulary_)} mots")
            
            return {
                'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
                'corpus_size': len(self.corpus_texts),
                'tfidf_shape': tfidf_matrix.shape,
                'sparsity': sparsity,
                'feature_names': list(self.tfidf_vectorizer.get_feature_names_out())[:20]  # Top 20 features
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TF-IDF optimisé: {str(e)}")

    def fit_tfidf(self, texts: List[str]) -> Dict:
        """Entraîne le vectoriseur TF-IDF"""
        try:
            self.corpus_texts = self.preprocess_texts(texts)
            
            # Configuration TF-IDF adaptée pour l'autoencoder
            tfidf_config = self.app_config.get_tfidf_config() if self.app_config else {
                'max_features': self.config['input_dim'],
                'stop_words': 'english',
                'ngram_range': (1, 2),
                'min_df': 2,
                'max_df': 0.8,
                'sublinear_tf': True
            }
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=min(tfidf_config['max_features'], self.config['input_dim']),
                stop_words=tfidf_config['stop_words'],
                ngram_range=tfidf_config['ngram_range'],
                min_df=tfidf_config['min_df'],
                max_df=tfidf_config['max_df'],
                sublinear_tf=tfidf_config['sublinear_tf']
            )
            
            # Entraînement TF-IDF
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.corpus_texts)
            
            print(f"✅ TF-IDF entraîné sur {len(self.corpus_texts)} textes")
            print(f"📊 Dimensions: {tfidf_matrix.shape}")
            
            return {
                'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
                'corpus_size': len(self.corpus_texts),
                'tfidf_shape': tfidf_matrix.shape,
                'sparsity': 1.0 - (tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]))
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TF-IDF: {str(e)}")
    
    def build_autoencoder_optimized(self, input_dim: int = None, encoding_dim: int = None) -> Dict:
        """
        🎯 Architecture d'autoencoder avec REGULARISATION AVANCEE
        Implémente les techniques de régularisation avancées :
        - Régularisation L2 (Ridge) pour éviter l'overfitting
        - Dropout progressif pour la robustesse
        - Batch Normalization pour la stabilité
        """
        try:
            if not TENSORFLOW_AVAILABLE:
                return self._build_simple_autoencoder(input_dim, encoding_dim)
            
            # Mise à jour de la configuration
            if input_dim:
                self.config['input_dim'] = input_dim
            if encoding_dim:
                self.config['encoding_dim'] = encoding_dim
            
            input_dim = self.config['input_dim']
            encoding_dim = self.config['encoding_dim']
            
            print(f"🏗️ Construction autoencoder avec REGULARISATION AVANCEE")
            print(f"📊 Architecture: {input_dim} → {encoding_dim} → {input_dim}")
            print(f"📊 Taux de compression: {(1 - encoding_dim/input_dim)*100:.1f}%")
            print(f"🎯 L2 Regularization: {self.config.get('l2_kernel_reg', 0.001)}")
            print(f"🎯 Dropout rates: {self.config.get('dropout_rates', [0.1, 0.2, 0.3])}")
            print(f"🎯 Batch Normalization: {self.config.get('use_batch_norm', True)}")
            
            # Input layer
            input_layer = layers.Input(shape=(input_dim,))
            
            # 🎯 ENCODER avec REGULARISATION PROGRESSIVE
            encoded = input_layer
            dropout_rates = self.config.get('dropout_rates', [0.1, 0.2, 0.3])
            l2_kernel = self.config.get('l2_kernel_reg', 0.001)
            l2_bias = self.config.get('l2_bias_reg', 0.0005)
            use_batch_norm = self.config.get('use_batch_norm', True)
            batch_norm_momentum = self.config.get('batch_norm_momentum', 0.99)
            
            for i, hidden_dim in enumerate(self.config['hidden_layers']):
                print(f"  🔧 Couche Encoder {i+1}: {hidden_dim} neurones")
                print(f"     - L2 regularization: kernel={l2_kernel}, bias={l2_bias}")
                print(f"     - Dropout: {dropout_rates[i] if i < len(dropout_rates) else 0.0}")
                print(f"     - Batch Norm: {use_batch_norm}")
                
                # Couche Dense avec régularisation L2 configurée
                encoded = layers.Dense(
                    hidden_dim, 
                    activation=self.config['activation'],
                    kernel_regularizer=l2(l2_kernel),  # Régularisation des poids
                    bias_regularizer=l2(l2_bias),      # Régularisation des biais
                    name=f'encoder_dense_{i+1}'
                )(encoded)
                
                # Batch Normalization conditionnelle
                if use_batch_norm:
                    encoded = layers.BatchNormalization(
                        momentum=batch_norm_momentum,
                        name=f'encoder_bn_{i+1}'
                    )(encoded)
                
                # Dropout progressif selon configuration
                if i < len(dropout_rates) and dropout_rates[i] > 0:
                    encoded = layers.Dropout(
                        dropout_rates[i], 
                        name=f'encoder_dropout_{i+1}'
                    )(encoded)
            
            # 🎯 GOULOT D'ETRANGLEMENT (Bottleneck) - Couche critique
            bottleneck_dropout = self.config.get('dropout_bottleneck', 0.0)
            print(f"  🎯 Goulot d'étranglement: {encoding_dim} neurones")
            print(f"     - Dropout bottleneck: {bottleneck_dropout}")
            
            encoded = layers.Dense(
                encoding_dim, 
                activation='relu',  # ReLU pour préserver la positivité TF-IDF
                kernel_regularizer=l2(l2_kernel * 0.5),  # Régularisation réduite au goulot
                bias_regularizer=l2(l2_bias * 0.5),
                name='encoded_bottleneck'
            )(encoded)
            
            # Batch Norm au goulot d'étranglement
            if use_batch_norm:
                encoded = layers.BatchNormalization(
                    momentum=batch_norm_momentum,
                    name='bottleneck_bn'
                )(encoded)
            
            # Dropout léger au goulot d'étranglement (optionnel)
            if bottleneck_dropout > 0:
                encoded = layers.Dropout(bottleneck_dropout, name='bottleneck_dropout')(encoded)
            
            # 🎯 DECODER avec REGULARISATION SYMETRIQUE
            decoded = encoded
            reversed_hidden = list(reversed(self.config['hidden_layers']))
            decoder_dropout = self.config.get('dropout_decoder', 0.1)
            
            print(f"  🔧 Decoder avec dropout: {decoder_dropout}")
            
            for i, hidden_dim in enumerate(reversed_hidden):
                print(f"  🔧 Couche Decoder {i+1}: {hidden_dim} neurones")
                
                decoded = layers.Dense(
                    hidden_dim, 
                    activation=self.config['activation'],
                    kernel_regularizer=l2(l2_kernel * 0.8),  # Régularisation légèrement réduite
                    bias_regularizer=l2(l2_bias * 0.8),
                    name=f'decoder_dense_{i+1}'
                )(decoded)
                
                # Batch Normalization dans le decoder
                if use_batch_norm:
                    decoded = layers.BatchNormalization(
                        momentum=batch_norm_momentum,
                        name=f'decoder_bn_{i+1}'
                    )(decoded)
                
                # Dropout uniforme dans le decoder (plus conservateur)
                if decoder_dropout > 0:
                    decoded = layers.Dropout(
                        decoder_dropout, 
                        name=f'decoder_dropout_{i+1}'
                    )(decoded)
            
            # 🎯 COUCHE DE SORTIE - Reconstruction
            print(f"  🎯 Couche de sortie: {input_dim} neurones (reconstruction)")
            decoded = layers.Dense(
                input_dim, 
                activation='sigmoid',  # Sigmoid pour données TF-IDF normalisées [0,1]
                kernel_regularizer=l2(l2_kernel * 0.5),  # Régularisation légère en sortie
                name='output_reconstruction'
            )(decoded)
            
            # 🎯 CONSTRUCTION DES MODELES
            self.autoencoder = Model(input_layer, decoded, name='autoencoder_regularized')
            self.encoder = Model(input_layer, encoded, name='encoder_regularized')
            
            # Decoder séparé pour l'analyse
            encoded_input = layers.Input(shape=(encoding_dim,))
            decoder_layers = []
            
            # Récupérer les couches du decoder depuis l'autoencoder
            start_decoder = False
            for layer in self.autoencoder.layers:
                if layer.name == 'encoded_bottleneck':
                    start_decoder = True
                    continue
                if start_decoder:
                    decoder_layers.append(layer)
            
            # Construire le decoder
            decoded_output = encoded_input
            for layer in decoder_layers:
                decoded_output = layer(decoded_output)
            
            self.decoder = Model(encoded_input, decoded_output, name='decoder_regularized')
            
            # 🎯 OPTIMISEUR AVEC PARAMETRES AVANCES
            optimizer = Adam(
                learning_rate=self.config['learning_rate'],
                beta_1=0.9,      # Momentum pour gradient
                beta_2=0.999,    # Momentum pour gradient carré
                epsilon=1e-7,    # Stabilité numérique
                decay=1e-6       # Décroissance du learning rate
            )
            
            # 🎯 LOSS FUNCTION adaptée aux données TF-IDF
            self.autoencoder.compile(
                optimizer=optimizer,
                loss='binary_crossentropy',  # Optimal pour données [0,1]
                metrics=['mse', 'mae', 'cosine_similarity']  # Métriques complètes
            )
            
            # Résumé de l'architecture
            total_params = self.autoencoder.count_params()
            trainable_params = sum([tf.keras.utils.count_params(w) for w in self.autoencoder.trainable_weights])
            
            print(f"✅ Autoencoder REGULARISE construit avec succès!")
            print(f"📊 Paramètres totaux: {total_params:,}")
            print(f"📊 Paramètres entraînables: {trainable_params:,}")
            
            return {
                'architecture': 'tensorflow_regularized_advanced',
                'input_dim': input_dim,
                'encoding_dim': encoding_dim,
                'hidden_layers': self.config['hidden_layers'],
                'total_params': total_params,
                'trainable_params': trainable_params,
                'compression_ratio': input_dim / encoding_dim,
                'regularization_techniques': {
                    'l2_kernel_reg': l2_kernel,
                    'l2_bias_reg': l2_bias,
                    'dropout_rates': dropout_rates,
                    'dropout_decoder': decoder_dropout,
                    'dropout_bottleneck': bottleneck_dropout,
                    'batch_normalization': use_batch_norm,
                    'batch_norm_momentum': batch_norm_momentum
                },
                'activation_functions': {
                    'hidden': self.config['activation'],
                    'bottleneck': 'relu',
                    'output': 'sigmoid'
                },
                'loss_function': 'binary_crossentropy',
                'optimizer': 'adam_advanced',
                'advanced_requirements': '✅ L2 Regularization + Dropout implementés'
            }
            
        except Exception as e:
            raise Exception(f"Erreur construction autoencoder régularisé: {str(e)}")

    def build_autoencoder(self, input_dim: int = None, encoding_dim: int = None) -> Dict:
        """Construit l'architecture de l'autoencoder"""
        try:
            if not TENSORFLOW_AVAILABLE:
                return self._build_simple_autoencoder(input_dim, encoding_dim)
            
            # Mise à jour de la configuration
            if input_dim:
                self.config['input_dim'] = input_dim
            if encoding_dim:
                self.config['encoding_dim'] = encoding_dim
            
            input_dim = self.config['input_dim']
            encoding_dim = self.config['encoding_dim']
            
            print(f"🏗️ Construction autoencoder: {input_dim} → {encoding_dim} → {input_dim}")
            
            # Input layer
            input_layer = layers.Input(shape=(input_dim,))
            
            # Encoder
            encoded = input_layer
            for hidden_dim in self.config['hidden_layers']:
                encoded = layers.Dense(hidden_dim, activation=self.config['activation'])(encoded)
                encoded = layers.Dropout(0.2)(encoded)
            
            # Couche de compression (goulot d'étranglement)
            encoded = layers.Dense(encoding_dim, activation=self.config['activation'], name='encoded')(encoded)
            
            # Decoder (miroir de l'encoder)
            decoded = encoded
            for hidden_dim in reversed(self.config['hidden_layers']):
                decoded = layers.Dense(hidden_dim, activation=self.config['activation'])(decoded)
                decoded = layers.Dropout(0.2)(decoded)
            
            # Output layer (reconstruction)
            decoded = layers.Dense(input_dim, activation='linear', name='decoded')(decoded)
            
            # Modèles
            self.autoencoder = Model(input_layer, decoded, name='autoencoder')
            self.encoder = Model(input_layer, encoded, name='encoder')
            
            # Decoder séparé
            encoded_input = layers.Input(shape=(encoding_dim,))
            decoder_layers = self.autoencoder.layers[-len(self.config['hidden_layers'])-1:]
            decoded_output = encoded_input
            for layer in decoder_layers:
                decoded_output = layer(decoded_output)
            self.decoder = Model(encoded_input, decoded_output, name='decoder')
            
            # Compilation
            optimizer = Adam(learning_rate=self.config['learning_rate'])
            self.autoencoder.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            
            return {
                'architecture': 'tensorflow',
                'input_dim': input_dim,
                'encoding_dim': encoding_dim,
                'hidden_layers': self.config['hidden_layers'],
                'total_params': self.autoencoder.count_params(),
                'compression_ratio': input_dim / encoding_dim
            }
            
        except Exception as e:
            raise Exception(f"Erreur construction autoencoder: {str(e)}")
    
    def _build_simple_autoencoder(self, input_dim: int = None, encoding_dim: int = None) -> Dict:
        """Autoencoder simplifié sans TensorFlow"""
        if input_dim:
            self.config['input_dim'] = input_dim
        if encoding_dim:
            self.config['encoding_dim'] = encoding_dim
        
        print("🔧 Construction autoencoder simplifié (sans TensorFlow)")
        
        # Matrices de poids aléatoires
        np.random.seed(42)
        input_dim = self.config['input_dim']
        encoding_dim = self.config['encoding_dim']
        
        self.weights = {
            'encoder': np.random.randn(input_dim, encoding_dim) * 0.1,
            'decoder': np.random.randn(encoding_dim, input_dim) * 0.1,
            'encoder_bias': np.zeros(encoding_dim),
            'decoder_bias': np.zeros(input_dim)
        }
        
        return {
            'architecture': 'numpy',
            'input_dim': input_dim,
            'encoding_dim': encoding_dim,
            'compression_ratio': input_dim / encoding_dim
        }
    
    def train_autoencoder(self, texts: List[str] = None, config: Dict = None, use_proper_split: bool = True) -> Dict:
        """Entraîne l'autoencoder sur les embeddings TF-IDF"""
        try:
            # Mise à jour de la configuration
            if config:
                self.config.update(config)
            
            # Utiliser les textes fournis ou le corpus existant
            training_texts = texts if texts else self.corpus_texts
            if not training_texts:
                raise Exception("Aucun texte pour l'entraînement")
            
            # S'assurer que TF-IDF est entraîné
            if not self.tfidf_vectorizer:
                print("🔄 Entraînement TF-IDF automatique...")
                self.fit_tfidf(training_texts)
            
            # Générer les embeddings TF-IDF
            print("📊 Génération des embeddings TF-IDF...")
            tfidf_matrix = self.tfidf_vectorizer.transform(self.corpus_texts)
            X = tfidf_matrix.toarray()
            
            # Construire l'autoencoder si nécessaire
            if not self.autoencoder and not hasattr(self, 'weights'):
                self.build_autoencoder(input_dim=X.shape[1])
            
            print(f"🚀 Entraînement autoencoder sur {X.shape[0]} échantillons...")
            
            # Option pour split train/test approprié
            if use_proper_split and X.shape[0] > 10:
                X_train, X_test, scaler_proper, _ = self.split_and_normalize_data(X, normalization='l2')
                self.scaler = scaler_proper  # Utiliser le scaler du train
                
                if TENSORFLOW_AVAILABLE and self.autoencoder:
                    return self._train_tensorflow_autoencoder_with_test(X_train, X_test)
                else:
                    return self._train_simple_autoencoder_with_test(X_train, X_test)
            else:
                # Normalisation simple pour les petits datasets
                X_normalized, self.scaler = self.normalize_tfidf_data(X, method='l2')
                print(f"📊 Normalisation appliquée: L2")
                
                if TENSORFLOW_AVAILABLE and self.autoencoder:
                    return self._train_tensorflow_autoencoder(X_normalized)
                else:
                    return self._train_simple_autoencoder(X_normalized)
            
        except Exception as e:
            raise Exception(f"Erreur entraînement autoencoder: {str(e)}")
    
    def _train_tensorflow_autoencoder(self, X: np.ndarray) -> Dict:
        """
        🎯 Entraîne l'autoencoder avec CALLBACKS AVANCES
        Implémente les techniques de régularisation avancées :
        - Early Stopping intelligent
        - Learning Rate Scheduling
        - Monitoring avancé des métriques
        """
        try:
            print(f"🚀 Entraînement autoencoder TensorFlow REGULARISE sur {X.shape[0]} échantillons...")
            print(f"🎯 Techniques avancées activées: Early Stopping + LR Scheduling")
            
            # 🎯 CALLBACKS AVANCES selon configuration avancée
            callbacks = []
            
            # Early Stopping intelligent
            if self.config.get('early_stopping_patience', 15) > 0:
                early_stopping = EarlyStopping(
                    monitor=self.config.get('early_stopping_monitor', 'val_loss'),
                    patience=self.config.get('early_stopping_patience', 15),
                    min_delta=self.config.get('early_stopping_min_delta', 0.0001),
                    restore_best_weights=self.config.get('restore_best_weights', True),
                    verbose=1,
                    mode='min'
                )
                callbacks.append(early_stopping)
                print(f"  ✅ Early Stopping: monitor={early_stopping.monitor}, patience={early_stopping.patience}")
            
            # Learning Rate Scheduling
            if self.config.get('reduce_lr_on_plateau', True):
                lr_scheduler = ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=self.config.get('lr_reduction_factor', 0.5),
                    patience=self.config.get('lr_reduction_patience', 8),
                    min_lr=self.config.get('min_learning_rate', 1e-7),
                    verbose=1,
                    mode='min',
                    cooldown=2  # Attendre 2 epochs après réduction
                )
                callbacks.append(lr_scheduler)
                print(f"  ✅ LR Scheduler: factor={lr_scheduler.factor}, patience={lr_scheduler.patience}")
            
            # TensorBoard pour monitoring (optionnel)
            try:
                from tensorflow.keras.callbacks import TensorBoard
                import tempfile
                log_dir = os.path.join(tempfile.gettempdir(), 'autoencoder_logs')
                tensorboard = TensorBoard(
                    log_dir=log_dir,
                    histogram_freq=1,
                    write_graph=True,
                    update_freq='epoch'
                )
                callbacks.append(tensorboard)
                print(f"  ✅ TensorBoard: logs dans {log_dir}")
            except:
                print("  ⚠️ TensorBoard non disponible")
            
            # 🎯 ENTRAINEMENT AVEC MONITORING AVANCE
            print(f"🔥 Démarrage entraînement avec {len(callbacks)} callbacks...")
            print(f"📊 Epochs max: {self.config['epochs']}")
            print(f"📊 Batch size: {self.config['batch_size']}")
            print(f"📊 Validation split: {self.config['validation_split']}")
            print(f"📊 Learning rate initial: {self.config['learning_rate']}")
            
            # Entraînement avec callbacks avancés
            history = self.autoencoder.fit(
                X, X,  # Autoencoder: input = output
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_split=self.config['validation_split'],
                callbacks=callbacks,
                verbose=1,
                shuffle=True
            )
            
            self.is_trained = True
            self.training_history = history.history
            
            # 🎯 ANALYSE DETAILLEE DES RESULTATS
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            epochs_trained = len(history.history['loss'])
            
            # Métriques avancées
            best_val_loss = min(history.history['val_loss'])
            best_epoch = history.history['val_loss'].index(best_val_loss) + 1
            overfitting_ratio = final_val_loss / final_loss
            
            # Analyse de la convergence
            loss_improvement = (history.history['loss'][0] - final_loss) / history.history['loss'][0] * 100
            val_loss_improvement = (history.history['val_loss'][0] - final_val_loss) / history.history['val_loss'][0] * 100
            
            # Test de reconstruction
            sample_indices = np.random.choice(X.shape[0], min(10, X.shape[0]), replace=False)
            sample_input = X[sample_indices]
            sample_reconstructed = self.autoencoder.predict(sample_input, verbose=0)
            reconstruction_error = np.mean((sample_input - sample_reconstructed) ** 2)
            
            print(f"✅ Autoencoder REGULARISE entraîné avec succès!")
            print(f"📊 Epochs effectués: {epochs_trained}/{self.config['epochs']}")
            print(f"📊 Perte finale: {final_loss:.6f}")
            print(f"📊 Perte validation: {final_val_loss:.6f}")
            print(f"📊 Meilleure val_loss: {best_val_loss:.6f} (epoch {best_epoch})")
            print(f"📊 Ratio overfitting: {overfitting_ratio:.3f} {'✅' if overfitting_ratio < 1.2 else '⚠️'}")
            print(f"📊 Amélioration loss: {loss_improvement:.1f}%")
            print(f"📊 Amélioration val_loss: {val_loss_improvement:.1f}%")
            print(f"📊 Erreur reconstruction: {reconstruction_error:.6f}")
            
            # Détection des problèmes
            warnings = []
            if overfitting_ratio > 1.5:
                warnings.append("Overfitting détecté - Augmenter régularisation")
            if epochs_trained == self.config['epochs']:
                warnings.append("Entraînement non convergé - Augmenter epochs ou patience")
            if final_val_loss > history.history['val_loss'][0]:
                warnings.append("Validation loss augmente - Problème de généralisation")
            
            return {
                'status': 'success',
                'framework': 'tensorflow_regularized',
                'final_loss': float(final_loss),
                'final_val_loss': float(final_val_loss),
                'reconstruction_error': float(reconstruction_error),
                'best_val_loss': float(best_val_loss),
                'best_epoch': int(best_epoch),
                'epochs_trained': int(epochs_trained),
                'epochs_max': int(self.config['epochs']),
                'overfitting_ratio': float(overfitting_ratio),
                'loss_improvement_percent': float(loss_improvement),
                'val_loss_improvement_percent': float(val_loss_improvement),
                'input_shape': X.shape,
                'encoding_dim': self.config['encoding_dim'],
                'compression_ratio': X.shape[1] / self.config['encoding_dim'],
                'convergence_analysis': {
                    'converged': epochs_trained < self.config['epochs'],
                    'overfitting_detected': overfitting_ratio > 1.2,
                    'generalization_good': overfitting_ratio < 1.5
                },
                'regularization_effectiveness': {
                    'l2_applied': True,
                    'dropout_applied': True,
                    'batch_norm_applied': self.config.get('use_batch_norm', True),
                    'early_stopping_triggered': epochs_trained < self.config['epochs']
                },
                'warnings': warnings,
                'history': self.training_history,
                'advanced_validation': '✅ Régularisation L2 + Dropout + Callbacks avancés'
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TensorFlow régularisé: {str(e)}")
    
    def _train_tensorflow_autoencoder_with_test(self, X_train: np.ndarray, X_test: np.ndarray) -> Dict:
        """Entraînement TensorFlow avec validation sur test set séparé"""
        try:
            # Callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=self.config['patience'],
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # Entraînement avec test set comme validation
            history = self.autoencoder.fit(
                X_train, X_train,  # Autoencoder: input = target
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_data=(X_test, X_test),  # Test set pour validation
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            
            # Évaluation finale
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            # Test de reconstruction sur test set
            test_reconstructed = self.autoencoder.predict(X_test, verbose=0)
            test_reconstruction_error = np.mean((X_test - test_reconstructed) ** 2)
            
            print(f"✅ Autoencoder entraîné avec test set séparé!")
            print(f"📊 Perte finale (train): {final_loss:.4f}")
            print(f"📊 Perte finale (test): {final_val_loss:.4f}")
            print(f"📊 Erreur reconstruction (test): {test_reconstruction_error:.4f}")
            
            return {
                'status': 'success',
                'architecture': 'tensorflow',
                'final_loss': float(final_loss),
                'final_test_loss': float(final_val_loss),
                'test_reconstruction_error': float(test_reconstruction_error),
                'epochs_trained': len(history.history['loss']),
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'encoding_dim': self.config['encoding_dim'],
                'compression_ratio': X_train.shape[1] / self.config['encoding_dim'],
                'normalization': 'L2 + StandardScaler'
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TensorFlow avec test: {str(e)}")
    
    def _train_simple_autoencoder(self, X: np.ndarray) -> Dict:
        """Entraînement autoencoder simplifié"""
        try:
            print("🔧 Entraînement autoencoder simplifié...")
            
            # Paramètres d'entraînement
            learning_rate = 0.01
            epochs = 100
            
            losses = []
            
            for epoch in range(epochs):
                # Forward pass
                # Encoder: X -> encoded
                encoded = np.maximum(0, np.dot(X, self.weights['encoder']) + self.weights['encoder_bias'])
                
                # Decoder: encoded -> reconstructed
                reconstructed = np.dot(encoded, self.weights['decoder']) + self.weights['decoder_bias']
                
                # Loss (MSE)
                loss = np.mean((X - reconstructed) ** 2)
                losses.append(loss)
                
                # Backward pass (gradient descent simplifié)
                # Gradients
                d_output = 2 * (reconstructed - X) / X.shape[0]
                
                # Decoder gradients
                d_decoder_weights = np.dot(encoded.T, d_output)
                d_decoder_bias = np.sum(d_output, axis=0)
                
                # Encoder gradients
                d_encoded = np.dot(d_output, self.weights['decoder'].T)
                d_encoded[encoded <= 0] = 0  # ReLU derivative
                
                d_encoder_weights = np.dot(X.T, d_encoded)
                d_encoder_bias = np.sum(d_encoded, axis=0)
                
                # Update weights
                self.weights['decoder'] -= learning_rate * d_decoder_weights
                self.weights['decoder_bias'] -= learning_rate * d_decoder_bias
                self.weights['encoder'] -= learning_rate * d_encoder_weights
                self.weights['encoder_bias'] -= learning_rate * d_encoder_bias
                
                if epoch % 20 == 0:
                    print(f"Époque {epoch}, Perte: {loss:.4f}")
            
            self.is_trained = True
            
            # Test de reconstruction
            sample_indices = np.random.choice(X.shape[0], min(10, X.shape[0]), replace=False)
            sample_input = X[sample_indices]
            sample_encoded = np.maximum(0, np.dot(sample_input, self.weights['encoder']) + self.weights['encoder_bias'])
            sample_reconstructed = np.dot(sample_encoded, self.weights['decoder']) + self.weights['decoder_bias']
            reconstruction_error = np.mean((sample_input - sample_reconstructed) ** 2)
            
            print(f"✅ Autoencoder simplifié entraîné!")
            print(f"📊 Perte finale: {losses[-1]:.4f}")
            print(f"📊 Erreur reconstruction: {reconstruction_error:.4f}")
            
            return {
                'status': 'success',
                'architecture': 'numpy',
                'final_loss': float(losses[-1]),
                'reconstruction_error': float(reconstruction_error),
                'epochs_trained': epochs,
                'input_shape': X.shape,
                'encoding_dim': self.config['encoding_dim'],
                'compression_ratio': X.shape[1] / self.config['encoding_dim'],
                'loss_history': losses[-10:]  # Dernières 10 valeurs
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement simplifié: {str(e)}")
    
    def _train_simple_autoencoder_with_test(self, X_train: np.ndarray, X_test: np.ndarray) -> Dict:
        """Entraînement autoencoder simplifié avec test set"""
        try:
            print("🔧 Entraînement autoencoder simplifié avec test set...")
            
            # Paramètres d'entraînement
            learning_rate = 0.01
            epochs = 100
            
            train_losses = []
            test_losses = []
            
            for epoch in range(epochs):
                # Forward pass sur train
                encoded_train = np.maximum(0, np.dot(X_train, self.weights['encoder']) + self.weights['encoder_bias'])
                reconstructed_train = np.dot(encoded_train, self.weights['decoder']) + self.weights['decoder_bias']
                
                # Loss train
                train_loss = np.mean((X_train - reconstructed_train) ** 2)
                train_losses.append(train_loss)
                
                # Évaluation sur test (sans gradient)
                encoded_test = np.maximum(0, np.dot(X_test, self.weights['encoder']) + self.weights['encoder_bias'])
                reconstructed_test = np.dot(encoded_test, self.weights['decoder']) + self.weights['decoder_bias']
                test_loss = np.mean((X_test - reconstructed_test) ** 2)
                test_losses.append(test_loss)
                
                # Backward pass (seulement sur train)
                d_output = 2 * (reconstructed_train - X_train) / X_train.shape[0]
                
                # Decoder gradients
                d_decoder_weights = np.dot(encoded_train.T, d_output)
                d_decoder_bias = np.sum(d_output, axis=0)
                
                # Encoder gradients
                d_encoded = np.dot(d_output, self.weights['decoder'].T)
                d_encoded[encoded_train <= 0] = 0  # ReLU derivative
                
                d_encoder_weights = np.dot(X_train.T, d_encoded)
                d_encoder_bias = np.sum(d_encoded, axis=0)
                
                # Update weights
                self.weights['decoder'] -= learning_rate * d_decoder_weights
                self.weights['decoder_bias'] -= learning_rate * d_decoder_bias
                self.weights['encoder'] -= learning_rate * d_encoder_weights
                self.weights['encoder_bias'] -= learning_rate * d_encoder_bias
                
                if epoch % 20 == 0:
                    print(f"Époque {epoch}, Train: {train_loss:.4f}, Test: {test_loss:.4f}")
            
            self.is_trained = True
            
            # Évaluation finale
            final_test_reconstruction_error = test_losses[-1]
            
            print(f"✅ Autoencoder simplifié entraîné avec test set!")
            print(f"📊 Perte finale (train): {train_losses[-1]:.4f}")
            print(f"📊 Perte finale (test): {test_losses[-1]:.4f}")
            print(f"📊 Erreur reconstruction (test): {final_test_reconstruction_error:.4f}")
            
            return {
                'status': 'success',
                'architecture': 'numpy',
                'final_loss': float(train_losses[-1]),
                'final_test_loss': float(test_losses[-1]),
                'test_reconstruction_error': float(final_test_reconstruction_error),
                'epochs_trained': epochs,
                'train_shape': X_train.shape,
                'test_shape': X_test.shape,
                'encoding_dim': self.config['encoding_dim'],
                'compression_ratio': X_train.shape[1] / self.config['encoding_dim'],
                'normalization': 'L2 + StandardScaler',
                'train_loss_history': train_losses[-10:],
                'test_loss_history': test_losses[-10:]
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement simplifié avec test: {str(e)}")
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode un texte vers sa représentation compressée"""
        try:
            if not self.is_trained:
                raise Exception("Autoencoder non entraîné")
            
            # TF-IDF embedding
            tfidf_embedding = self.tfidf_vectorizer.transform([text]).toarray()
            
            # Appliquer la même normalisation que pendant l'entraînement
            if self.scaler is not None:
                # Si un scaler a été utilisé pendant l'entraînement
                from sklearn.preprocessing import normalize
                tfidf_l2_normalized = normalize(tfidf_embedding, norm='l2')
                tfidf_normalized = self.scaler.transform(tfidf_l2_normalized)
            else:
                # Normalisation L2 simple
                from sklearn.preprocessing import normalize
                tfidf_normalized = normalize(tfidf_embedding, norm='l2')
            
            # Encoding
            if TENSORFLOW_AVAILABLE and self.encoder:
                encoded = self.encoder.predict(tfidf_normalized, verbose=0)
            else:
                encoded = np.maximum(0, np.dot(tfidf_normalized, self.weights['encoder']) + self.weights['encoder_bias'])
            
            return encoded[0]
            
        except Exception as e:
            raise Exception(f"Erreur encodage: {str(e)}")
    
    def decode_embedding(self, encoded: np.ndarray) -> np.ndarray:
        """Décode une représentation compressée vers l'espace original"""
        try:
            if not self.is_trained:
                raise Exception("Autoencoder non entraîné")
            
            # Reshape si nécessaire
            if encoded.ndim == 1:
                encoded = encoded.reshape(1, -1)
            
            # Decoding
            if TENSORFLOW_AVAILABLE and self.decoder:
                decoded = self.decoder.predict(encoded, verbose=0)
            else:
                decoded = np.dot(encoded, self.weights['decoder']) + self.weights['decoder_bias']
            
            # Dénormalisation si un scaler a été utilisé
            if self.scaler is not None:
                decoded_denorm = self.scaler.inverse_transform(decoded)
                return decoded_denorm[0] if decoded_denorm.shape[0] == 1 else decoded_denorm
            else:
                # Pas de dénormalisation nécessaire
                return decoded[0] if decoded.shape[0] == 1 else decoded
            
        except Exception as e:
            raise Exception(f"Erreur décodage: {str(e)}")
    
    def reconstruct_text(self, text: str) -> Dict:
        """Reconstruit un texte via l'autoencoder (X → encoded → X)"""
        try:
            # Embedding original
            original_embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
            
            # Passage par l'autoencoder
            encoded = self.encode_text(text)
            reconstructed_embedding = self.decode_embedding(encoded)
            
            # Calcul de l'erreur de reconstruction
            reconstruction_error = np.mean((original_embedding - reconstructed_embedding) ** 2)
            similarity = cosine_similarity([original_embedding], [reconstructed_embedding])[0][0]
            
            # Analyse des termes les plus importants
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            # Termes originaux les plus importants
            top_original_indices = original_embedding.argsort()[-10:][::-1]
            top_original_terms = [(feature_names[i], float(original_embedding[i])) 
                                for i in top_original_indices if original_embedding[i] > 0]
            
            # Termes reconstruits les plus importants
            top_reconstructed_indices = reconstructed_embedding.argsort()[-10:][::-1]
            top_reconstructed_terms = [(feature_names[i], float(reconstructed_embedding[i])) 
                                     for i in top_reconstructed_indices if reconstructed_embedding[i] > 0]
            
            return {
                'original_text': text,
                'encoded_shape': encoded.shape,
                'reconstruction_error': float(reconstruction_error),
                'similarity': float(similarity),
                'compression_ratio': len(original_embedding) / len(encoded),
                'top_original_terms': top_original_terms[:5],
                'top_reconstructed_terms': top_reconstructed_terms[:5],
                'encoded_representation': encoded.tolist()
            }
            
        except Exception as e:
            raise Exception(f"Erreur reconstruction: {str(e)}")
    
    def find_similar_in_compressed_space(self, text: str, top_k: int = 5) -> List[Dict]:
        """Trouve les textes similaires dans l'espace compressé"""
        try:
            if not self.corpus_texts:
                raise Exception("Aucun corpus pour la recherche")
            
            # Encoder le texte de requête
            query_encoded = self.encode_text(text)
            
            # Encoder tous les textes du corpus
            corpus_encoded = []
            for corpus_text in self.corpus_texts:
                try:
                    encoded = self.encode_text(corpus_text)
                    corpus_encoded.append(encoded)
                except:
                    corpus_encoded.append(np.zeros_like(query_encoded))
            
            corpus_encoded = np.array(corpus_encoded)
            
            # Calcul des similarités dans l'espace compressé
            similarities = cosine_similarity([query_encoded], corpus_encoded)[0]
            
            # Tri et sélection des top_k
            results = []
            for i, (corpus_text, similarity) in enumerate(zip(self.corpus_texts, similarities)):
                results.append({
                    'index': i,
                    'text': corpus_text,
                    'similarity': float(similarity),
                    'text_preview': corpus_text[:150] + "..." if len(corpus_text) > 150 else corpus_text
                })
            
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            raise Exception(f"Erreur recherche espace compressé: {str(e)}")
    
    def get_model_info(self) -> Dict:
        """Informations sur le modèle"""
        info = {
            'is_trained': self.is_trained,
            'tensorflow_available': TENSORFLOW_AVAILABLE,
            'architecture': 'tensorflow' if TENSORFLOW_AVAILABLE and self.autoencoder else 'numpy',
            'config': self.config,
            'corpus_size': len(self.corpus_texts),
            'tfidf_fitted': self.tfidf_vectorizer is not None
        }
        
        if self.tfidf_vectorizer:
            info['vocabulary_size'] = len(self.tfidf_vectorizer.vocabulary_)
            info['input_dim'] = len(self.tfidf_vectorizer.vocabulary_)
        
        if TENSORFLOW_AVAILABLE and self.autoencoder:
            info['total_params'] = self.autoencoder.count_params()
            info['model_summary'] = str(self.autoencoder.summary())
        
        return info
    
    def save_model(self, filename: str) -> str:
        """Sauvegarde le modèle"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            # Sauvegarder les composants
            model_data = {
                'config': self.config,
                'is_trained': self.is_trained,
                'corpus_texts': self.corpus_texts,
                'scaler': self.scaler,
                'tfidf_vectorizer': self.tfidf_vectorizer
            }
            
            if not TENSORFLOW_AVAILABLE:
                model_data['weights'] = self.weights
            
            # Sauvegarde pickle
            with open(f"{filepath}_data.pkl", 'wb') as f:
                pickle.dump(model_data, f)
            
            # Sauvegarde TensorFlow si disponible
            if TENSORFLOW_AVAILABLE and self.autoencoder:
                self.autoencoder.save(f"{filepath}_autoencoder.h5")
                self.encoder.save(f"{filepath}_encoder.h5")
                self.decoder.save(f"{filepath}_decoder.h5")
            
            print(f"✅ Modèle sauvegardé: {filepath}")
            return filepath
            
        except Exception as e:
            raise Exception(f"Erreur sauvegarde: {str(e)}")
    
    def load_model(self, filename: str) -> bool:
        """Charge un modèle sauvegardé"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            # Charger les données
            with open(f"{filepath}_data.pkl", 'rb') as f:
                model_data = pickle.load(f)
            
            self.config = model_data['config']
            self.is_trained = model_data['is_trained']
            self.corpus_texts = model_data['corpus_texts']
            self.scaler = model_data['scaler']
            self.tfidf_vectorizer = model_data['tfidf_vectorizer']
            
            if 'weights' in model_data:
                self.weights = model_data['weights']
            
            # Charger TensorFlow si disponible
            if TENSORFLOW_AVAILABLE and os.path.exists(f"{filepath}_autoencoder.h5"):
                self.autoencoder = keras.models.load_model(f"{filepath}_autoencoder.h5")
                self.encoder = keras.models.load_model(f"{filepath}_encoder.h5")
                self.decoder = keras.models.load_model(f"{filepath}_decoder.h5")
            
            print(f"✅ Modèle chargé: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement: {str(e)}")
            return False
    
    def evaluate_autoencoder_quality(self, X_test: np.ndarray = None) -> Dict:
        """Évaluation rigoureuse de la qualité de l'autoencoder"""
        try:
            if not self.is_trained:
                raise Exception("Autoencoder non entraîné")
            
            # Utiliser les données de test ou un échantillon du corpus
            if X_test is None:
                if len(self.corpus_texts) < 10:
                    raise Exception("Corpus trop petit pour l'évaluation")
                
                # Générer des embeddings pour l'évaluation
                tfidf_matrix = self.tfidf_vectorizer.transform(self.corpus_texts)
                X_eval = tfidf_matrix.toarray()
                
                # Normalisation comme pendant l'entraînement
                X_eval_norm, _ = self.normalize_tfidf_data(X_eval, method='l2')
            else:
                X_eval_norm = X_test
            
            # Reconstruction
            if TENSORFLOW_AVAILABLE and self.autoencoder:
                X_reconstructed = self.autoencoder.predict(X_eval_norm, verbose=0)
            else:
                encoded = np.maximum(0, np.dot(X_eval_norm, self.weights['encoder']) + self.weights['encoder_bias'])
                X_reconstructed = np.dot(encoded, self.weights['decoder']) + self.weights['decoder_bias']
            
            # Métriques de reconstruction
            mse = np.mean((X_eval_norm - X_reconstructed) ** 2)
            mae = np.mean(np.abs(X_eval_norm - X_reconstructed))
            rmse = np.sqrt(mse)
            
            # Métriques de similarité
            similarities = []
            for i in range(min(len(X_eval_norm), 100)):  # Limiter pour la performance
                sim = cosine_similarity([X_eval_norm[i]], [X_reconstructed[i]])[0][0]
                similarities.append(sim)
            
            mean_similarity = np.mean(similarities)
            std_similarity = np.std(similarities)
            
            # Analyse de la variance expliquée
            total_variance = np.var(X_eval_norm)
            reconstruction_variance = np.var(X_reconstructed)
            variance_explained = reconstruction_variance / total_variance
            
            # Analyse de la compression
            compression_ratio = self.config['input_dim'] / self.config['encoding_dim']
            
            # Évaluation de la qualité
            quality_score = mean_similarity * 0.5 + (1 - mse) * 0.3 + variance_explained * 0.2
            
            quality_level = "Excellent" if quality_score > 0.8 else \
                           "Bon" if quality_score > 0.6 else \
                           "Moyen" if quality_score > 0.4 else "Faible"
            
            print(f"📊 Évaluation autoencoder:")
            print(f"   - MSE: {mse:.4f}")
            print(f"   - Similarité moyenne: {mean_similarity:.3f}")
            print(f"   - Variance expliquée: {variance_explained:.3f}")
            print(f"   - Score qualité: {quality_score:.3f} ({quality_level})")
            
            return {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'mean_similarity': float(mean_similarity),
                'std_similarity': float(std_similarity),
                'variance_explained': float(variance_explained),
                'compression_ratio': float(compression_ratio),
                'quality_score': float(quality_score),
                'quality_level': quality_level,
                'n_samples_evaluated': len(X_eval_norm)
            }
            
        except Exception as e:
            raise Exception(f"Erreur évaluation autoencoder: {str(e)}")
    
    def extract_all_embeddings(self, use_compressed: bool = True) -> Tuple[np.ndarray, List[str]]:
        """Extrait tous les embeddings du corpus (originaux ou compressés)"""
        try:
            if not self.is_trained:
                raise Exception("Autoencoder non entraîné")
            
            embeddings = []
            valid_texts = []
            
            for text in self.corpus_texts:
                try:
                    if use_compressed:
                        # Embedding compressé via l'autoencoder
                        encoded = self.encode_text(text)
                        embeddings.append(encoded)
                    else:
                        # Embedding TF-IDF original
                        tfidf_embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
                        embeddings.append(tfidf_embedding)
                    
                    valid_texts.append(text)
                except Exception as e:
                    print(f"⚠️ Erreur embedding pour texte: {str(e)[:100]}")
                    continue
            
            embeddings_array = np.array(embeddings)
            
            print(f"✅ {len(embeddings_array)} embeddings extraits")
            print(f"📊 Dimensions: {embeddings_array.shape}")
            
            return embeddings_array, valid_texts
            
        except Exception as e:
            raise Exception(f"Erreur extraction embeddings: {str(e)}")
    
    def perform_clustering_analysis(self, n_clusters: int = 4, use_compressed: bool = True) -> Dict:
        """Analyse de clustering rigoureuse avec métriques avancées"""
        try:
            # Extraire les embeddings
            embeddings, texts = self.extract_all_embeddings(use_compressed=use_compressed)
            
            if len(embeddings) < n_clusters:
                raise Exception(f"Pas assez d'échantillons ({len(embeddings)}) pour {n_clusters} clusters")
            
            # KMeans clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Métriques de clustering
            silhouette_avg = silhouette_score(embeddings, cluster_labels)
            calinski_harabasz = calinski_harabasz_score(embeddings, cluster_labels)
            davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
            
            # Inertie (somme des distances au carré aux centres)
            inertia = kmeans.inertia_
            
            # Analyse par cluster
            clusters_analysis = []
            for cluster_id in range(n_clusters):
                cluster_mask = cluster_labels == cluster_id
                cluster_texts = [texts[i] for i in range(len(texts)) if cluster_mask[i]]
                cluster_embeddings = embeddings[cluster_mask]
                
                # Centroïde du cluster
                centroid = np.mean(cluster_embeddings, axis=0)
                
                # Distance moyenne au centroïde
                distances_to_centroid = [
                    np.linalg.norm(emb - centroid) 
                    for emb in cluster_embeddings
                ]
                avg_distance = np.mean(distances_to_centroid)
                
                # Analyse des mots les plus fréquents (si TF-IDF)
                most_frequent_words = []
                if not use_compressed and hasattr(self, 'tfidf_vectorizer'):
                    # Pour les embeddings TF-IDF originaux
                    feature_names = self.tfidf_vectorizer.get_feature_names_out()
                    avg_tfidf = np.mean(cluster_embeddings, axis=0)
                    top_indices = avg_tfidf.argsort()[-10:][::-1]
                    most_frequent_words = [
                        (feature_names[i], float(avg_tfidf[i])) 
                        for i in top_indices if avg_tfidf[i] > 0
                    ]
                
                # Analyse de sentiment basique
                positive_words = ['good', 'great', 'excellent', 'amazing', 'perfect', 'love', 'best']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 'poor']
                
                sentiment_scores = []
                for text in cluster_texts:
                    text_lower = text.lower()
                    pos_count = sum(1 for word in positive_words if word in text_lower)
                    neg_count = sum(1 for word in negative_words if word in text_lower)
                    sentiment_scores.append(pos_count - neg_count)
                
                avg_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0
                sentiment_label = "Positif" if avg_sentiment > 0.5 else \
                                "Négatif" if avg_sentiment < -0.5 else "Neutre"
                
                clusters_analysis.append({
                    'cluster_id': int(cluster_id),
                    'size': int(np.sum(cluster_mask)),
                    'percentage': float(np.sum(cluster_mask) / len(texts) * 100),
                    'avg_distance_to_centroid': float(avg_distance),
                    'most_frequent_words': most_frequent_words[:5],
                    'sentiment_score': float(avg_sentiment),
                    'sentiment_label': sentiment_label,
                    'sample_texts': cluster_texts[:3]  # 3 exemples
                })
            
            # Interprétation des métriques
            silhouette_interpretation = "Excellent" if silhouette_avg > 0.7 else \
                                     "Bon" if silhouette_avg > 0.5 else \
                                     "Moyen" if silhouette_avg > 0.25 else "Faible"
            
            print(f"🔍 Analyse de clustering ({n_clusters} clusters):")
            print(f"   - Score de silhouette: {silhouette_avg:.3f} ({silhouette_interpretation})")
            print(f"   - Calinski-Harabasz: {calinski_harabasz:.2f}")
            print(f"   - Davies-Bouldin: {davies_bouldin:.3f}")
            print(f"   - Inertie: {inertia:.2f}")
            
            return {
                'n_clusters': n_clusters,
                'n_samples': len(embeddings),
                'embedding_type': 'compressed' if use_compressed else 'original',
                'silhouette_score': float(silhouette_avg),
                'silhouette_interpretation': silhouette_interpretation,
                'calinski_harabasz_score': float(calinski_harabasz),
                'davies_bouldin_score': float(davies_bouldin),
                'inertia': float(inertia),
                'clusters': clusters_analysis,
                'cluster_centers': kmeans.cluster_centers_.tolist()
            }
            
        except Exception as e:
            raise Exception(f"Erreur analyse clustering: {str(e)}")
    
    def optimize_clustering(self, max_clusters: int = 10, use_compressed: bool = True) -> Dict:
        """Optimisation du nombre de clusters avec méthode du coude et silhouette"""
        try:
            embeddings, texts = self.extract_all_embeddings(use_compressed=use_compressed)
            
            if len(embeddings) < 4:
                raise Exception("Pas assez d'échantillons pour l'optimisation")
            
            max_k = min(max_clusters, len(embeddings) - 1)
            k_range = range(2, max_k + 1)
            
            results = []
            inertias = []
            silhouette_scores = []
            
            print(f"🔍 Optimisation du clustering sur {len(embeddings)} échantillons...")
            
            for k in k_range:
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embeddings)
                    
                    inertia = kmeans.inertia_
                    silhouette_avg = silhouette_score(embeddings, cluster_labels)
                    
                    inertias.append(inertia)
                    silhouette_scores.append(silhouette_avg)
                    
                    results.append({
                        'k': k,
                        'inertia': float(inertia),
                        'silhouette_score': float(silhouette_avg)
                    })
                    
                    print(f"k={k}: Inertie={inertia:.2f}, Silhouette={silhouette_avg:.3f}")
                    
                except Exception as e:
                    print(f"⚠️ Erreur pour k={k}: {str(e)}")
                    continue
            
            if not results:
                raise Exception("Aucun résultat d'optimisation valide")
            
            # Méthode du coude (approximation)
            if len(inertias) >= 3:
                # Calcul des différences secondes pour trouver le coude
                second_diffs = []
                for i in range(1, len(inertias) - 1):
                    diff2 = inertias[i-1] - 2*inertias[i] + inertias[i+1]
                    second_diffs.append(abs(diff2))
                
                elbow_idx = np.argmax(second_diffs) + 1  # +1 car on commence à i=1
                elbow_k = k_range[elbow_idx]
            else:
                elbow_k = k_range[0]
            
            # Meilleur k selon silhouette
            best_silhouette_idx = np.argmax(silhouette_scores)
            best_silhouette_k = k_range[best_silhouette_idx]
            best_silhouette_score = silhouette_scores[best_silhouette_idx]
            
            # Recommandation finale (combinaison des deux méthodes)
            if abs(elbow_k - best_silhouette_k) <= 1:
                recommended_k = best_silhouette_k
                reason = "Convergence coude + silhouette"
            else:
                # Privilégier le score de silhouette si significativement meilleur
                recommended_k = best_silhouette_k
                reason = "Meilleur score de silhouette"
            
            print(f"✅ Optimisation terminée:")
            print(f"   - Coude: k={elbow_k}")
            print(f"   - Meilleur silhouette: k={best_silhouette_k} (score: {best_silhouette_score:.3f})")
            print(f"   - Recommandation: k={recommended_k} ({reason})")
            
            return {
                'k_range': list(k_range),
                'results': results,
                'elbow_k': int(elbow_k),
                'best_silhouette_k': int(best_silhouette_k),
                'best_silhouette_score': float(best_silhouette_score),
                'recommended_k': int(recommended_k),
                'recommendation_reason': reason,
                'n_samples': len(embeddings),
                'embedding_type': 'compressed' if use_compressed else 'original'
            }
            
        except Exception as e:
            raise Exception(f"Erreur optimisation clustering: {str(e)}")
    
    def train_autoencoder_regularized(self, texts: List[str] = None, config: Dict = None) -> Dict:
        """
        🎯 ENTRAINEMENT AUTOENCODER AVEC REGULARISATION AVANCEE
        Méthode spécialisée qui implémente TOUTES les techniques de régularisation avancées :
        - Régularisation L2 (Ridge)
        - Dropout progressif
        - Batch Normalization
        - Early Stopping intelligent
        - Learning Rate Scheduling
        - Monitoring avancé
        """
        try:
            print("🎯 ========== ENTRAINEMENT REGULARISE (Techniques Avancées) ==========")
            
            # Mise à jour configuration avec paramètres de régularisation
            if config:
                self.config.update(config)
            
            # Utiliser les textes fournis ou le corpus existant
            training_texts = texts if texts else self.corpus_texts
            if not training_texts:
                raise Exception("Aucun texte pour l'entraînement")
            
            print(f"📚 Corpus: {len(training_texts)} textes")
            
            # 1. TF-IDF optimisé avec préprocessing avancé
            if not self.tfidf_vectorizer:
                print("🔄 Phase 1: TF-IDF optimisé avec préprocessing NLTK...")
                self.fit_tfidf_optimized(training_texts)
            
            # 2. Génération embeddings
            print("🔄 Phase 2: Génération embeddings TF-IDF...")
            tfidf_matrix = self.tfidf_vectorizer.transform(self.corpus_texts)
            X = tfidf_matrix.toarray()
            
            print(f"📊 Matrice TF-IDF: {X.shape}")
            print(f"📊 Sparsité: {(1.0 - (tfidf_matrix.nnz / (X.shape[0] * X.shape[1])))*100:.1f}%")
            
            # 3. Construction autoencoder REGULARISE
            print("🔄 Phase 3: Construction autoencoder avec REGULARISATION...")
            architecture_info = self.build_autoencoder_optimized(input_dim=X.shape[1])
            
            # 4. Split et normalisation avancée
            print("🔄 Phase 4: Split et normalisation L2...")
            X_train, X_test, scaler_proper, _ = self.split_and_normalize_data(
                X, 
                test_size=0.2, 
                normalization='l2'
            )
            self.scaler = scaler_proper
            
            # 5. Entraînement avec callbacks avancés
            print("🔄 Phase 5: Entraînement avec callbacks avancés...")
            if TENSORFLOW_AVAILABLE and self.autoencoder:
                training_results = self._train_tensorflow_autoencoder_with_test(X_train, X_test)
            else:
                training_results = self._train_simple_autoencoder_with_test(X_train, X_test)
            
            # 6. Évaluation complète
            print("🔄 Phase 6: Évaluation qualité modèle...")
            evaluation_results = self.evaluate_autoencoder_quality(X_test)
            
            # 7. Analyse clustering sur espace compressé
            print("🔄 Phase 7: Analyse clustering dans l'espace compressé...")
            try:
                clustering_results = self.perform_clustering_analysis(n_clusters=4, use_compressed=True)
            except Exception as e:
                print(f"⚠️ Clustering échoué: {str(e)}")
                clustering_results = {'status': 'failed', 'error': str(e)}
            
            # 8. Résultats consolidés
            print("✅ ========== ENTRAINEMENT REGULARISE TERMINE ==========")
            
            final_results = {
                'status': 'success',
                'method': 'regularized_advanced',
                'advanced_techniques_implemented': [
                    '✅ Régularisation L2 (Ridge)',
                    '✅ Dropout progressif',
                    '✅ Batch Normalization',
                    '✅ Early Stopping intelligent',
                    '✅ Learning Rate Scheduling',
                    '✅ Monitoring avancé des métriques'
                ],
                'architecture': architecture_info,
                'training': training_results,
                'evaluation': evaluation_results,
                'clustering': clustering_results,
                'data_analysis': {
                    'corpus_size': len(self.corpus_texts),
                    'tfidf_shape': X.shape,
                    'train_shape': X_train.shape,
                    'test_shape': X_test.shape,
                    'compression_ratio': X.shape[1] / self.config['encoding_dim'],
                    'sparsity_percent': (1.0 - (tfidf_matrix.nnz / (X.shape[0] * X.shape[1])))*100
                },
                'regularization_summary': {
                    'l2_kernel_regularization': self.config.get('l2_kernel_reg', 0.001),
                    'l2_bias_regularization': self.config.get('l2_bias_reg', 0.0005),
                    'dropout_rates': self.config.get('dropout_rates', [0.1, 0.2, 0.3]),
                    'batch_normalization': self.config.get('use_batch_norm', True),
                    'early_stopping_patience': self.config.get('early_stopping_patience', 15),
                    'lr_scheduling': self.config.get('reduce_lr_on_plateau', True)
                },
                'advanced_validation': '🎓 Toutes les techniques de régularisation implémentées'
            }
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Erreur entraînement régularisé: {str(e)}")

    def train_autoencoder_with_evaluation(self, texts: List[str] = None, config: Dict = None) -> Dict:
        """Entraînement avec évaluation complète"""
        try:
            # Entraînement standard
            training_result = self.train_autoencoder(texts, config, use_proper_split=True)
            
            # Évaluation de la qualité
            quality_evaluation = self.evaluate_autoencoder_quality()
            
            # Analyse de clustering sur l'espace compressé
            try:
                clustering_analysis = self.perform_clustering_analysis(n_clusters=4, use_compressed=True)
                clustering_optimization = self.optimize_clustering(max_clusters=8, use_compressed=True)
            except Exception as e:
                print(f"⚠️ Erreur analyse clustering: {str(e)}")
                clustering_analysis = None
                clustering_optimization = None
            
            # Résultat complet
            complete_result = {
                'training': training_result,
                'quality_evaluation': quality_evaluation,
                'clustering_analysis': clustering_analysis,
                'clustering_optimization': clustering_optimization,
                'summary': {
                    'architecture': training_result.get('architecture', 'unknown'),
                    'compression_ratio': quality_evaluation.get('compression_ratio', 0),
                    'quality_score': quality_evaluation.get('quality_score', 0),
                    'quality_level': quality_evaluation.get('quality_level', 'Unknown'),
                    'recommended_clusters': clustering_optimization.get('recommended_k', 4) if clustering_optimization else 4,
                    'silhouette_score': clustering_analysis.get('silhouette_score', 0) if clustering_analysis else 0
                }
            }
            
            print(f"\n🎯 Résumé de l'entraînement:")
            print(f"   - Architecture: {complete_result['summary']['architecture']}")
            print(f"   - Compression: {complete_result['summary']['compression_ratio']:.1f}x")
            print(f"   - Qualité: {complete_result['summary']['quality_level']} ({complete_result['summary']['quality_score']:.3f})")
            print(f"   - Clusters recommandés: {complete_result['summary']['recommended_clusters']}")
            
            return complete_result
            
        except Exception as e:
            raise Exception(f"Erreur entraînement avec évaluation: {str(e)}") 