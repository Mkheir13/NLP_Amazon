import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import re
import pickle

# Essayer d'importer TensorFlow/Keras
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model, Sequential
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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
        self.scaler = StandardScaler()
        self.is_trained = False
        self.corpus_texts = []
        
        # Configuration des répertoires
        if self.app_config:
            self.models_dir = self.app_config.AUTOENCODER_DIR
            self.config = self.app_config.get_autoencoder_config()
        else:
            self.models_dir = "./models/embeddings"
            self.config = {
                'input_dim': 1000,
                'encoding_dim': 64,
                'hidden_layers': [512, 128],
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
    
    def preprocess_texts(self, texts: List[str]) -> List[str]:
        """Préprocessing des textes pour TF-IDF"""
        processed = []
        for text in texts:
            # Nettoyage basique
            text = re.sub(r'[^\w\s]', ' ', text.lower())
            text = re.sub(r'\s+', ' ', text.strip())
            if len(text) > 10:  # Filtrer les textes trop courts
                processed.append(text)
        return processed
    
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
    
    def train_autoencoder(self, texts: List[str] = None, config: Dict = None) -> Dict:
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
            
            # Normalisation
            X_normalized = self.scaler.fit_transform(X)
            
            # Construire l'autoencoder si nécessaire
            if not self.autoencoder and not hasattr(self, 'weights'):
                self.build_autoencoder(input_dim=X.shape[1])
            
            print(f"🚀 Entraînement autoencoder sur {X.shape[0]} échantillons...")
            
            if TENSORFLOW_AVAILABLE and self.autoencoder:
                return self._train_tensorflow_autoencoder(X_normalized)
            else:
                return self._train_simple_autoencoder(X_normalized)
            
        except Exception as e:
            raise Exception(f"Erreur entraînement autoencoder: {str(e)}")
    
    def _train_tensorflow_autoencoder(self, X: np.ndarray) -> Dict:
        """Entraînement avec TensorFlow"""
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
            
            # Entraînement
            history = self.autoencoder.fit(
                X, X,  # Autoencoder: input = target
                epochs=self.config['epochs'],
                batch_size=self.config['batch_size'],
                validation_split=self.config['validation_split'],
                callbacks=callbacks,
                verbose=1
            )
            
            self.is_trained = True
            
            # Évaluation
            final_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            
            # Test de reconstruction
            sample_indices = np.random.choice(X.shape[0], min(10, X.shape[0]), replace=False)
            sample_input = X[sample_indices]
            sample_reconstructed = self.autoencoder.predict(sample_input, verbose=0)
            reconstruction_error = np.mean((sample_input - sample_reconstructed) ** 2)
            
            print(f"✅ Autoencoder entraîné avec succès!")
            print(f"📊 Perte finale: {final_loss:.4f}")
            print(f"📊 Perte validation: {final_val_loss:.4f}")
            print(f"📊 Erreur reconstruction: {reconstruction_error:.4f}")
            
            return {
                'status': 'success',
                'architecture': 'tensorflow',
                'final_loss': float(final_loss),
                'final_val_loss': float(final_val_loss),
                'reconstruction_error': float(reconstruction_error),
                'epochs_trained': len(history.history['loss']),
                'input_shape': X.shape,
                'encoding_dim': self.config['encoding_dim'],
                'compression_ratio': X.shape[1] / self.config['encoding_dim']
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TensorFlow: {str(e)}")
    
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
    
    def encode_text(self, text: str) -> np.ndarray:
        """Encode un texte vers sa représentation compressée"""
        try:
            if not self.is_trained:
                raise Exception("Autoencoder non entraîné")
            
            # TF-IDF embedding
            tfidf_embedding = self.tfidf_vectorizer.transform([text]).toarray()
            tfidf_normalized = self.scaler.transform(tfidf_embedding)
            
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
            
            # Dénormalisation
            decoded_denorm = self.scaler.inverse_transform(decoded)
            
            return decoded_denorm[0] if decoded_denorm.shape[0] == 1 else decoded_denorm
            
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