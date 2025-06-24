"""
Service Word2Vec simplifié pour de meilleurs embeddings sémantiques
Compatible Windows sans compilation
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import json
import os
from typing import List, Dict, Optional
import re
from collections import Counter, defaultdict
import pickle

class SimpleWord2Vec:
    """Implémentation simplifiée de Word2Vec basée sur la co-occurrence"""
    
    def __init__(self, vector_size=100, window=5, min_count=2):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.vocabulary = {}
        self.word_vectors = {}
        self.word_counts = Counter()
        self.cooccurrence_matrix = defaultdict(lambda: defaultdict(int))
        
    def preprocess_text(self, text: str) -> List[str]:
        """Préprocessing du texte"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = [token for token in text.split() if len(token) >= 2]
        return tokens
    
    def build_vocabulary(self, texts: List[str]):
        """Construit le vocabulaire"""
        for text in texts:
            tokens = self.preprocess_text(text)
            self.word_counts.update(tokens)
        
        # Filtrer par fréquence minimale
        self.vocabulary = {
            word: idx for idx, (word, count) in enumerate(self.word_counts.items())
            if count >= self.min_count
        }
        
        print(f"📚 Vocabulaire: {len(self.vocabulary)} mots")
    
    def build_cooccurrence_matrix(self, texts: List[str]):
        """Construit la matrice de co-occurrence"""
        for text in texts:
            tokens = self.preprocess_text(text)
            
            for i, target_word in enumerate(tokens):
                if target_word not in self.vocabulary:
                    continue
                
                # Fenêtre de contexte
                start = max(0, i - self.window)
                end = min(len(tokens), i + self.window + 1)
                
                for j in range(start, end):
                    if i != j and tokens[j] in self.vocabulary:
                        context_word = tokens[j]
                        distance = abs(i - j)
                        weight = 1.0 / distance  # Pondération par distance
                        self.cooccurrence_matrix[target_word][context_word] += weight
        
        print(f"🔗 Matrice de co-occurrence construite")
    
    def train_embeddings(self, texts: List[str]):
        """Entraîne les embeddings"""
        self.build_vocabulary(texts)
        self.build_cooccurrence_matrix(texts)
        
        # Initialisation aléatoire des vecteurs
        vocab_size = len(self.vocabulary)
        self.word_vectors = {}
        
        for word in self.vocabulary:
            self.word_vectors[word] = np.random.normal(0, 0.1, self.vector_size)
        
        # Optimisation par descente de gradient simplifiée
        learning_rate = 0.01
        epochs = 10
        
        print(f"🧠 Entraînement des embeddings ({epochs} epochs)...")
        
        for epoch in range(epochs):
            total_loss = 0
            updates = 0
            
            for target_word in self.vocabulary:
                if target_word not in self.cooccurrence_matrix:
                    continue
                
                target_vector = self.word_vectors[target_word]
                
                for context_word, cooccur_count in self.cooccurrence_matrix[target_word].items():
                    if context_word not in self.vocabulary:
                        continue
                    
                    context_vector = self.word_vectors[context_word]
                    
                    # Calcul du score de similarité
                    dot_product = np.dot(target_vector, context_vector)
                    
                    # Loss simplifié (différence avec co-occurrence log)
                    expected_score = np.log(cooccur_count + 1)
                    loss = (dot_product - expected_score) ** 2
                    total_loss += loss
                    
                    # Gradient
                    gradient = 2 * (dot_product - expected_score)
                    
                    # Mise à jour des vecteurs
                    self.word_vectors[target_word] -= learning_rate * gradient * context_vector
                    self.word_vectors[context_word] -= learning_rate * gradient * target_vector
                    
                    updates += 1
            
            if epoch % 2 == 0:
                avg_loss = total_loss / max(updates, 1)
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")
        
        # Normalisation des vecteurs
        for word in self.word_vectors:
            norm = np.linalg.norm(self.word_vectors[word])
            if norm > 0:
                self.word_vectors[word] = self.word_vectors[word] / norm
        
        print(f"✅ Embeddings entraînés pour {len(self.word_vectors)} mots")

class Word2VecService:
    """Service Word2Vec avec fallback TF-IDF"""
    
    def __init__(self):
        self.model = None
        self.models_dir = "./models/embeddings"
        self.corpus_texts = []
        
        # Essayer d'importer des bibliothèques avancées
        try:
            import gensim
            self.has_gensim = True
            print("✅ Gensim disponible")
        except ImportError:
            self.has_gensim = False
            print("⚠️ Gensim non disponible - utilisation du Word2Vec simplifié")
        
        os.makedirs(self.models_dir, exist_ok=True)
        print("🚀 Service Word2Vec initialisé")
    
    def train_word2vec(self, texts: List[str], config: Dict = None) -> Dict:
        """Entraîne un modèle Word2Vec"""
        try:
            if self.has_gensim:
                return self._train_gensim_word2vec(texts, config)
            else:
                return self._train_simple_word2vec(texts, config)
        except Exception as e:
            # Fallback vers le modèle simple en cas d'erreur
            print(f"⚠️ Erreur Gensim, fallback vers Word2Vec simple: {e}")
            return self._train_simple_word2vec(texts, config)
    
    def _train_gensim_word2vec(self, texts: List[str], config: Dict = None) -> Dict:
        """Entraîne avec Gensim (si disponible)"""
        from gensim.models import Word2Vec
        
        # Configuration par défaut
        w2v_config = {
            'vector_size': 100,
            'window': 5,
            'min_count': 2,
            'workers': 1,  # Réduire pour éviter les problèmes Windows
            'epochs': 10,
            'sg': 1
        }
        
        if config:
            w2v_config.update(config)
        
        # Préparation du corpus
        corpus = []
        for text in texts:
            tokens = self._preprocess_text(text)
            if len(tokens) > 0:
                corpus.append(tokens)
        
        # Entraînement
        self.model = Word2Vec(
            sentences=corpus,
            **w2v_config
        )
        
        self.corpus_texts = texts
        
        return {
            'model_type': 'gensim_word2vec',
            'vocabulary_size': len(self.model.wv.key_to_index),
            'vector_size': w2v_config['vector_size'],
            'corpus_size': len(corpus)
        }
    
    def _train_simple_word2vec(self, texts: List[str], config: Dict = None) -> Dict:
        """Entraîne avec le modèle simple"""
        w2v_config = {
            'vector_size': 100,
            'window': 5,
            'min_count': 2
        }
        
        if config:
            w2v_config.update(config)
        
        self.model = SimpleWord2Vec(**w2v_config)
        self.model.train_embeddings(texts)
        self.corpus_texts = texts
        
        return {
            'model_type': 'simple_word2vec',
            'vocabulary_size': len(self.model.vocabulary),
            'vector_size': w2v_config['vector_size'],
            'corpus_size': len(texts)
        }
    
    def _preprocess_text(self, text: str) -> List[str]:
        """Préprocessing du texte"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        tokens = [token for token in text.split() if len(token) >= 2]
        return tokens
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Obtient l'embedding d'un mot"""
        if not self.model:
            raise Exception("Modèle non entraîné")
        
        processed_word = self._preprocess_text(word)
        if not processed_word:
            raise KeyError(f"Mot vide: {word}")
        
        processed_word = processed_word[0]
        
        if hasattr(self.model, 'wv'):  # Gensim
            if processed_word in self.model.wv:
                return self.model.wv[processed_word]
        elif hasattr(self.model, 'word_vectors'):  # Simple
            if processed_word in self.model.word_vectors:
                return self.model.word_vectors[processed_word]
        
        raise KeyError(f"Mot non trouvé: {word}")
    
    def get_text_embedding(self, text: str, method: str = 'average') -> np.ndarray:
        """Obtient l'embedding d'un texte"""
        tokens = self._preprocess_text(text)
        if not tokens:
            raise Exception("Texte vide")
        
        embeddings = []
        for token in tokens:
            try:
                embedding = self.get_word_embedding(token)
                embeddings.append(embedding)
            except KeyError:
                continue
        
        if not embeddings:
            raise Exception("Aucun mot trouvé dans le vocabulaire")
        
        embeddings = np.array(embeddings)
        
        if method == 'average':
            return np.mean(embeddings, axis=0)
        elif method == 'sum':
            return np.sum(embeddings, axis=0)
        elif method == 'max':
            return np.max(embeddings, axis=0)
        else:
            return np.mean(embeddings, axis=0)
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Dict]:
        """Trouve les mots similaires"""
        try:
            word_embedding = self.get_word_embedding(word)
            
            similarities = []
            
            if hasattr(self.model, 'wv'):  # Gensim
                similar_words = self.model.wv.most_similar(word, topn=top_k)
                return [
                    {'word': w, 'similarity': float(s), 'original_query': word}
                    for w, s in similar_words
                ]
            
            elif hasattr(self.model, 'word_vectors'):  # Simple
                for other_word, other_embedding in self.model.word_vectors.items():
                    if other_word != word:
                        similarity = cosine_similarity([word_embedding], [other_embedding])[0][0]
                        similarities.append((other_word, similarity))
                
                similarities.sort(key=lambda x: x[1], reverse=True)
                
                return [
                    {'word': w, 'similarity': float(s), 'original_query': word}
                    for w, s in similarities[:top_k]
                ]
            
            return []
            
        except Exception as e:
            raise Exception(f"Erreur recherche mots similaires: {str(e)}")
    
    def semantic_search(self, query: str, texts: List[str] = None, top_k: int = 5) -> List[Dict]:
        """Recherche sémantique"""
        try:
            if not self.model:
                raise Exception("Modèle non entraîné")
            
            search_texts = texts if texts else self.corpus_texts
            if not search_texts:
                raise Exception("Aucun texte pour la recherche")
            
            # Embedding de la requête
            query_embedding = self.get_text_embedding(query)
            
            # Calcul des similarités
            results = []
            for i, text in enumerate(search_texts):
                try:
                    text_embedding = self.get_text_embedding(text)
                    similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
                    
                    results.append({
                        'index': i,
                        'text': text,
                        'similarity': float(similarity),
                        'text_preview': text[:200] + "..." if len(text) > 200 else text,
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                except Exception:
                    continue
            
            # Tri par similarité
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            raise Exception(f"Erreur recherche sémantique: {str(e)}")
    
    def visualize_words(self, words: List[str], method: str = 'tsne') -> Dict:
        """Visualise les mots dans l'espace vectoriel"""
        try:
            embeddings = []
            valid_words = []
            
            for word in words:
                try:
                    embedding = self.get_word_embedding(word)
                    embeddings.append(embedding)
                    valid_words.append(word)
                except KeyError:
                    continue
            
            if len(embeddings) < 2:
                raise Exception("Pas assez de mots valides")
            
            embeddings = np.array(embeddings)
            
            # Réduction de dimensionnalité
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            else:
                perplexity = min(5, len(embeddings) - 1)
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            
            embeddings_2d = reducer.fit_transform(embeddings)
            
            return {
                'words': valid_words,
                'coordinates': embeddings_2d.tolist(),
                'method': method,
                'words_found': len(valid_words),
                'words_not_found': [w for w in words if w not in valid_words]
            }
            
        except Exception as e:
            raise Exception(f"Erreur visualisation: {str(e)}")
    
    def is_trained(self) -> bool:
        """Vérifie si le modèle est entraîné"""
        return self.model is not None
    
    def get_model_info(self) -> Dict:
        """Informations sur le modèle"""
        if not self.is_trained():
            return {'trained': False}
        
        info = {'trained': True}
        
        if hasattr(self.model, 'wv'):  # Gensim
            info.update({
                'model_type': 'gensim_word2vec',
                'vocabulary_size': len(self.model.wv.key_to_index),
                'vector_size': self.model.wv.vector_size
            })
        elif hasattr(self.model, 'word_vectors'):  # Simple
            info.update({
                'model_type': 'simple_word2vec',
                'vocabulary_size': len(self.model.vocabulary),
                'vector_size': self.model.vector_size
            })
        
        return info
    
    def save_model(self, filename: str) -> str:
        """Sauvegarde le modèle"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            if hasattr(self.model, 'save'):  # Gensim
                self.model.save(filepath + '.model')
            else:  # Simple
                with open(filepath + '.pkl', 'wb') as f:
                    pickle.dump(self.model, f)
            
            return filepath
            
        except Exception as e:
            raise Exception(f"Erreur sauvegarde: {str(e)}")
    
    def load_model(self, filename: str) -> bool:
        """Charge un modèle"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            if os.path.exists(filepath + '.model') and self.has_gensim:
                from gensim.models import Word2Vec
                self.model = Word2Vec.load(filepath + '.model')
                return True
            elif os.path.exists(filepath + '.pkl'):
                with open(filepath + '.pkl', 'rb') as f:
                    self.model = pickle.load(f)
                return True
            
            return False
            
        except Exception as e:
            print(f"Erreur chargement: {str(e)}")
            return False 