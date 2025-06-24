import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import re
import pickle
from collections import Counter
import logging

# Configuration pour éviter les warnings
logging.getLogger("gensim").setLevel(logging.WARNING)

try:
    # Essayer d'importer gensim pour Word2Vec
    from gensim.models import Word2Vec
    from gensim.models.phrases import Phrases, Phraser
    GENSIM_AVAILABLE = True
    print("✅ Gensim disponible - Word2Vec activé")
except ImportError:
    GENSIM_AVAILABLE = False
    print("⚠️ Gensim non disponible - Fallback vers TF-IDF")
    from sklearn.feature_extraction.text import TfidfVectorizer

class EmbeddingServiceWord2Vec:
    def __init__(self):
        self.model = None
        self.word2vec_model = None
        self.bigram_model = None
        self.trigram_model = None
        self.vocabulary = set()
        self.embeddings_cache = {}
        self.models_dir = "./models/embeddings"
        
        # Configuration Word2Vec
        self.w2v_config = {
            'vector_size': 100,
            'window': 5,
            'min_count': 2,
            'workers': 4,
            'epochs': 10,
            'sg': 1,  # Skip-gram
            'hs': 0,  # Hierarchical softmax
            'negative': 5,  # Negative sampling
            'alpha': 0.025,
            'min_alpha': 0.0001
        }
        
        # Fallback TF-IDF si Word2Vec non disponible
        if not GENSIM_AVAILABLE:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.tfidf_vectorizer = None
            self.tfidf_matrix = None
        
        os.makedirs(self.models_dir, exist_ok=True)
        
        if GENSIM_AVAILABLE:
            print("🚀 Service Word2Vec initialisé")
        else:
            print("🔄 Service en mode TF-IDF (Word2Vec non disponible)")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Préprocessing pour Word2Vec"""
        # Minuscules
        text = text.lower()
        
        # Supprimer la ponctuation mais garder les apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Supprimer les nombres
        text = re.sub(r'\d+', '', text)
        
        # Tokenisation
        tokens = text.split()
        
        # Filtrer les mots trop courts
        tokens = [token for token in tokens if len(token) >= 2]
        
        return tokens
    
    def prepare_corpus(self, texts: List[str]) -> List[List[str]]:
        """Prépare le corpus pour Word2Vec"""
        corpus = []
        for text in texts:
            tokens = self.preprocess_text(text)
            if len(tokens) > 0:
                corpus.append(tokens)
        
        return corpus
    
    def train_word2vec(self, texts: List[str], config: Dict = None) -> Dict:
        """Entraîne un modèle Word2Vec"""
        if not GENSIM_AVAILABLE:
            return self._train_tfidf_fallback(texts, config)
        
        try:
            if config:
                self.w2v_config.update(config)
            
            print(f"🔧 Configuration Word2Vec: {self.w2v_config}")
            
            # Préparation du corpus
            corpus = self.prepare_corpus(texts)
            print(f"📝 Corpus préparé: {len(corpus)} documents")
            
            # Entraînement des bigrammes et trigrammes
            print("🔗 Entraînement des phrases (bigrammes/trigrammes)...")
            bigram = Phrases(corpus, min_count=5, threshold=100)
            self.bigram_model = Phraser(bigram)
            
            # Application des bigrammes
            corpus_bigrams = [self.bigram_model[doc] for doc in corpus]
            
            # Trigrammes
            trigram = Phrases(corpus_bigrams, threshold=100)
            self.trigram_model = Phraser(trigram)
            
            # Application des trigrammes
            corpus_trigrams = [self.trigram_model[self.bigram_model[doc]] for doc in corpus]
            
            # Entraînement Word2Vec
            print("🧠 Entraînement Word2Vec...")
            self.word2vec_model = Word2Vec(
                sentences=corpus_trigrams,
                vector_size=self.w2v_config['vector_size'],
                window=self.w2v_config['window'],
                min_count=self.w2v_config['min_count'],
                workers=self.w2v_config['workers'],
                epochs=self.w2v_config['epochs'],
                sg=self.w2v_config['sg'],
                hs=self.w2v_config['hs'],
                negative=self.w2v_config['negative'],
                alpha=self.w2v_config['alpha'],
                min_alpha=self.w2v_config['min_alpha']
            )
            
            # Construire le vocabulaire
            self.vocabulary = set(self.word2vec_model.wv.key_to_index.keys())
            
            # Statistiques
            vocab_size = len(self.vocabulary)
            
            print(f"✅ Word2Vec entraîné sur {len(corpus)} documents")
            print(f"📊 Vocabulaire : {vocab_size} mots")
            print(f"🎯 Dimensions : {self.w2v_config['vector_size']}")
            
            return {
                'model_type': 'word2vec',
                'vocabulary_size': vocab_size,
                'corpus_size': len(corpus),
                'vector_size': self.w2v_config['vector_size'],
                'config': self.w2v_config,
                'has_bigrams': True,
                'has_trigrams': True
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement Word2Vec: {str(e)}")
    
    def _train_tfidf_fallback(self, texts: List[str], config: Dict = None) -> Dict:
        """Fallback TF-IDF si Word2Vec non disponible"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            processed_texts = [' '.join(self.preprocess_text(text)) for text in texts]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            self.vocabulary = set(self.tfidf_vectorizer.get_feature_names_out())
            
            return {
                'model_type': 'tfidf_fallback',
                'vocabulary_size': len(self.vocabulary),
                'corpus_size': len(texts),
                'vector_size': self.tfidf_matrix.shape[1]
            }
            
        except Exception as e:
            raise Exception(f"Erreur fallback TF-IDF: {str(e)}")
    
    def get_word_embedding(self, word: str) -> np.ndarray:
        """Obtient l'embedding d'un mot"""
        if not GENSIM_AVAILABLE or not self.word2vec_model:
            return self._get_tfidf_word_embedding(word)
        
        try:
            # Préprocessing du mot
            processed_word = self.preprocess_text(word)
            if not processed_word:
                raise KeyError(f"Mot vide après preprocessing: {word}")
            
            processed_word = processed_word[0]
            
            # Appliquer les transformations de phrases
            if self.bigram_model and self.trigram_model:
                phrase = self.trigram_model[self.bigram_model[[processed_word]]]
                if phrase:
                    processed_word = phrase[0]
            
            # Récupérer l'embedding
            if processed_word in self.word2vec_model.wv:
                return self.word2vec_model.wv[processed_word]
            else:
                raise KeyError(f"Mot non trouvé dans le vocabulaire: {processed_word}")
                
        except Exception as e:
            raise KeyError(f"Erreur embedding mot '{word}': {str(e)}")
    
    def _get_tfidf_word_embedding(self, word: str) -> np.ndarray:
        """Fallback TF-IDF pour l'embedding d'un mot"""
        if not self.tfidf_vectorizer:
            raise Exception("Aucun modèle entraîné")
        
        try:
            processed_word = ' '.join(self.preprocess_text(word))
            embedding = self.tfidf_vectorizer.transform([processed_word])
            return embedding.toarray()[0]
        except Exception as e:
            raise KeyError(f"Mot non trouvé: {word}")
    
    def get_text_embedding(self, text: str, method: str = 'average') -> np.ndarray:
        """Obtient l'embedding d'un texte"""
        try:
            tokens = self.preprocess_text(text)
            if not tokens:
                raise Exception("Texte vide après preprocessing")
            
            embeddings = []
            found_words = []
            
            for token in tokens:
                try:
                    embedding = self.get_word_embedding(token)
                    embeddings.append(embedding)
                    found_words.append(token)
                except KeyError:
                    continue
            
            if not embeddings:
                raise Exception("Aucun mot trouvé dans le vocabulaire")
            
            embeddings = np.array(embeddings)
            
            # Méthodes d'agrégation
            if method == 'average':
                return np.mean(embeddings, axis=0)
            elif method == 'sum':
                return np.sum(embeddings, axis=0)
            elif method == 'max':
                return np.max(embeddings, axis=0)
            elif method == 'weighted_average':
                # Pondération par fréquence inverse
                word_counts = Counter(found_words)
                weights = [1.0 / word_counts[word] for word in found_words]
                weights = np.array(weights) / np.sum(weights)
                return np.average(embeddings, axis=0, weights=weights)
            else:
                return np.mean(embeddings, axis=0)
                
        except Exception as e:
            raise Exception(f"Erreur embedding texte: {str(e)}")
    
    def find_similar_words(self, word: str, top_k: int = 10) -> List[Dict]:
        """Trouve les mots les plus similaires"""
        if not GENSIM_AVAILABLE or not self.word2vec_model:
            return self._find_similar_words_tfidf(word, top_k)
        
        try:
            processed_word = self.preprocess_text(word)[0]
            
            # Appliquer les transformations de phrases
            if self.bigram_model and self.trigram_model:
                phrase = self.trigram_model[self.bigram_model[[processed_word]]]
                if phrase:
                    processed_word = phrase[0]
            
            if processed_word not in self.word2vec_model.wv:
                raise KeyError(f"Mot non trouvé: {word}")
            
            similar_words = self.word2vec_model.wv.most_similar(processed_word, topn=top_k)
            
            return [
                {
                    'word': similar_word,
                    'similarity': float(similarity),
                    'original_query': word
                }
                for similar_word, similarity in similar_words
            ]
            
        except Exception as e:
            raise Exception(f"Erreur recherche mots similaires: {str(e)}")
    
    def _find_similar_words_tfidf(self, word: str, top_k: int = 10) -> List[Dict]:
        """Fallback TF-IDF pour mots similaires"""
        try:
            if not self.tfidf_vectorizer:
                raise Exception("Modèle non entraîné")
            
            word_embedding = self._get_tfidf_word_embedding(word)
            
            # Calculer similarités avec tous les mots du vocabulaire
            similarities = []
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            
            for i, feature in enumerate(feature_names):
                if feature != word:
                    feature_embedding = np.zeros(len(feature_names))
                    feature_embedding[i] = 1.0
                    similarity = cosine_similarity([word_embedding], [feature_embedding])[0][0]
                    similarities.append((feature, similarity))
            
            # Trier par similarité
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            return [
                {
                    'word': similar_word,
                    'similarity': float(similarity),
                    'original_query': word
                }
                for similar_word, similarity in similarities[:top_k]
            ]
            
        except Exception as e:
            return []
    
    def semantic_search(self, query: str, texts: List[str], top_k: int = 5) -> List[Dict]:
        """Recherche sémantique améliorée"""
        try:
            if not self.is_trained():
                raise Exception("Modèle non entraîné")
            
            # Embedding de la requête
            query_embedding = self.get_text_embedding(query, method='weighted_average')
            
            # Déduplication
            unique_texts = self._deduplicate_texts(texts)
            
            # Embeddings des textes
            results = []
            for i, text in enumerate(unique_texts):
                try:
                    text_embedding = self.get_text_embedding(text, method='weighted_average')
                    similarity = cosine_similarity([query_embedding], [text_embedding])[0][0]
                    
                    # Métriques supplémentaires
                    word_overlap = self._calculate_word_overlap(query, text)
                    
                    results.append({
                        'index': i,
                        'text': text,
                        'similarity': float(similarity),
                        'word_overlap': word_overlap,
                        'combined_score': float(similarity * 0.8 + word_overlap * 0.2),
                        'text_preview': self._create_smart_preview(text, query),
                        'word_count': len(text.split()),
                        'char_count': len(text)
                    })
                    
                except Exception:
                    continue
            
            # Tri par score combiné
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Diversification
            diversified_results = self._diversify_results(results, top_k)
            
            return diversified_results[:top_k]
            
        except Exception as e:
            raise Exception(f"Erreur recherche sémantique: {str(e)}")
    
    def _deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Déduplication basée sur les mots principaux"""
        unique_texts = []
        seen_fingerprints = set()
        
        for text in texts:
            words = set(self.preprocess_text(text))
            fingerprint = hash(frozenset(list(words)[:10]))
            
            if fingerprint not in seen_fingerprints and len(text.strip()) > 20:
                unique_texts.append(text)
                seen_fingerprints.add(fingerprint)
        
        return unique_texts
    
    def _calculate_word_overlap(self, query: str, text: str) -> float:
        """Calcule le chevauchement de mots"""
        query_words = set(self.preprocess_text(query))
        text_words = set(self.preprocess_text(text))
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
    
    def _create_smart_preview(self, text: str, query: str) -> str:
        """Crée un aperçu intelligent"""
        sentences = re.split(r'[.!?]+', text)
        query_words = set(self.preprocess_text(query))
        
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                sentence_words = set(self.preprocess_text(sentence))
                overlap = len(query_words.intersection(sentence_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence + "..."
        else:
            return text[:150] + "..." if len(text) > 150 else text
    
    def _diversify_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Diversifie les résultats"""
        if len(results) <= top_k:
            return results
        
        diversified = [results[0]]
        
        for result in results[1:]:
            if len(diversified) >= top_k:
                break
            
            is_diverse = True
            for selected in diversified:
                similarity = self._text_similarity(result['text'], selected['text'])
                if similarity > 0.7:
                    is_diverse = False
                    break
            
            if is_diverse:
                diversified.append(result)
        
        return diversified
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Similarité entre deux textes"""
        words1 = set(self.preprocess_text(text1))
        words2 = set(self.preprocess_text(text2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
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
                raise Exception("Pas assez de mots valides pour la visualisation")
            
            embeddings = np.array(embeddings)
            
            # Réduction de dimensionnalité
            if method.lower() == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            else:  # t-SNE
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
    
    def save_model(self, filename: str) -> str:
        """Sauvegarde le modèle"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            if GENSIM_AVAILABLE and self.word2vec_model:
                self.word2vec_model.save(filepath + '_w2v.model')
                
                # Sauvegarder les modèles de phrases
                if self.bigram_model:
                    with open(filepath + '_bigram.pkl', 'wb') as f:
                        pickle.dump(self.bigram_model, f)
                
                if self.trigram_model:
                    with open(filepath + '_trigram.pkl', 'wb') as f:
                        pickle.dump(self.trigram_model, f)
            
            elif self.tfidf_vectorizer:
                with open(filepath + '_tfidf.pkl', 'wb') as f:
                    pickle.dump({
                        'vectorizer': self.tfidf_vectorizer,
                        'matrix': self.tfidf_matrix
                    }, f)
            
            return filepath
            
        except Exception as e:
            raise Exception(f"Erreur sauvegarde: {str(e)}")
    
    def load_model(self, filename: str) -> bool:
        """Charge un modèle sauvegardé"""
        try:
            filepath = os.path.join(self.models_dir, filename)
            
            # Essayer de charger Word2Vec
            if GENSIM_AVAILABLE and os.path.exists(filepath + '_w2v.model'):
                self.word2vec_model = Word2Vec.load(filepath + '_w2v.model')
                self.vocabulary = set(self.word2vec_model.wv.key_to_index.keys())
                
                # Charger les modèles de phrases
                if os.path.exists(filepath + '_bigram.pkl'):
                    with open(filepath + '_bigram.pkl', 'rb') as f:
                        self.bigram_model = pickle.load(f)
                
                if os.path.exists(filepath + '_trigram.pkl'):
                    with open(filepath + '_trigram.pkl', 'rb') as f:
                        self.trigram_model = pickle.load(f)
                
                return True
            
            # Essayer de charger TF-IDF
            elif os.path.exists(filepath + '_tfidf.pkl'):
                with open(filepath + '_tfidf.pkl', 'rb') as f:
                    data = pickle.load(f)
                    self.tfidf_vectorizer = data['vectorizer']
                    self.tfidf_matrix = data['matrix']
                    self.vocabulary = set(self.tfidf_vectorizer.get_feature_names_out())
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Erreur chargement: {str(e)}")
            return False
    
    def is_trained(self) -> bool:
        """Vérifie si un modèle est entraîné"""
        if GENSIM_AVAILABLE:
            return self.word2vec_model is not None
        else:
            return self.tfidf_vectorizer is not None
    
    def get_model_info(self) -> Dict:
        """Informations sur le modèle"""
        if not self.is_trained():
            return {'trained': False}
        
        info = {
            'trained': True,
            'vocabulary_size': len(self.vocabulary),
            'model_type': 'word2vec' if GENSIM_AVAILABLE and self.word2vec_model else 'tfidf_fallback'
        }
        
        if GENSIM_AVAILABLE and self.word2vec_model:
            info.update({
                'vector_size': self.word2vec_model.wv.vector_size,
                'has_bigrams': self.bigram_model is not None,
                'has_trigrams': self.trigram_model is not None,
                'config': self.w2v_config
            })
        
        return info 