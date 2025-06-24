import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from collections import Counter
import math

# Télécharger les données NLTK nécessaires
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

class EmbeddingServiceEnhanced:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.corpus_texts = []
        self.embeddings_cache = {}
        self.models_dir = "./models/embeddings"
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english')) if nltk.data.find('corpora/stopwords') else set()
        
        # Paramètres avancés
        self.advanced_config = {
            'use_stemming': True,
            'use_bigrams': True,
            'use_trigrams': False,
            'remove_numbers': True,
            'min_word_length': 2,
            'max_features': 10000,
            'use_idf_weighting': True,
            'sublinear_tf': True,
            'use_l2_norm': True
        }
        
        os.makedirs(self.models_dir, exist_ok=True)
        print("✅ Service d'embedding avancé initialisé")
    
    def preprocess_text_advanced(self, text: str) -> str:
        """Préprocessing avancé du texte"""
        try:
            # Conversion en minuscules
            text = text.lower()
            
            # Suppression des caractères spéciaux mais garde les espaces
            text = re.sub(r'[^\w\s]', ' ', text)
            
            # Suppression des nombres si activé
            if self.advanced_config['remove_numbers']:
                text = re.sub(r'\d+', '', text)
            
            # Tokenisation
            try:
                tokens = word_tokenize(text)
            except:
                tokens = text.split()
            
            # Filtrage et stemming
            processed_tokens = []
            for token in tokens:
                # Filtrer par longueur
                if len(token) >= self.advanced_config['min_word_length']:
                    # Supprimer les mots vides
                    if token not in self.stop_words:
                        # Stemming si activé
                        if self.advanced_config['use_stemming']:
                            try:
                                token = self.stemmer.stem(token)
                            except:
                                pass
                        processed_tokens.append(token)
            
            return ' '.join(processed_tokens)
            
        except Exception as e:
            print(f"⚠️ Erreur preprocessing: {e}")
            return text.lower()
    
    def create_ngrams(self, tokens: List[str]) -> List[str]:
        """Crée des n-grammes pour enrichir le vocabulaire"""
        result = tokens.copy()
        
        # Bigrammes
        if self.advanced_config['use_bigrams'] and len(tokens) > 1:
            bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
            result.extend(bigrams)
        
        # Trigrammes
        if self.advanced_config['use_trigrams'] and len(tokens) > 2:
            trigrams = [f"{tokens[i]}_{tokens[i+1]}_{tokens[i+2]}" for i in range(len(tokens)-2)]
            result.extend(trigrams)
        
        return result
    
    def fit_tfidf_enhanced(self, texts: List[str], config: Dict = None):
        """TF-IDF amélioré avec préprocessing avancé"""
        try:
            if config:
                self.advanced_config.update(config)
            
            print(f"🔧 Configuration: {self.advanced_config}")
            
            # Préprocessing avancé
            processed_texts = []
            for text in texts:
                processed = self.preprocess_text_advanced(text)
                if len(processed.strip()) > 0:
                    processed_texts.append(processed)
            
            self.corpus_texts = processed_texts
            
            # Configuration TF-IDF avancée
            ngram_range = (1, 1)
            if self.advanced_config['use_bigrams']:
                ngram_range = (1, 2)
            if self.advanced_config['use_trigrams']:
                ngram_range = (1, 3)
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=self.advanced_config['max_features'],
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.8,
                sublinear_tf=self.advanced_config['sublinear_tf'],
                use_idf=self.advanced_config['use_idf_weighting'],
                norm='l2' if self.advanced_config['use_l2_norm'] else None,
                smooth_idf=True,
                stop_words=None  # Déjà fait dans le preprocessing
            )
            
            # Entraînement
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(processed_texts)
            
            # Statistiques avancées
            vocab_size = len(self.tfidf_vectorizer.vocabulary_)
            sparsity = 1.0 - (self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1]))
            
            print(f"✅ TF-IDF avancé entraîné sur {len(processed_texts)} textes")
            print(f"📊 Vocabulaire : {vocab_size} termes")
            print(f"📈 Sparsité : {sparsity:.3f}")
            
            return {
                'vocabulary_size': vocab_size,
                'corpus_size': len(processed_texts),
                'features': self.tfidf_matrix.shape[1],
                'sparsity': sparsity,
                'config': self.advanced_config
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TF-IDF avancé: {str(e)}")
    
    def semantic_search_enhanced(self, query: str, texts: List[str] = None, top_k: int = 5, 
                               similarity_threshold: float = 0.01) -> List[Dict]:
        """Recherche sémantique améliorée avec clustering et re-ranking"""
        try:
            if not self.tfidf_vectorizer:
                if texts:
                    self.fit_tfidf_enhanced(texts)
                else:
                    raise Exception("Modèle non entraîné")
            
            search_texts = texts if texts else self.corpus_texts
            if not search_texts:
                raise Exception("Aucun texte pour la recherche")
            
            # Préprocessing de la requête
            processed_query = self.preprocess_text_advanced(query)
            
            # Déduplication intelligente
            unique_texts = self._deduplicate_texts(search_texts)
            print(f"🔍 Recherche dans {len(unique_texts)} textes uniques")
            
            # Embeddings
            query_embedding = self.tfidf_vectorizer.transform([processed_query])
            text_embeddings = self.tfidf_vectorizer.transform([
                self.preprocess_text_advanced(text) for text in unique_texts
            ])
            
            # Calcul des similarités
            similarities = cosine_similarity(query_embedding, text_embeddings)[0]
            
            # Création des résultats avec métriques avancées
            results = []
            for i, (text, similarity) in enumerate(zip(unique_texts, similarities)):
                if similarity > similarity_threshold:
                    # Métriques avancées
                    word_overlap = self._calculate_word_overlap(query, text)
                    semantic_score = self._calculate_semantic_score(query, text)
                    
                    results.append({
                        'index': i,
                        'text': text,
                        'similarity': float(similarity),
                        'word_overlap': word_overlap,
                        'semantic_score': semantic_score,
                        'combined_score': float(similarity * 0.7 + semantic_score * 0.3),
                        'text_preview': self._create_smart_preview(text, query),
                        'word_count': len(text.split()),
                        'char_count': len(text),
                        'similarity_category': self._categorize_similarity(similarity)
                    })
            
            # Tri par score combiné
            results.sort(key=lambda x: x['combined_score'], reverse=True)
            
            # Re-ranking par diversité
            final_results = self._diversify_results(results, top_k)
            
            print(f"📊 Résultats: {len(final_results)} sélectionnés avec diversité")
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Erreur recherche avancée: {str(e)}")
    
    def _deduplicate_texts(self, texts: List[str]) -> List[str]:
        """Déduplication avancée basée sur la similarité"""
        unique_texts = []
        seen_fingerprints = set()
        
        for text in texts:
            # Empreinte basée sur les mots principaux
            words = set(self.preprocess_text_advanced(text).split())
            fingerprint = hash(frozenset(list(words)[:10]))  # 10 premiers mots
            
            if fingerprint not in seen_fingerprints and len(text.strip()) > 20:
                unique_texts.append(text)
                seen_fingerprints.add(fingerprint)
        
        return unique_texts
    
    def _calculate_word_overlap(self, query: str, text: str) -> float:
        """Calcule le chevauchement de mots entre requête et texte"""
        query_words = set(self.preprocess_text_advanced(query).split())
        text_words = set(self.preprocess_text_advanced(text).split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
    
    def _calculate_semantic_score(self, query: str, text: str) -> float:
        """Score sémantique basé sur la fréquence des termes"""
        query_words = self.preprocess_text_advanced(query).split()
        text_words = self.preprocess_text_advanced(text).split()
        
        if not query_words or not text_words:
            return 0.0
        
        # TF des mots de la requête dans le texte
        text_word_count = Counter(text_words)
        total_words = len(text_words)
        
        score = 0.0
        for word in query_words:
            tf = text_word_count.get(word, 0) / total_words
            score += tf
        
        return score / len(query_words)
    
    def _create_smart_preview(self, text: str, query: str) -> str:
        """Crée un aperçu intelligent centré sur la requête"""
        sentences = re.split(r'[.!?]+', text)
        query_words = set(self.preprocess_text_advanced(query).split())
        
        # Trouve la phrase avec le plus de mots de la requête
        best_sentence = ""
        max_overlap = 0
        
        for sentence in sentences:
            if len(sentence.strip()) > 20:
                sentence_words = set(self.preprocess_text_advanced(sentence).split())
                overlap = len(query_words.intersection(sentence_words))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence + "..."
        else:
            return text[:150] + "..." if len(text) > 150 else text
    
    def _diversify_results(self, results: List[Dict], top_k: int) -> List[Dict]:
        """Diversifie les résultats pour éviter la redondance"""
        if len(results) <= top_k:
            return results
        
        diversified = [results[0]]  # Prendre le meilleur
        
        for result in results[1:]:
            if len(diversified) >= top_k:
                break
            
            # Vérifier la diversité avec les résultats déjà sélectionnés
            is_diverse = True
            for selected in diversified:
                similarity = self._text_similarity(result['text'], selected['text'])
                if similarity > 0.8:  # Trop similaire
                    is_diverse = False
                    break
            
            if is_diverse:
                diversified.append(result)
        
        return diversified
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calcule la similarité entre deux textes"""
        words1 = set(self.preprocess_text_advanced(text1).split())
        words2 = set(self.preprocess_text_advanced(text2).split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _categorize_similarity(self, similarity: float) -> str:
        """Catégorise le niveau de similarité"""
        if similarity >= 0.8:
            return "très_similaire"
        elif similarity >= 0.5:
            return "assez_similaire"
        elif similarity >= 0.2:
            return "peu_similaire"
        else:
            return "très_différent"
    
    def cluster_texts(self, texts: List[str] = None, n_clusters: int = 5) -> Dict:
        """Clustering des textes pour analyse thématique"""
        try:
            search_texts = texts if texts else self.corpus_texts
            if not search_texts or not self.tfidf_vectorizer:
                raise Exception("Pas de textes ou modèle non entraîné")
            
            # Embeddings des textes
            text_embeddings = self.tfidf_vectorizer.transform([
                self.preprocess_text_advanced(text) for text in search_texts
            ])
            
            # Réduction de dimensionnalité pour le clustering
            if text_embeddings.shape[1] > 100:
                svd = TruncatedSVD(n_components=100, random_state=42)
                text_embeddings = svd.fit_transform(text_embeddings)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(search_texts)), random_state=42)
            cluster_labels = kmeans.fit_predict(text_embeddings)
            
            # Analyse des clusters
            clusters = {}
            for i, (text, label) in enumerate(zip(search_texts, cluster_labels)):
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append({
                    'text': text,
                    'preview': text[:100] + "..." if len(text) > 100 else text
                })
            
            # Mots-clés par cluster
            cluster_keywords = {}
            for label in clusters:
                cluster_texts = [item['text'] for item in clusters[label]]
                cluster_embedding = self.tfidf_vectorizer.transform([
                    self.preprocess_text_advanced(' '.join(cluster_texts))
                ])
                
                # Top mots du cluster
                feature_names = self.tfidf_vectorizer.get_feature_names_out()
                tfidf_scores = cluster_embedding.toarray()[0]
                top_indices = tfidf_scores.argsort()[-10:][::-1]
                
                cluster_keywords[label] = [
                    feature_names[i] for i in top_indices if tfidf_scores[i] > 0
                ]
            
            return {
                'clusters': clusters,
                'cluster_keywords': cluster_keywords,
                'n_clusters': len(clusters),
                'silhouette_score': self._calculate_silhouette_score(text_embeddings, cluster_labels)
            }
            
        except Exception as e:
            raise Exception(f"Erreur clustering: {str(e)}")
    
    def _calculate_silhouette_score(self, embeddings, labels) -> float:
        """Calcule le score de silhouette pour évaluer la qualité du clustering"""
        try:
            from sklearn.metrics import silhouette_score
            if len(set(labels)) > 1:
                return float(silhouette_score(embeddings, labels))
            return 0.0
        except:
            return 0.0
    
    def get_text_statistics(self, text: str) -> Dict:
        """Statistiques avancées d'un texte"""
        processed = self.preprocess_text_advanced(text)
        words = processed.split()
        
        # Statistiques de base
        stats = {
            'original_length': len(text),
            'processed_length': len(processed),
            'word_count': len(words),
            'unique_words': len(set(words)),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'vocabulary_richness': len(set(words)) / len(words) if words else 0
        }
        
        # Analyse TF-IDF si disponible
        if self.tfidf_vectorizer:
            try:
                embedding = self.tfidf_vectorizer.transform([processed])
                stats.update({
                    'tfidf_norm': float(np.linalg.norm(embedding.toarray())),
                    'tfidf_sparsity': float(1.0 - embedding.nnz / embedding.shape[1]),
                    'top_tfidf_terms': self._get_top_tfidf_terms(embedding, 5)
                })
            except:
                pass
        
        return stats
    
    def _get_top_tfidf_terms(self, embedding, top_k: int) -> List[Tuple[str, float]]:
        """Récupère les termes TF-IDF les plus importants"""
        try:
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = embedding.toarray()[0]
            top_indices = tfidf_scores.argsort()[-top_k:][::-1]
            
            return [(feature_names[i], float(tfidf_scores[i])) 
                   for i in top_indices if tfidf_scores[i] > 0]
        except:
            return []
    
    @property
    def is_fitted(self):
        """Vérifie si le modèle est entraîné"""
        return self.tfidf_vectorizer is not None
    
    def get_available_models(self) -> List[Dict]:
        """Retourne les modèles disponibles"""
        return [{
            'id': 'tfidf_enhanced',
            'type': 'tfidf-enhanced',
            'name': 'TF-IDF Avancé',
            'description': 'TF-IDF avec preprocessing avancé, n-grammes et clustering',
            'dimensions': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            'available': self.tfidf_vectorizer is not None,
            'features': ['stemming', 'ngrams', 'clustering', 'diversification']
        }] 