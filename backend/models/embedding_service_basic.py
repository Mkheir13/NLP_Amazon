import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import re

class EmbeddingServiceBasic:
    def __init__(self):
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.corpus_texts = []
        self.embeddings_cache = {}
        self.models_dir = "./models/embeddings"
        os.makedirs(self.models_dir, exist_ok=True)
        
        print("✅ Service d'embedding basique initialisé avec TF-IDF")
    
    @property
    def is_fitted(self):
        """Vérifie si le modèle TF-IDF est entraîné"""
        return self.tfidf_vectorizer is not None
    
    def fit_tfidf(self, texts: List[str]):
        """Entraîne le vectoriseur TF-IDF sur un corpus de textes"""
        try:
            self.corpus_texts = texts
            
            # Configuration TF-IDF
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.8
            )
            
            # Entraînement
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            
            print(f"✅ TF-IDF entraîné sur {len(texts)} textes")
            print(f"📊 Vocabulaire : {len(self.tfidf_vectorizer.vocabulary_)} termes")
            
            return {
                'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
                'corpus_size': len(texts),
                'features': self.tfidf_matrix.shape[1]
            }
            
        except Exception as e:
            raise Exception(f"Erreur entraînement TF-IDF: {str(e)}")
    
    def get_text_embedding(self, text: str) -> np.ndarray:
        """Obtient l'embedding TF-IDF d'un texte"""
        try:
            if self.tfidf_vectorizer is None:
                raise Exception("TF-IDF non entraîné")
            
            # Cache pour éviter les recalculs
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]
            
            # Transformer le texte
            embedding = self.tfidf_vectorizer.transform([text]).toarray()[0]
            self.embeddings_cache[text] = embedding
            
            return embedding
        except Exception as e:
            raise Exception(f"Erreur embedding texte: {str(e)}")
    
    def semantic_search(self, query: str, texts: List[str] = None, top_k: int = 5) -> List[Dict]:
        """Recherche sémantique dans une collection de textes"""
        try:
            if self.tfidf_vectorizer is None:
                # Auto-entraînement si pas encore fait
                if texts:
                    self.fit_tfidf(texts)
                else:
                    raise Exception("TF-IDF non entraîné et aucun texte fourni")
            
            # Utiliser le corpus existant ou les textes fournis
            search_texts = texts if texts else self.corpus_texts
            
            if not search_texts:
                raise Exception("Aucun texte disponible pour la recherche")
            
            # Supprimer les doublons tout en gardant l'ordre
            unique_texts = []
            seen_texts = set()
            for text in search_texts:
                # Normaliser le texte pour la comparaison (enlever espaces, ponctuation)
                normalized = re.sub(r'[^\w\s]', '', text.lower().strip())
                if normalized not in seen_texts and len(normalized) > 10:  # Éviter les textes trop courts
                    unique_texts.append(text)
                    seen_texts.add(normalized)
            
            if not unique_texts:
                raise Exception("Aucun texte unique trouvé après déduplication")
            
            print(f"🔍 Recherche dans {len(unique_texts)} textes uniques (était {len(search_texts)})")
            
            # Embedding de la requête
            query_embedding = self.get_text_embedding(query)
            
            # Embeddings des textes uniques
            text_embeddings = []
            for text in unique_texts:
                embedding = self.get_text_embedding(text)
                text_embeddings.append(embedding)
            text_embeddings = np.array(text_embeddings)
            
            # Calcul des similarités
            similarities = cosine_similarity([query_embedding], text_embeddings)[0]
            
            # Créer les résultats avec plus d'informations
            results = []
            for i, (text, similarity) in enumerate(zip(unique_texts, similarities)):
                # Calculer des métriques supplémentaires
                word_count = len(text.split())
                char_count = len(text)
                
                # Créer un preview intelligent (première phrase ou 150 caractères)
                sentences = re.split(r'[.!?]+', text)
                if len(sentences) > 1 and len(sentences[0]) > 20:
                    preview = sentences[0].strip() + "..."
                else:
                    preview = text[:150] + "..." if len(text) > 150 else text
                
                results.append({
                    'index': i,
                    'text': text,
                    'similarity': float(similarity),
                    'text_preview': preview,
                    'word_count': word_count,
                    'char_count': char_count,
                    'similarity_category': self._categorize_similarity(similarity)
                })
            
            # Tri par similarité décroissante
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # Filtrer les résultats avec similarité très faible (< 0.01)
            filtered_results = [r for r in results if r['similarity'] > 0.01]
            
            # Retourner les top_k résultats
            final_results = filtered_results[:top_k]
            
            print(f"📊 Résultats: {len(final_results)} sur {len(filtered_results)} (seuil > 1%)")
            
            return final_results
            
        except Exception as e:
            raise Exception(f"Erreur recherche sémantique: {str(e)}")
    
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
    
    def find_similar_texts(self, reference_text: str, texts: List[str], top_k: int = 10) -> List[Dict]:
        """Trouve les textes les plus similaires à un texte de référence"""
        try:
            # S'assurer que le TF-IDF est entraîné
            if self.tfidf_vectorizer is None:
                all_texts = [reference_text] + texts
                self.fit_tfidf(all_texts)
            
            # Embedding du texte de référence
            ref_embedding = self.get_text_embedding(reference_text)
            
            # Embeddings des textes
            similarities = []
            for text in texts:
                if text != reference_text:  # Exclure le texte de référence
                    embedding = self.get_text_embedding(text)
                    similarity = cosine_similarity([ref_embedding], [embedding])[0][0]
                    similarities.append({
                        'text': text,
                        'similarity': float(similarity),
                        'text_preview': text[:150] + "..." if len(text) > 150 else text
                    })
            
            # Tri par similarité décroissante
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:top_k]
            
        except Exception as e:
            raise Exception(f"Erreur recherche similarité: {str(e)}")
    
    def visualize_text_embeddings(self, texts, labels=None, method='pca'):
        """Visualise les embeddings de textes en 2D"""
        if not self.is_fitted:
            raise ValueError("Le modèle TF-IDF doit être entraîné d'abord")
        
        if not texts:
            raise ValueError("Aucun texte fourni pour la visualisation")
        
        try:
            # Pour la visualisation de mots individuels, on les traite comme des textes
            # Si ce sont des mots seuls, on les garde tels quels
            processed_texts = []
            for text in texts:
                if isinstance(text, str):
                    # Si c'est un mot seul (sans espaces), on le garde tel quel
                    # Sinon on le traite comme un texte complet
                    processed_texts.append(text.strip())
                else:
                    processed_texts.append(str(text))
            
            # Générer les embeddings TF-IDF
            embeddings = self.tfidf_vectorizer.transform(processed_texts)
            embeddings_dense = embeddings.toarray()
            
            # Vérifier si on a des embeddings valides
            if embeddings_dense.shape[0] == 0:
                raise ValueError("Aucun embedding généré")
            
            # Réduction de dimensionnalité
            if method.lower() == 'pca':
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
            elif method.lower() == 'tsne':
                from sklearn.manifold import TSNE
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(5, len(processed_texts)-1))
            else:  # umap ou autre
                from sklearn.decomposition import PCA
                reducer = PCA(n_components=2, random_state=42)
            
            # Appliquer la réduction
            embeddings_2d = reducer.fit_transform(embeddings_dense)
            
            # Préparer les données pour Plotly
            plot_data = {
                'data': [{
                    'x': embeddings_2d[:, 0].tolist(),
                    'y': embeddings_2d[:, 1].tolist(),
                    'text': processed_texts,
                    'mode': 'markers+text',
                    'type': 'scatter',
                    'marker': {
                        'size': 12,
                        'color': labels if labels else 'rgba(55, 128, 191, 0.8)',
                        'line': {'width': 2, 'color': 'white'}
                    },
                    'textposition': 'top center',
                    'textfont': {'size': 12, 'color': 'white'}
                }],
                'layout': {
                    'title': {
                        'text': f'Visualisation des Embeddings ({method.upper()})',
                        'font': {'size': 18, 'color': 'white'}
                    },
                    'xaxis': {
                        'title': f'{method.upper()} Dimension 1',
                        'color': 'white',
                        'gridcolor': 'rgba(255,255,255,0.2)'
                    },
                    'yaxis': {
                        'title': f'{method.upper()} Dimension 2',
                        'color': 'white',
                        'gridcolor': 'rgba(255,255,255,0.2)'
                    },
                    'plot_bgcolor': 'rgba(0,0,0,0)',
                    'paper_bgcolor': 'rgba(0,0,0,0)',
                    'font': {'color': 'white'},
                    'showlegend': False
                }
            }
            
            # Identifier les mots non trouvés (ceux avec des embeddings vides)
            words_found = []
            words_not_found = []
            
            for i, text in enumerate(processed_texts):
                if embeddings_dense[i].sum() > 0:  # Si l'embedding n'est pas vide
                    words_found.append(text)
                else:
                    words_not_found.append(text)
            
            return {
                'plot': json.dumps(plot_data),
                'method': method,
                'words_count': len(words_found),
                'words_not_found': words_not_found,
                'coordinates': embeddings_2d.tolist(),
                'words': processed_texts
            }
            
        except Exception as e:
            print(f"❌ Erreur visualisation: {str(e)}")
            raise ValueError(f"Erreur lors de la visualisation: {str(e)}")
    
    def analyze_text_semantics(self, text: str) -> Dict:
        """Analyse sémantique complète d'un texte"""
        try:
            # S'assurer que le TF-IDF est entraîné
            if self.tfidf_vectorizer is None:
                # Entraîner sur un corpus minimal
                sample_texts = [text, "This is a sample text", "Another example text"]
                self.fit_tfidf(sample_texts)
            
            # Embedding du texte
            embedding = self.get_text_embedding(text)
            
            # Analyse des mots
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = list(set(words))
            
            # Mots les plus importants selon TF-IDF
            feature_names = self.tfidf_vectorizer.get_feature_names_out()
            tfidf_scores = dict(zip(feature_names, embedding))
            top_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                'text': text,
                'embedding_shape': embedding.shape,
                'embedding_norm': float(np.linalg.norm(embedding)),
                'word_count': len(words),
                'unique_words': len(unique_words),
                'tfidf_features': len(feature_names),
                'top_terms': [(term, float(score)) for term, score in top_terms if score > 0],
                'sparsity': float(np.sum(embedding == 0) / len(embedding))
            }
            
        except Exception as e:
            raise Exception(f"Erreur analyse sémantique: {str(e)}")
    
    def compare_texts_similarity(self, text1: str, text2: str) -> Dict:
        """Compare la similarité entre deux textes"""
        try:
            # S'assurer que le TF-IDF est entraîné
            if self.tfidf_vectorizer is None:
                self.fit_tfidf([text1, text2])
            
            embedding1 = self.get_text_embedding(text1)
            embedding2 = self.get_text_embedding(text2)
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            return {
                'text1': text1,
                'text2': text2,
                'similarity': float(similarity),
                'similarity_percentage': float(similarity * 100),
                'interpretation': self._interpret_similarity(similarity),
                'method': 'TF-IDF + Cosine Similarity'
            }
            
        except Exception as e:
            raise Exception(f"Erreur comparaison textes: {str(e)}")
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interprète le score de similarité"""
        if similarity >= 0.8:
            return "Très similaire"
        elif similarity >= 0.6:
            return "Similaire"
        elif similarity >= 0.4:
            return "Moyennement similaire"
        elif similarity >= 0.2:
            return "Peu similaire"
        else:
            return "Très différent"
    
    def get_available_models(self) -> List[Dict]:
        """Retourne les modèles disponibles"""
        return [{
            'id': 'tfidf',
            'type': 'tfidf-vectorizer',
            'name': 'TF-IDF Vectorizer',
            'description': 'Vectorisation TF-IDF avec scikit-learn',
            'dimensions': len(self.tfidf_vectorizer.vocabulary_) if self.tfidf_vectorizer else 0,
            'available': self.tfidf_vectorizer is not None
        }]
    
    def is_service_available(self) -> bool:
        """Vérifie si le service est disponible"""
        return True  # TF-IDF est toujours disponible avec scikit-learn
    
    def get_vocabulary_stats(self) -> Dict:
        """Retourne les statistiques du vocabulaire"""
        if self.tfidf_vectorizer is None:
            return {'error': 'TF-IDF non entraîné'}
        
        return {
            'vocabulary_size': len(self.tfidf_vectorizer.vocabulary_),
            'corpus_size': len(self.corpus_texts),
            'features': self.tfidf_matrix.shape[1] if self.tfidf_matrix is not None else 0,
            'sparsity': float(self.tfidf_matrix.nnz / (self.tfidf_matrix.shape[0] * self.tfidf_matrix.shape[1])) if self.tfidf_matrix is not None else 0
        } 