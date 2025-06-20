import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import re

class EmbeddingServiceSimple:
    def __init__(self):
        self.sentence_transformer = None
        self.embeddings_cache = {}
        self.models_dir = "./models/embeddings"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Charger le modèle pré-entraîné
        self._load_pretrained_models()
    
    def _load_pretrained_models(self):
        """Charge les modèles pré-entraînés"""
        try:
            # Modèle Sentence-BERT pour les embeddings de phrases
            print("Chargement du modèle Sentence-BERT...")
            self.sentence_transformer = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Sentence-BERT chargé avec succès")
        except Exception as e:
            print(f"⚠️ Erreur chargement Sentence-BERT: {e}")
    
    def get_sentence_embedding(self, text: str) -> np.ndarray:
        """Obtient l'embedding d'une phrase avec Sentence-BERT"""
        try:
            if self.sentence_transformer is None:
                raise Exception("Sentence-BERT non disponible")
            
            # Cache pour éviter les recalculs
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]
            
            embedding = self.sentence_transformer.encode(text)
            self.embeddings_cache[text] = embedding
            
            return embedding
        except Exception as e:
            raise Exception(f"Erreur embedding phrase: {str(e)}")
    
    def semantic_search(self, query: str, texts: List[str], top_k: int = 5) -> List[Dict]:
        """Recherche sémantique dans une collection de textes"""
        try:
            if self.sentence_transformer is None:
                raise Exception("Sentence-BERT non disponible")
            
            # Embedding de la requête
            query_embedding = self.get_sentence_embedding(query)
            
            # Embeddings des textes
            text_embeddings = []
            for text in texts:
                embedding = self.get_sentence_embedding(text)
                text_embeddings.append(embedding)
            
            # Calcul des similarités
            similarities = cosine_similarity([query_embedding], text_embeddings)[0]
            
            # Tri par similarité décroissante
            results = []
            for i, (text, similarity) in enumerate(zip(texts, similarities)):
                results.append({
                    'index': i,
                    'text': text,
                    'similarity': float(similarity),
                    'text_preview': text[:200] + "..." if len(text) > 200 else text
                })
            
            # Retourner les top_k résultats
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            raise Exception(f"Erreur recherche sémantique: {str(e)}")
    
    def find_similar_texts(self, reference_text: str, texts: List[str], top_k: int = 10) -> List[Dict]:
        """Trouve les textes les plus similaires à un texte de référence"""
        try:
            if self.sentence_transformer is None:
                raise Exception("Sentence-BERT non disponible")
            
            # Embedding du texte de référence
            ref_embedding = self.get_sentence_embedding(reference_text)
            
            # Embeddings des textes
            text_embeddings = []
            for text in texts:
                if text != reference_text:  # Exclure le texte de référence
                    embedding = self.get_sentence_embedding(text)
                    text_embeddings.append((text, embedding))
            
            # Calcul des similarités
            similarities = []
            for text, embedding in text_embeddings:
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
    
    def visualize_text_embeddings(self, texts: List[str], labels: List[str] = None, method: str = 'pca') -> Dict:
        """Visualise les embeddings de textes en 2D"""
        try:
            if self.sentence_transformer is None:
                raise Exception("Sentence-BERT non disponible")
            
            if len(texts) < 2:
                raise Exception("Au moins 2 textes sont nécessaires pour la visualisation")
            
            # Récupérer les embeddings des textes
            embeddings = []
            valid_texts = []
            
            for i, text in enumerate(texts):
                try:
                    embedding = self.get_sentence_embedding(text)
                    embeddings.append(embedding)
                    valid_texts.append(text[:50] + "..." if len(text) > 50 else text)
                except Exception as e:
                    print(f"Erreur embedding texte {i}: {e}")
                    continue
            
            if len(embeddings) < 2:
                raise Exception("Pas assez d'embeddings valides pour la visualisation")
            
            embeddings = np.array(embeddings)
            
            # Réduction de dimensionnalité
            if method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'tsne':
                perplexity = min(30, len(embeddings) - 1)
                reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            else:
                raise Exception(f"Méthode non supportée: {method}")
            
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Utiliser les labels fournis ou créer des labels par défaut
            if labels is None:
                labels = [f"Texte {i+1}" for i in range(len(valid_texts))]
            
            # Créer le graphique Plotly
            fig = go.Figure()
            
            # Couleurs pour différents labels
            unique_labels = list(set(labels))
            colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
            color_map = {label: colors[i % len(colors)] for i, label in enumerate(unique_labels)}
            
            for label in unique_labels:
                indices = [i for i, l in enumerate(labels) if l == label]
                fig.add_trace(go.Scatter(
                    x=[embeddings_2d[i, 0] for i in indices],
                    y=[embeddings_2d[i, 1] for i in indices],
                    mode='markers+text',
                    text=[valid_texts[i] for i in indices],
                    textposition='top center',
                    name=label,
                    marker=dict(
                        size=10,
                        color=color_map[label],
                        opacity=0.7
                    ),
                    hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
                ))
            
            fig.update_layout(
                title=f'Visualisation des Embeddings de Textes ({method.upper()})',
                xaxis_title=f'{method.upper()} Dimension 1',
                yaxis_title=f'{method.upper()} Dimension 2',
                template='plotly_dark',
                showlegend=True
            )
            
            return {
                'plot': fig.to_json(),
                'method': method,
                'texts_count': len(valid_texts),
                'texts_processed': valid_texts
            }
            
        except Exception as e:
            raise Exception(f"Erreur visualisation: {str(e)}")
    
    def analyze_text_semantics(self, text: str) -> Dict:
        """Analyse sémantique complète d'un texte"""
        try:
            # Embedding de la phrase
            sentence_embedding = self.get_sentence_embedding(text)
            
            # Analyse des mots individuels
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = list(set(words))
            
            # Calculer l'embedding de chaque mot individuellement
            word_embeddings = {}
            for word in unique_words:
                try:
                    word_embedding = self.get_sentence_embedding(word)
                    word_embeddings[word] = word_embedding.tolist()
                except Exception:
                    continue
            
            return {
                'text': text,
                'sentence_embedding_shape': sentence_embedding.shape,
                'sentence_embedding_norm': float(np.linalg.norm(sentence_embedding)),
                'word_count': len(words),
                'unique_words': len(unique_words),
                'words_with_embeddings': len(word_embeddings),
                'semantic_density': len(word_embeddings) / len(unique_words) if len(unique_words) > 0 else 0
            }
            
        except Exception as e:
            raise Exception(f"Erreur analyse sémantique: {str(e)}")
    
    def compare_texts_similarity(self, text1: str, text2: str) -> Dict:
        """Compare la similarité entre deux textes"""
        try:
            embedding1 = self.get_sentence_embedding(text1)
            embedding2 = self.get_sentence_embedding(text2)
            
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            
            return {
                'text1': text1,
                'text2': text2,
                'similarity': float(similarity),
                'similarity_percentage': float(similarity * 100),
                'interpretation': self._interpret_similarity(similarity)
            }
            
        except Exception as e:
            raise Exception(f"Erreur comparaison textes: {str(e)}")
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interprète le score de similarité"""
        if similarity >= 0.9:
            return "Très similaire"
        elif similarity >= 0.7:
            return "Similaire"
        elif similarity >= 0.5:
            return "Moyennement similaire"
        elif similarity >= 0.3:
            return "Peu similaire"
        else:
            return "Très différent"
    
    def get_available_models(self) -> List[Dict]:
        """Retourne les modèles disponibles (pour l'instant juste Sentence-BERT)"""
        return [{
            'id': 'sentence-bert',
            'type': 'sentence-transformer',
            'name': 'all-MiniLM-L6-v2',
            'description': 'Modèle Sentence-BERT pré-entraîné',
            'dimensions': 384,
            'available': self.sentence_transformer is not None
        }]
    
    def is_service_available(self) -> bool:
        """Vérifie si le service est disponible"""
        return self.sentence_transformer is not None 