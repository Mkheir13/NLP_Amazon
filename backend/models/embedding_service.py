import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import plotly.graph_objects as go
import plotly.express as px
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import pickle
import re

class EmbeddingService:
    def __init__(self):
        self.word2vec_model = None
        self.sentence_transformer = None
        self.embeddings_cache = {}
        self.models_dir = "./models/embeddings"
        os.makedirs(self.models_dir, exist_ok=True)
        
        # Charger les modèles pré-entraînés
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
    
    def train_word2vec(self, texts: List[str], config: Dict = None) -> Dict:
        """Entraîne un modèle Word2Vec sur les textes fournis"""
        try:
            if config is None:
                config = {}
            
            # Préprocessing des textes
            processed_texts = []
            for text in texts:
                # Nettoyage et tokenisation
                text = re.sub(r'[^\w\s]', '', text.lower())
                tokens = text.split()
                if len(tokens) > 0:
                    processed_texts.append(tokens)
            
            if len(processed_texts) == 0:
                raise ValueError("Aucun texte valide pour l'entraînement")
            
            # Configuration par défaut
            default_config = {
                'vector_size': 100,
                'window': 5,
                'min_count': 2,
                'workers': 4,
                'epochs': 10,
                'sg': 1  # Skip-gram
            }
            default_config.update(config)
            
            # Entraînement Word2Vec
            print(f"Entraînement Word2Vec sur {len(processed_texts)} textes...")
            self.word2vec_model = Word2Vec(
                sentences=processed_texts,
                vector_size=default_config['vector_size'],
                window=default_config['window'],
                min_count=default_config['min_count'],
                workers=default_config['workers'],
                epochs=default_config['epochs'],
                sg=default_config['sg']
            )
            
            # Sauvegarder le modèle
            model_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"{self.models_dir}/word2vec_{model_id}.model"
            self.word2vec_model.save(model_path)
            
            # Métadonnées
            model_info = {
                'id': model_id,
                'type': 'word2vec',
                'path': model_path,
                'config': default_config,
                'vocabulary_size': len(self.word2vec_model.wv.key_to_index),
                'trained_on': len(processed_texts),
                'created_at': datetime.now().isoformat()
            }
            
            # Sauvegarder les métadonnées
            with open(f"{self.models_dir}/word2vec_{model_id}_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            return model_info
            
        except Exception as e:
            raise Exception(f"Erreur entraînement Word2Vec: {str(e)}")
    
    def get_word_embedding(self, word: str, model_id: str = None) -> Optional[np.ndarray]:
        """Obtient l'embedding d'un mot"""
        try:
            if model_id:
                # Charger un modèle spécifique
                model_path = f"{self.models_dir}/word2vec_{model_id}.model"
                if os.path.exists(model_path):
                    model = Word2Vec.load(model_path)
                    if word in model.wv:
                        return model.wv[word]
            else:
                # Utiliser le modèle courant
                if self.word2vec_model and word in self.word2vec_model.wv:
                    return self.word2vec_model.wv[word]
            
            return None
        except Exception as e:
            print(f"Erreur récupération embedding: {e}")
            return None
    
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
    
    def find_similar_words(self, word: str, top_k: int = 10, model_id: str = None) -> List[Tuple[str, float]]:
        """Trouve les mots les plus similaires"""
        try:
            model = self.word2vec_model
            
            if model_id:
                model_path = f"{self.models_dir}/word2vec_{model_id}.model"
                if os.path.exists(model_path):
                    model = Word2Vec.load(model_path)
            
            if model is None:
                return []
            
            if word not in model.wv:
                return []
            
            similar_words = model.wv.most_similar(word, topn=top_k)
            return similar_words
            
        except Exception as e:
            print(f"Erreur recherche similarité: {e}")
            return []
    
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
    
    def visualize_embeddings(self, words: List[str], method: str = 'tsne', model_id: str = None) -> Dict:
        """Visualise les embeddings de mots en 2D"""
        try:
            model = self.word2vec_model
            
            if model_id:
                model_path = f"{self.models_dir}/word2vec_{model_id}.model"
                if os.path.exists(model_path):
                    model = Word2Vec.load(model_path)
            
            if model is None:
                raise Exception("Aucun modèle Word2Vec disponible")
            
            # Récupérer les embeddings des mots disponibles
            embeddings = []
            valid_words = []
            
            for word in words:
                if word in model.wv:
                    embeddings.append(model.wv[word])
                    valid_words.append(word)
            
            if len(embeddings) == 0:
                raise Exception("Aucun mot trouvé dans le vocabulaire")
            
            embeddings = np.array(embeddings)
            
            # Réduction de dimensionnalité
            if method == 'pca':
                reducer = PCA(n_components=2, random_state=42)
            elif method == 'tsne':
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
            elif method == 'umap':
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                raise Exception(f"Méthode non supportée: {method}")
            
            embeddings_2d = reducer.fit_transform(embeddings)
            
            # Créer le graphique Plotly
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=embeddings_2d[:, 0],
                y=embeddings_2d[:, 1],
                mode='markers+text',
                text=valid_words,
                textposition='top center',
                marker=dict(
                    size=10,
                    color=np.arange(len(valid_words)),
                    colorscale='Viridis',
                    showscale=True
                ),
                hovertemplate='<b>%{text}</b><br>X: %{x:.2f}<br>Y: %{y:.2f}<extra></extra>'
            ))
            
            fig.update_layout(
                title=f'Visualisation des Embeddings ({method.upper()})',
                xaxis_title=f'{method.upper()} Dimension 1',
                yaxis_title=f'{method.upper()} Dimension 2',
                showlegend=False,
                template='plotly_dark'
            )
            
            return {
                'plot': fig.to_json(),
                'method': method,
                'words_count': len(valid_words),
                'words_found': valid_words,
                'words_not_found': [w for w in words if w not in valid_words]
            }
            
        except Exception as e:
            raise Exception(f"Erreur visualisation: {str(e)}")
    
    def get_embedding_statistics(self, model_id: str = None) -> Dict:
        """Obtient les statistiques d'un modèle d'embedding"""
        try:
            model = self.word2vec_model
            
            if model_id:
                model_path = f"{self.models_dir}/word2vec_{model_id}.model"
                if os.path.exists(model_path):
                    model = Word2Vec.load(model_path)
            
            if model is None:
                return {'error': 'Aucun modèle disponible'}
            
            # Statistiques du vocabulaire
            vocab_size = len(model.wv.key_to_index)
            vector_size = model.wv.vector_size
            
            # Mots les plus fréquents
            word_counts = [(word, model.wv.get_vecattr(word, "count")) 
                          for word in list(model.wv.key_to_index.keys())[:20]]
            word_counts.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'vocabulary_size': vocab_size,
                'vector_size': vector_size,
                'most_frequent_words': word_counts[:10],
                'model_type': 'Word2Vec'
            }
            
        except Exception as e:
            return {'error': str(e)}
    
    def get_available_models(self) -> List[Dict]:
        """Retourne la liste des modèles d'embedding disponibles"""
        models = []
        
        try:
            for filename in os.listdir(self.models_dir):
                if filename.endswith('_info.json'):
                    info_path = os.path.join(self.models_dir, filename)
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                        models.append(model_info)
        except Exception as e:
            print(f"Erreur chargement modèles: {e}")
        
        return models
    
    def analyze_text_semantics(self, text: str) -> Dict:
        """Analyse sémantique complète d'un texte"""
        try:
            # Embedding de la phrase
            sentence_embedding = self.get_sentence_embedding(text)
            
            # Analyse des mots individuels
            words = re.findall(r'\b\w+\b', text.lower())
            word_embeddings = {}
            word_similarities = {}
            
            if self.word2vec_model:
                for word in set(words):
                    if word in self.word2vec_model.wv:
                        word_embeddings[word] = self.word2vec_model.wv[word].tolist()
                        # Similarités avec d'autres mots du texte
                        similarities = []
                        for other_word in set(words):
                            if other_word != word and other_word in self.word2vec_model.wv:
                                sim = self.word2vec_model.wv.similarity(word, other_word)
                                similarities.append((other_word, float(sim)))
                        word_similarities[word] = sorted(similarities, key=lambda x: x[1], reverse=True)[:3]
            
            return {
                'text': text,
                'sentence_embedding_shape': sentence_embedding.shape,
                'sentence_embedding_norm': float(np.linalg.norm(sentence_embedding)),
                'word_count': len(words),
                'unique_words': len(set(words)),
                'words_in_vocabulary': len(word_embeddings),
                'word_similarities': word_similarities,
                'semantic_density': len(word_embeddings) / len(set(words)) if len(set(words)) > 0 else 0
            }
            
        except Exception as e:
            raise Exception(f"Erreur analyse sémantique: {str(e)}") 