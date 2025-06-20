from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
from datetime import datetime
from models.embedding_service_basic import EmbeddingServiceBasic

# Télécharger les ressources NLTK nécessaires
nltk.download('vader_lexicon', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)
CORS(app)

class BERTTrainer:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.trained_models = []
        
    def prepare_data(self, reviews_data):
        """Prépare les données pour l'entraînement BERT"""
        df = pd.DataFrame(reviews_data)
        
        # Convertir les labels en format numérique si nécessaire
        if 'sentiment' in df.columns:
            label_map = {'negative': 0, 'positive': 1}
            df['labels'] = df['sentiment'].map(label_map)
        elif 'label' in df.columns:
            df['labels'] = df['label']
        else:
            # Assumer que les labels sont déjà numériques
            df['labels'] = [1 if 'good' in text.lower() or 'great' in text.lower() else 0 for text in df['text']]
        
        return df
    
    def tokenize_data(self, texts, labels, tokenizer, max_length=512):
        """Tokenise les données pour BERT"""
        encodings = tokenizer(
            texts.tolist(),
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        dataset = Dataset.from_dict({
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels.tolist()
        })
        
        return dataset
    
    def train_bert_model(self, data, config):
        """Entraîne un modèle BERT"""
        try:
            # Préparer les données
            df = self.prepare_data(data)
            
            # Diviser les données
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                df['text'], df['labels'], 
                test_size=config.get('test_size', 0.2), 
                random_state=42
            )
            
            # Charger le tokenizer et le modèle
            model_name = config.get('model_name', 'distilbert-base-uncased')
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name, 
                num_labels=2
            )
            
            # Tokeniser les données
            train_dataset = self.tokenize_data(train_texts, train_labels, self.tokenizer)
            val_dataset = self.tokenize_data(val_texts, val_labels, self.tokenizer)
            
            # Configuration d'entraînement
            training_args = TrainingArguments(
                output_dir=f'./models/bert_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                num_train_epochs=config.get('epochs', 3),
                per_device_train_batch_size=config.get('batch_size', 8),
                per_device_eval_batch_size=config.get('batch_size', 8),
                warmup_steps=500,
                weight_decay=0.01,
                logging_dir='./logs',
                logging_steps=10,
                eval_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                learning_rate=config.get('learning_rate', 2e-5)
            )
            
            # Créer le trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                tokenizer=self.tokenizer
            )
            
            # Entraîner le modèle
            trainer.train()
            
            # Évaluer le modèle
            eval_results = trainer.evaluate()
            
            # Prédictions sur le set de validation
            predictions = trainer.predict(val_dataset)
            pred_labels = np.argmax(predictions.predictions, axis=1)
            
            # Calculer les métriques
            accuracy = accuracy_score(val_labels, pred_labels)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, pred_labels, average='weighted'
            )
            
            # Sauvegarder le modèle
            model_info = {
                'id': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'name': f"BERT_{config.get('model_name', 'distilbert')}",
                'type': 'bert',
                'model_name': model_name,
                'config': config,
                'metrics': {
                    'accuracy': float(accuracy),
                    'precision': float(precision),
                    'recall': float(recall),
                    'f1_score': float(f1),
                    'eval_loss': float(eval_results.get('eval_loss', 0))
                },
                'trained_on': len(train_texts),
                'created_at': datetime.now().isoformat()
            }
            
            # Sauvegarder le modèle et le tokenizer
            model_dir = training_args.output_dir
            trainer.save_model(model_dir)
            self.tokenizer.save_pretrained(model_dir)
            
            # Sauvegarder les métadonnées
            with open(f"{model_dir}/model_info.json", 'w') as f:
                json.dump(model_info, f, indent=2)
            
            self.trained_models.append(model_info)
            
            return model_info
            
        except Exception as e:
            raise Exception(f"Erreur lors de l'entraînement BERT: {str(e)}")

class NLTKAnalyzer:
    def __init__(self):
        self.sia = SentimentIntensityAnalyzer()
        
    def analyze_sentiment(self, text):
        """Analyse le sentiment avec NLTK VADER"""
        scores = self.sia.polarity_scores(text)
        
        # Déterminer le sentiment principal
        if scores['compound'] >= 0.05:
            sentiment = 'positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
            
        return {
            'sentiment': sentiment,
            'confidence': abs(scores['compound']),
            'scores': scores,
            'polarity': scores['compound']
        }
    
    def batch_analyze(self, texts):
        """Analyse un lot de textes"""
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results

# Instances globales
bert_trainer = BERTTrainer()
nltk_analyzer = NLTKAnalyzer()
embedding_service = EmbeddingServiceBasic()

@app.route('/api/train/bert', methods=['POST'])
def train_bert():
    """Endpoint pour entraîner un modèle BERT"""
    try:
        data = request.json
        reviews_data = data.get('data', [])
        config = data.get('config', {})
        
        if not reviews_data:
            return jsonify({'error': 'Aucune donnée fournie'}), 400
        
        # Entraîner le modèle
        model_info = bert_trainer.train_bert_model(reviews_data, config)
        
        return jsonify({
            'success': True,
            'model': model_info,
            'message': 'Modèle BERT entraîné avec succès'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/nltk', methods=['POST'])
def analyze_nltk():
    """Endpoint pour analyser avec NLTK"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        result = nltk_analyzer.analyze_sentiment(text)
        
        return jsonify({
            'success': True,
            'result': result
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze/nltk/batch', methods=['POST'])
def analyze_nltk_batch():
    """Endpoint pour analyser plusieurs textes avec NLTK"""
    try:
        data = request.json
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        results = nltk_analyzer.batch_analyze(texts)
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Récupère la liste des modèles entraînés"""
    return jsonify({
        'success': True,
        'models': bert_trainer.trained_models
    })

@app.route('/api/predict/bert/<model_id>', methods=['POST'])
def predict_bert(model_id):
    """Prédiction avec un modèle BERT entraîné"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        # Charger le modèle spécifique
        model_dir = f'./models/bert_{model_id}'
        
        if not os.path.exists(model_dir):
            return jsonify({'error': 'Modèle non trouvé'}), 404
        
        # Charger le modèle et le tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        
        # Tokeniser et prédire
        inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=-1).item()
            confidence = torch.max(predictions).item()
        
        sentiment = 'positive' if predicted_class == 1 else 'negative'
        
        return jsonify({
            'success': True,
            'prediction': {
                'sentiment': sentiment,
                'confidence': float(confidence),
                'class': int(predicted_class)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état du serveur"""
    return jsonify({
        'status': 'healthy',
        'message': 'Backend NLP opérationnel',
        'features': ['BERT Training', 'NLTK Analysis', 'Word Embeddings', 'Semantic Search']
    })

# ========== ENDPOINTS EMBEDDINGS (VERSION BASIQUE TF-IDF) ==========

@app.route('/api/embeddings/train/tfidf', methods=['POST'])
def train_tfidf():
    """Entraîne le vectoriseur TF-IDF"""
    try:
        data = request.json
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'error': 'Aucun texte fourni pour l\'entraînement'}), 400
        
        stats = embedding_service.fit_tfidf(texts)
        
        return jsonify({
            'success': True,
            'stats': stats,
            'message': 'TF-IDF entraîné avec succès'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/text', methods=['POST'])
def get_text_embedding():
    """Obtient l'embedding TF-IDF d'un texte"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        embedding = embedding_service.get_text_embedding(text)
        
        return jsonify({
            'success': True,
            'text': text,
            'embedding': embedding.tolist(),
            'dimension': len(embedding)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/similar', methods=['POST'])
def find_similar_texts():
    """Trouve les textes similaires"""
    try:
        data = request.json
        reference_text = data.get('reference_text', '')
        texts = data.get('texts', [])
        top_k = data.get('top_k', 10)
        
        if not reference_text:
            return jsonify({'error': 'Aucun texte de référence fourni'}), 400
        
        if not texts:
            return jsonify({'error': 'Aucun texte fourni pour la comparaison'}), 400
        
        similar_texts = embedding_service.find_similar_texts(reference_text, texts, top_k)
        
        return jsonify({
            'success': True,
            'reference_text': reference_text,
            'similar_texts': similar_texts,
            'count': len(similar_texts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/search', methods=['POST'])
def semantic_search():
    """Recherche sémantique"""
    try:
        data = request.json
        query = data.get('query', '')
        texts = data.get('texts', [])
        top_k = data.get('top_k', 5)
        
        if not query:
            return jsonify({'error': 'Aucune requête fournie'}), 400
        
        if not texts:
            return jsonify({'error': 'Aucun texte fourni pour la recherche'}), 400
        
        results = embedding_service.semantic_search(query, texts, top_k)
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'total_searched': len(texts)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/visualize', methods=['POST'])
def visualize_embeddings():
    """Visualise les embeddings de textes"""
    try:
        data = request.json
        texts = data.get('texts', [])
        labels = data.get('labels', None)
        method = data.get('method', 'pca')
        
        if not texts:
            return jsonify({'error': 'Aucun texte fourni pour la visualisation'}), 400
        
        visualization = embedding_service.visualize_text_embeddings(texts, labels, method)
        
        return jsonify({
            'success': True,
            'visualization': visualization
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/compare', methods=['POST'])
def compare_texts():
    """Compare la similarité entre deux textes"""
    try:
        data = request.json
        text1 = data.get('text1', '')
        text2 = data.get('text2', '')
        
        if not text1 or not text2:
            return jsonify({'error': 'Deux textes sont nécessaires pour la comparaison'}), 400
        
        comparison = embedding_service.compare_texts_similarity(text1, text2)
        
        return jsonify({
            'success': True,
            'comparison': comparison
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/analyze', methods=['POST'])
def analyze_text_semantics():
    """Analyse sémantique complète d'un texte"""
    try:
        data = request.json
        text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'Aucun texte fourni'}), 400
        
        analysis = embedding_service.analyze_text_semantics(text)
        
        return jsonify({
            'success': True,
            'analysis': analysis
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/models', methods=['GET'])
def get_embedding_models():
    """Obtient la liste des modèles d'embedding"""
    try:
        models = embedding_service.get_available_models()
        
        return jsonify({
            'success': True,
            'models': models,
            'count': len(models)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/embeddings/status', methods=['GET'])
def get_embedding_status():
    """Vérifie le statut du service d'embedding"""
    try:
        available = embedding_service.is_service_available()
        
        return jsonify({
            'success': True,
            'available': available,
            'service': 'TF-IDF',
            'message': 'Service TF-IDF disponible' if available else 'Service non disponible'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Créer les dossiers nécessaires
    os.makedirs('./models', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 