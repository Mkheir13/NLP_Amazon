#!/usr/bin/env python3
"""
Script pour créer les modèles BERT nécessaires au projet.
À exécuter après avoir cloné le repository.
"""

import os
import json
from datetime import datetime
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

def create_model_structure():
    """Crée la structure de base pour les modèles BERT."""
    
    # Créer le dossier models s'il n'existe pas
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    # Créer un modèle de base avec DistilBERT
    model_name = f"bert_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_path = os.path.join(models_dir, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Création du modèle dans {model_path}...")
    
    # Télécharger et sauvegarder le modèle de base
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=2  # Pour classification binaire (positif/négatif)
    )
    
    # Sauvegarder le tokenizer et le modèle
    tokenizer.save_pretrained(model_path)
    model.save_pretrained(model_path)
    
    # Créer un fichier d'info sur le modèle
    model_info = {
        "model_name": model_name,
        "base_model": "distilbert-base-uncased",
        "num_labels": 2,
        "task": "sentiment_analysis",
        "created_at": datetime.now().isoformat(),
        "training_status": "base_model",
        "accuracy": 0.0,
        "dataset": "none"
    }
    
    with open(os.path.join(model_path, "model_info.json"), "w") as f:
        json.dump(model_info, f, indent=2)
    
    print(f"✅ Modèle de base créé : {model_name}")
    print(f"📁 Chemin : {model_path}")
    print("\n🚀 Vous pouvez maintenant lancer l'application et entraîner le modèle via l'interface web !")
    
    return model_path

if __name__ == "__main__":
    print("🤖 Création des modèles BERT de base...")
    print("⏳ Téléchargement du modèle DistilBERT (peut prendre quelques minutes)...")
    
    try:
        model_path = create_model_structure()
        print(f"\n✅ Modèle créé avec succès dans {model_path}")
    except Exception as e:
        print(f"❌ Erreur lors de la création du modèle : {e}")
        print("💡 Assurez-vous d'avoir installé les dépendances : pip install -r requirements.txt") 