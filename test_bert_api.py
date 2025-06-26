#!/usr/bin/env python3
"""
Script de test pour vérifier les endpoints BERT
Aide au diagnostic des erreurs 404
"""

import requests
import json
import sys

# Configuration
BASE_URL = "http://localhost:5000/api"

def test_health():
    """Test de l'endpoint health"""
    print("🔍 Test de l'endpoint health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health OK: {data}")
            return True
        else:
            print(f"❌ Health failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur health: {e}")
        return False

def test_models_list():
    """Test de l'endpoint models"""
    print("\n🔍 Test de l'endpoint models...")
    try:
        response = requests.get(f"{BASE_URL}/models")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Models OK: {len(data.get('models', []))} modèles trouvés")
            for model in data.get('models', []):
                print(f"   - {model.get('id', 'N/A')}: {model.get('name', 'N/A')}")
            return data.get('models', [])
        else:
            print(f"❌ Models failed: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Erreur models: {e}")
        return []

def test_nltk_analysis():
    """Test de l'endpoint NLTK"""
    print("\n🔍 Test de l'endpoint NLTK...")
    try:
        test_text = "This is a great product! I love it."
        response = requests.post(f"{BASE_URL}/analyze/nltk", 
                               json={"text": test_text})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ NLTK OK: {data}")
            return True
        else:
            print(f"❌ NLTK failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur NLTK: {e}")
        return False

def test_bert_prediction(model_id):
    """Test de l'endpoint BERT prediction"""
    print(f"\n🔍 Test de la prédiction BERT avec modèle {model_id}...")
    try:
        test_text = "This is a great product! I love it."
        response = requests.post(f"{BASE_URL}/predict/bert/{model_id}", 
                               json={"text": test_text})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ BERT Prediction OK: {data}")
            return True
        elif response.status_code == 404:
            print(f"❌ BERT Model not found (404): {model_id}")
            print(f"   Response: {response.text}")
            return False
        else:
            print(f"❌ BERT Prediction failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Erreur BERT Prediction: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🚀 Test des endpoints BERT API")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health():
        print("\n❌ Backend non disponible - Arrêt des tests")
        sys.exit(1)
    
    # Test 2: Liste des modèles
    models = test_models_list()
    
    # Test 3: NLTK (doit toujours fonctionner)
    test_nltk_analysis()
    
    # Test 4: BERT (seulement si des modèles existent)
    if models:
        print(f"\n🎯 Test des modèles BERT disponibles:")
        for model in models:
            model_id = model.get('id')
            if model_id:
                test_bert_prediction(model_id)
    else:
        print("\n⚠️ Aucun modèle BERT disponible pour les tests")
        print("   Entraînez un modèle BERT via l'interface web d'abord")
    
    print("\n" + "=" * 50)
    print("🏁 Tests terminés")
    
    # Résumé
    if not models:
        print("\n💡 Conseil: Pour tester BERT complètement:")
        print("   1. Ouvrez l'interface web")
        print("   2. Allez dans 'BERT Training'")
        print("   3. Entraînez un modèle")
        print("   4. Relancez ce script")

if __name__ == "__main__":
    main() 