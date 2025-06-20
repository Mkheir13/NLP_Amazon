#!/usr/bin/env python3
"""
Script de test pour les endpoints d'embeddings
"""

import requests
import json

BASE_URL = "http://localhost:5000/api/embeddings"

def test_status():
    """Test du statut du service"""
    print("🔍 Test du statut du service...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_train_tfidf():
    """Test de l'entraînement TF-IDF"""
    print("\n🚀 Test de l'entraînement TF-IDF...")
    
    texts = [
        "Ce produit est excellent et je le recommande vivement",
        "La qualité est décevante, je ne suis pas satisfait",
        "Service client parfait, très professionnel",
        "Livraison rapide et emballage soigné",
        "Prix trop élevé pour la qualité proposée",
        "Très bon rapport qualité-prix, je recommande",
        "Produit défaillant, j'ai dû le retourner",
        "Parfait pour mes besoins, très content de mon achat"
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/train/tfidf",
            headers={"Content-Type": "application/json"},
            json={"texts": texts}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_text_embedding():
    """Test de génération d'embedding"""
    print("\n📊 Test de génération d'embedding...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/text",
            headers={"Content-Type": "application/json"},
            json={"text": "Ce produit est vraiment excellent"}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        if result.get('success'):
            print(f"Dimension: {result['dimension']}")
            print(f"Embedding (premiers 10 éléments): {result['embedding'][:10]}")
        else:
            print(f"Erreur: {result.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_semantic_search():
    """Test de recherche sémantique"""
    print("\n🔍 Test de recherche sémantique...")
    
    texts = [
        "Excellent produit, très satisfait",
        "Qualité médiocre, je déconseille",
        "Bon rapport qualité-prix",
        "Service client décevant",
        "Livraison très rapide"
    ]
    
    try:
        response = requests.post(
            f"{BASE_URL}/search",
            headers={"Content-Type": "application/json"},
            json={
                "query": "produit de bonne qualité",
                "texts": texts,
                "top_k": 3
            }
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        if result.get('success'):
            print("Résultats de recherche:")
            for i, res in enumerate(result['results'][:3]):
                print(f"  {i+1}. Similarité: {res['similarity']:.3f} - {res['text']}")
        else:
            print(f"Erreur: {result.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_compare_texts():
    """Test de comparaison de textes"""
    print("\n⚖️ Test de comparaison de textes...")
    
    try:
        response = requests.post(
            f"{BASE_URL}/compare",
            headers={"Content-Type": "application/json"},
            json={
                "text1": "Ce produit est excellent",
                "text2": "Cette article est fantastique"
            }
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        if result.get('success'):
            comp = result['comparison']
            print(f"Similarité: {comp['similarity']:.3f} ({comp['similarity_percentage']:.1f}%)")
            print(f"Interprétation: {comp['interpretation']}")
        else:
            print(f"Erreur: {result.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_models():
    """Test de récupération des modèles"""
    print("\n📋 Test de récupération des modèles...")
    
    try:
        response = requests.get(f"{BASE_URL}/models")
        print(f"Status: {response.status_code}")
        result = response.json()
        if result.get('success'):
            print(f"Modèles disponibles: {result['count']}")
            for model in result['models']:
                print(f"  - {model['name']} ({model['type']}) - Disponible: {model['available']}")
        else:
            print(f"Erreur: {result.get('error')}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def main():
    """Fonction principale de test"""
    print("🧪 Tests des Endpoints d'Embeddings")
    print("=" * 50)
    
    tests = [
        ("Status", test_status),
        ("Entraînement TF-IDF", test_train_tfidf),
        ("Embedding de texte", test_text_embedding),
        ("Recherche sémantique", test_semantic_search),
        ("Comparaison de textes", test_compare_texts),
        ("Modèles disponibles", test_models)
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
            print(f"{'✅' if success else '❌'} {name}: {'OK' if success else 'ÉCHEC'}")
        except Exception as e:
            print(f"❌ {name}: ERREUR - {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("📊 Résumé des tests:")
    passed = sum(1 for _, success in results if success)
    total = len(results)
    print(f"Tests réussis: {passed}/{total}")
    
    if passed == total:
        print("🎉 Tous les tests sont passés avec succès !")
    else:
        print("⚠️ Certains tests ont échoué. Vérifiez le backend.")

if __name__ == "__main__":
    main() 