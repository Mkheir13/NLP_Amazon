#!/usr/bin/env python3
"""
Test complet de toutes les fonctionnalités du site web NLP Amazon
Vérifie que TOUTES les fonctionnalités sont accessibles et fonctionnelles
"""

import requests
import json
import time
import sys

# Configuration
API_BASE = "http://localhost:5000/api"
FRONTEND_URL = "http://localhost:5177"

def test_api_health():
    """Test 1: Vérification de l'état de l'API"""
    print("🔍 Test 1: Vérification de l'état de l'API")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API opérationnelle: {data['message']}")
            print(f"📋 Features: {', '.join(data['features'])}")
            return True
        else:
            print(f"❌ API non accessible: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur connexion API: {e}")
        return False

def test_autoencoder_info():
    """Test 2: Informations du modèle autoencoder"""
    print("\n🔍 Test 2: Informations du modèle autoencoder")
    try:
        response = requests.get(f"{API_BASE}/autoencoder/info")
        if response.status_code == 200:
            data = response.json()
            model_info = data.get('model_info', data)  # Support both structures
            print(f"✅ Modèle entraîné: {model_info.get('is_trained', 'Unknown')}")
            print(f"🔧 Architecture: {model_info.get('architecture', 'Unknown')}")
            config = model_info.get('config', {})
            if config:
                print(f"📊 Configuration: {config.get('input_dim', 'N/A')}D → {config.get('encoding_dim', 'N/A')}D")
            return True
        else:
            print(f"❌ Erreur info modèle: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_standard_training():
    """Test 3: Entraînement standard"""
    print("\n🔍 Test 3: Entraînement standard de l'autoencoder")
    try:
        config = {
            "input_dim": 1000,
            "encoding_dim": 128,
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 16
        }
        
        response = requests.post(f"{API_BASE}/autoencoder/train", json={"config": config})
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            print(f"✅ Entraînement standard réussi")
            print(f"📊 Architecture: {result.get('architecture', 'Unknown')}")
            print(f"📉 Perte finale: {result.get('final_loss', 0):.4f}")
            print(f"🔄 Compression: {result.get('compression_ratio', 1):.1f}:1")
            return True
        else:
            print(f"❌ Erreur entraînement: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_optimized_training():
    """Test 4: Entraînement optimisé (nouvelle fonctionnalité)"""
    print("\n🔍 Test 4: Entraînement optimisé (Data Science)")
    try:
        config = {
            "input_dim": 1000,
            "encoding_dim": 128,
            "learning_rate": 0.001,
            "epochs": 20,
            "batch_size": 16
        }
        
        response = requests.post(f"{API_BASE}/autoencoder/train_optimized", json={"config": config})
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            print(f"✅ Entraînement optimisé réussi")
            print(f"📊 Qualité: {result.get('quality_level', 'Unknown')} ({result.get('quality_score', 0):.3f})")
            print(f"📈 Variance expliquée: {result.get('variance_explained', 0)*100:.1f}%")
            print(f"🎯 Clusters recommandés: {result.get('recommended_k', 'N/A')}")
            return True
        else:
            print(f"❌ Erreur entraînement optimisé: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_model_evaluation():
    """Test 5: Évaluation du modèle (nouvelle fonctionnalité)"""
    print("\n🔍 Test 5: Évaluation avancée du modèle")
    try:
        response = requests.post(f"{API_BASE}/autoencoder/evaluate", json={})
        if response.status_code == 200:
            data = response.json()
            eval_data = data['evaluation']
            print(f"✅ Évaluation réussie")
            print(f"🎯 Score qualité: {eval_data['quality_score']:.3f} ({eval_data['quality_level']})")
            print(f"📊 MSE: {eval_data['mse']:.4f}")
            print(f"📈 Similarité moyenne: {eval_data['mean_similarity']*100:.1f}%")
            print(f"📉 Variance expliquée: {eval_data['variance_explained']*100:.1f}%")
            return True
        else:
            print(f"❌ Erreur évaluation: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_clustering_basic():
    """Test 6: Clustering basique"""
    print("\n🔍 Test 6: Clustering KMeans basique")
    try:
        response = requests.post(f"{API_BASE}/autoencoder/kmeans", json={"n_clusters": 3})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Clustering basique réussi")
            print(f"📊 Score silhouette: {data['result']['silhouette_score']:.3f}")
            print(f"🔢 Nombre de clusters: {data['result']['n_clusters']}")
            print(f"📈 Inertie: {data['result']['inertia']:.2f}")
            return True
        else:
            print(f"❌ Erreur clustering: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_advanced_clustering():
    """Test 7: Analyse avancée du clustering (nouvelle fonctionnalité)"""
    print("\n🔍 Test 7: Analyse avancée du clustering")
    try:
        response = requests.post(f"{API_BASE}/autoencoder/clustering_advanced", json={
            "n_clusters": 3,
            "use_compressed": True
        })
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            clustering = result.get('clustering', result)  # Support nested structure
            print(f"✅ Analyse avancée réussie")
            print(f"📊 Silhouette: {clustering.get('silhouette_score', 0):.3f}")
            print(f"📈 Calinski-Harabasz: {clustering.get('calinski_harabasz', 0):.2f}")
            print(f"📉 Davies-Bouldin: {clustering.get('davies_bouldin', 0):.3f}")
            if 'clusters_analysis' in clustering:
                print(f"🔍 Analyse détaillée: {len(clustering['clusters_analysis'])} clusters analysés")
            return True
        else:
            print(f"❌ Erreur analyse avancée: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_cluster_optimization():
    """Test 8: Optimisation automatique des clusters (nouvelle fonctionnalité)"""
    print("\n🔍 Test 8: Optimisation automatique des clusters")
    try:
        response = requests.post(f"{API_BASE}/autoencoder/optimize_clusters", json={
            "max_clusters": 8,
            "use_compressed": True
        })
        if response.status_code == 200:
            data = response.json()
            result = data['result']
            print(f"✅ Optimisation réussie")
            print(f"🎯 k optimal: {result['recommended_k']}")
            print(f"📊 Meilleur silhouette: {result['best_silhouette_score']:.3f}")
            print(f"🔄 Méthode: {result['recommendation_reason']}")
            if 'elbow_k' in result:
                print(f"📈 Coude détecté: k={result['elbow_k']}")
            return True
        else:
            print(f"❌ Erreur optimisation: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_text_reconstruction():
    """Test 9: Reconstruction de texte"""
    print("\n🔍 Test 9: Reconstruction de texte")
    try:
        test_text = "This product has excellent quality and great value for money!"
        response = requests.post(f"{API_BASE}/autoencoder/reconstruct", json={"text": test_text})
        if response.status_code == 200:
            data = response.json()
            result = data.get('reconstruction', data)  # Support both structures
            print(f"✅ Reconstruction réussie")
            print(f"📝 Texte original: {result.get('original_text', test_text)[:50]}...")
            print(f"📊 Erreur reconstruction: {result.get('reconstruction_error', 0):.4f}")
            print(f"📈 Similarité: {result.get('similarity', 0)*100:.1f}%")
            print(f"🔄 Compression: {result.get('compression_ratio', 1):.1f}:1")
            return True
        else:
            print(f"❌ Erreur reconstruction: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_semantic_search():
    """Test 10: Recherche sémantique"""
    print("\n🔍 Test 10: Recherche sémantique")
    try:
        query = "excellent product quality"
        response = requests.post(f"{API_BASE}/autoencoder/search", json={
            "query": query,
            "top_k": 3
        })
        if response.status_code == 200:
            data = response.json()
            results = data['results']
            print(f"✅ Recherche sémantique réussie")
            print(f"🔍 Requête: {query}")
            print(f"📊 Résultats trouvés: {len(results)}")
            if results:
                best_match = results[0]
                print(f"🎯 Meilleur match: {best_match['similarity']*100:.1f}% de similarité")
                print(f"📝 Texte: {best_match['text_preview'][:50]}...")
            return True
        else:
            print(f"❌ Erreur recherche: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_model_save():
    """Test 11: Sauvegarde de modèle (nouvelle fonctionnalité)"""
    print("\n🔍 Test 11: Sauvegarde de modèle")
    try:
        model_name = f"test_model_{int(time.time())}"
        response = requests.post(f"{API_BASE}/autoencoder/save", json={"filename": model_name})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Sauvegarde réussie")
            print(f"💾 Fichier: {data['filepath']}")
            return True, model_name
        else:
            print(f"❌ Erreur sauvegarde: {response.status_code}")
            return False, None
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False, None

def test_model_load(model_name):
    """Test 12: Chargement de modèle (nouvelle fonctionnalité)"""
    print("\n🔍 Test 12: Chargement de modèle")
    if not model_name:
        print("⚠️ Pas de modèle à charger (sauvegarde échouée)")
        return False
    
    try:
        response = requests.post(f"{API_BASE}/autoencoder/load", json={"filename": model_name})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Chargement réussi")
            print(f"📂 Modèle chargé: {model_name}")
            if 'model_info' in data:
                print(f"📊 Info modèle: {data['model_info']}")
            return True
        else:
            print(f"❌ Erreur chargement: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return False

def test_health():
    """Test de l'endpoint health"""
    print("🔍 Test de l'endpoint health...")
    try:
        response = requests.get(f"{API_BASE}/health")
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
    """Test de l'endpoint models BERT"""
    print("\n🔍 Test de la liste des modèles BERT...")
    try:
        response = requests.get(f"{API_BASE}/models")
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            print(f"✅ Modèles trouvés: {len(models)}")
            for model in models:
                print(f"   - {model.get('id', 'N/A')}: {model.get('name', 'N/A')}")
            return models
        else:
            print(f"❌ Erreur liste modèles: {response.status_code}")
            return []
    except Exception as e:
        print(f"❌ Erreur models: {e}")
        return []

def test_nltk_analysis():
    """Test de l'analyse NLTK avec vérification des scores"""
    print("\n🔍 Test de l'analyse NLTK...")
    test_text = "This product is absolutely amazing! I love it so much, it's perfect!"
    
    try:
        response = requests.post(f"{API_BASE}/analyze/nltk", 
                               json={"text": test_text})
        if response.status_code == 200:
            data = response.json()
            result = data.get('result', {})
            
            print(f"✅ Analyse NLTK réussie")
            print(f"   - Sentiment: {result.get('sentiment', 'N/A')}")
            print(f"   - Confiance: {result.get('confidence', 0):.3f}")
            print(f"   - Polarité: {result.get('polarity', 0):.3f}")
            
            # Vérifications importantes
            confidence = result.get('confidence', 0)
            polarity = result.get('polarity', 0)
            
            if confidence == 0:
                print("⚠️  PROBLÈME: Confiance = 0")
                return False
            if polarity == 0:
                print("⚠️  PROBLÈME: Polarité = 0")
                return False
                
            print("✅ Scores NLTK valides (non-zéro)")
            return True
        else:
            print(f"❌ Erreur NLTK: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur NLTK: {e}")
        return False

def test_bert_prediction(model_id):
    """Test de prédiction BERT avec un modèle spécifique"""
    print(f"\n🔍 Test de prédiction BERT avec modèle {model_id}...")
    test_text = "This is an excellent product, I highly recommend it!"
    
    try:
        response = requests.post(f"{API_BASE}/predict/bert/{model_id}",
                               json={"text": test_text})
        if response.status_code == 200:
            data = response.json()
            prediction = data.get('prediction', {})
            
            print(f"✅ Prédiction BERT réussie")
            print(f"   - Sentiment: {prediction.get('sentiment', 'N/A')}")
            print(f"   - Confiance: {prediction.get('confidence', 0):.3f}")
            print(f"   - Classe: {prediction.get('class', 'N/A')}")
            return True
        elif response.status_code == 404:
            print(f"❌ Erreur 404: Modèle {model_id} non trouvé")
            return False
        else:
            print(f"❌ Erreur BERT: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Détails: {error_data}")
            except:
                pass
            return False
    except Exception as e:
        print(f"❌ Erreur BERT: {e}")
        return False

def test_models_reload():
    """Test de rechargement des modèles"""
    print("\n🔍 Test de rechargement des modèles...")
    try:
        response = requests.post(f"{API_BASE}/models/reload")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Rechargement réussi: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Erreur rechargement: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur rechargement: {e}")
        return False

def test_autoencoder_training():
    """Test de l'entraînement autoencoder"""
    print("\n🔍 Test de l'entraînement autoencoder...")
    try:
        # Test avec des données minimales
        response = requests.post(f"{API_BASE}/autoencoder/train",
                               json={
                                   "num_samples": 50,
                                   "config": {"epochs": 10}
                               })
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Entraînement autoencoder réussi")
            print(f"   - Loss: {data.get('loss', 'N/A')}")
            print(f"   - Accuracy: {data.get('accuracy', 'N/A')}")
            return True
        else:
            print(f"❌ Erreur autoencoder: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Erreur autoencoder: {e}")
        return False

def run_complete_test():
    """Lance tous les tests"""
    print("🚀 DÉBUT DES TESTS COMPLETS - TOUTES LES FONCTIONNALITÉS")
    print("=" * 70)
    
    tests = [
        test_api_health,
        test_autoencoder_info,
        test_standard_training,
        test_optimized_training,
        test_model_evaluation,
        test_clustering_basic,
        test_advanced_clustering,
        test_cluster_optimization,
        test_text_reconstruction,
        test_semantic_search,
    ]
    
    passed = 0
    total = len(tests) + 2  # +2 pour save/load
    
    # Tests standards
    for test in tests:
        try:
            if test():
                passed += 1
            time.sleep(0.5)  # Pause entre les tests
        except Exception as e:
            print(f"❌ Erreur dans {test.__name__}: {e}")
    
    # Tests save/load
    save_success, model_name = test_model_save()
    if save_success:
        passed += 1
    time.sleep(0.5)
    
    if test_model_load(model_name):
        passed += 1
    
    # Résumé final
    print("\n" + "=" * 70)
    print("📊 RÉSUMÉ DES TESTS")
    print(f"✅ Tests réussis: {passed}/{total}")
    print(f"❌ Tests échoués: {total - passed}/{total}")
    print(f"📈 Taux de réussite: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 TOUS LES TESTS SONT PASSÉS!")
        print("🌟 Le site web est entièrement fonctionnel!")
        print(f"🌐 Vous pouvez accéder à l'interface: {FRONTEND_URL}")
        print("\n📋 FONCTIONNALITÉS DISPONIBLES:")
        print("   • Entraînement standard et optimisé")
        print("   • Évaluation avancée de la qualité")
        print("   • Clustering avec optimisation automatique")
        print("   • Analyse avancée des clusters")
        print("   • Reconstruction de texte")
        print("   • Recherche sémantique")
        print("   • Sauvegarde/chargement de modèles")
        print("   • Métriques Data Science complètes")
    else:
        print(f"\n⚠️  {total - passed} tests ont échoué")
        print("🔧 Vérifiez que le backend est démarré sur le port 5000")
        print("🔧 Vérifiez que le frontend est démarré sur le port 5177")
    
    return passed == total

if __name__ == "__main__":
    print("🔍 Vérification des prérequis...")
    print(f"🌐 Backend attendu: {API_BASE}")
    print(f"🌐 Frontend attendu: {FRONTEND_URL}")
    print()
    
    success = run_complete_test()
    sys.exit(0 if success else 1) 