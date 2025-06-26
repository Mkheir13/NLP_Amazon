#!/usr/bin/env python3
"""
Script de test pour les améliorations Data Science
Teste toutes les nouvelles fonctionnalités implémentées
"""

import requests
import json
import time

# Configuration
BASE_URL = "http://localhost:5000"

def test_api_endpoint(endpoint, method="GET", data=None):
    """Teste un endpoint de l'API"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=data)
        
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Erreur {response.status_code}: {response.text}"
    except Exception as e:
        return False, str(e)

def test_data_science_improvements():
    """Test complet des améliorations Data Science"""
    
    print("🧪 Test des Améliorations Data Science")
    print("=" * 50)
    
    # 1. Test de santé de l'API
    print("\n1. 📋 Test de santé de l'API...")
    success, result = test_api_endpoint("/api/health")
    if success:
        print("✅ API opérationnelle")
    else:
        print(f"❌ Erreur API: {result}")
        return
    
    # 2. Test des informations de l'autoencoder
    print("\n2. 🤖 Test des informations autoencoder...")
    success, result = test_api_endpoint("/api/autoencoder/info")
    if success:
        print(f"✅ Configuration autoencoder: {result['model_info']['config']}")
    else:
        print(f"❌ Erreur info autoencoder: {result}")
    
    # 3. Test de l'entraînement avec évaluation complète
    print("\n3. 🎯 Test d'entraînement avec évaluation complète...")
    training_data = {
        "use_optimization": True,
        "config": {
            "epochs": 50,  # Réduire pour le test
            "batch_size": 16
        }
    }
    
    success, result = test_api_endpoint("/api/autoencoder/train", "POST", training_data)
    if success:
        print("✅ Entraînement avec évaluation réussi")
        
        # Afficher les résultats d'évaluation
        if 'result' in result and 'quality_evaluation' in result['result']:
            quality = result['result']['quality_evaluation']
            print(f"   📊 Qualité autoencoder: {quality.get('quality_level', 'N/A')}")
            print(f"   📊 Score qualité: {quality.get('quality_score', 0):.3f}")
            print(f"   📊 Compression: {quality.get('compression_ratio', 0):.1f}x")
        
        if 'result' in result and 'clustering_analysis' in result['result']:
            clustering = result['result']['clustering_analysis']
            if clustering:
                print(f"   🔍 Score silhouette: {clustering.get('silhouette_score', 0):.3f}")
                print(f"   🔍 Interprétation: {clustering.get('silhouette_interpretation', 'N/A')}")
        
    else:
        print(f"❌ Erreur entraînement: {result}")
    
    # 4. Test de l'entraînement optimisé
    print("\n4. ⚡ Test d'entraînement optimisé...")
    success, result = test_api_endpoint("/api/autoencoder/train_optimized", "POST", {})
    if success:
        print("✅ Entraînement optimisé réussi")
        if 'tfidf_stats' in result:
            stats = result['tfidf_stats']
            print(f"   📊 Vocabulaire: {stats.get('vocabulary_size', 0)} mots")
            print(f"   📊 Sparsité: {stats.get('sparsity', 0)*100:.1f}%")
    else:
        print(f"❌ Erreur entraînement optimisé: {result}")
    
    # 5. Test d'évaluation standalone
    print("\n5. 📈 Test d'évaluation standalone...")
    success, result = test_api_endpoint("/api/autoencoder/evaluate", "POST", {})
    if success:
        print("✅ Évaluation standalone réussie")
        evaluation = result['evaluation']
        print(f"   📊 MSE: {evaluation.get('mse', 0):.4f}")
        print(f"   📊 Similarité moyenne: {evaluation.get('mean_similarity', 0):.3f}")
        print(f"   📊 Variance expliquée: {evaluation.get('variance_explained', 0):.3f}")
    else:
        print(f"❌ Erreur évaluation: {result}")
    
    # 6. Test de clustering avancé
    print("\n6. 🔍 Test de clustering avancé...")
    clustering_data = {
        "n_clusters": 4,
        "use_compressed": True
    }
    
    success, result = test_api_endpoint("/api/autoencoder/kmeans", "POST", clustering_data)
    if success:
        print("✅ Clustering avancé réussi")
        clustering_result = result['result']
        print(f"   🔍 Nombre de clusters: {clustering_result.get('n_clusters', 0)}")
        print(f"   🔍 Score silhouette: {clustering_result.get('silhouette_score', 0):.3f}")
        print(f"   🔍 Échantillons: {clustering_result.get('n_samples', 0)}")
        
        # Afficher info sur les clusters
        clusters = clustering_result.get('clusters', [])
        for cluster in clusters[:2]:  # Afficher les 2 premiers clusters
            print(f"   📊 Cluster {cluster['cluster_id']}: {cluster['size']} échantillons ({cluster['percentage']:.1f}%)")
            print(f"      Sentiment: {cluster['sentiment_label']}")
    else:
        print(f"❌ Erreur clustering: {result}")
    
    # 7. Test d'optimisation des clusters
    print("\n7. 🎯 Test d'optimisation des clusters...")
    optimization_data = {
        "max_clusters": 8,
        "use_compressed": True
    }
    
    success, result = test_api_endpoint("/api/autoencoder/optimize_clusters", "POST", optimization_data)
    if success:
        print("✅ Optimisation clusters réussie")
        optimization_result = result['result']
        print(f"   🎯 k recommandé: {optimization_result.get('recommended_k', 0)}")
        print(f"   🎯 Raison: {optimization_result.get('recommendation_reason', 'N/A')}")
        print(f"   🎯 Meilleur score silhouette: {optimization_result.get('best_silhouette_score', 0):.3f}")
    else:
        print(f"❌ Erreur optimisation: {result}")
    
    # 8. Test d'analyse complète
    print("\n8. 🚀 Test d'analyse complète...")
    advanced_data = {
        "n_clusters": 4,
        "use_compressed": True
    }
    
    success, result = test_api_endpoint("/api/autoencoder/clustering_advanced", "POST", advanced_data)
    if success:
        print("✅ Analyse complète réussie")
        analysis_result = result['result']
        
        # Résumé de l'analyse
        summary = analysis_result.get('analysis_summary', {})
        print(f"   🎯 Efficacité compression: {summary.get('compression_efficiency', 0):.1f}x")
        print(f"   🎯 Qualité reconstruction: {summary.get('reconstruction_quality', 'N/A')}")
        print(f"   🎯 Qualité clustering: {summary.get('clustering_quality', 'N/A')}")
        print(f"   🎯 Points de données: {summary.get('data_points', 0)}")
    else:
        print(f"❌ Erreur analyse complète: {result}")
    
    # 9. Test de reconstruction de texte
    print("\n9. 🔄 Test de reconstruction de texte...")
    reconstruction_data = {
        "text": "This product is excellent quality and amazing value"
    }
    
    success, result = test_api_endpoint("/api/autoencoder/reconstruct", "POST", reconstruction_data)
    if success:
        print("✅ Reconstruction de texte réussie")
        reconstruction = result['reconstruction']
        print(f"   🔄 Erreur reconstruction: {reconstruction.get('reconstruction_error', 0):.4f}")
        print(f"   🔄 Similarité: {reconstruction.get('similarity', 0)*100:.1f}%")
        print(f"   🔄 Ratio compression: {reconstruction.get('compression_ratio', 0):.1f}x")
    else:
        print(f"❌ Erreur reconstruction: {result}")
    
    # 10. Test de recherche dans l'espace compressé
    print("\n10. 🔍 Test de recherche sémantique...")
    search_data = {
        "query": "excellent product quality",
        "top_k": 3
    }
    
    success, result = test_api_endpoint("/api/autoencoder/search", "POST", search_data)
    if success:
        print("✅ Recherche sémantique réussie")
        results = result['results']
        print(f"   🔍 Résultats trouvés: {len(results)}")
        for i, res in enumerate(results[:2]):
            print(f"   📄 Résultat {i+1}: Similarité {res['similarity']*100:.1f}%")
    else:
        print(f"❌ Erreur recherche: {result}")
    
    print("\n🎉 Test des améliorations Data Science terminé !")
    print("=" * 50)

if __name__ == "__main__":
    # Attendre que le serveur soit prêt
    print("⏳ Attente du démarrage du serveur...")
    time.sleep(3)
    
    # Lancer les tests
    test_data_science_improvements() 