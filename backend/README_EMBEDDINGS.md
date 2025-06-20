# 🔗 Module d'Embeddings - NLP Amazon Analysis

## Vue d'ensemble

Le module d'embeddings ajoute des capacités avancées de représentation vectorielle et de recherche sémantique à votre projet NLP Amazon Analysis. Il permet de :

- **Entraîner des modèles Word2Vec** personnalisés sur vos données
- **Visualiser les embeddings** en 2D/3D avec t-SNE, PCA et UMAP
- **Effectuer des recherches sémantiques** dans des collections de textes
- **Analyser les relations sémantiques** entre les mots

## 🚀 Installation

### Dépendances requises

```bash
pip install gensim>=4.3.0
pip install sentence-transformers>=2.2.0
pip install plotly>=5.17.0
pip install umap-learn>=0.5.4
pip install matplotlib>=3.7.0
pip install seaborn>=0.12.0
```

### Installation automatique
Les dépendances sont listées dans `requirements.txt` et s'installent avec :
```bash
pip install -r requirements.txt
```

## 📚 API Endpoints

### Entraînement de modèles

#### `POST /api/embeddings/train/word2vec`
Entraîne un nouveau modèle Word2Vec.

**Paramètres :**
```json
{
  "texts": ["liste", "de", "textes"],
  "config": {
    "vector_size": 100,
    "window": 5,
    "min_count": 2,
    "workers": 4,
    "epochs": 10,
    "sg": 1
  }
}
```

**Réponse :**
```json
{
  "success": true,
  "model": {
    "id": "20241220_143022",
    "type": "word2vec",
    "vocabulary_size": 1500,
    "trained_on": 1000,
    "config": {...}
  }
}
```

### Récupération d'embeddings

#### `GET /api/embeddings/word/{word}`
Obtient l'embedding d'un mot spécifique.

**Paramètres :**
- `model_id` (optionnel) : ID du modèle à utiliser

**Réponse :**
```json
{
  "success": true,
  "word": "good",
  "embedding": [0.1, -0.2, 0.3, ...],
  "dimension": 100
}
```

#### `POST /api/embeddings/sentence`
Obtient l'embedding d'une phrase avec Sentence-BERT.

**Paramètres :**
```json
{
  "text": "This is a sample sentence"
}
```

### Recherche de similarité

#### `GET /api/embeddings/similar/{word}`
Trouve les mots les plus similaires à un mot donné.

**Paramètres :**
- `top_k` : nombre de résultats (défaut: 10)
- `model_id` (optionnel) : ID du modèle

**Réponse :**
```json
{
  "success": true,
  "word": "good",
  "similar_words": [
    ["great", 0.85],
    ["excellent", 0.82],
    ["nice", 0.78]
  ]
}
```

### Recherche sémantique

#### `POST /api/embeddings/search`
Effectue une recherche sémantique dans une collection de textes.

**Paramètres :**
```json
{
  "query": "good product quality",
  "texts": ["list of texts to search"],
  "top_k": 5
}
```

**Réponse :**
```json
{
  "success": true,
  "results": [
    {
      "index": 0,
      "text": "This product has excellent quality",
      "similarity": 0.89,
      "text_preview": "This product has excellent..."
    }
  ]
}
```

### Visualisation

#### `POST /api/embeddings/visualize`
Génère une visualisation 2D des embeddings.

**Paramètres :**
```json
{
  "words": ["good", "bad", "excellent", "terrible"],
  "method": "tsne",
  "model_id": "20241220_143022"
}
```

**Méthodes disponibles :**
- `pca` : Analyse en Composantes Principales
- `tsne` : t-SNE (t-Distributed Stochastic Neighbor Embedding)
- `umap` : UMAP (Uniform Manifold Approximation and Projection)

### Gestion des modèles

#### `GET /api/embeddings/models`
Liste tous les modèles d'embedding disponibles.

#### `GET /api/embeddings/stats`
Obtient les statistiques d'un modèle.

**Paramètres :**
- `model_id` (optionnel) : ID du modèle

## 🎯 Utilisation Frontend

### Composants disponibles

1. **EmbeddingTraining** : Interface pour entraîner des modèles Word2Vec
2. **EmbeddingVisualizer** : Visualisation interactive des embeddings
3. **SemanticSearch** : Interface de recherche sémantique

### Exemple d'utilisation

```tsx
import { EmbeddingService } from '../services/EmbeddingService';

// Entraîner un modèle
const model = await EmbeddingService.trainWord2Vec(texts, {
  vector_size: 100,
  window: 5,
  epochs: 10
});

// Recherche sémantique
const results = await EmbeddingService.semanticSearch(
  "good quality product",
  reviewTexts,
  10
);

// Visualiser des embeddings
const visualization = await EmbeddingService.visualizeEmbeddings(
  ["good", "bad", "excellent"],
  "tsne",
  modelId
);
```

## ⚙️ Configuration

### Paramètres Word2Vec

- **vector_size** : Taille des vecteurs d'embedding (50-300)
- **window** : Taille de la fenêtre contextuelle (3-10)
- **min_count** : Fréquence minimale des mots (1-10)
- **workers** : Nombre de threads parallèles (1-8)
- **epochs** : Nombre d'époques d'entraînement (5-20)
- **sg** : Algorithme (1=Skip-gram, 0=CBOW)

### Recommandations

- **Petits datasets** (<1000 textes) : vector_size=50, window=3, epochs=15
- **Datasets moyens** (1000-10000) : vector_size=100, window=5, epochs=10
- **Gros datasets** (>10000) : vector_size=200, window=7, epochs=5

## 🔧 Architecture technique

### Backend (Python)
- **EmbeddingService** : Service principal pour la gestion des embeddings
- **Gensim** : Entraînement des modèles Word2Vec
- **Sentence-Transformers** : Embeddings de phrases pré-entraînés
- **Plotly** : Génération des visualisations interactives
- **UMAP/t-SNE** : Réduction de dimensionnalité

### Frontend (React/TypeScript)
- **EmbeddingService.ts** : Client API pour les embeddings
- **Composants React** : Interfaces utilisateur interactives
- **Plotly.js** : Rendu des visualisations côté client

## 🎨 Fonctionnalités avancées

### 1. Analyse sémantique complète
```python
analysis = embedding_service.analyze_text_semantics(text)
# Retourne : embedding de phrase, similarités entre mots, densité sémantique
```

### 2. Visualisation interactive
- Zoom et pan sur les graphiques
- Hover pour voir les détails des mots
- Export des visualisations
- Comparaison de différentes méthodes

### 3. Recherche sémantique avancée
- Recherche dans le dataset Amazon (500+ avis)
- Recherche dans des textes personnalisés
- Tri par score de similarité
- Historique des recherches

## 🚨 Dépannage

### Erreurs communes

1. **"Service d'embedding non disponible"**
   - Vérifiez que les dépendances sont installées
   - Redémarrez le backend
   - Vérifiez les logs pour les erreurs d'import

2. **"Aucun modèle Word2Vec disponible"**
   - Entraînez d'abord un modèle via l'interface
   - Vérifiez que le dossier `models/embeddings` existe

3. **"Mot non trouvé dans le vocabulaire"**
   - Le mot n'était pas assez fréquent lors de l'entraînement
   - Réduisez le paramètre `min_count`
   - Entraînez avec plus de données

### Optimisation des performances

1. **Entraînement lent**
   - Augmentez le nombre de `workers`
   - Réduisez `vector_size` ou `epochs`
   - Utilisez CBOW au lieu de Skip-gram

2. **Visualisation lente**
   - Limitez le nombre de mots (<50)
   - Utilisez PCA au lieu de t-SNE pour de gros datasets
   - Réduisez la taille des embeddings

## 📈 Exemples d'utilisation

### Analyse de sentiment avec embeddings
```python
# Entraîner sur les avis Amazon
model = train_word2vec(amazon_reviews)

# Analyser les relations sémantiques
similar_to_good = model.wv.most_similar('good', topn=10)
# ['great', 'excellent', 'amazing', 'wonderful', ...]

# Recherche sémantique
results = semantic_search("poor quality", reviews, top_k=5)
# Trouve les avis parlant de mauvaise qualité même sans les mots exacts
```

### Exploration de domaine
```python
# Visualiser les mots d'un domaine
tech_words = ['phone', 'battery', 'screen', 'camera', 'performance']
visualization = visualize_embeddings(tech_words, method='tsne')
# Voir comment ces concepts se regroupent dans l'espace sémantique
```

## 🔮 Développements futurs

- **Embeddings contextuels** : Intégration de BERT/RoBERTa
- **Embeddings multilingues** : Support de plusieurs langues
- **Clustering automatique** : Détection de groupes sémantiques
- **Comparaison de modèles** : Interface pour comparer différents embeddings
- **Export/Import** : Sauvegarde et partage de modèles

## 📞 Support

Pour toute question ou problème :
1. Consultez les logs du backend
2. Vérifiez la documentation des dépendances
3. Ouvrez une issue sur GitHub avec les détails de l'erreur 