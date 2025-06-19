# Backend NLP - BERT & NLTK Training

Backend Python Flask pour l'entraînement de modèles BERT et l'analyse de sentiment avec NLTK.

## 🚀 Installation

1. **Créer un environnement virtuel** :
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate     # Windows
```

2. **Installer les dépendances** :
```bash
pip install -r requirements.txt
```

3. **Créer les modèles de base** :
```bash
python create_models.py
```

4. **Télécharger les données NLTK** :
```python
import nltk
nltk.download('vader_lexicon')
```

## 🏃‍♂️ Lancement

### Méthode 1 : Script automatique
```bash
# Windows
start_backend.bat

# Linux/Mac
chmod +x start_backend.sh
./start_backend.sh
```

### Méthode 2 : Manuel
```bash
cd backend
python app.py
```

Le serveur démarre sur `http://localhost:5000`

## 📁 Structure des Modèles

Les modèles BERT sont stockés dans `backend/models/` mais ne sont pas inclus dans Git à cause de leur taille (plusieurs GB).

### Recréer les modèles manquants

Si vous clonez le repository et que le dossier `models/` est vide :

```bash
cd backend
python create_models.py
```

Ce script va :
- Télécharger DistilBERT de base depuis Hugging Face
- Créer la structure de dossiers nécessaire
- Générer les fichiers de configuration
- Préparer le modèle pour l'entraînement via l'interface web

## 🔧 API Endpoints

- `GET /api/health` - Status du serveur
- `POST /api/train/bert` - Entraîner un modèle BERT
- `POST /api/analyze/nltk` - Analyser avec NLTK VADER
- `GET /api/models` - Liste des modèles disponibles
- `POST /api/predict/bert` - Prédiction avec BERT

## 📊 Fonctionnalités

### BERT Training
- Support DistilBERT, BERT, RoBERTa
- Configuration personnalisable (epochs, batch size, learning rate)
- Métriques complètes (accuracy, precision, recall, F1)
- Sauvegarde automatique des checkpoints

### NLTK Analysis
- Sentiment analysis avec VADER
- Scores détaillés (positive, negative, neutral, compound)
- Analyse en temps réel

## 🎯 Utilisation

1. **Lancer le backend** : `python app.py`
2. **Lancer le frontend** : `npm run dev` (dans le dossier racine)
3. **Aller sur** : `http://localhost:5173`
4. **Naviguer vers** : "Entraîner Modèles"
5. **Configurer et entraîner** vos modèles !

## 🔍 Dépannage

### Erreur "No models found"
```bash
python create_models.py
```

### Erreur NLTK
```python
import nltk
nltk.download('vader_lexicon')
```

### Erreur de dépendances
```bash
pip install -r requirements.txt
```

## 📚 Technologies

- **Flask** : Serveur web Python
- **Transformers** : Modèles BERT (Hugging Face)
- **NLTK** : Analyse de sentiment VADER
- **PyTorch** : Framework de deep learning
- **Datasets** : Gestion des datasets (Hugging Face)

## 📡 Endpoints API

### Santé du serveur
- **GET** `/api/health` - Vérifier l'état du serveur

### Entraînement BERT
- **POST** `/api/train/bert` - Entraîner un modèle BERT

**Body :**
```json
{
  "data": [
    {"text": "Great product!", "label": 1, "sentiment": "positive"},
    {"text": "Terrible quality", "label": 0, "sentiment": "negative"}
  ],
  "config": {
    "model_name": "distilbert-base-uncased",
    "epochs": 3,
    "batch_size": 8,
    "learning_rate": 2e-5,
    "test_size": 0.2
  }
}
```

### Analyse NLTK
- **POST** `/api/analyze/nltk` - Analyser un texte avec NLTK VADER

**Body :**
```json
{
  "text": "This is a great product!"
}
```

- **POST** `/api/analyze/nltk/batch` - Analyser plusieurs textes

**Body :**
```json
{
  "texts": ["Great product!", "Terrible quality"]
}
```

### Gestion des modèles
- **GET** `/api/models` - Récupérer la liste des modèles entraînés
- **POST** `/api/predict/bert/{model_id}` - Prédiction avec un modèle BERT

## 🔧 Configuration

### Modèles supportés
- `distilbert-base-uncased` (Recommandé pour débuter)
- `bert-base-uncased`
- `roberta-base`

### Paramètres d'entraînement
- **epochs** : Nombre d'époques (1-10)
- **batch_size** : Taille des lots (4-32)
- **learning_rate** : Taux d'apprentissage (1e-5 à 5e-5)
- **test_size** : Pourcentage pour le test (0.1-0.5)

## 📁 Structure des fichiers

```
backend/
├── app.py              # Serveur Flask principal
├── requirements.txt    # Dépendances Python
├── models/            # Modèles BERT entraînés
├── logs/              # Logs d'entraînement
└── README.md          # Ce fichier
```

## ⚠️ Notes importantes

1. **GPU recommandé** : L'entraînement BERT est beaucoup plus rapide avec un GPU
2. **Mémoire** : BERT nécessite au moins 4GB de RAM
3. **Temps d'entraînement** : Peut prendre 10-30 minutes selon le dataset et le matériel
4. **Stockage** : Chaque modèle fait environ 250MB

## 🐛 Dépannage

### Erreur de mémoire
Réduisez le `batch_size` à 4 ou moins.

### Modèle trop lent
Utilisez `distilbert-base-uncased` au lieu de `bert-base-uncased`.

### Erreur CUDA
Si vous n'avez pas de GPU, PyTorch utilisera automatiquement le CPU.

## 📊 Métriques retournées

Après l'entraînement, vous obtenez :
- **Accuracy** : Précision globale
- **Precision** : Précision par classe
- **Recall** : Rappel par classe  
- **F1-Score** : Score F1 pondéré
- **Eval Loss** : Perte sur le set de validation 