# 🔧 Dépannage BERT - Erreur 404 "Modèle non trouvé"

## 🚨 Problème Identifié

**Erreur** : `Failed to load resource: the server responded with a status of 404 (NOT FOUND)`
**Message** : `Erreur prédiction BERT: Error: Modèle non trouvé`

## 🔍 Cause du Problème

L'erreur 404 se produit quand l'interface essaie d'utiliser un modèle BERT qui n'existe plus sur le serveur. Cela peut arriver dans plusieurs cas :

1. **Redémarrage du serveur** : Les modèles en mémoire sont perdus
2. **ID de modèle incorrect** : L'interface référence un modèle avec un mauvais ID
3. **Cache navigateur** : L'interface garde en mémoire des modèles obsolètes
4. **Modèle supprimé** : Le modèle a été supprimé du serveur

## ✅ Solutions Rapides

### 1. **Actualiser la Liste des Modèles**
- Dans l'interface BERT Training, cliquez sur le bouton **🔄 Rafraîchir** à côté de "Modèles BERT"
- Cela recharge la liste des modèles disponibles depuis le serveur

### 2. **Entraîner un Nouveau Modèle**
Si aucun modèle n'est disponible :
1. Vérifiez que le **Backend Python** est **"En ligne"** (voyant vert)
2. Configurez les paramètres BERT selon vos besoins
3. Cliquez sur **"Entraîner BERT"**
4. Attendez la fin de l'entraînement (2-5 minutes)

### 3. **Redémarrer le Backend**
Si le problème persiste :
```bash
# Arrêtez le serveur (Ctrl+C)
# Puis redémarrez
python backend/app.py
```

### 4. **Vider le Cache Navigateur**
- Rechargez la page avec **Ctrl+F5** (Windows) ou **Cmd+Shift+R** (Mac)
- Ou ouvrez les outils développeur (F12) → Onglet Network → Cochez "Disable cache"

## 🛠️ Solutions Avancées

### Vérifier les Modèles Disponibles
Vous pouvez vérifier manuellement quels modèles sont disponibles :
```bash
curl http://localhost:5000/api/models
```

### Nettoyer les Modèles Corrompus
Si vous avez des modèles corrompus dans le dossier `models/bert/` :
```bash
# Supprimer tous les modèles BERT (ATTENTION : perte de données)
rm -rf models/bert/*
```

### Logs du Serveur
Consultez les logs du serveur Flask pour plus de détails :
- Les erreurs 404 apparaissent dans le terminal où vous avez lancé `python backend/app.py`
- Recherchez les lignes contenant `404` ou `predict/bert`

## 🎯 Prévention

### 1. **Utilisation Recommandée**
- **Toujours vérifier** que le backend est "En ligne" avant d'analyser
- **Actualiser la liste** des modèles après un redémarrage du serveur
- **Entraîner un modèle** avant de l'utiliser pour l'analyse

### 2. **Workflow Optimal**
1. Démarrer le backend : `python backend/app.py`
2. Vérifier le statut (voyant vert)
3. Entraîner un modèle BERT
4. Sélectionner le modèle entraîné
5. Analyser les textes

### 3. **Gestion des Modèles**
- Les modèles sont **temporaires** en mémoire par défaut
- Pour une persistance, utilisez la fonctionnalité "Sauvegarder" (si disponible)
- Notez l'**ID du modèle** affiché dans l'interface

## 🔄 Amélioration Continue

### Correctifs Implémentés
- ✅ **Gestion d'erreur améliorée** : L'analyse NLTK continue même si BERT échoue
- ✅ **Messages explicites** : Affichage clair des erreurs avec conseils
- ✅ **Affichage des IDs** : Les IDs de modèles sont maintenant visibles
- ✅ **Indicateur de sélection** : Le modèle actif est clairement identifié

### Fonctionnalités à Venir
- 🔄 **Auto-refresh** des modèles toutes les 30 secondes
- 💾 **Persistance** automatique des modèles entraînés
- 🔍 **Validation** des modèles avant utilisation
- ⚠️ **Alertes proactives** en cas de modèle indisponible

## 📞 Support

Si le problème persiste après avoir essayé ces solutions :

1. **Vérifiez les logs** du serveur Flask
2. **Redémarrez complètement** backend + frontend
3. **Testez avec NLTK seulement** (sans sélectionner de modèle BERT)
4. **Entraînez un nouveau modèle** avec des paramètres simples :
   - Model: DistilBERT (Rapide)
   - Epochs: 2
   - Batch Size: 8

---

🎯 **L'objectif est de rendre l'expérience BERT plus robuste et user-friendly !** 