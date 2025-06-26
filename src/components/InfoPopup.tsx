import React from 'react';
import { X, BookOpen, Target, Lightbulb, Play } from 'lucide-react';

interface InfoPopupProps {
  isOpen: boolean;
  onClose: () => void;
  stepId: string;
}

const InfoPopup: React.FC<InfoPopupProps> = ({ isOpen, onClose, stepId }) => {
  if (!isOpen) return null;

  const getInfoContent = (id: string) => {
    const infoData: Record<string, {
      title: string;
      objective: string;
      explanation: string;
      howToUse: string;
      tips: string[];
      category: string;
    }> = {
      // Exploration de Données
      'home': {
        title: 'Accueil & Guide du Projet',
        objective: 'Découvrir et comprendre le projet NLP Amazon',
        explanation: 'Cette section présente une vue d\'ensemble complète du projet d\'analyse de sentiment sur les avis Amazon. Vous y trouverez l\'architecture générale, les technologies utilisées (BERT, RNN, Autoencoder), et le workflow complet du projet.',
        howToUse: 'Parcourez cette section en premier pour comprendre les objectifs et la structure du projet. Utilisez-la comme référence tout au long de votre apprentissage.',
        tips: [
          'Lisez attentivement l\'introduction pour comprendre le contexte',
          'Familiarisez-vous avec les technologies mentionnées',
          'Revenez à cette section si vous vous perdez dans le projet'
        ],
        category: 'Découverte'
      },
      'explore': {
        title: 'Exploration du Dataset Amazon',
        objective: 'Analyser et comprendre les données d\'avis Amazon',
        explanation: 'Le dataset contient plus de 1000 avis Amazon avec des sentiments positifs et négatifs. Cette étape vous permet d\'explorer la distribution des données, la longueur des textes, les mots les plus fréquents, et de comprendre la structure des données.',
        howToUse: 'Utilisez les graphiques interactifs pour explorer les données. Examinez des exemples d\'avis positifs et négatifs pour comprendre les patterns linguistiques.',
        tips: [
          'Observez la distribution équilibrée entre avis positifs/négatifs',
          'Notez les mots-clés récurrents dans chaque catégorie',
          'Analysez la longueur moyenne des avis pour le preprocessing'
        ],
        category: 'Analyse'
      },

      // Analyse & Sentiment
      'analyze': {
        title: 'Analyseur NLTK + BERT',
        objective: 'Comparer les approches traditionnelles et modernes d\'analyse de sentiment',
        explanation: 'Cette fonctionnalité combine NLTK (approche traditionnelle basée sur des règles) avec BERT (modèle transformer moderne). Vous pouvez analyser n\'importe quel texte et voir comment chaque approche interprète le sentiment.',
        howToUse: 'Saisissez un texte dans le champ prévu, cliquez sur "Analyser" et comparez les résultats NLTK vs BERT. Testez avec différents types de textes pour voir les différences.',
        tips: [
          'Testez des phrases ironiques pour voir les limites de NLTK',
          'Essayez des textes complexes où BERT excelle',
          'Comparez les scores de confiance entre les deux approches'
        ],
        category: 'Analyse Comparative'
      },
      'pipeline': {
        title: 'Pipeline NLP Complet',
        objective: 'Comprendre toutes les étapes du preprocessing NLP',
        explanation: 'Le pipeline NLP montre toutes les étapes de transformation d\'un texte brut : tokenisation, nettoyage, suppression des stop words, lemmatisation, etc. Chaque étape est visualisée en temps réel.',
        howToUse: 'Entrez un texte et observez chaque transformation étape par étape. Chaque phase est expliquée avec des exemples concrets.',
        tips: [
          'Observez comment le texte se transforme à chaque étape',
          'Comprenez l\'impact de chaque preprocessing sur le résultat final',
          'Testez avec des textes contenant des émojis, URLs, etc.'
        ],
        category: 'Preprocessing'
      },
      'results': {
        title: 'Résultats Détaillés',
        objective: 'Analyser en profondeur les performances des modèles',
        explanation: 'Cette section présente une analyse détaillée des résultats de tous les modèles : métriques de performance, matrices de confusion, courbes ROC, et comparaisons entre modèles.',
        howToUse: 'Explorez les différentes métriques, comparez les performances des modèles, et analysez les cas d\'erreur pour comprendre les forces/faiblesses de chaque approche.',
        tips: [
          'Focalisez-vous sur le F1-score pour une évaluation équilibrée',
          'Analysez les faux positifs/négatifs pour comprendre les erreurs',
          'Comparez les temps d\'inférence entre les modèles'
        ],
        category: 'Évaluation'
      },

      // Entraînement Modèles
      'training': {
        title: 'Hub d\'Entraînement Central',
        objective: 'Entraîner et comparer BERT, RNN et Autoencoder',
        explanation: 'Le hub central permet d\'entraîner trois types de modèles différents : BERT (transformer), RNN (réseau récurrent), et Autoencoder (compression). Chaque modèle a ses propres paramètres et objectifs.',
        howToUse: 'Sélectionnez un modèle, configurez les hyperparamètres, lancez l\'entraînement et suivez les métriques en temps réel. Comparez les résultats entre modèles.',
        tips: [
          'Commencez par des époques faibles pour tester rapidement',
          'Surveillez l\'overfitting avec les courbes de validation',
          'Sauvegardez les meilleurs modèles pour les réutiliser'
        ],
        category: 'Machine Learning'
      },
      'simple_autoencoder': {
        title: 'Autoencoder Simplifié',
        objective: 'Apprendre les bases de la compression de données textuelles',
        explanation: 'Version simplifiée de l\'autoencoder pour comprendre les concepts fondamentaux : encodage, espace latent, décodage. Idéal pour débuter avec les autoencoders.',
        howToUse: 'Configurez les dimensions d\'encodage, entraînez le modèle et observez comment il compresse et reconstruit les données textuelles.',
        tips: [
          'Commencez avec des dimensions d\'encodage importantes',
          'Observez l\'erreur de reconstruction pour évaluer la qualité',
          'Testez différentes architectures pour comprendre l\'impact'
        ],
        category: 'Apprentissage'
      },

      // Embeddings & Vectorisation
      'embeddings': {
        title: 'Entraînement d\'Embeddings',
        objective: 'Créer des représentations vectorielles de mots',
        explanation: 'Les embeddings transforment les mots en vecteurs numériques qui capturent leur sens sémantique. Cette section permet d\'entraîner Word2Vec, GloVe, ou d\'utiliser TF-IDF.',
        howToUse: 'Choisissez un type d\'embedding, configurez les paramètres (dimensions, fenêtre de contexte), entraînez et explorez les mots similaires.',
        tips: [
          'Utilisez 100-300 dimensions pour un bon compromis qualité/vitesse',
          'Testez la similarité entre mots pour valider la qualité',
          'Comparez les différents types d\'embeddings sur votre domaine'
        ],
        category: 'Représentation'
      },
      'visualize': {
        title: 'Visualisation d\'Embeddings',
        objective: 'Explorer visuellement l\'espace vectoriel des mots',
        explanation: 'Visualisation 2D/3D des embeddings avec t-SNE ou PCA. Permet de voir comment les mots similaires se regroupent dans l\'espace vectoriel.',
        howToUse: 'Sélectionnez un modèle d\'embedding, choisissez la méthode de réduction de dimension, et explorez interactivement la carte des mots.',
        tips: [
          'Cherchez des clusters de mots sémantiquement liés',
          'Utilisez t-SNE pour une meilleure séparation des clusters',
          'Zoomez sur des régions intéressantes pour explorer en détail'
        ],
        category: 'Visualisation'
      },
      'search': {
        title: 'Recherche Sémantique',
        objective: 'Trouver des documents similaires par le sens, pas par les mots',
        explanation: 'La recherche sémantique utilise les embeddings pour trouver des textes similaires en sens, même s\'ils n\'utilisent pas les mêmes mots. Plus puissant que la recherche par mots-clés.',
        howToUse: 'Entrez une requête, sélectionnez le type d\'embedding, et découvrez les avis Amazon les plus similaires sémantiquement.',
        tips: [
          'Testez des synonymes pour voir la recherche sémantique en action',
          'Comparez avec une recherche par mots-clés classique',
          'Utilisez des requêtes conceptuelles (ex: "déception qualité")'
        ],
        category: 'Recherche'
      },

      // Fonctionnalités Avancées
      'interactive_charts': {
        title: 'Graphiques Interactifs',
        objective: 'Visualiser dynamiquement les données et résultats',
        explanation: 'Graphiques interactifs avancés pour explorer les données : distributions, corrélations, évolution temporelle, comparaisons de modèles. Tous les graphiques sont interactifs et exportables.',
        howToUse: 'Naviguez entre les différents types de graphiques, utilisez les filtres, zoomez sur les zones d\'intérêt, et exportez les visualisations.',
        tips: [
          'Utilisez les filtres pour analyser des sous-ensembles de données',
          'Comparez plusieurs modèles simultanément',
          'Exportez les graphiques pour vos rapports'
        ],
        category: 'Visualisation'
      },
      'word_cloud': {
        title: 'Nuage de Mots Dynamique',
        objective: 'Visualiser la fréquence et l\'importance des mots',
        explanation: 'Génération de nuages de mots interactifs basés sur la fréquence TF-IDF. Permet de voir rapidement les termes les plus importants dans chaque catégorie de sentiment.',
        howToUse: 'Sélectionnez une catégorie (positif/négatif), ajustez les paramètres de filtrage, et explorez le nuage de mots généré.',
        tips: [
          'Comparez les nuages entre sentiments positifs et négatifs',
          'Ajustez le nombre de mots pour plus ou moins de détail',
          'Cliquez sur les mots pour voir leur contexte d\'usage'
        ],
        category: 'Visualisation'
      },
      'sentiment_analyzer': {
        title: 'Analyseur de Sentiment Avancé',
        objective: 'Analyse fine des émotions et sentiments',
        explanation: 'Analyseur multi-modèles qui combine plusieurs approches pour une analyse de sentiment robuste. Détecte non seulement positif/négatif mais aussi l\'intensité et les émotions spécifiques.',
        howToUse: 'Saisissez un texte, sélectionnez les modèles à utiliser, et obtenez une analyse détaillée avec scores de confiance et explications.',
        tips: [
          'Utilisez plusieurs modèles pour une analyse plus robuste',
          'Observez les scores de confiance pour évaluer la fiabilité',
          'Testez des textes ambigus pour voir les nuances détectées'
        ],
        category: 'Analyse Avancée'
      },
      'nlp_pipeline': {
        title: 'Pipeline NLP Avancé',
        objective: 'Workflow complet d\'analyse NLP de bout en bout',
        explanation: 'Pipeline complet qui enchaîne toutes les étapes : preprocessing, extraction de features, modélisation, et post-processing. Configurable et extensible.',
        howToUse: 'Configurez chaque étape du pipeline, chargez vos données, lancez le traitement complet et analysez les résultats finaux.',
        tips: [
          'Sauvegardez vos configurations de pipeline réussies',
          'Testez différentes combinaisons d\'étapes',
          'Surveillez les performances à chaque étape'
        ],
        category: 'Workflow'
      },

      // Code & API
      'code': {
        title: 'Explorateur de Code',
        objective: 'Comprendre l\'implémentation technique de chaque fonctionnalité',
        explanation: 'Code source complet et documenté pour chaque fonctionnalité. Permet de comprendre les détails d\'implémentation et d\'apprendre les bonnes pratiques.',
        howToUse: 'Naviguez entre les différentes sections de code, lisez les commentaires explicatifs, et copiez les extraits utiles pour vos projets.',
        tips: [
          'Lisez les commentaires pour comprendre la logique',
          'Testez les extraits de code dans votre environnement',
          'Adaptez le code à vos propres besoins'
        ],
        category: 'Technique'
      }
    };

    return infoData[id] || {
      title: 'Information non disponible',
      objective: 'Cette fonctionnalité est en cours de documentation.',
      explanation: 'Les détails de cette fonctionnalité seront bientôt disponibles.',
      howToUse: 'Consultez la documentation générale en attendant.',
      tips: ['Revenez plus tard pour plus d\'informations'],
      category: 'En développement'
    };
  };

  const info = getInfoContent(stepId);

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-slate-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto border border-slate-700 shadow-2xl">
        {/* Header */}
        <div className="sticky top-0 bg-slate-800 border-b border-slate-700 p-6 flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="p-2 bg-blue-500/20 rounded-lg">
              <BookOpen className="h-6 w-6 text-blue-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white">{info.title}</h2>
              <span className="text-sm text-blue-400 bg-blue-500/20 px-2 py-1 rounded-full">
                {info.category}
              </span>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 hover:bg-slate-700 rounded-lg transition-colors"
          >
            <X className="h-5 w-5 text-slate-400" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6 space-y-6">
          {/* Objectif */}
          <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/20 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Target className="h-5 w-5 text-green-400" />
              <h3 className="font-semibold text-green-400">Objectif Pédagogique</h3>
            </div>
            <p className="text-slate-300 leading-relaxed">{info.objective}</p>
          </div>

          {/* Explication */}
          <div className="bg-slate-700/50 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Lightbulb className="h-5 w-5 text-yellow-400" />
              <h3 className="font-semibold text-yellow-400">Qu'est-ce que c'est ?</h3>
            </div>
            <p className="text-slate-300 leading-relaxed">{info.explanation}</p>
          </div>

          {/* Comment utiliser */}
          <div className="bg-gradient-to-r from-blue-500/10 to-cyan-500/10 border border-blue-500/20 rounded-xl p-4">
            <div className="flex items-center space-x-2 mb-3">
              <Play className="h-5 w-5 text-blue-400" />
              <h3 className="font-semibold text-blue-400">Comment l'utiliser ?</h3>
            </div>
            <p className="text-slate-300 leading-relaxed">{info.howToUse}</p>
          </div>

          {/* Conseils */}
          <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-4">
            <h3 className="font-semibold text-purple-400 mb-3">💡 Conseils & Astuces</h3>
            <ul className="space-y-2">
              {info.tips.map((tip, index) => (
                <li key={index} className="flex items-start space-x-2">
                  <div className="w-1.5 h-1.5 bg-purple-400 rounded-full mt-2 flex-shrink-0" />
                  <span className="text-slate-300 text-sm leading-relaxed">{tip}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Footer */}
        <div className="border-t border-slate-700 p-4 bg-slate-800/50">
          <div className="flex items-center justify-between">
            <p className="text-xs text-slate-500">
              Cliquez sur "Fermer" ou appuyez sur Échap pour continuer
            </p>
            <button
              onClick={onClose}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm font-medium"
            >
              Fermer
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InfoPopup; 